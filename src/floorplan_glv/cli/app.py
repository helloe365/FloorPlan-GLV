from __future__ import annotations

import json
import os
import tempfile
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, cast

import numpy as np
import torch
import typer
import yaml
from PIL import Image

from floorplan_glv import __version__
from floorplan_glv.config.load import load_config
from floorplan_glv.config.models import AppConfig, ConfigurationError
from floorplan_glv.data import dataset as dataset_module
from floorplan_glv.data import output_schema
from floorplan_glv.data.annotation_schema import AnnotationError
from floorplan_glv.data.audit import inspect_dataset
from floorplan_glv.data.collate import ModelBatch
from floorplan_glv.data.index import prepare_cubicasa_dataset
from floorplan_glv.evaluation import (
    Stage1EvaluationError,
    Stage1EvaluationOptions,
    WeightSelection,
    write_stage1_artifacts,
)
from floorplan_glv.evaluation import (
    evaluate_stage1 as evaluate_stage1_masks,
)
from floorplan_glv.geometry.primitives import GeometryError
from floorplan_glv.metrics.geometry import GeometryMetricReport, evaluate_geometry
from floorplan_glv.models import types as model_types
from floorplan_glv.models.acceptance_stub import (
    ACCEPTANCE_IMPLEMENTATION,
    build_acceptance_model,
)
from floorplan_glv.models.encoders import CheckpointError
from floorplan_glv.models.model import FloorPlanGLV
from floorplan_glv.postprocess import tiled_merge
from floorplan_glv.postprocess.openings import decode_openings
from floorplan_glv.postprocess.pipeline import build_floorplan_result, export_result
from floorplan_glv.postprocess.wall_graph import build_wall_graph
from floorplan_glv.training import checkpoint as checkpoint_io
from floorplan_glv.training.engine import seed_everything
from floorplan_glv.training.runner import run_training
from floorplan_glv.visualization.render import write_debug_visualizations

app = typer.Typer(no_args_is_help=True)
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_ACCEPTANCE_CHECKPOINT = _PROJECT_ROOT / "tests/assets/stub_checkpoint.pt"
_ACCEPTANCE_CHECKPOINT_SHA256 = (
    "2849064edece5a2f0802fcf82443eae9a18324414e075a2f0df5762dd68af8bc"
)

ConfigPath = Annotated[Path, typer.Option(exists=True, dir_okay=False, readable=True)]
ReadableFile = Annotated[Path, typer.Option(exists=True, dir_okay=False, readable=True)]
OutputDirectory = Annotated[Path, typer.Option("--output", file_okay=False)]
EvaluateConfigPath = Annotated[
    Path | None,
    typer.Option(
        "--config",
        exists=True,
        dir_okay=False,
        readable=True,
        help="Validated application configuration for evaluation.",
    ),
]
EvaluateCheckpoint = Annotated[
    Path | None,
    typer.Option(
        "--checkpoint",
        exists=True,
        dir_okay=False,
        readable=True,
        help="Optional checkpoint to validate with the evaluation configuration.",
    ),
]
EvaluateResult = Annotated[
    Path | None,
    typer.Option(exists=True, dir_okay=False, readable=True),
]
IndexFile = Annotated[
    Path, typer.Option("--index", exists=True, dir_okay=False, readable=True)
]
WaivedUnresolvedSources = Annotated[
    list[str] | None,
    typer.Option(
        "--waive-unresolved-host-source",
        help="Explicit source whose unresolved hosts are excluded from the gate.",
    ),
]


@app.callback()
def main(
    version: Annotated[bool, typer.Option("--version", is_eager=True)] = False,
) -> None:
    """FloorPlan-GLV command group."""
    if version:
        typer.echo(__version__)
        raise typer.Exit


def _acceptance_checkpoint_authorized(checkpoint: Path) -> bool:
    """Return whether ``checkpoint`` is the immutable repository acceptance fixture."""
    return _same_file(checkpoint, _ACCEPTANCE_CHECKPOINT) and (
        checkpoint_io.file_sha256(checkpoint) == _ACCEPTANCE_CHECKPOINT_SHA256
    )


def _same_file(left: Path, right: Path) -> bool:
    """Compare existing files by identity without rewriting the user path."""
    try:
        return left.resolve().samefile(right)
    except OSError:
        return False


def _validate_ema_state(
    checkpoint: Path, raw: Mapping[str, object], model: FloorPlanGLV
) -> None:
    """Validate exact EMA state keys, tensor shapes, and dtypes before loading."""
    ema_state = raw.get("ema_state")
    shadow = ema_state.get("shadow") if isinstance(ema_state, Mapping) else None
    if not isinstance(shadow, Mapping):
        raise CheckpointError(f"checkpoint {checkpoint} lacks EMA model tensors")
    expected_state = model.state_dict()
    missing = set(expected_state) - set(shadow)
    extra = set(shadow) - set(expected_state)
    if missing or extra:
        raise CheckpointError(
            f"checkpoint {checkpoint} EMA tensor keys differ; "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )
    for name, expected in expected_state.items():
        value = shadow[name]
        if not isinstance(value, torch.Tensor):
            raise CheckpointError(
                f"checkpoint {checkpoint} EMA tensor {name} is not a tensor"
            )
        contracts = (
            ("shape", tuple(value.shape), tuple(expected.shape)),
            ("dtype", value.dtype, expected.dtype),
        )
        for contract, actual, wanted in contracts:
            if actual != wanted:
                raise CheckpointError(
                    f"checkpoint {checkpoint} EMA tensor {name} {contract} differs: "
                    f"checkpoint={actual}, model={wanted}"
                )
    try:
        model.load_state_dict(cast(Mapping[str, torch.Tensor], shadow), strict=True)
    except RuntimeError as exc:
        raise CheckpointError(
            f"checkpoint {checkpoint} EMA state is incompatible: {exc}"
        ) from exc


def _checkpoint_model(
    checkpoint: Path,
    config: AppConfig,
) -> tuple[FloorPlanGLV, bool]:
    raw = checkpoint_io._read_checkpoint(checkpoint, map_location="cpu")
    resolved = raw.get("resolved_config")
    checkpoint_model = resolved.get("model") if isinstance(resolved, Mapping) else None
    if not isinstance(checkpoint_model, Mapping):
        raise CheckpointError(f"checkpoint {checkpoint} lacks resolved model config")
    requested = config.model.model_dump(mode="json")
    differences = [
        f"model.{key}: checkpoint={checkpoint_model.get(key)}, "
        f"requested={requested.get(key)}"
        for key in sorted(set(checkpoint_model) | set(requested))
        if checkpoint_model.get(key) != requested.get(key)
    ]
    if differences:
        raise CheckpointError(
            "checkpoint architecture/config mismatch:\n  " + "\n  ".join(differences)
        )
    metadata = raw.get("run_metadata")
    has_acceptance_marker = (
        isinstance(metadata, Mapping)
        and metadata.get("inference_implementation") == ACCEPTANCE_IMPLEMENTATION
    )
    is_acceptance_path = _same_file(checkpoint, _ACCEPTANCE_CHECKPOINT)
    is_acceptance = _acceptance_checkpoint_authorized(checkpoint)
    if (is_acceptance_path and not is_acceptance) or (
        has_acceptance_marker != is_acceptance
    ):
        raise CheckpointError(
            "acceptance checkpoint is only authorized at "
            f"{_ACCEPTANCE_CHECKPOINT} with SHA256 "
            f"{_ACCEPTANCE_CHECKPOINT_SHA256}"
        )
    model = (
        build_acceptance_model(config.model)
        if is_acceptance
        else FloorPlanGLV(config.model)
    )
    _validate_ema_state(checkpoint, raw, model)
    return model, is_acceptance


def _validate_evaluation_checkpoint(checkpoint: Path, config: AppConfig) -> None:
    """Validate checkpoint metadata and EMA compatibility for CLI evaluation."""
    raw = checkpoint_io._read_checkpoint(checkpoint, map_location="cpu")
    resolved = raw.get("resolved_config")
    checkpoint_model = resolved.get("model") if isinstance(resolved, Mapping) else None
    if not isinstance(checkpoint_model, Mapping):
        raise CheckpointError(f"checkpoint {checkpoint} lacks resolved model config")
    requested = config.model.model_dump(mode="json")
    differences = [
        f"model.{key}: checkpoint={checkpoint_model.get(key)}, "
        f"requested={requested.get(key)}"
        for key in sorted(set(checkpoint_model) | set(requested))
        if checkpoint_model.get(key) != requested.get(key)
    ]
    if differences:
        raise CheckpointError(
            "checkpoint architecture/config mismatch:\n  " + "\n  ".join(differences)
        )
    metadata = raw.get("run_metadata")
    has_acceptance_marker = (
        isinstance(metadata, Mapping)
        and metadata.get("inference_implementation") == ACCEPTANCE_IMPLEMENTATION
    )
    is_acceptance_path = _same_file(checkpoint, _ACCEPTANCE_CHECKPOINT)
    is_acceptance = _acceptance_checkpoint_authorized(checkpoint)
    if (is_acceptance_path and not is_acceptance) or (
        has_acceptance_marker != is_acceptance
    ):
        raise CheckpointError(
            "acceptance checkpoint is only authorized at "
            f"{_ACCEPTANCE_CHECKPOINT} with SHA256 "
            f"{_ACCEPTANCE_CHECKPOINT_SHA256}"
        )
    model_state = raw.get("model_state")
    ema_state = raw.get("ema_state")
    shadow = ema_state.get("shadow") if isinstance(ema_state, Mapping) else None
    if not isinstance(model_state, Mapping) or not isinstance(shadow, Mapping):
        raise CheckpointError(f"checkpoint {checkpoint} lacks model and EMA tensors")
    if not all(isinstance(name, str) for name in model_state) or not all(
        isinstance(name, str) for name in shadow
    ):
        raise CheckpointError(f"checkpoint {checkpoint} tensor keys must be strings")
    if not model_state or set(model_state) != set(shadow):
        raise CheckpointError(
            f"checkpoint {checkpoint} model and EMA tensor keys are incompatible"
        )
    for name in sorted(model_state):
        model_tensor = model_state[name]
        ema_tensor = shadow[name]
        if not isinstance(model_tensor, torch.Tensor) or not isinstance(
            ema_tensor, torch.Tensor
        ):
            raise CheckpointError(
                f"checkpoint {checkpoint} model and EMA tensor {name} are not tensors"
            )
        if model_tensor.shape != ema_tensor.shape:
            raise CheckpointError(
                f"checkpoint {checkpoint} model and EMA tensor {name} shapes differ"
            )
        if model_tensor.dtype != ema_tensor.dtype:
            raise CheckpointError(
                f"checkpoint {checkpoint} model and EMA tensor {name} dtypes differ"
            )
    if is_acceptance:
        _checkpoint_model(checkpoint, config)


def _load_source_image(path: Path, config: AppConfig) -> Image.Image:
    try:
        with Image.open(path) as encoded:
            image = dataset_module._decode_rgb(encoded)
    except OSError as exc:
        raise AnnotationError(f"failed to decode input image {path}: {exc}") from exc
    if min(image.size) < (minimum := config.data.minimum_image_size):
        raise AnnotationError(f"input image {path} smaller than {minimum} x {minimum}")
    return image


def _prediction_batches(
    image: Image.Image,
    global_image: torch.Tensor,
    transform: dataset_module.ImageTransform,
    grid: tiled_merge.PatchGrid,
    config: AppConfig,
    model: FloorPlanGLV,
    device: torch.device,
) -> tuple[tiled_merge.PatchOutput, ...]:
    patch_size = config.model.patch_size
    output_size = patch_size // config.model.output_stride
    minibatch = config.train.patches_per_image
    global_batch = global_image.unsqueeze(0).to(device)
    cached_global: model_types.FeaturePyramid | None = None
    if config.model.global_enabled:
        if model.global_encoder is None:
            raise CheckpointError("global model checkpoint lacks a global encoder")
        cached_global = cast(
            model_types.FeaturePyramid, model.global_encoder(global_batch)
        )
    outputs: list[tiled_merge.PatchOutput] = []
    for offset in range(0, len(grid), minibatch):
        placements = grid.placements[offset : offset + minibatch]
        boxes = [
            (item.x, item.y, item.x + patch_size, item.y + patch_size)
            for item in placements
        ]
        local = torch.stack(
            [
                dataset_module._normalize_image(
                    dataset_module._crop_with_padding(
                        image,
                        box,
                        padding_value=config.model.padding_value,
                    )
                )
                for box in boxes
            ]
        ).to(device)
        valid_masks = torch.stack(
            [
                _valid_patch_mask(item.valid_width, item.valid_height, output_size)
                for item in placements
            ]
        ).to(device)
        batch = ModelBatch(
            global_images=global_batch,
            local_patches=local,
            patch_to_image=torch.zeros(
                len(placements), dtype=torch.int64, device=device
            ),
            patch_boxes_global_xyxy=torch.tensor(
                [transform.map_box(box) for box in boxes],
                dtype=torch.float32,
                device=device,
            ),
            patch_valid_masks=valid_masks,
            targets={},
        )
        prediction = _forward_prediction(model, batch, cached_global)
        for index in range(len(placements)):
            raw_prediction = cast(Mapping[str, torch.Tensor], prediction)
            per_patch = cast(
                model_types.FloorPlanModelOutput,
                {
                    name: value[index].detach().to("cpu")
                    for name, value in raw_prediction.items()
                },
            )
            outputs.append(
                tiled_merge.PatchOutput(per_patch, valid_masks[index].to("cpu"))
            )
    return tuple(outputs)


def _forward_prediction(
    model: FloorPlanGLV,
    batch: ModelBatch,
    cached_global: model_types.FeaturePyramid | None,
) -> model_types.FloorPlanModelOutput:
    features = cast(
        model_types.FeaturePyramid, model.local_encoder(batch.local_patches)
    )
    if cached_global is not None:
        features = model_types.FeaturePyramid(
            *[
                fusion(
                    local,
                    global_feature,
                    batch.patch_to_image,
                    batch.patch_boxes_global_xyxy,
                )
                for fusion, local, global_feature in zip(
                    model.fusion_blocks, features, cached_global, strict=True
                )
            ]
        )
    shared = model.decoder(features, batch.local_patches)
    return cast(model_types.FloorPlanModelOutput, model.heads(shared))


def _valid_patch_mask(width: int, height: int, output_size: int) -> torch.Tensor:
    starts = torch.arange(output_size, dtype=torch.float32) * 2.0
    return ((starts[:, None] < height) & (starts[None, :] < width)).unsqueeze(0)


def _write_prediction_maps(path: Path, maps: tiled_merge.FullResolutionMaps) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        raw_maps = cast(Mapping[str, torch.Tensor], maps)
        with os.fdopen(descriptor, "wb") as handle:
            with zipfile.ZipFile(handle, "w") as archive:
                for name in sorted(raw_maps):
                    info = zipfile.ZipInfo(
                        f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0)
                    )
                    info.compress_type = zipfile.ZIP_DEFLATED
                    with archive.open(info, "w", force_zip64=True) as member:
                        np.lib.format.write_array(
                            member, raw_maps[name].detach().cpu().numpy()
                        )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _run_prediction(
    config_path: Path,
    checkpoint: Path,
    input_path: Path,
    output: Path,
) -> Path:
    config = load_config(config_path)
    model, is_acceptance = _checkpoint_model(checkpoint, config)
    image = _load_source_image(input_path, config)
    seed_everything(config.train.seed, deterministic_algorithms=True)
    device = torch.device(
        "cpu" if is_acceptance or not torch.cuda.is_available() else "cuda"
    )
    model.to(device).eval()
    global_image, transform = dataset_module._prepare_global_image(
        image,
        global_size=config.model.global_size,
        padding_value=config.model.padding_value,
    )
    grid = tiled_merge.PatchGrid(
        image.size, config.model.patch_size, config.model.stride
    )
    with torch.inference_mode():
        patch_outputs = _prediction_batches(
            image, global_image, transform, grid, config, model, device
        )
    maps = tiled_merge.merge_predictions(
        grid,
        patch_outputs,
        image.size,
        hann_weight_floor=config.postprocess.hann_weight_floor,
    )
    graph = build_wall_graph(maps, config.postprocess.wall)
    openings = decode_openings(maps, graph, config.postprocess)
    metadata = checkpoint_io.build_run_metadata(
        seed=config.train.seed,
        dataset_indexes=(),
        checkpoint=checkpoint,
        postprocess_config=config.postprocess.model_dump(mode="json"),
        deterministic_algorithms=True,
    )
    source_sha256 = checkpoint_io.file_sha256(input_path)
    metadata["source_image_sha256"] = source_sha256
    metadata["patch_count"] = len(grid)
    metadata["patch_batch_size"] = config.train.patches_per_image
    model_name = "global-local" if config.model.global_enabled else "local"
    result = build_floorplan_result(
        graph,
        openings,
        image=output_schema.ImageInfo(
            file_name=input_path.name,
            width_px=image.width,
            height_px=image.height,
            sha256=source_sha256,
        ),
        metadata=output_schema.RunMetadata(
            model_name=f"floorplan-glv-{model_name}-b4",
            checkpoint_sha256=cast(str, metadata["checkpoint_sha256"]),
            postprocess_config_sha256=cast(str, metadata["postprocess_config_sha256"]),
        ),
    )
    export_result(result, output)
    write_debug_visualizations(
        np.asarray(image, dtype=np.uint8),
        result,
        output,
        dropped_openings=[item for item in openings if item.drop_reason is not None],
    )
    _write_prediction_maps(output / "prediction_maps.npz", maps)
    checkpoint_io.atomic_write_text(
        output / "resolved_config.yaml",
        yaml.safe_dump(config.model_dump(mode="json"), sort_keys=True),
    )
    checkpoint_io.atomic_write_json(output / "run_metadata.json", metadata)
    return output / "result.json"


@app.command("prepare-cubicasa")
def prepare_cubicasa(config: ConfigPath) -> None:
    """Convert CubiCasa samples into normalized annotations."""
    try:
        report = prepare_cubicasa_dataset(config)
    except (AnnotationError, ConfigurationError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(
        f"Converted {report.converted_samples}/{report.discovered_samples} samples; "
        f"rejected {report.rejected_objects} objects and "
        f"{report.rejected_samples} samples."
    )


@app.command("inspect-dataset")
def inspect_dataset_command(
    index: IndexFile,
    output: OutputDirectory,
    waive_unresolved_host_source: WaivedUnresolvedSources = None,
) -> None:
    """Audit a normalized dataset without modifying its source files."""
    try:
        report = inspect_dataset(
            index,
            output,
            waived_unresolved_host_sources=tuple(waive_unresolved_host_source or ()),
        )
    except (AnnotationError, OSError) as exc:
        typer.echo(f"{index}: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    failed = [name for name, gate in report.hard_gates.items() if not gate.passed]
    status = "PASS" if not failed else f"FAIL ({', '.join(failed)})"
    typer.echo(
        f"{index}: audited {report.indexed_samples} samples; hard gates: {status}; "
        f"report: {output / 'audit.json'}"
    )
    if failed:
        raise typer.Exit(code=1)


@app.command("train")
def train(config: ConfigPath) -> None:
    """Train or exactly resume a configured FloorPlan-GLV stage."""
    try:
        checkpoint = run_training(config)
    except torch.cuda.OutOfMemoryError as exc:
        typer.echo(
            f"{config}: CUDA out of memory. Reduce patches_per_image or "
            "source_images_per_gpu and preserve the effective batch with "
            f"gradient_accumulation_steps. Details: {exc}",
            err=True,
        )
        raise typer.Exit(code=1) from exc
    except (AnnotationError, CheckpointError, ConfigurationError, RuntimeError) as exc:
        typer.echo(f"{config}: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Training complete: {checkpoint}")


@app.command("predict")
def predict(
    config: ConfigPath,
    checkpoint: ReadableFile,
    input: ReadableFile,
    output: OutputDirectory,
) -> None:
    """Recognize one raster floor plan and write V1 artifacts."""
    try:
        result = _run_prediction(config, checkpoint, input, output)
    except (
        AnnotationError,
        CheckpointError,
        ConfigurationError,
        GeometryError,
        output_schema.JsonContractError,
        OSError,
        RuntimeError,
    ) as exc:
        typer.echo(f"{input}: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Prediction complete: {result}")


@app.command("validate-json")
def validate_json(
    result: Annotated[Path, typer.Argument(exists=True, dir_okay=False, readable=True)],
) -> None:
    """Validate a result against the exact V1 JSON contract."""
    try:
        output_schema.validate_result_json(result)
    except output_schema.JsonContractError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(f"Valid FloorPlanResult: {result}")


@app.command("evaluate")
def evaluate(
    config: EvaluateConfigPath = None,
    checkpoint: EvaluateCheckpoint = None,
    prediction: EvaluateResult = None,
    ground_truth: EvaluateResult = None,
) -> None:
    """Evaluate validated geometry or preflight an evaluation configuration."""
    report: GeometryMetricReport | dict[str, object]
    try:
        resolved_config: AppConfig | None = None
        if config is not None:
            resolved_config = load_config(config)
            if checkpoint is None:
                raise ConfigurationError("--checkpoint is required with --config")
            _validate_evaluation_checkpoint(checkpoint, resolved_config)
        if (prediction is None) != (ground_truth is None):
            raise ConfigurationError(
                "--prediction and --ground-truth must be provided together"
            )
        if prediction is not None and ground_truth is not None:
            predicted_result = output_schema.validate_result_json(prediction)
            target_result = output_schema.validate_result_json(ground_truth)
            report = evaluate_geometry(predicted_result, target_result)
        elif config is not None:
            report = {
                "status": "evaluation_ready",
                "config": str(config),
                "checkpoint": str(checkpoint),
            }
        else:
            raise ConfigurationError(
                "evaluate requires --config or a prediction/ground-truth pair"
            )
    except (
        CheckpointError,
        ConfigurationError,
        GeometryError,
        output_schema.JsonContractError,
    ) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    payload = report if isinstance(report, dict) else report.as_dict()
    typer.echo(json.dumps(payload, sort_keys=True, separators=(",", ":")))


@app.command("evaluate-stage1")
def evaluate_stage1_command(
    config: ConfigPath,
    checkpoint: ReadableFile,
    output: OutputDirectory,
    weights: Annotated[str, typer.Option("--weights")] = "both",
    wall_threshold: Annotated[float, typer.Option("--wall-threshold")] = 0.45,
    opening_threshold: Annotated[float, typer.Option("--opening-threshold")] = 0.50,
    threshold_min: Annotated[float, typer.Option("--threshold-min")] = 0.30,
    threshold_max: Annotated[float, typer.Option("--threshold-max")] = 0.70,
    threshold_step: Annotated[float, typer.Option("--threshold-step")] = 0.05,
    visual_samples: Annotated[int, typer.Option("--visual-samples")] = 12,
) -> None:
    """Evaluate deterministic epoch-0 Stage 1 validation patches."""
    try:
        options = Stage1EvaluationOptions(
            weights=cast(WeightSelection, weights),
            wall_threshold=wall_threshold,
            opening_threshold=opening_threshold,
            threshold_min=threshold_min,
            threshold_max=threshold_max,
            threshold_step=threshold_step,
            visual_samples=visual_samples,
        )
        result = evaluate_stage1_masks(
            config,
            checkpoint,
            output,
            options=options,
        )
        artifact_paths = write_stage1_artifacts(result, output)
    except (
        Stage1EvaluationError,
        ConfigurationError,
        AnnotationError,
        CheckpointError,
        OSError,
    ) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    receipt = {
        "artifacts": {
            name: str(artifact_paths[name])
            for name in ("summary", "patch_metrics", "threshold_sweep")
        },
        "output": str(output),
        "status": "complete",
        "weights": options.weights,
    }
    typer.echo(
        json.dumps(receipt, sort_keys=True, allow_nan=False, separators=(",", ":"))
    )
