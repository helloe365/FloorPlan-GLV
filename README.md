# FloorPlan-GLV V1

[中文 README](README_ZH.md)

FloorPlan-GLV is an engineering-oriented research implementation for recognizing walls, doors, and windows in raster floor plans. It converts PNG or JPEG floor plans into deterministic, pixel-space geometry and debug artifacts.

## Scope

V1 uses a global-local SegFormer pipeline with MiT-B2 and MiT-B4 branches, gated ROI fusion, multi-task prediction heads, and deterministic Python/OpenCV/Shapely postprocessing.

The system produces wall probability maps, wall centerline and junction maps, wall orientation and thickness fields, opening maps and attributes, a deterministic wall graph, and schema-validated pixel-coordinate JSON.

V1 does not infer real-world scale and does not export CAD, BIM, DXF, Revit, or 3D data. It also excludes OCR, furniture recognition, room-type classification, door swing direction, curved-wall primitives, autoregressive decoding, graph Transformer refinement, VLM correction, and mandatory super-resolution.

## Requirements and installation

- Python 3.11 or newer.
- PyTorch `>=2.5,<3`.
- A CUDA-specific PyTorch wheel installed separately on GPU machines.
- The project dependencies declared in `pyproject.toml`.

Install the project and development dependencies with:

```bash
python -m pip install -e ".[dev]"
```

## Configuration

Configuration is validated with Pydantic. The main configuration files are:

- `configs/data/cubicasa.yaml`: source paths, conversion settings, and split proportions.
- `configs/model/local_b4.yaml`: local model settings.
- `configs/model/global_local_b4.yaml`: approved global-local model settings.
- `configs/postprocess/default.yaml`: postprocessing defaults.
- `configs/train/*.yaml`: ordered training-stage settings.

Image sizes, thresholds, NMS radii, geometric tolerances, loss weights, sampling ratios, batching, precision, and output switches are configuration values. Geometry remains in source-image pixels throughout the model, postprocessor, and JSON exporter.

## Commands

Prepare and audit a CubiCasa dataset:

```bash
floorplan-glv prepare-cubicasa --config configs/data/cubicasa.yaml
floorplan-glv inspect-dataset --index data/processed/index.jsonl --output runs/data_audit
```

Train the ordered stages:

```bash
floorplan-glv train --config configs/train/stage1_local_masks.yaml
floorplan-glv train --config configs/train/stage2_local_geometry.yaml
floorplan-glv train --config configs/train/stage3_global_local.yaml
floorplan-glv train --config configs/train/stage4_domain_finetune.yaml
```

### Early stopping and checkpoints

Each training epoch validates the EMA weights. `best.pt` is selected by the
lowest EMA `validation.total`; use it for inference and as the next stage's
`initial_checkpoint`. `last.pt` retains the most recently completed training
state and is for `resume_checkpoint` only. The approved policy monitors
`validation.total` in `min` mode with a relative improvement threshold of
`0.001`.

| Stage | Patience | Minimum epochs |
| --- | ---: | ---: |
| 1: local masks | 8 | 10 |
| 2: local geometry | 8 | 10 |
| 3: global-local | 8 | 10 |
| 4: domain fine-tune | 5 | 5 |

Stage 1 uses `runs/train/stage1_local_masks/last.pt` as an
`initial_checkpoint`, so its legacy schema 1.0 state starts a fresh
policy/optimizer. Stages 2–4 initialize from the preceding stage's `best.pt`.
Use `resume_checkpoint` only to continue the same run. A schema 1.0 checkpoint
with validation enabled and early stopping disabled can resume only if the
configured output directory contains the paired `best.pt`; otherwise use it
as an `initial_checkpoint` for a fresh run. The approved stages enable early
stopping, so schema 1.0 checkpoints cannot resume them exactly. They must be
used as `initial_checkpoint` values.

Evaluate a checkpoint or a pair of validated geometry results:

```bash
floorplan-glv evaluate --config configs/train/stage3_global_local.yaml --checkpoint /path/to/best.pt
floorplan-glv evaluate --prediction runs/predict/sample/result.json --ground-truth /path/to/ground_truth.json
```

Evaluate the deterministic epoch-0 validation patches for Stage 1 masks:

```bash
floorplan-glv evaluate-stage1 \
  --config configs/train/stage1_local_masks.yaml \
  --checkpoint runs/train/stage1_local_masks/last.pt \
  --weights both \
  --output runs/eval/stage1
```

The evaluator ignores invalid pixels and writes `summary.json`,
`patch_metrics.jsonl`, `threshold_sweep.json`, and `visualizations/*.png`.
It compares raw `model_state` with `ema_state.shadow`. These metrics are
descriptive, not a hard pass/fail gate; full-image geometry acceptance remains
separate.

Run prediction and validate the exported JSON:

```bash
floorplan-glv predict --config configs/model/global_local_b4.yaml --checkpoint /path/to/best.pt --input /path/to/floorplan.png --output runs/predict/sample
floorplan-glv validate-json runs/predict/sample/result.json
```

## Prediction outputs

A successful prediction writes the following files under the selected output directory:

```text
result.json
overlay.png
wall_graph.png
openings.png
prediction_maps.npz
resolved_config.yaml
run_metadata.json
```

`result.json` uses source-image pixels with a top-left origin, x increasing to the right, and y increasing downward. IDs and serialized arrays are sorted so identical model outputs and postprocessing settings produce deterministic JSON.

## Repository layout

```text
configs/            validated data, model, postprocess, and training YAML
src/floorplan_glv/  CLI, data, models, losses, postprocessing, and metrics
pyproject.toml      package metadata and dependencies
```
