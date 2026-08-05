# FloorPlan-GLV V1

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

Evaluate a checkpoint or a pair of validated geometry results:

```bash
floorplan-glv evaluate --config configs/train/stage3_global_local.yaml --checkpoint /path/to/best.pt
floorplan-glv evaluate --prediction runs/predict/sample/result.json --ground-truth /path/to/ground_truth.json
```

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
