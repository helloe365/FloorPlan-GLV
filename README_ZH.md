# FloorPlan-GLV V1

FloorPlan-GLV 是一个面向工程验证的研究实现，用于从栅格户型图中识别墙体、门和窗。它接收 PNG 或 JPEG 户型图，输出确定性的像素坐标几何结果和调试产物。

## 项目范围

V1 使用全局—局部 SegFormer 网络，由 MiT-B2 和 MiT-B4 分支、门控 ROI 融合、多任务预测头，以及确定性的 Python/OpenCV/Shapely 后处理组成。

系统会生成墙体概率图、墙体中心线和连接点概率图、墙体方向与厚度场、开口概率图及其属性、确定性的墙体图，以及经过契约校验的像素坐标 JSON。

V1 不推断真实世界比例，不输出 CAD、BIM、DXF、Revit 或三维数据，也不包含 OCR、家具识别、房间类型分类、门扇开启方向、曲墙图元、自回归解码、图 Transformer 精修、VLM 修正和强制性的超分辨率阶段。

## 环境与安装

- Python 3.11 或更高版本。
- PyTorch `>=2.5,<3`。
- GPU 机器需单独安装匹配的 CUDA PyTorch wheel。
- 安装 `pyproject.toml` 中声明的项目依赖。

安装项目和开发依赖：

```bash
python -m pip install -e ".[dev]"
```

## 配置

配置由 Pydantic 模型校验。主要配置文件如下：

- `configs/data/cubicasa.yaml`：源路径、转换参数和数据划分比例。
- `configs/model/local_b4.yaml`：局部模型设置。
- `configs/model/global_local_b4.yaml`：批准的全局—局部模型设置。
- `configs/postprocess/default.yaml`：后处理默认值。
- `configs/train/*.yaml`：有顺序的训练阶段设置。

图像尺寸、阈值、NMS 半径、几何容差、损失权重、采样比例、批处理、精度和输出开关都属于配置值。模型、后处理和 JSON 导出始终使用源图像像素坐标。

## 命令

准备并审计 CubiCasa 数据集：

```bash
floorplan-glv prepare-cubicasa --config configs/data/cubicasa.yaml
floorplan-glv inspect-dataset --index data/processed/index.jsonl --output runs/data_audit
```

按顺序训练各阶段：

```bash
floorplan-glv train --config configs/train/stage1_local_masks.yaml
floorplan-glv train --config configs/train/stage2_local_geometry.yaml
floorplan-glv train --config configs/train/stage3_global_local.yaml
floorplan-glv train --config configs/train/stage4_domain_finetune.yaml
```

评估 checkpoint 或一对已经验证的几何结果：

```bash
floorplan-glv evaluate --config configs/train/stage3_global_local.yaml --checkpoint /path/to/best.pt
floorplan-glv evaluate --prediction runs/predict/sample/result.json --ground-truth /path/to/ground_truth.json
```

运行预测并验证导出的 JSON：

```bash
floorplan-glv predict --config configs/model/global_local_b4.yaml --checkpoint /path/to/best.pt --input /path/to/floorplan.png --output runs/predict/sample
floorplan-glv validate-json runs/predict/sample/result.json
```

## 预测输出

成功预测后，会在指定的输出目录下写出以下文件：

```text
result.json
overlay.png
wall_graph.png
openings.png
prediction_maps.npz
resolved_config.yaml
run_metadata.json
```

`result.json` 使用源图像像素坐标：原点在左上角，x 轴向右，y 轴向下。ID 和序列化数组均经过排序，因此相同的模型输出与后处理设置会产生确定性的 JSON。

## 仓库结构

```text
configs/            数据、模型、后处理和训练 YAML
src/floorplan_glv/  CLI、数据、模型、损失、后处理和指标
pyproject.toml      项目元数据和依赖
```
