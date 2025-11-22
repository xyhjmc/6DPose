# 6DPose / PVNet 快速上手指南

本仓库基于 PVNet 思路实现了 6D 位姿估计的完整训练与评估流程，并提供了数据预处理、调试工具与可复用的配置系统。本文档概述数据准备、训练/评估流程、常见超参数含义及调试脚本用法。

## 目录
- [数据准备](#数据准备)
- [配置系统](#配置系统)
- [模型训练](#模型训练)
- [模型评估](#模型评估)
- [核心超参数说明](#核心超参数说明)
- [调试脚本与常用命令](#调试脚本与常用命令)

## 数据准备
使用 `tools/prepare_pvnet_data.py` 将 BOP 数据集（如 LINEMOD、T-LESS、YCB-V 等）转换为 PVNet 训练所需的 `.npz` 文件与 `index.json`。

示例命令：
```bash
python tools/prepare_pvnet_data.py \
  --data-root /path/to/BOP/lm \
  --dataset-split train_pbr \
  --obj-id 8 \
  --out-dir /path/to/output/linemod_pvnet/driller_all \
  --num-kp 9 \
  --use-offset \
  --num-workers 8 \
  --resume
```
关键参数（均在脚本开头的 `parse_args` 中可调）：
- `--data-root`：BOP 数据根目录（包含 `models/` 与 `scene_gt.json` 等）。
- `--dataset-split`：处理的划分（如 `train_pbr`、`train`、`test`）。
- `--obj-id`：物体 ID（整数）。
- `--out-dir`：输出 `.npz` 与 `index.json` 的目录；脚本会自动创建并可断点续跑（`--resume`）。
- `--num-kp` / `--kp3d-path`：关键点数量或自定义 3D 关键点文件。
- `--use-offset`：生成像素偏移场（SmoothL1）或单位向量场；需与训练配置 `transforms.use_offset`、`model.vertex_scale` 对齐。
- `--render-if-no-mask`：当缺失 mask 时调用 `pyrender` 渲染（需要安装 `pyrender`）。

输出：每张图片生成一个 `.npz`（包含 `vertex`、`mask`、`kp2d`、`kp3d`、`K`、`R`、`t`、`rgb_path` 等）以及汇总的 `index.json` 列表，供训练与评估加载。【F:tools/prepare_pvnet_data.py†L1-L121】【F:tools/prepare_pvnet_data.py†L560-L651】

## 配置系统
- 基础配置位于 `configs/default.yaml`，实验配置（如 `configs/pvnet_linemod_driller_all.yaml`）通过 `defaults: [default]` 继承并覆盖关键路径。
- 加载逻辑在 `configs/config.py`：支持多基类合并、必填字段校验（如 `run.log_dir`、`dataset.train_data_dir`）、并将 YAML 转为可点式访问的 `SimpleNamespace`。【F:configs/config.py†L1-L168】【F:configs/config.py†L197-L244】
- 需要修改参数时，建议在对应实验 YAML 覆盖路径/超参，而不要直接改 `default.yaml`，以保证可重复性。

## 模型训练
主入口：`train_pvnet.py`。

基本用法：
```bash
python train_pvnet.py --config configs/pvnet_linemod_driller_all.yaml
```
主要流程：
1. 读取配置并根据 `transforms.use_offset` 同步 `model.vertex_scale`。
2. 构建训练/验证数据增强（`RandomAffine`、`RandomFlip`、`ColorJitter`、`Resize`、`NormalizeAndToTensor`）。
3. 初始化 `BopPvnetDataset` + `DataLoader`，使用 `pvnet_collate_fn` 对 batch 进行张量堆叠。
4. 构建模型（`src.models.pvnet.factory.build_model_from_cfg`）、损失（`PVNetLoss`）、优化器、学习率调度器。
5. 交由 `src.engines.trainer.Trainer` 运行，自动记录日志、保存 checkpoint、可选 AMP。【F:train_pvnet.py†L1-L188】【F:train_pvnet.py†L188-L238】

调整训练参数：
- 训练超参（epoch、日志频率、AMP）：`train.*`。
- 数据增强：`transforms.*`（需与数据生成 `--use-offset` 保持一致）。
- 优化器/调度器：`optimizer.*`、`scheduler.*`。
- 数据路径、batch size：`dataset.*`。
在实验 YAML 中覆盖即可，例如修改 `configs/pvnet_linemod_driller_all.yaml` 中的 `run.log_dir`、`dataset.train_data_dir` 等。【F:configs/pvnet_linemod_driller_all.yaml†L10-L77】

## 模型评估
主入口：`eval_pvnet.py`。

基本用法：
```bash
python eval_pvnet.py \
  --config configs/pvnet_linemod_driller_all.yaml \
  --checkpoint best.pth \
  --out_dir eval_results
```
流程概要：
1. 加载与训练一致的配置并构建验证集（仅 `Resize` + `NormalizeAndToTensor`）。
2. 创建模型并从 `run.checkpoint_dir` 读取给定 checkpoint（默认 `best.pth`）。
3. `src.engines.evaluator.Evaluator` 执行推理与 BOP 指标计算，结果写入 `eval_results/summary_metrics.json` 等。【F:eval_pvnet.py†L1-L153】【F:eval_pvnet.py†L153-L212】

调试模式：
```bash
python eval_pvnet.py --config ... --checkpoint best.pth --debug
```
会运行 `debug_full_pipeline.run_full_debug` 对单样本打印前向输出、RANSAC、PnP 中间结果，便于检查 vertex 缩放、关键点解码等问题。【F:eval_pvnet.py†L20-L44】【F:debug_full_pipeline.py†L10-L86】

## 核心超参数说明
参考 `configs/default.yaml`：
- `transforms.use_offset`：是否使用像素偏移监督。`True` 时 `model.vertex_scale` 应为较大值（如 100），且数据生成需 `--use-offset`；`False` 时自动强制 `vertex_scale=1.0` 并改为单位向量监督。【F:configs/default.yaml†L23-L62】【F:train_pvnet.py†L30-L58】
- `transforms.input_size_hw`：网络输入尺寸，需与数据生成和评价期望一致。
- `model.ransac_voting.*`：RANSAC 投票数量、内点阈值、最大迭代，用于从 vertex 场解码 2D 关键点。【F:configs/default.yaml†L63-L104】【F:configs/pvnet_linemod_driller_all.yaml†L53-L77】
- `loss.seg_weight` / `loss.vote_weight`：分割与关键点投票损失权重。
- `optimizer` / `scheduler`：学习率、权重衰减、多步衰减里程碑等。
- `pnp.reproj_error_thresh`：PnP RANSAC 重投影误差阈值，用于评估阶段姿态解算。【F:configs/default.yaml†L104-L138】

## 调试脚本与常用命令
- `debug_full_pipeline.py`：在评估阶段单样本端到端调试（前向、RANSAC、PnP），通过 `eval_pvnet.py --debug` 入口调用；也可手动导入后调用 `run_full_debug(model, dataset, cfg)`。【F:debug_full_pipeline.py†L10-L86】
- `debug_dataset_vertex_to_kp.py`：读取 `BopPvnetDataset` 输出，使用 GT vertex + mask 在变换后坐标系重建 `kp2d` 并对比误差，检查数据增强几何一致性。可在脚本顶部修改 `VERTEX_SCALE` 与 `USE_AUGMENT` 控制缩放和增强开关。【F:debug_dataset_vertex_to_kp.py†L1-L38】
- `debug.py`：快速检查单个 `.npz` 的关键字段形状与取值范围，默认读取 `data/linemod_pvnet/driller_less` 下首个文件，可根据需要修改路径。运行 `python debug.py`。【F:debug.py†L1-L15】
- 其它调试脚本：`debug_mask_vertex.py`、`debug_vertex_to_kp.py`、`debug_viz.py` 等，可按需参考源码；通常通过修改脚本顶部常量或 `main` 中的路径/参数进行定制。

## 运行与日志位置
- 训练日志与检查点目录由配置文件 `run.log_dir`、`run.checkpoint_dir` 控制；默认示例配置将输出到 `./logs/...` 与 `./checkpoints/...`，请根据本机路径修改。
- 数据路径（`dataset.train_data_dir` / `dataset.val_data_dir`）需指向 `prepare_pvnet_data.py` 生成的目录。

如需新增实验，建议复制已有 YAML 修改路径与超参，并通过 `train_pvnet.py` / `eval_pvnet.py` 运行，以保持实验可追踪与可复现。
