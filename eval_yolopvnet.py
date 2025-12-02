# eval_yolopvnet.py
"""
YOLO-backed PVNet 评估脚本。

流程与 ``eval_pvnet.py`` 保持一致，默认使用 YOLO11 主干配置：
1. 解析命令行参数，加载配置。
2. 构建无随机增强的验证集 DataLoader。
3. 初始化模型并加载检查点（默认 'best.pth'）。
4. 调用 Evaluator 运行完整的 BOP 度量评估。
"""

import argparse
import json
import os
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from configs.config import load_config
from datasets.bop_pvnet_dataset import BopPvnetDataset, pvnet_collate_fn
from datasets.transforms import Compose, NormalizeAndToTensor, Resize
from src.engines.evaluator import Evaluator
from src.models.pvnet.factory import build_model_from_cfg
from src.utils.model_stats import summarize_model_stats
from src.utils.train_model_utils import load_network


def build_val_loader(cfg):
    val_transforms = Compose(
        [
            Resize(output_size_hw=cfg.transforms.input_size_hw, use_offset=cfg.transforms.use_offset),
            NormalizeAndToTensor(
                mean=np.array(cfg.transforms.mean),
                std=np.array(cfg.transforms.std),
                vertex_scale=cfg.model.vertex_scale,
                use_offset=cfg.transforms.use_offset,
            ),
        ]
    )

    print(f"加载验证集: {cfg.dataset.val_data_dir}")
    val_dataset = BopPvnetDataset(
        data_dir=cfg.dataset.val_data_dir,
        transforms=val_transforms,
        split_name="val",
        kp3d_path=getattr(cfg.dataset, "kp3d_path", None),
    )

    eval_batch_size = cfg.dataset.loader.batch_size * 2
    val_loader = DataLoader(
        val_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=cfg.dataset.num_workers,
        collate_fn=pvnet_collate_fn,
        pin_memory=cfg.dataset.pin_memory,
        drop_last=False,
    )
    return val_loader, val_dataset


def main():
    parser = argparse.ArgumentParser(description="YOLO-backed PVNet 评估脚本")
    parser.add_argument(
        "--config",
        required=False,
        default="configs/pvnet_yolo_linemod_driller_mini.yaml",
        type=str,
        help="指向实验配置 .yaml 文件的路径",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="best.pth",
        help="要加载的检查点文件名 (默认: 'best.pth')",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="eval_results",
        help="保存详细 .json 评估结果的目录",
    )
    parser.add_argument(
        "--debug",
        default=False,
        action="store_true",
        help="运行单样本全流程调试，然后退出",
    )
    parser.add_argument(
        "--debug_metrics",
        default=True,
        action="store_true",
        help="开启耗时的调试度量（PnP/ADD/投票实验）",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    if not cfg.transforms.use_offset:
        if getattr(cfg.model, "vertex_scale", 1.0) != 1.0:
            print("[配置提示] use_offset=False 时自动将 vertex_scale 重置为 1.0。")
        cfg.model.vertex_scale = 1.0
    cfg.model.use_offset = cfg.transforms.use_offset

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    device = torch.device(cfg.device)

    val_loader, val_dataset = build_val_loader(cfg)

    print(f"初始化模型: {cfg.model.name}")
    model = build_model_from_cfg(cfg)
    model.to(device)

    try:
        input_h, input_w = cfg.transforms.input_size_hw
        model_stats = summarize_model_stats(model, (1, 3, input_h, input_w), device)
        print(
            f"模型参数量: {model_stats['param_millions']:.3f} M ({int(model_stats['param_count'])} 参数)"
            f" | 单次前向计算量: {model_stats['gflops']:.3f} GFLOPs"
        )
    except Exception as exc:  # pragma: no cover - 信息输出
        model_stats = None
        print(f"[警告] 无法统计模型参数/FLOPs: {exc}")

    checkpoint_path = os.path.join(cfg.run.checkpoint_dir, args.checkpoint)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"未找到检查点: {checkpoint_path}")

    print(f"正在从 {checkpoint_path} 加载模型权重...")
    load_network(
        net=model,
        model_dir=cfg.run.checkpoint_dir,
        resume=True,
        epoch=args.checkpoint.split(".")[0],
    )

    if args.debug:
        from debug_full_pipeline import run_full_debug

        print("\n[INFO] 进入调试模式...")
        run_full_debug(model, val_dataset, cfg)
        return

    eval_output_dir = os.path.join(cfg.run.checkpoint_dir, args.out_dir)
    os.makedirs(eval_output_dir, exist_ok=True)

    evaluator = Evaluator(
        model=model,
        dataloader=val_loader,
        device=device,
        cfg=cfg,
        out_dir=eval_output_dir,
        verbose=True,
        enable_debug=args.debug_metrics,
    )

    print(f"--- 评估开始 (Config: {args.config}) ---")
    summary = evaluator.evaluate()

    print("\n--- 评估完成 ---")
    if model_stats is not None:
        print(
            f"模型参数量: {model_stats['param_millions']:.3f} M | 计算量: {model_stats['gflops']:.3f} GFLOPs"
        )
    print("最终指标概要:")
    print(json.dumps(summary, indent=4))

    summary_path = os.path.join(eval_output_dir, "summary_metrics.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)
    print(f"评估摘要已保存到: {summary_path}")


if __name__ == "__main__":
    main()
