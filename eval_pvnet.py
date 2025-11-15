# evaluate.py
"""
项目主评估脚本 (Main Evaluation Entrypoint)

功能:
1. 解析命令行参数 (获取配置文件路径)。
2. 调用 src.config.config.py 加载和合并配置。
3. [核心] 初始化所有模块:
    - Validation Transforms (无随机增强)
    - Validation Dataset & DataLoader
    - Model (PVNet)
4. [核心] 加载 'best.pth' 检查点权重。
5. [核心] 初始化评估引擎 (src.engines.evaluator.Evaluator)。
6. 调用 evaluator.evaluate() 启动评估并打印最终指标 (例如 ADD-S)。
"""

import os
import argparse
import torch
import numpy as np
import random
import json
from torch.utils.data import DataLoader
from debug_full_pipeline import run_full_debug
# --- 1. 导入我们所有的自定义模块 ---
# 配置加载器
from configs.config import load_config
# 模型
from src.models.pvnet.PVNet import PVNet
# 数据集
from datasets.bop_pvnet_dataset import BopPvnetDataset, pvnet_collate_fn
# 数据增强
from datasets.transforms import (
    Compose,
    Resize,
    NormalizeAndToTensor
)
# 评估引擎
from src.engines.evaluator import Evaluator
# 模型加载工具
from src.utils.train_model_utils import load_network
from debug_full_pipeline import run_full_debug



def main():
    # --- 1. 解析配置 ---
    parser = argparse.ArgumentParser(description="PVNet 评估主脚本")
    parser.add_argument("--config",
                        required=False,
                        default="configs/pvnet_linemod_driller_less.yaml",
                        type=str,
                        help="指向实验配置 .yaml 文件的路径 (例如: configs/pvnet_linemod_ape.yaml)")

    parser.add_argument("--checkpoint",
                        type=str,
                        default="best.pth",
                        help="要加载的检查点文件名 (默认: 'best.pth')")

    parser.add_argument("--out_dir",
                        type=str,
                        default="eval_results",
                        help="保存详细 .json 评估结果的目录")

    parser.add_argument("--debug", default=False,action="store_true", help="运行单样本全流程调试，然后退出")
    args = parser.parse_args()

    # 加载配置 (cfg 是一个 SimpleNamespace)
    cfg = load_config(args.config)

    # --- 2. 设置 (Seed & Device) ---
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    device = torch.device(cfg.device)

    # --- 3. [评估] 数据增强 (Validation Transforms) ---
    # [关键] 验证集不使用任何随机增强 (没有 RandomAffine, RandomFlip, ColorJitter)
    val_transforms = Compose([
        Resize(
            output_size_hw=cfg.transforms.input_size_hw,
            use_offset=cfg.transforms.use_offset
        ),
        NormalizeAndToTensor(
            mean=np.array(cfg.transforms.mean),
            std=np.array(cfg.transforms.std),
            vertex_scale=cfg.model.vertex_scale
        )
    ])

    # --- 4. [评估] 数据集 (Dataset) & 加载器 (DataLoader) ---
    print(f"加载验证集: {cfg.dataset.val_data_dir}")
    val_dataset = BopPvnetDataset(
        data_dir=cfg.dataset.val_data_dir,
        transforms=val_transforms,
        split_name="val"
    )

    # 评估时 batch_size 可以设置得稍大，以加快速度
    eval_batch_size = cfg.dataset.loader.batch_size * 2

    val_loader = DataLoader(
        val_dataset,
        batch_size=eval_batch_size,
        shuffle=False,  # 评估时绝不能打乱
        num_workers=cfg.dataset.num_workers,
        collate_fn=pvnet_collate_fn,
        pin_memory=cfg.dataset.pin_memory,
        drop_last=False
    )

    # --- 5. 模型 (Model) ---
    # [接口对齐]
    print(f"初始化模型: {cfg.model.name}")
    model = PVNet(
        ver_dim=cfg.model.ver_dim,
        seg_dim=cfg.model.seg_dim,
        # 传入 RANSAC 投票参数 (供 Evaluator 内部使用)
        vote_num=cfg.model.ransac_voting.vote_num,
        inlier_thresh=cfg.model.ransac_voting.inlier_thresh,
        max_trials=cfg.model.ransac_voting.max_trials
    )
    model.to(device)

    # --- 6. [核心] 加载训练好的权重 ---
    checkpoint_path = os.path.join(cfg.run.checkpoint_dir, args.checkpoint)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"未找到检查点: {checkpoint_path}")

    print(f"正在从 {checkpoint_path} 加载模型权重...")
    # [关键] 使用 load_network 只加载模型权重 (net.state_dict())
    load_network(
        net=model,
        model_dir=cfg.run.checkpoint_dir,
        resume=True,
        epoch=args.checkpoint.split('.')[0]  # 传入 'best' 或 epoch '49'
    )
    # ========================================================
    # [插入] 调试模块入口
    # ========================================================
    if args.debug:
        print("\n[INFO] 进入调试模式...")
        run_full_debug(model, val_dataset, cfg)
        return  # 调试完成后直接退出
    # --- 7. [核心] 初始化评估引擎 (Evaluator) ---
    # [接口对齐]
    print("初始化评估引擎 (Evaluator)...")

    # 确保输出目录存在
    eval_output_dir = os.path.join(cfg.run.checkpoint_dir, args.out_dir)
    os.makedirs(eval_output_dir, exist_ok=True)

    evaluator = Evaluator(
        model=model,
        dataloader=val_loader,
        device=device,
        cfg=cfg,  # Evaluator 需要完整的 cfg 来获取 BOP 路径和 PnP 参数
        out_dir=eval_output_dir,  # 保存每个样本的 .json 结果
        verbose=True
    )

    # --- 8. 启动评估！ ---
    print(f"--- 评估开始 (Config: {args.config}) ---")

    # evaluator.evaluate() 将运行完整的推理和 BOP 度量计算
    summary = evaluator.evaluate()

    print("\n--- 评估完成 ---")
    print("最终指标概要:")
    print(json.dumps(summary, indent=4))

    # 将最终摘要也保存到输出目录
    summary_path = os.path.join(eval_output_dir, "summary_metrics.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=4)
    print(f"评估摘要已保存到: {summary_path}")


if __name__ == "__main__":
    main()

