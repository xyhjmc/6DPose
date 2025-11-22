# train_pvnet.py
"""
项目主训练脚本 (Main Training Entrypoint)

功能:
1. 解析命令行参数 (获取配置文件路径)。
2. 调用 src.config.config.py 加载和合并配置。
3. [核心] 初始化所有模块:
    - Transforms (训练/验证)
    - Datasets (训练/验证)
    - DataLoaders (训练/验证)
    - Model (PVNet)
    - Loss Function (PVNetLoss)
    - Optimizer (Adam/SGD)
    - Scheduler (MultiStepLR)
4. [核心] 初始化训练引擎 (src.engines.trainer.Trainer)。
5. 调用 trainer.run() 启动训练。
"""

import os
import argparse
import torch
from torch.utils.data import DataLoader
import numpy as np
import random

# --- 1. 导入我们所有的自定义模块 ---
# 配置加载器
from configs.config import load_config
# 模型工厂（支持 PVNet / PVNetPlus）
from src.models.pvnet.factory import build_model_from_cfg
# 损失函数
from src.losses.pvnet.pvnet_loss import PVNetLoss
# 数据集
from datasets.bop_pvnet_dataset import BopPvnetDataset, pvnet_collate_fn
# 数据增强
from datasets.transforms import (
    Compose,
    RandomAffine,
    RandomFlip,
    Resize,
    ColorJitter,
    NormalizeAndToTensor
)
# 训练引擎
from src.engines.trainer import Trainer


def main():
    # --- 1. 解析配置 ---
    parser = argparse.ArgumentParser(description="PVNet 训练主脚本")
    parser.add_argument("--config",
                        required=False,
                        default="configs/pvnet_linemod_driller_all.yaml",
                        type=str,
                        help="指向实验配置 .yaml 文件的路径 (例如: configs/pvnet_linemod_ape.yaml)")
    args = parser.parse_args()

    # 加载配置 (cfg 是一个 SimpleNamespace)
    cfg = load_config(args.config)

    # 如果使用单位方向模式，禁用 vertex 缩放并同步到 model 配置
    if not cfg.transforms.use_offset:
        if getattr(cfg.model, "vertex_scale", 1.0) != 1.0:
            print("[配置提示] use_offset=False 时自动将 vertex_scale 重置为 1.0。")
        cfg.model.vertex_scale = 1.0
    cfg.model.use_offset = cfg.transforms.use_offset

    # --- 2. 设置 (Seed & Device) ---
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    device = torch.device(cfg.device)

    # --- 3. 数据增强 (Transforms) ---

    # 训练集变换管道
    train_transforms = Compose([
        # 几何变换
        RandomAffine(
            degrees=cfg.transforms.augmentation.degrees,
            scale_range=cfg.transforms.augmentation.scale_range,
            use_offset=cfg.transforms.use_offset
        ),
        RandomFlip(p=cfg.transforms.augmentation.flip_p),

        # 颜色变换
        ColorJitter(
            brightness=cfg.transforms.color_jitter.brightness,
            contrast=cfg.transforms.color_jitter.contrast,
            saturation=cfg.transforms.color_jitter.saturation
        ),

        # 缩放与格式化
        Resize(output_size_hw=cfg.transforms.input_size_hw, use_offset=cfg.transforms.use_offset),
        NormalizeAndToTensor(
            mean=np.array(cfg.transforms.mean),
            std=np.array(cfg.transforms.std),
            vertex_scale = cfg.model.vertex_scale,
            use_offset=cfg.transforms.use_offset
        )
    ])

    # 验证集变换管道 (没有随机增强)
    val_transforms = Compose([
        Resize(output_size_hw=cfg.transforms.input_size_hw, use_offset=cfg.transforms.use_offset),
        NormalizeAndToTensor(
            mean=np.array(cfg.transforms.mean),
            std=np.array(cfg.transforms.std),
            vertex_scale = cfg.model.vertex_scale,
            use_offset=cfg.transforms.use_offset
        )
    ])

    # --- 4. 数据集 (Datasets) & 加载器 (DataLoaders) ---
    print(f"加载训练集: {cfg.dataset.train_data_dir}")
    train_dataset = BopPvnetDataset(
        data_dir=cfg.dataset.train_data_dir,
        transforms=train_transforms,
        split_name="train",
        max_retry=1,
        max_aug_retry=1,
        debug_fallback=False,

    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.dataset.loader.batch_size,
        shuffle=cfg.dataset.loader.shuffle,
        num_workers=cfg.dataset.num_workers,
        collate_fn=pvnet_collate_fn,
        pin_memory=cfg.dataset.pin_memory,
        drop_last=cfg.dataset.loader.drop_last
    )

    print(f"加载验证集: {cfg.dataset.val_data_dir}")
    val_dataset = BopPvnetDataset(
        data_dir=cfg.dataset.val_data_dir,
        transforms=val_transforms,
        split_name="val"
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.dataset.loader.batch_size,  # 验证时 batch_size 可以稍大
        shuffle=False,  # 验证集不应 shuffle
        num_workers=cfg.dataset.num_workers,
        collate_fn=pvnet_collate_fn,
        pin_memory=cfg.dataset.pin_memory,
        drop_last=False
    )

    # --- 5. 模型 (Model) ---
    # [接口对齐]
    print(f"初始化模型: {cfg.model.name}")
    model = build_model_from_cfg(cfg)
    model.to(device)

    # --- 6. 损失函数 (Loss Function) ---
    # [接口对齐]
    loss_fn = PVNetLoss(
        seg_weight=cfg.loss.seg_weight,
        vote_weight=cfg.loss.vote_weight
    )
    loss_fn.to(device)

    # --- 7. 优化器 (Optimizer) ---
    # [接口对齐]
    if cfg.optimizer.type.lower() == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg.optimizer.lr,
            weight_decay=cfg.optimizer.weight_decay
        )
    elif cfg.optimizer.type.lower() == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.optimizer.lr,
            weight_decay=cfg.optimizer.weight_decay
        )
    elif cfg.optimizer.type.lower() == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=cfg.optimizer.lr,
            weight_decay=cfg.optimizer.weight_decay,
            momentum=cfg.optimizer.get('momentum', 0.9)  # SGD 通常需要 momentum
        )
    else:
        raise ValueError(f"不支持的优化器: {cfg.optimizer.type}")
    # --- 8. 调度器 (Scheduler) ---
    # [接口对齐]
    if cfg.scheduler.type.lower() == 'multisteplr':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=cfg.scheduler.milestones,
            gamma=cfg.scheduler.gamma
        )
    elif cfg.scheduler.type.lower() == 'steplr':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=cfg.scheduler.step_size,
            gamma=cfg.scheduler.gamma
        )
    else:
        raise ValueError(f"不支持的调度器: {cfg.scheduler.type}")

    # --- 9. [核心] 初始化训练引擎 (Trainer) ---
    # [接口对齐]
    print("初始化训练引擎 (Trainer)...")
    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        log_dir=cfg.run.log_dir,
        checkpoint_dir=cfg.run.checkpoint_dir,
        max_epochs=cfg.train.max_epochs,
        log_interval=cfg.train.log_interval,
        use_amp=cfg.train.use_amp,
        resume=cfg.run.resume
    )

    # --- 10. 启动训练！ ---
    print(f"--- 训练开始 (Config: {args.config}) ---")
    trainer.run()
    print(f"--- 训练完成 ---")


if __name__ == "__main__":
    main()