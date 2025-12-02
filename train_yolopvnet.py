# train_yolopvnet.py
"""
使用 YOLO11 主干的 PVNet 训练脚本。

功能与 ``train_pvnet.py`` 保持一致，但默认使用 YOLO-backed 模型配置。
流程：
1. 解析命令行参数 (获取配置文件路径)。
2. 调用 configs/config.py 加载和合并配置。
3. 初始化所有模块：Transforms、Datasets、DataLoaders、Model、Loss、Optimizer、Scheduler。
4. 初始化训练引擎 (src.engines.trainer.Trainer)。
5. 调用 trainer.run() 启动训练。
"""

import argparse
import os
import random
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

from configs.config import load_config
from datasets.bop_pvnet_dataset import BopPvnetDataset, pvnet_collate_fn
from datasets.transforms import (
    ColorJitter,
    Compose,
    NormalizeAndToTensor,
    RandomAffine,
    RandomCropResize,
    RandomFlip,
    Resize,
)
from src.engines.trainer import Trainer
from src.losses.pvnet.pvnet_loss import PVNetLoss
from src.models.pvnet.factory import build_model_from_cfg
from src.utils.model_stats import summarize_model_stats


def build_transforms(cfg):
    crop_cfg = getattr(cfg.transforms, "crop", None)
    crop_enabled = getattr(crop_cfg, "enabled", False) if crop_cfg is not None else False

    train_geo_transforms = []
    if crop_enabled:
        pad_scale_range = getattr(crop_cfg, "pad_scale_range", [1.1, 1.4])
        jitter_ratio = getattr(crop_cfg, "jitter_ratio", 0.1)
        min_side = getattr(crop_cfg, "min_side", 32)
        train_geo_transforms.append(
            RandomCropResize(
                output_size_hw=cfg.transforms.input_size_hw,
                use_offset=cfg.transforms.use_offset,
                pad_scale_range=tuple(pad_scale_range),
                min_side=min_side,
                jitter_ratio=jitter_ratio,
            )
        )

    train_geo_transforms.extend(
        [
            RandomAffine(
                degrees=cfg.transforms.augmentation.degrees,
                scale_range=cfg.transforms.augmentation.scale_range,
                use_offset=cfg.transforms.use_offset,
            ),
            RandomFlip(p=cfg.transforms.augmentation.flip_p),
        ]
    )

    train_transforms = Compose(
        train_geo_transforms
        + [
            ColorJitter(
                brightness=cfg.transforms.color_jitter.brightness,
                contrast=cfg.transforms.color_jitter.contrast,
                saturation=cfg.transforms.color_jitter.saturation,
            ),
            Resize(output_size_hw=cfg.transforms.input_size_hw, use_offset=cfg.transforms.use_offset)
            if not crop_enabled
            else lambda x: x,
            NormalizeAndToTensor(
                mean=np.array(cfg.transforms.mean),
                std=np.array(cfg.transforms.std),
                vertex_scale=cfg.model.vertex_scale,
                use_offset=cfg.transforms.use_offset,
            ),
        ]
    )

    val_geo_transforms = []
    if crop_enabled:
        pad_scale_range = getattr(crop_cfg, "pad_scale_range", [1.1, 1.4])
        min_side = getattr(crop_cfg, "min_side", 32)
        pad_scale = float(pad_scale_range[0]) if len(pad_scale_range) > 0 else 1.0
        val_geo_transforms.append(
            RandomCropResize(
                output_size_hw=cfg.transforms.input_size_hw,
                use_offset=cfg.transforms.use_offset,
                pad_scale_range=(pad_scale, pad_scale),
                min_side=min_side,
                jitter_ratio=0.0,
            )
        )

    val_transforms = Compose(
        val_geo_transforms
        + [
            Resize(output_size_hw=cfg.transforms.input_size_hw, use_offset=cfg.transforms.use_offset)
            if not crop_enabled
            else (lambda x: x),
            NormalizeAndToTensor(
                mean=np.array(cfg.transforms.mean),
                std=np.array(cfg.transforms.std),
                vertex_scale=cfg.model.vertex_scale,
                use_offset=cfg.transforms.use_offset,
            ),
        ]
    )
    return train_transforms, val_transforms


def build_dataloaders(cfg, train_transforms, val_transforms):
    print(f"加载训练集: {cfg.dataset.train_data_dir}")
    train_dataset = BopPvnetDataset(
        data_dir=cfg.dataset.train_data_dir,
        transforms=train_transforms,
        split_name="train",
        kp3d_path=getattr(cfg.dataset, "kp3d_path", None),
        max_retry=1,
        max_aug_retry=1,
        debug_fallback=False,
        fallback_resize_hw=tuple(cfg.transforms.input_size_hw),
        fallback_use_offset=cfg.transforms.use_offset,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.dataset.loader.batch_size,
        shuffle=cfg.dataset.loader.shuffle,
        num_workers=cfg.dataset.num_workers,
        collate_fn=pvnet_collate_fn,
        pin_memory=cfg.dataset.pin_memory,
        drop_last=cfg.dataset.loader.drop_last,
    )

    print(f"加载验证集: {cfg.dataset.val_data_dir}")
    val_dataset = BopPvnetDataset(
        data_dir=cfg.dataset.val_data_dir,
        transforms=val_transforms,
        split_name="val",
        kp3d_path=getattr(cfg.dataset, "kp3d_path", None),
        fallback_resize_hw=tuple(cfg.transforms.input_size_hw),
        fallback_use_offset=cfg.transforms.use_offset,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.dataset.loader.batch_size,
        shuffle=False,
        num_workers=cfg.dataset.num_workers,
        collate_fn=pvnet_collate_fn,
        pin_memory=cfg.dataset.pin_memory,
        drop_last=False,
    )
    return train_loader, val_loader


def build_optimizer(cfg, model):
    if cfg.optimizer.type.lower() == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=cfg.optimizer.lr, weight_decay=cfg.optimizer.weight_decay
        )
    elif cfg.optimizer.type.lower() == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=cfg.optimizer.lr,
            momentum=getattr(cfg.optimizer, "momentum", 0.9),
            weight_decay=cfg.optimizer.weight_decay,
            nesterov=getattr(cfg.optimizer, "nesterov", False),
        )
    else:
        raise ValueError(f"不支持的优化器类型: {cfg.optimizer.type}")
    return optimizer


def build_scheduler(cfg, optimizer):
    if cfg.scheduler.type.lower() == "multisteplr":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=cfg.scheduler.milestones, gamma=cfg.scheduler.gamma
        )
    elif cfg.scheduler.type.lower() == "reducelronplateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=cfg.scheduler.factor,
            patience=cfg.scheduler.patience,
            threshold=cfg.scheduler.threshold,
            cooldown=cfg.scheduler.cooldown,
            min_lr=cfg.scheduler.min_lr,
        )
    else:
        raise ValueError(f"不支持的学习率调度器类型: {cfg.scheduler.type}")
    return scheduler


def main():
    parser = argparse.ArgumentParser(description="YOLO-backed PVNet 训练脚本")
    parser.add_argument(
        "--config",
        required=False,
        default="configs/pvnet_yolo_linemod_driller_mini.yaml",
        type=str,
        help="指向实验配置 .yaml 文件的路径",
    )
    parser.add_argument(
        "--variant",
        required=False,
        choices=["n", "s", "m", "l", "x"],
        help="覆盖配置文件中的 YOLO 主干尺度（n/s/m/l/x）",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.variant:
        if getattr(cfg.model, "yolo", None) is None:
            cfg.model.yolo = SimpleNamespace()
        cfg.model.yolo.variant = args.variant
    if getattr(cfg.model, "yolo", None) is not None:
        print(f"使用的 YOLO 主干尺度: {getattr(cfg.model.yolo, 'variant', 'n')}")

    if not cfg.transforms.use_offset:
        if getattr(cfg.model, "vertex_scale", 1.0) != 1.0:
            print("[配置提示] use_offset=False 时自动将 vertex_scale 重置为 1.0。")
        cfg.model.vertex_scale = 1.0
    cfg.model.use_offset = cfg.transforms.use_offset

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    device = torch.device(cfg.device)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    train_transforms, val_transforms = build_transforms(cfg)
    train_loader, val_loader = build_dataloaders(cfg, train_transforms, val_transforms)

    print(f"初始化模型: {cfg.model.name}")
    model = build_model_from_cfg(cfg)
    model.to(device)

    try:
        input_h, input_w = cfg.transforms.input_size_hw
        stats = summarize_model_stats(model, (1, 3, input_h, input_w), device)
        print(
            f"模型参数量: {stats['param_millions']:.3f} M ({int(stats['param_count'])} 参数) | "
            f"计算量: {stats['gflops']:.3f} GFLOPs"
        )
    except Exception as exc:  # pragma: no cover - 信息输出
        print(f"[警告] 无法统计模型参数/FLOPs: {exc}")

    loss_fn = PVNetLoss(
        seg_loss_type=cfg.loss.seg_loss_type,
        seg_focal_gamma=cfg.loss.seg_focal_gamma,
        seg_focal_alpha=cfg.loss.seg_focal_alpha,
        seg_class_weights=cfg.loss.seg_class_weights,
        seg_weight=cfg.loss.seg_weight,
        vote_weight=cfg.loss.vote_weight,
        vote_smooth_l1_beta=cfg.loss.vote_smooth_l1_beta,
        vote_normalize_eps=cfg.loss.vote_normalize_eps,
        skip_vote_if_no_fg=cfg.loss.skip_vote_if_no_fg,
    )

    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer)

    os.makedirs(cfg.run.log_dir, exist_ok=True)
    os.makedirs(cfg.run.checkpoint_dir, exist_ok=True)

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
        grad_clip_norm=cfg.train.grad_clip_norm,
    )
    trainer.run()


if __name__ == "__main__":
    main()
