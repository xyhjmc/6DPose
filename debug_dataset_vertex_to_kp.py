# debug_dataset_vertex_to_kp.py
"""
实验 H（升级版）：
- 通过 BopPvnetDataset + 实际 transforms（与 train/eval 一致）读数据
- 用 GT vertex + mask 在“transform 之后的坐标系”里重建 kp2d
- 计算 vertex → kp2d 的像素误差，看几何关系是否被 transforms 破坏

注意：
- 这里会使用 datasets.transforms 里的 Resize / RandomAffine / RandomFlip / NormalizeAndToTensor
- NormalizeAndToTensor 里对 vertex 除以 vertex_scale，这里会乘回来再算几何
"""

import os
import numpy as np
import torch

from datasets.bop_pvnet_dataset import BopPvnetDataset
from datasets.transforms import (
    Compose,
    RandomAffine,
    RandomFlip,
    Resize,
    ColorJitter,
    NormalizeAndToTensor,
    DEFAULT_MEAN,
    DEFAULT_STD,
)

# 和 config 里的 model.vertex_scale 保持一致
VERTEX_SCALE = 100.0

# 控制是否使用数据增强（仿射 + 翻转 + 颜色）
# - False：类似 eval 阶段，只做 Resize + Normalize
# - True：类似 train 阶段，附带 RandomAffine / RandomFlip / ColorJitter
USE_AUGMENT = True


def build_transforms_for_debug(use_aug: bool):
    """
    构造和实际 train/eval 一致的 transforms 流程。

    这里硬编码与你的 yaml 配置保持一致：
      use_offset = True
      input_size_hw = [480, 640]
      augmentation:
        degrees = 60.0
        scale_range = [0.8, 1.3]
        flip_p = 0.5
      color_jitter:
        brightness = 0.5
        contrast = 0.2
        saturation = 0.2
      mean/std = DEFAULT_MEAN/DEFAULT_STD
      vertex_scale = 100.0
    """
    use_offset = True
    H_out, W_out = 480, 640

    tfs = []

    if use_aug:
        # 几何增强：旋转 + 缩放
        tfs.append(RandomAffine(
            degrees=60.0,
            scale_range=(0.8, 1.3),
            use_offset=use_offset
        ))
        # 随机水平翻转
        tfs.append(RandomFlip(p=0.5))
        # 尺度统一
        tfs.append(Resize((H_out, W_out), use_offset=use_offset))
        # 颜色抖动（只影响 image，不影响几何）
        tfs.append(ColorJitter(
            brightness=0.5,
            contrast=0.2,
            saturation=0.2
        ))
    else:
        # eval 阶段一般只做 Resize
        tfs.append(Resize((H_out, W_out), use_offset=use_offset))

    # 归一化 + ToTensor + vertex_scale
    tfs.append(NormalizeAndToTensor(
        mean=DEFAULT_MEAN,
        std=DEFAULT_STD,
        vertex_scale=VERTEX_SCALE
    ))

    return Compose(tfs)


def check_dataset(name: str, data_dir: str, max_samples: int = 50):
    print(f"\n===== 实验 H：{name} ({data_dir})，Dataset 输出后前 {max_samples} 个样本 =====")

    transforms = build_transforms_for_debug(USE_AUGMENT)
    dataset = BopPvnetDataset(data_dir, transforms=transforms, split_name=name)

    num = min(max_samples, len(dataset))

    all_errs = []
    img_mean_errs = []
    num_fg_imgs = 0

    for idx in range(num):
        sample = dataset[idx]

        # 经过 NormalizeAndToTensor 之后：
        #   mask: torch.LongTensor(H, W)
        #   vertex: torch.FloatTensor(2K, H, W)，数值已经被 / VERTEX_SCALE
        #   kp2d: torch.FloatTensor(K, 2)
        mask = sample['mask'].cpu().numpy()          # (H, W)
        vertex = sample['vertex'].cpu().numpy()      # (2K, H, W)，此时是 "offset / VERTEX_SCALE"
        kp2d_gt = sample['kp2d'].cpu().numpy()       # (K, 2)

        # 恢复到像素偏移量（和 npz 里的 GT 一致的量纲）
        vertex = vertex * VERTEX_SCALE

        # 找前景
        ys, xs = np.where(mask > 0)
        if ys.size == 0:
            print(f"[{name}] idx={idx}, npz={sample['meta']['npz_file']}: 前景像素数 = 0，跳过")
            continue

        num_fg_imgs += 1
        H, W = mask.shape
        K = kp2d_gt.shape[0]

        per_img_errs = []

        # 这里我们用最简单的“所有前景像素投票求平均”来重建 kp2d
        # 重点是验证几何关系有没有被 transforms 破坏
        for k in range(K):
            vx = vertex[2 * k, ys, xs]      # (N_fg,)
            vy = vertex[2 * k + 1, ys, xs]  # (N_fg,)

            px = xs + vx
            py = ys + vy

            # 简单求均值，作为 kp2d_hat
            kp_hat = np.array([px.mean(), py.mean()], dtype=np.float32)
            err = np.linalg.norm(kp_hat - kp2d_gt[k])
            all_errs.append(err)
            per_img_errs.append(err)

        img_mean_errs.append(float(np.mean(per_img_errs)))

    if num_fg_imgs == 0:
        print(f"[{name}] 前 {num} 个样本里竟然没有前景像素，这不科学，检查一下 mask。")
        return

    all_errs = np.array(all_errs, dtype=np.float32)
    img_mean_errs = np.array(img_mean_errs, dtype=np.float32)

    print(f"[{name}] 有前景的样本数: {num_fg_imgs}")
    print(f"[{name}] 所有关键点像素误差：min={all_errs.min():.2f}, max={all_errs.max():.2f}, mean={all_errs.mean():.2f}")
    print(f"[{name}] 按图像平均误差：min={img_mean_errs.min():.2f}, max={img_mean_errs.max():.2f}, mean={img_mean_errs.mean():.2f}")


if __name__ == "__main__":
    # 你自己的路径
    driller_test_dir = "/home/xyh/PycharmProjects/6DPose/data/linemod_pvnet/driller_test"
    driller_mini_dir = "/home/xyh/PycharmProjects/6DPose/data/linemod_pvnet/driller_all"

    check_dataset("driller_test", driller_test_dir, max_samples=100)
    check_dataset("driller_mini", driller_mini_dir, max_samples=100)
