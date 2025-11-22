# datasets/bop_pvnet_dataset.py
"""
BOP / PVNet 数据集加载器 (Dataset Class)
负责：
1. 加载 npz 中的 kpt, mask, vertex, K, R, t 等信息
2. 加载 RGB 图像
3. 组织 sample 字典
4. 调用 transforms (外部数据增强模块)
"""
import copy

import os
import json
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from typing import Optional, Callable, Dict, Any, List


class BopPvnetDataset(Dataset):

    def __init__(
        self,
        data_dir: str,
        transforms: Optional[Callable] = None,
        split_name: str = "train",
        # ==== 新增：fallback 相关参数 ====
        min_fg_pixels: int = 10,      # 前景最少像素数阈值
        min_fg_ratio: float = 0.1,    # 增强后前景面积至少是原来的多少比例
        max_retry: int = 1,           # 读取样本失败时的最大重试次数（原来写在 __getitem__ 里）
        max_aug_retry: int = 1,# 同一样本增强重试次数
        debug_fallback: bool = False
    ):
        super().__init__()
        self.data_dir = data_dir
        self.transforms = transforms
        self.split_name = split_name

        # 保存参数
        self.min_fg_pixels = min_fg_pixels
        self.min_fg_ratio = min_fg_ratio
        self.max_retry = max_retry
        self.max_aug_retry = max_aug_retry
        self.debug_fallback = debug_fallback

        index_path = os.path.join(data_dir, "index.json")
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"未找到 index.json: {index_path}")

        print(f"[{split_name}] 加载 index.json ...")

        with open(index_path, "r") as f:
            self.index_list = json.load(f)

        print(f"[{split_name}] 共加载 {len(self.index_list)} 个样本.")


    def __len__(self):
        return len(self.index_list)

    def _resolve_rgb_path(self, path: str):
        """
        解决可能的：
        - 字符串 ndarray
        - 相对路径
        """
        # 处理 numpy 标量
        if isinstance(path, np.ndarray):
            path = path.item()

        path = str(path)

        # 如果是绝对路径，直接返回
        if os.path.isabs(path):
            return path

        # 尝试相对 data_dir
        candidate = os.path.normpath(os.path.join(self.data_dir, path))
        if os.path.exists(candidate):
            return candidate

        # 其它 fallback：原路径
        return path

    def __getitem__(self, idx: int) -> Dict[str, Any]:

        retry = 0
        # 原来这里写死的 max_retry = 5，现在用 self.max_retry
        max_retry = self.max_retry

        while retry < max_retry:
            try:
                meta = self.index_list[idx]
                npz_path = os.path.join(self.data_dir, meta["file"])

                data = np.load(npz_path, allow_pickle=True)

                rgb_path = self._resolve_rgb_path(data["rgb_path"])
                image = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
                if image is None:
                    raise RuntimeError(f"无法读取图像：{rgb_path}")

                # BGR → RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # ---------------------------
                # 1) 记录“原始 mask”的前景信息
                # ---------------------------
                mask_raw_np = None
                fg_raw = -1
                raw_shape = None

                if "mask" in data:  # 或者换成 mask_visib
                    mask_raw_np = np.asarray(data["mask"])
                elif "mask_visib" in data:  # 兜底
                    mask_raw_np = np.asarray(data["mask_visib"])

                if mask_raw_np is not None:
                    fg_raw = int((mask_raw_np > 0).sum())
                    raw_shape = tuple(mask_raw_np.shape)

                # 构建 sample 字典（先是 np / 原始数据）
                sample = {k: v for k, v in data.items()}
                sample["image"] = image
                sample["meta"] = {
                    "npz_file": meta["file"],
                    "rgb_path": rgb_path,
                    "idx": idx,
                    "obj_id": int(data['obj_id'].item()),
                    "scene_id": int(data['scene_id'].item()),
                    "im_id": int(data['im_id'].item())
                }

                # ---------------------------
                # 2) 应用 transforms + fallback
                # ---------------------------
                sample = self._apply_transforms_with_fallback(
                    sample,
                    mask_raw_np=mask_raw_np,
                    fg_raw=fg_raw,
                    meta=meta
                )

                # ---------------------------
                # 3) 不再在这里因为 fg_aug==0 而跳过样本
                #    因为前面 fallback 已经处理过了
                # ---------------------------
                # 如果你还想在这里做一个 assert，可以写个简单检查：
                if "mask" in sample:
                    fg_final = self._count_fg(sample["mask"])
                    if fg_final == 0 and self.debug_fallback:  # ★ 只有 debug 时打印
                        print(
                            f"[警告] fallback 后样本 {meta['file']} 仍然前景为 0，"
                            f"但不再跳过（仅提示一次）。raw_shape={raw_shape}"
                        )

                        # 不再 retry / continue，直接返回，让你后面看到效果

                # =========================
                # 一切正常，返回 sample
                # =========================
                return sample

            except Exception as e:
                print(f"[警告] 样本 {idx} 加载失败: {e}")
                retry += 1
                idx = (idx + 1) % len(self.index_list)

        raise RuntimeError("连续多次尝试依然失败，可能数据损坏。")

    def _count_fg(self, mask) -> int:
        """统计 mask 中前景像素数（>0 即视为前景）"""
        if mask is None:
            return -1
        if torch.is_tensor(mask):
            return int((mask > 0).sum().item())
        mask_np = np.asarray(mask)
        return int((mask_np > 0).sum())

    def _mask_ok(self, fg_raw: int, fg_aug: int) -> bool:
        """
        判断增强后的前景是否“够用”
        - 绝对像素阈值：fg_aug >= min_fg_pixels
        - 相对阈值：fg_aug >= min_fg_ratio * fg_raw
        """
        if fg_aug <= 0:
            return False
        if fg_aug < self.min_fg_pixels:
            return False
        if fg_raw > 0 and fg_aug < self.min_fg_ratio * fg_raw:
            return False
        return True

    def _basic_tensor_convert(
        self,
        sample_raw: Dict[str, Any],
        mask_raw_np: Optional[np.ndarray]
    ) -> Dict[str, Any]:
        """
        在 fallback 情况下，把原始 sample 转成最基本的张量格式：
        只保留训练需要的字段，避免处理任何字符串 np.ndarray（比如 rgb_path）。
        输出字段与正常 transforms 后保持一致：
        - inp: (3, H, W) float32, [0,1]
        - mask: (H, W) uint8
        - vertex: (...)
        - kp2d, kp3d, K, R, t
        - meta: 原样保留
        """
        out: Dict[str, Any] = {}

        # ---- 1) image -> inp (CHW, float32 [0,1]) ----
        image = sample_raw["image"]         # HWC, uint8
        img_np = np.asarray(image, dtype=np.float32) / 255.0
        img_chw = img_np.transpose(2, 0, 1)     # (3, H, W)
        out["inp"] = torch.from_numpy(img_chw.astype(np.float32))

        # ---- 2) mask ----
        if mask_raw_np is not None:
            out["mask"] = torch.from_numpy(mask_raw_np.astype(np.uint8))
        else:
            # 兜底：从 sample_raw 里再找一次
            if "mask" in sample_raw:
                out["mask"] = torch.from_numpy(
                    np.asarray(sample_raw["mask"]).astype(np.uint8)
                )
            elif "mask_visib" in sample_raw:
                out["mask"] = torch.from_numpy(
                    np.asarray(sample_raw["mask_visib"]).astype(np.uint8)
                )

        # ---- 3) vertex ----
        if "vertex" in sample_raw:
            v_np = np.asarray(sample_raw["vertex"])
            out["vertex"] = torch.from_numpy(v_np.astype(np.float32))

        # ---- 4) kp2d / kp3d（注意命名对齐 collate_fn） ----
        # 2D keypoints
        if "kp2d" in sample_raw:
            kp2d_np = np.asarray(sample_raw["kp2d"])
            out["kp2d"] = torch.from_numpy(kp2d_np.astype(np.float32))
        elif "kpt_2d" in sample_raw:
            kp2d_np = np.asarray(sample_raw["kpt_2d"])
            out["kp2d"] = torch.from_numpy(kp2d_np.astype(np.float32))

        # 3D keypoints
        if "kp3d" in sample_raw:
            kp3d_np = np.asarray(sample_raw["kp3d"])
            out["kp3d"] = torch.from_numpy(kp3d_np.astype(np.float32))
        elif "kpt_3d" in sample_raw:
            kp3d_np = np.asarray(sample_raw["kpt_3d"])
            out["kp3d"] = torch.from_numpy(kp3d_np.astype(np.float32))

        # ---- 5) K / R / t ----
        for key in ["K", "R", "t"]:
            if key in sample_raw:
                arr = np.asarray(sample_raw[key])
                out[key] = torch.from_numpy(arr.astype(np.float32))

        # ---- 6) meta 保留，用于后面 eval / debug ----
        if "meta" in sample_raw:
            out["meta"] = sample_raw["meta"]

        return out


    def _apply_transforms_with_fallback(
        self,
        sample_raw: Dict[str, Any],           # 注意这里叫 sample_raw
        mask_raw_np: Optional[np.ndarray],
        fg_raw: int,
        meta: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        对单个 sample 做：
        1. 多次尝试 self.transforms（随机增强）
        2. 如果每次增强后的前景都太少 / 为 0，则回退到“无几何增强版本”
        注意：每次增强都用 sample_raw 的 deepcopy，避免 transforms 原地改 sample 导致 KeyError('image')
        """
        # 完全没传增强：直接基础转换
        if self.transforms is None:
            return self._basic_tensor_convert(sample_raw, mask_raw_np)

        # 没有 mask 信息，就没法做前景判断，老老实实只增强一次
        if mask_raw_np is None or fg_raw <= 0:
            # 这里也用 deepcopy，避免 transforms 把原始 sample 改坏
            sample_copy = copy.deepcopy(sample_raw)
            return self.transforms(sample_copy)

        npz_name = meta.get("file", "unknown")
        last_sample = None

        for i in range(self.max_aug_retry):
            # ★ 关键：每次都从 sample_raw deepcopy 一份出来增强
            sample_copy = copy.deepcopy(sample_raw)
            aug_sample = self.transforms(sample_copy)
            last_sample = aug_sample

            mask_aug = aug_sample.get("mask", None)
            fg_aug = self._count_fg(mask_aug)

            if self._mask_ok(fg_raw, fg_aug):
                if i > 0 and self.debug_fallback:  # ★ 只在 debug 时打印
                    print(
                        f"[fallback] 样本 {npz_name} 第 {i + 1} 次增强通过: "
                        f"fg_raw={fg_raw}, fg_aug={fg_aug}"
                    )
                return aug_sample
            else:
                if self.debug_fallback:  # ★ 只在 debug 时打印
                    print(
                        f"[fallback] 样本 {npz_name} 第 {i + 1} 次增强前景过少: "
                        f"fg_raw={fg_raw}, fg_aug={fg_aug}"
                    )

        # ★ 能跑到这里，说明多次增强都不合格：回退到原始样本
        if self.debug_fallback:  # ★ 只在 debug 时打印
            print(
                f"[fallback] 样本 {npz_name} 多次增强后前景像素仍过少，"
                f"回退到未增强版本。fg_raw={fg_raw}"
            )
        safe_sample = self._basic_tensor_convert(sample_raw, mask_raw_np)
        return safe_sample


# --------------------------------------------------------------
# Collate Function
# --------------------------------------------------------------

# [修复后的代码]
def pvnet_collate_fn(batch: List[Dict]):
    """
    将 Dataset 输出的 Dict 合并为批次
    """
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return {}

    # [修复] 我们需要堆叠 (stack) 所有的张量，包括 R, t, kp3d
    keys_to_stack = ['inp', 'mask', 'vertex', 'kp2d', 'K', 'R', 't', 'kp3d']
    collated = {}

    for key in keys_to_stack:
        if not all(key in b for b in batch):
            # 有些样本缺少该键，跳过堆叠以避免 KeyError
            continue
        collated[key] = torch.stack([b[key] for b in batch], dim=0)

    # [修复] 'meta' 是非张量数据，单独收集
    if "meta" in batch[0]:
        collated["meta"] = [b["meta"] for b in batch]

    return collated
