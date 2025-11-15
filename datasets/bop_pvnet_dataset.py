# datasets/bop_pvnet_dataset.py
"""
BOP / PVNet 数据集加载器 (Dataset Class)
负责：
1. 加载 npz 中的 kpt, mask, vertex, K, R, t 等信息
2. 加载 RGB 图像
3. 组织 sample 字典
4. 调用 transforms (外部数据增强模块)
"""

import os
import json
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from typing import Optional, Callable, Dict, Any, List


class BopPvnetDataset(Dataset):

    def __init__(self, data_dir: str, transforms: Optional[Callable] = None, split_name: str = "train"):
        super().__init__()
        self.data_dir = data_dir
        self.transforms = transforms
        self.split_name = split_name

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
        max_retry = 3

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

                # 构建 sample 字典
                sample = {k: v for k, v in data.items()}
                sample["image"] = image
                sample["meta"] = {
                    "npz_file": meta["file"],
                    "rgb_path": rgb_path,
                    "idx": idx,
                    # [修复] 从 .npz 文件 (加载到 'data' 字典中) 读取 ID
                    # 我们使用 .item() 将 0 维 numpy 数组转为 python int
                    "obj_id": int(data['obj_id'].item()),
                    "scene_id": int(data['scene_id'].item()),
                    "im_id": int(data['im_id'].item())
                }

                # 应用 transforms
                if self.transforms:
                    sample = self.transforms(sample)

                return sample

            except Exception as e:
                print(f"[警告] 样本 {idx} 加载失败: {e}")
                retry += 1
                idx = (idx + 1) % len(self.index_list)

        raise RuntimeError("连续多次尝试依然失败，可能数据损坏。")


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
        if key in batch[0]:
            collated[key] = torch.stack([b[key] for b in batch], dim=0)

    # [修复] 'meta' 是非张量数据，单独收集
    if "meta" in batch[0]:
        collated["meta"] = [b["meta"] for b in batch]

    return collated