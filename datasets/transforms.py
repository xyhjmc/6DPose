# datasets/transforms.py
"""
数据增强模块（Data Transformations for 6D Pose Estimation, PVNet）

本模块包含：
1. Compose                      —— 多个变换串联
2. 几何变换（对 image/mask/kp2d/vertex/K 同步变换）
   - RandomAffine
   - RandomFlip
   - Resize
3. 颜色变换
   - ColorJitter
4. 格式变换
   - NormalizeAndToTensor

注意：
- use_offset = True 时，vertex 表示像素偏移（dx, dy），需随图像缩放；
- use_offset = False 时，vertex 为单位方向向量，需要保持归一化；
"""

import torch
import numpy as np
import cv2
import random
from typing import Dict, Any, Tuple, List, Callable


# -------------------------
# ImageNet 默认归一化参数
# -------------------------
DEFAULT_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
DEFAULT_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ==============================================================
#   辅助函数（Helpers）
# ==============================================================

def _warp_affine_points(pts: np.ndarray, M: np.ndarray) -> np.ndarray:
    """
    使用 2x3 仿射矩阵 M 变换点 (N, 2)。
    """
    pts = pts.astype(np.float32)
    N = pts.shape[0]
    hom_pts = np.concatenate([pts, np.ones((N, 1), dtype=np.float32)], axis=1)
    pts2d = (M.astype(np.float32) @ hom_pts.T).T
    return pts2d.astype(np.float32)


def _affine_apply_to_vertex(vertex: np.ndarray,
                            M: np.ndarray,
                            mask: np.ndarray,
                            use_offset: bool) -> np.ndarray:
    """
    对顶点场 (2K, H, W) 应用仿射变换 M。

    正确几何应该是：
      v'(p') = L · v(A^{-1} p')
    即：
      1) 先像 image 一样对每个通道做 warpAffine（空间重采样）
      2) 再用线性部分 L(2x2) 旋转/缩放向量
      3) 单位向量模式下重新归一化
      4) 应用新的 mask
    """
    import cv2

    mask = (mask > 0).astype(np.float32)
    H, W = mask.shape
    num_kp = vertex.shape[0] // 2

    # reshape -> (K, 2, H, W)
    v = vertex.reshape(num_kp, 2, H, W)
    vx = v[:, 0, :, :]    # (K, H, W)
    vy = v[:, 1, :, :]    # (K, H, W)

    # ---- 1) 先对 vx, vy 做和 image/mask 一样的 warpAffine（空间变换） ----
    vx_warp = np.zeros_like(vx, dtype=np.float32)
    vy_warp = np.zeros_like(vy, dtype=np.float32)

    for i in range(num_kp):
        vx_warp[i] = cv2.warpAffine(
            vx[i].astype(np.float32), M, (W, H),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
        vy_warp[i] = cv2.warpAffine(
            vy[i].astype(np.float32), M, (W, H),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )

    # ---- 2) 对向量值应用线性部分 L（包含旋转+缩放）----
    L = M[:, :2].astype(np.float32)  # (2,2)

    new_vx = L[0, 0] * vx_warp + L[0, 1] * vy_warp
    new_vy = L[1, 0] * vx_warp + L[1, 1] * vy_warp

    new_v = np.stack([new_vx, new_vy], axis=1)  # (K, 2, H, W)

    # ---- 3) 单位向量模式：重新归一化 ----
    if not use_offset:
        norm = np.linalg.norm(new_v, axis=1, keepdims=True)  # (K,1,H,W)
        norm[norm < 1e-8] = 1.0
        new_v = new_v / norm

    # ---- 4) 应用 mask ----
    new_v *= mask[None, None, :, :]  # (K,2,H,W) * (1,1,H,W)

    return new_v.reshape(2 * num_kp, H, W).astype(np.float32)



def _update_K_after_affine(K: np.ndarray, M: np.ndarray) -> np.ndarray:
    """
    K_new[:2, :] = M @ K[:3, :]
    """
    K = K.astype(np.float32).copy()

    L = M[:, :2]
    t = M[:, 2]

    K[:2, :2] = L @ K[:2, :2]
    K[:2, 2] = L @ K[:2, 2] + t
    return K


# ==============================================================
#   变换类
# ==============================================================

class Compose:
    """多个变换顺序执行"""
    def __init__(self, transforms: List[Callable]):
        self.transforms = transforms

    def __call__(self, sample: Dict[str, Any]):
        for t in self.transforms:
            sample = t(sample)
        return sample


# --------------------------------------------------------------
#   RandomAffine（旋转 + 缩放）
# --------------------------------------------------------------

class RandomAffine:
    def __init__(self, degrees: float, scale_range: Tuple[float, float], use_offset: bool):
        self.degrees = degrees
        self.scale_min, self.scale_max = scale_range
        self.use_offset = use_offset

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:

        img = sample['image']
        H, W = img.shape[:2]

        angle = random.uniform(-self.degrees, self.degrees)
        scale = random.uniform(self.scale_min, self.scale_max)

        cx, cy = W / 2.0, H / 2.0
        M = cv2.getRotationMatrix2D((cx, cy), angle, scale)

        # image & mask
        sample['image'] = cv2.warpAffine(img, M, (W, H),
                                         flags=cv2.INTER_LINEAR,
                                         borderMode=cv2.BORDER_REFLECT)

        sample['mask'] = cv2.warpAffine(sample['mask'], M, (W, H),
                                        flags=cv2.INTER_NEAREST,
                                        borderMode=cv2.BORDER_CONSTANT,
                                        borderValue=0)

        # kp2d
        sample['kp2d'] = _warp_affine_points(sample['kp2d'], M)

        # vertex
        sample['vertex'] = _affine_apply_to_vertex(
            sample['vertex'], M, sample['mask'], self.use_offset
        )

        # K
        sample['K'] = _update_K_after_affine(sample['K'], M)

        return sample


# --------------------------------------------------------------
#   RandomFlip（水平翻转）
# --------------------------------------------------------------

class RandomFlip:
    def __init__(self, p: float):
        self.p = p

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        if random.random() > self.p:
            return sample

        img = sample['image']
        H, W = img.shape[:2]

        # flip image & mask
        sample['image'] = np.ascontiguousarray(img[:, ::-1, :])
        sample['mask'] = np.ascontiguousarray(sample['mask'][:, ::-1])

        # flip keypoints
        sample['kp2d'][:, 0] = (W - 1) - sample['kp2d'][:, 0]

        # flip vertex
        vertex = np.flip(sample['vertex'], axis=2).copy()  # flip w-axis
        vertex[0::2, :, :] *= -1  # invert dx
        sample['vertex'] = vertex

        # flip K cx
        sample['K'][0, 2] = (W - 1) - sample['K'][0, 2]

        return sample


# --------------------------------------------------------------
#   Resize
# --------------------------------------------------------------

class Resize:
    def __init__(self, output_size_hw: Tuple[int, int], use_offset: bool):
        self.output_size_hw = output_size_hw
        self.use_offset = use_offset

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:

        H_in, W_in = sample['image'].shape[:2]
        H_out, W_out = self.output_size_hw

        if (H_in, W_in) == (H_out, W_out):
            return sample

        scale_w = W_out / W_in
        scale_h = H_out / H_in

        # image & mask
        sample['image'] = cv2.resize(sample['image'], (W_out, H_out),
                                     interpolation=cv2.INTER_LINEAR)

        sample['mask'] = cv2.resize(sample['mask'].astype(np.uint8), (W_out, H_out),
                                    interpolation=cv2.INTER_NEAREST).astype(np.uint8)

        # keypoints
        sample['kp2d'][:, 0] *= scale_w
        sample['kp2d'][:, 1] *= scale_h

        # K
        Kmat = sample['K']
        Kmat[0, 0] *= scale_w    # fx
        Kmat[0, 2] *= scale_w    # cx
        Kmat[1, 1] *= scale_h    # fy
        Kmat[1, 2] *= scale_h    # cy
        sample['K'] = Kmat

        # vertex field
        vertex = sample['vertex']
        num_kp = vertex.shape[0] // 2
        vertex_new = np.zeros((2 * num_kp, H_out, W_out), dtype=np.float32)

        for i in range(num_kp):
            vx = cv2.resize(vertex[2 * i], (W_out, H_out), interpolation=cv2.INTER_LINEAR)
            vy = cv2.resize(vertex[2 * i + 1], (W_out, H_out), interpolation=cv2.INTER_LINEAR)

            if self.use_offset:
                vx *= scale_w
                vy *= scale_h

            vertex_new[2 * i] = vx
            vertex_new[2 * i + 1] = vy

        # 单位向量模式：重新归一化
        if not self.use_offset:
            v = vertex_new.reshape(num_kp, 2, H_out, W_out)
            norm = np.linalg.norm(v, axis=1, keepdims=True)
            norm[norm < 1e-8] = 1.0
            v = v / norm
            v *= (sample['mask'] > 0).astype(np.float32)[None, None, :, :]
            vertex_new = v.reshape(2 * num_kp, H_out, W_out)
        else:
            vertex_new *= (sample['mask'] > 0).astype(np.float32)[None, :, :]

        sample['vertex'] = vertex_new
        return sample


# --------------------------------------------------------------
#   ColorJitter（亮度/对比度/饱和度）
# --------------------------------------------------------------

class ColorJitter:
    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:

        img = sample['image']
        if img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0

        # brightness
        b = 1.0 + random.uniform(-self.brightness, self.brightness)
        img = np.clip(img * b, 0.0, 1.0)

        # contrast
        c = 1.0 + random.uniform(-self.contrast, self.contrast)
        mean = img.mean(axis=(0, 1), keepdims=True)
        img = np.clip((img - mean) * c + mean, 0.0, 1.0)

        # saturation
        s = 1.0 + random.uniform(-self.saturation, self.saturation)
        gray = img.mean(axis=2, keepdims=True)
        img = np.clip((img - gray) * s + gray, 0.0, 1.0)

        sample['image'] = img
        return sample


# --------------------------------------------------------------
#   NormalizeAndToTensor
# --------------------------------------------------------------

class NormalizeAndToTensor:
    def __init__(self, mean: np.ndarray = DEFAULT_MEAN, std: np.ndarray = DEFAULT_STD, vertex_scale: float = 1.0):
        """
        Args:
            vertex_scale: 顶点场的缩放因子。
                          如果使用 pixel offset (数值约 -130~130)，建议设为 100.0。
                          这样输入给网络的数值就会变成 -1.3~1.3，利于收敛。
        """
        self.mean = mean.reshape(1, 1, 3)
        self.std = std.reshape(1, 1, 3)
        self.vertex_scale = vertex_scale

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:

        img = sample['image']

        # 1. 确保是 float[0, 1]
        if img.dtype == np.uint8:
            img = img.astype(np.float32) / 255.0
        elif img.max() > 1.0:  # 假设是 float[0, 255]
            img = img.astype(np.float32) / 255.0

        # 2. 归一化
        img = (img - self.mean) / self.std

        # 3. HWC -> CHW 并转为 Tensor
        sample['image'] = torch.from_numpy(img.transpose(2, 0, 1)).float()

        # --- 其他数据转为 Tensor ---
        if 'mask' in sample:
            sample['mask'] = torch.from_numpy(sample['mask'].astype(np.int64)).long()  # (H, W)

        if 'vertex' in sample:
            # [新增] 对顶点场进行缩放 (归一化)
            # 例如：偏移量 130.0 -> 1.3
            vertex = sample['vertex']
            if self.vertex_scale != 1.0:
                vertex = vertex / self.vertex_scale
            sample['vertex'] = torch.from_numpy(vertex).float()  # (2K, H, W)

        if 'kp2d' in sample:
            sample['kp2d'] = torch.from_numpy(sample['kp2d']).float()  # (K, 2)
        if 'K' in sample:
            sample['K'] = torch.from_numpy(sample['K']).float()  # (3, 3)
        if 'R' in sample:
            sample['R'] = torch.from_numpy(sample['R']).float()  # (3, 3)
        if 't' in sample:
            sample['t'] = torch.from_numpy(sample['t']).float()  # (3,)

        # [您已有的修复] 添加对 'kp3d' 的转换
        if 'kp3d' in sample:
            sample['kp3d'] = torch.from_numpy(sample['kp3d']).float()  # (K, 3)

        # 兼容我们的 Trainer，将图像命名为 'inp'
        if 'image' in sample:
            sample['inp'] = sample.pop('image')

        return sample