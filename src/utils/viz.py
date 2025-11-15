# src/utils/viz.py
"""
通用可视化工具函数。

功能:
- draw_pose_bbox: 在图像上绘制 6D 姿态（3D 包围盒）
- draw_keypoints: 在图像上绘制 2D 关键点
- draw_mask_overlay: 在图像上绘制半透明的分割掩码
- get_bbox_corners: 从 3D 点云计算 8 个角点
"""

import numpy as np
import cv2
from typing import Optional, Tuple, List

# 我们可以使用我们自己的 mesh 工具来加载模型
try:
    from src.utils.mesh import get_model_points
except ImportError:
    print("警告: 'src.utils.mesh' 未找到。get_model_points_from_path 将不可用。")
    get_model_points = None


# --- 3D 包围盒相关的辅助函数 ---

def get_bbox_corners(model_points: np.ndarray) -> np.ndarray:
    """
    从 (N, 3) 的 3D 点云计算 8 个 3D 包围盒角点。

    返回:
      (8, 3) Numpy 数组
    """
    min_x, min_y, min_z = np.min(model_points, axis=0)
    max_x, max_y, max_z = np.max(model_points, axis=0)

    corners = np.array([
        [min_x, min_y, min_z],
        [min_x, min_y, max_z],
        [min_x, max_y, min_z],
        [min_x, max_y, max_z],
        [max_x, min_y, min_z],
        [max_x, min_y, max_z],
        [max_x, max_y, min_z],
        [max_x, max_y, max_z],
    ], dtype=np.float32)
    return corners


def get_bbox_edges() -> List[Tuple[int, int]]:
    """
    返回定义 3D 包围盒的 12 条边的索引对。
    (索引对应 get_bbox_corners 的 0-7)
    """
    return [
        (0, 1), (0, 2), (0, 4), (1, 3),
        (1, 5), (2, 3), (2, 6), (3, 7),
        (4, 5), (4, 6), (5, 7), (6, 7)
    ]


# --- 主要的可视化函数 ---

def draw_mask_overlay(image: np.ndarray,
                      mask: np.ndarray,
                      color: Tuple[int, int, int] = (0, 255, 0),
                      alpha: float = 0.4) -> np.ndarray:
    """
    在原图上绘制一个半透明的分割掩码。

    参数:
      image: (H, W, 3) BGR 图像 (uint8)
      mask: (H, W) 二值掩码 (0 或 1)

    返回:
      (H, W, 3) 融合后的 BGR 图像
    """
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)

    overlay = image.copy()
    mask_bool = (mask > 0)

    # 创建一个纯色的层
    color_layer = np.zeros_like(image, dtype=np.uint8)
    color_layer[mask_bool] = color

    # 融合
    # cv2.addWeighted(src1, alpha, src2, beta, gamma)
    fused_image = cv2.addWeighted(color_layer, alpha, overlay, 1.0 - alpha, 0)
    return fused_image


def draw_keypoints(image: np.ndarray,
                   kpts_2d: np.ndarray,
                   radius: int = 3,
                   color: Tuple[int, int, int] = (0, 0, 255)) -> np.ndarray:
    """
    在图像上绘制 2D 关键点 (圆形)。

    参数:
      image: (H, W, 3) BGR 图像
      kpts_2d: (K, 2) 关键点 (x, y) 坐标

    返回:
      (H, W, 3) 带关键点的 BGR 图像
    """
    output_image = image.copy()
    for (x, y) in kpts_2d:
        x_int, y_int = int(round(x)), int(round(y))
        cv2.circle(output_image, (x_int, y_int), radius, color, -1)  # -1 表示实心
    return output_image


def draw_pose_bbox(image: np.ndarray,
                   K: np.ndarray,
                   R: np.ndarray,
                   t: np.ndarray,
                   bbox_corners_3d: np.ndarray,
                   line_color: Tuple[int, int, int] = (0, 255, 0),
                   line_thickness: int = 2) -> np.ndarray:
    """
    [核心] 将 3D 包围盒根据 6D 姿态 (R, t) 和相机内参 (K) 投影到图像上。

    参数:
      image: (H, W, 3) BGR 图像
      K: (3, 3) 相机内参矩阵
      R: (3, 3) 旋转矩阵
      t: (3,) 平移向量
      bbox_corners_3d: (8, 3) 3D 包围盒的 8 个角点

    返回:
      (H, W, 3) 绘制了 3D 包围盒的 BGR 图像
    """
    output_image = image.copy()

    # --- 1. 投影 3D 角点到 2D ---
    # cv2.projectPoints 需要 (N, 1, 3) 格式的点
    points_3d = bbox_corners_3d.reshape(-1, 1, 3).astype(np.float32)
    K = K.astype(np.float32)

    # rvec (旋转向量)
    rvec, _ = cv2.Rodrigues(R.astype(np.float32))
    tvec = t.reshape(3, 1).astype(np.float32)

    # 投影
    points_2d, _ = cv2.projectPoints(points_3d, rvec, tvec, K, None)

    # (N, 1, 2) -> (N, 2)
    points_2d = points_2d.reshape(-1, 2).astype(np.int32)

    # --- 2. 绘制 12 条边 ---
    edges = get_bbox_edges()
    for (i, j) in edges:
        p1 = tuple(points_2d[i])
        p2 = tuple(points_2d[j])
        cv2.line(output_image, p1, p2, line_color, line_thickness)

    # --- 3. (可选) 绘制坐标轴 (Z 轴) ---
    axis_3d = np.array([[0, 0, 0], [0, 0, 50]], dtype=np.float32)  # 50mm 长的 Z 轴
    axis_2d, _ = cv2.projectPoints(axis_3d, rvec, tvec, K, None)
    axis_2d = axis_2d.reshape(-1, 2).astype(np.int32)

    # 绘制 Z 轴 (蓝色)
    cv2.line(output_image, tuple(axis_2d[0]), tuple(axis_2d[1]), (255, 0, 0), 3)

    return output_image