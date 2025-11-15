# src/utils/geometry.py
"""
用于 6D 姿态估计的几何工具库。

包含:
 - PyTorch 张量友好的变换 (四元数<->矩阵, 点云变换, 投影)
 - Numpy/OpenCV PnP 封装 (solve_pnp_np, solve_pnp_ransac)
 - 接受 torch 或 numpy 输入的便利封装 solve_pnp()
 - 使用 Open3D (软依赖) 的 ICP 精配准
 - 基于 trace 分支的、数值稳健的 matrix->quaternion 实现

作者: (已为您的项目适配和改进)
"""

from typing import Optional, Tuple, Union
import numpy as np
import math
import warnings

import torch

# 尝试导入 OpenCV (用于 PnP)
try:
    import cv2
except Exception as e:
    cv2 = None
    warnings.warn("OpenCV (cv2) 未找到。PnP 相关函数将无法使用。请安装: pip install opencv-python")

# 尝试导入 Open3D (软依赖，仅 ICP 需要)
try:
    import open3d as o3d  # type: ignore
except Exception:
    o3d = None

# Epsilon (极小值)，用于防止除零
EPS = 1e-12

# ==============================================================
# �� PyTorch 基础几何函数
# (用于网络内部、损失计算等需要张量的场合)
# ==============================================================

def quaternion_to_matrix(q: torch.Tensor) -> torch.Tensor:
    """
    将四元数 (或一批四元数) 转换为旋转矩阵。

    参数:
      q: (..., 4) 形状的张量, 格式为 (x, y, z, w)。
    返回:
      R: (..., 3, 3) 形状的旋转矩阵张量。
    """
    q_norm = q.norm(p=2, dim=-1, keepdim=True)
    # 归一化，加上 epsilon 防止除零
    q = q / (q_norm + EPS)

    # 提取分量
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    # 预计算乘积项
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, xw = x * y, x * z, x * w
    yz, yw, zw = y * z, y * w, z * w

    # 构造旋转矩阵
    R_shape = q.shape[:-1] + (3, 3)
    R = torch.empty(R_shape, dtype=q.dtype, device=q.device)
    
    R[..., 0, 0] = 1 - 2 * (yy + zz)
    R[..., 0, 1] = 2 * (xy - zw)
    R[..., 0, 2] = 2 * (xz + yw)

    R[..., 1, 0] = 2 * (xy + zw)
    R[..., 1, 1] = 1 - 2 * (xx + zz)
    R[..., 1, 2] = 2 * (yz - xw)

    R[..., 2, 0] = 2 * (xz - yw)
    R[..., 2, 1] = 2 * (yz + xw)
    R[..., 2, 2] = 1 - 2 * (xx + yy)
    
    return R


def matrix_to_quaternion(R: torch.Tensor) -> torch.Tensor:
    """
    将旋转矩阵 (或一批矩阵) 转换为四元数。
    格式为 (x, y, z, w)。
    
    此实现是数值稳健的 (使用 trace 分支)，并且是完全向量化的 (使用 torch.where)。

    参数:
      R: (..., 3, 3) 形状的旋转矩阵张量。
    返回:
      q: (..., 4) 形状的四元数张量。
    """
    if R.dim() < 2 or R.shape[-2:] != (3, 3):
        raise ValueError("R 必须具有 (..., 3, 3) 形状")

    # 提取对角线元素和迹 (trace)
    r00, r11, r22 = R[..., 0, 0], R[..., 1, 1], R[..., 2, 2]
    trace = r00 + r11 + r22

    # --- Case 1: trace > 0 (最常见，最稳定的情况) ---
    s1 = torch.sqrt(trace + 1.0) * 2.0  # s = 4 * w
    qw1 = 0.25 * s1
    qx1 = (R[..., 2, 1] - R[..., 1, 2]) / s1
    qy1 = (R[..., 0, 2] - R[..., 2, 0]) / s1
    qz1 = (R[..., 1, 0] - R[..., 0, 1]) / s1

    # --- Case 2: trace <= 0, 且 R[0,0] 是最大对角元素 ---
    s2 = torch.sqrt(1.0 + r00 - r11 - r22) * 2.0  # s = 4 * x
    qw2 = (R[..., 2, 1] - R[..., 1, 2]) / s2
    qx2 = 0.25 * s2
    qy2 = (R[..., 0, 1] + R[..., 1, 0]) / s2
    qz2 = (R[..., 0, 2] + R[..., 2, 0]) / s2

    # --- Case 3: trace <= 0, 且 R[1,1] 是最大对角元素 ---
    s3 = torch.sqrt(1.0 + r11 - r00 - r22) * 2.0  # s = 4 * y
    qw3 = (R[..., 0, 2] - R[..., 2, 0]) / s3
    qx3 = (R[..., 0, 1] + R[..., 1, 0]) / s3
    qy3 = 0.25 * s3
    qz3 = (R[..., 1, 2] + R[..., 2, 1]) / s3

    # --- Case 4: trace <= 0, 且 R[2,2] 是最大对角元素 ---
    s4 = torch.sqrt(1.0 + r22 - r00 - r11) * 2.0  # s = 4 * z
    qw4 = (R[..., 1, 0] - R[..., 0, 1]) / s4
    qx4 = (R[..., 0, 2] + R[..., 2, 0]) / s4
    qy4 = (R[..., 1, 2] + R[..., 2, 1]) / s4
    qz4 = 0.25 * s4

    # --- 使用 torch.where (嵌套) 来根据条件选择正确的分支 ---
    # 这会并行计算所有分支，然后根据掩码进行选择
    
    # 1. 定义条件
    mask_trace = trace > 0
    # 2. 定义 trace <= 0 时的子条件
    cond0 = (r00 >= r11) & (r00 >= r22)
    cond1 = (r11 > r00) & (r11 >= r22)
    # cond2 = (r22 > r00) & (r22 > r11) # (最后一个分支，自动)

    # 3. 嵌套 `where`
    #    格式: torch.where(condition, value_if_true, value_if_false)
    qx = torch.where(mask_trace, qx1, torch.where(cond0, qx2, torch.where(cond1, qx3, qx4)))
    qy = torch.where(mask_trace, qy1, torch.where(cond0, qy2, torch.where(cond1, qy3, qy4)))
    qz = torch.where(mask_trace, qz1, torch.where(cond0, qz2, torch.where(cond1, qz3, qz4)))
    qw = torch.where(mask_trace, qw1, torch.where(cond0, qw2, torch.where(cond1, qw3, qw4)))

    # 4. 组合并归一化
    q = torch.stack([qx, qy, qz, qw], dim=-1)
    q = q / (q.norm(p=2, dim=-1, keepdim=True) + EPS)
    return q


def transform_points_torch(points_3d: torch.Tensor,
                           R: torch.Tensor,
                           t: torch.Tensor) -> torch.Tensor:
    """
    [PyTorch版] 变换点云 p' = R @ p + t。
    支持广播 (Broadcasting):
      - points_3d: (N, 3) 或 (B, N, 3)
      - R: (3, 3) 或 (B, 3, 3)
      - t: (3,) 或 (B, 3)
    
    返回:
      与输入点云批次维度匹配的变换后点云。
    """
    if points_3d.ndim == 2:
        # ----- 单个点云 (N, 3) -----
        if R.ndim == 2:
            # (3,3) @ (3,N) -> (3,N) -> (N,3)
            return (R @ points_3d.T).T + t.reshape(1, 3)
        else:
            # R 是 (B, 3, 3), t 是 (B, 3)
            # (N,3) @ (B,3,3).T -> bmm( (1,N,3), (B,3,3).T ) -> (B,N,3)
            # (B, N, 3)
            return torch.bmm(points_3d.unsqueeze(0).expand(R.shape[0], -1, -1), 
                             R.transpose(1, 2)) + t.unsqueeze(1)
            
    elif points_3d.ndim == 3:
        # ----- 批量点云 (B, N, 3) -----
        B, N, _ = points_3d.shape
        if R.ndim == 2:
            # 将 R (3,3) 扩展到 (B, 3, 3)
            R = R.unsqueeze(0).expand(B, -1, -1)
        if t.ndim == 1:
            # 将 t (3,) 扩展到 (B, 3)
            t = t.unsqueeze(0).expand(B, -1)
        
        # (B, N, 3) @ (B, 3, 3).T -> (B, N, 3)
        return torch.bmm(points_3d, R.transpose(1, 2)) + t.unsqueeze(1)
    else:
        raise ValueError("points_3d 必须是 (N, 3) 或 (B, N, 3) 形状")


def project_points_torch(points_3d: torch.Tensor,
                         R: torch.Tensor,
                         t: torch.Tensor,
                         K: torch.Tensor) -> torch.Tensor:
    """
    [PyTorch版] 将3D点投影到2D图像平面。
    p_cam = R @ p_world + t
    p_img = K @ p_cam
    (u, v) = (p_img_x / p_img_z, p_img_y / p_img_z)

    输入参数的广播规则同 transform_points_torch。
    返回: (..., N, 2) 或 (N, 2)
    """
    # 1. 变换到相机坐标系
    pts_cam = transform_points_torch(points_3d, R, t)
    
    # 2. 投影到图像平面 (齐次坐标)
    if pts_cam.ndim == 2:
        # ----- 单个实例 (N, 3) -----
        # (3,3) @ (3,N) -> (3,N) -> (N,3)
        proj_hom = (K @ pts_cam.T).T
        # 透视除法
        z = proj_hom[:, 2:3].clamp(min=EPS) # (N, 1)
        return proj_hom[:, :2] / z # (N, 2)
    else:
        # ----- 批量实例 (B, N, 3) -----
        B, N, _ = pts_cam.shape
        if K.ndim == 2:
            # 将 K (3,3) 扩展到 (B, 3, 3)
            K_batch = K.unsqueeze(0).expand(B, -1, -1)
        else:
            K_batch = K
            
        # (B, 3, 3) @ (B, 3, N) -> (B, 3, N)
        proj_hom_t = torch.bmm(K_batch, pts_cam.transpose(1, 2))
        # (B, 3, N) -> (B, N, 3)
        proj_hom = proj_hom_t.transpose(1, 2)
        
        # 透视除法
        z = proj_hom[..., 2:3].clamp(min=EPS) # (B, N, 1)
        return proj_hom[..., :2] / z # (B, N, 2)


# ==============================================================
# �� Numpy/OpenCV 封装 (用于后处理)
# ==============================================================

def _to_numpy(x: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """工具函数：将 PyTorch 张量转换为 Numpy 数组（如果需要的话）。"""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def solve_pnp_np(object_points: np.ndarray,
                 image_points: np.ndarray,
                 K: np.ndarray,
                 method: int = cv2.SOLVEPNP_EPNP) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    [Numpy版] 封装 OpenCV PnP 求解器。
    所有输入输出均为 Numpy 数组。

    参数:
      object_points: (N, 3) 3D 物体点
      image_points: (N, 2) 2D 图像点
      K: (3, 3) 相机内参
      method: OpenCV PnP 方法 (例如 cv2.SOLVEPNP_EPNP, cv2.SOLVEPNP_AP3P)
      
    返回:
      (R, t) 元组，其中 R 是 (3, 3) 旋转矩阵, t 是 (3,) 平移向量。
      如果求解失败，返回 None。
    """
    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) 未找到, 无法执行 solve_pnp_np。")

    # OpenCV 要求 float64 (或 float32)
    obj = object_points.astype(np.float64)
    img = image_points.astype(np.float64)
    K_np = K.astype(np.float64)
    
    try:
        # D=None 表示无畸变
        success, rvec, tvec = cv2.solvePnP(obj, img, K_np, None, flags=method)
        if not success:
            return None
        
        # 将旋转向量 (rvec) 转换为旋转矩阵 (R)
        R, _ = cv2.Rodrigues(rvec)
        return R, tvec.flatten() # 返回 (3,3) 和 (3,)
        
    except cv2.error as e:
        warnings.warn(f"OpenCV solvePnP 发生错误: {e}")
        return None


def solve_pnp_ransac(object_points: np.ndarray,
                       image_points: np.ndarray,
                       K: np.ndarray,
                       reproj_thresh: float = 3.0,
                       num_iter: int = 100,
                       confidence: float = 0.999,
                       method: int = cv2.SOLVEPNP_EPNP) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    [Numpy版] 封装 OpenCV PnP-RANSAC 稳健求解器。
    
    参数:
      reproj_thresh: RANSAC 重投影误差阈值 (像素)
      num_iter: RANSAC 迭代次数
      confidence: 置信度
      
    返回:
      (R, t, inliers_mask) 元组。
      inliers_mask 是一个 (N,) 形状的布尔数组，标记了哪些点是内点。
      如果求解失败，返回 None。
    """
    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) 未找到, 无法执行 solve_pnp_ransac。")

    obj = object_points.astype(np.float64)
    img = image_points.astype(np.float64)
    K_np = K.astype(np.float64)
    
    try:
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            obj, img, K_np, None,
            iterationsCount=num_iter,
            reprojectionError=reproj_thresh,
            confidence=confidence,
            flags=method
        )
        
        if not success:
            return None
            
        R, _ = cv2.Rodrigues(rvec)
        
        # 将 OpenCV 返回的内点索引列表 (K, 1) 转换为 (N,) 布尔掩码
        inliers_mask = np.zeros(len(object_points), dtype=bool)
        if inliers is not None:
            inliers_mask[inliers.squeeze()] = True
            
        return R, tvec.flatten(), inliers_mask
        
    except cv2.error as e:
        warnings.warn(f"OpenCV solvePnPRansac 发生错误: {e}")
        return None


def solve_pnp(object_points: Union[np.ndarray, torch.Tensor],
              image_points: Union[np.ndarray, torch.Tensor],
              K: Union[np.ndarray, torch.Tensor],
              ransac: bool = False,
              **kwargs):
    """
    PnP 求解的统一便利封装器 (主调用接口)。
    - 自动处理 PyTorch/Numpy 输入。
    - 返回值总是 Numpy 数组 (R, t) 或 (R, t, inliers_mask)。
    
    参数:
      object_points: (N, 3) 3D 物体点
      image_points: (N, 2) 2D 图像点
      K: (3, 3) 相机内参
      ransac: (bool) 是否使用 RANSAC (solve_pnp_ransac)
      **kwargs: 传递给 solve_pnp_np 或 solve_pnp_ransac 的额外参数
                (例如 reproj_thresh, num_iter 等)
    """
    # 自动将输入转换为 Numpy
    obj_np = _to_numpy(object_points)
    img_np = _to_numpy(image_points)
    K_np = _to_numpy(K)
    
    if ransac:
        return solve_pnp_ransac(obj_np, img_np, K_np, **kwargs)
    else:
        # PnP 通常需要至少 4 个点 (AP3P) 或 6 个点 (EPNP)
        if len(obj_np) < 4:
            warnings.warn(f"PnP 至少需要 4 个点, 但只收到了 {len(obj_np)} 个。")
            return None
        return solve_pnp_np(obj_np, img_np, K_np, **kwargs)


# ==============================================================
# �� ICP 精配准 (使用 Open3D，软依赖)
# ==============================================================

def icp_refine_o3d(src_points: np.ndarray,
                   tgt_points: np.ndarray,
                   init_transform: Optional[np.ndarray] = None,
                   max_iter: int = 50,
                   threshold: float = 0.01) -> np.ndarray:
    """
    [Numpy版] 使用 Open3D (点对点) ICP 来精配准姿态。
    
    目标: 找到一个变换 T，使得 T @ src_points 尽可能接近 tgt_points。
    
    参数:
      src_points: (N, 3) 源点云 (例如, 模型点云根据 *初始姿态* 变换后)
      tgt_points: (M, 3) 目标点云 (例如, 从深度图反投影的点云)
      init_transform: (4, 4) 初始变换矩阵 T
      max_iter: ICP 最大迭代次数
      threshold: ICP 中点对的最大对应距离
      
    返回:
      (4, 4) 优化后的变换矩阵 T_refined。
    """
    if o3d is None:
        raise RuntimeError("Open3D (o3d) 未找到。请 `pip install open3d` 来使用 ICP 精配准。")
        
    if init_transform is None:
        init_transform = np.eye(4)

    # 1. 构建 Open3D 点云对象
    src_pcd = o3d.geometry.PointCloud()
    src_pcd.points = o3d.utility.Vector3dVector(np.asarray(src_points))
    
    tgt_pcd = o3d.geometry.PointCloud()
    tgt_pcd.points = o3d.utility.Vector3dVector(np.asarray(tgt_points))

    # 2. 运行 ICP
    # TransformationEstimationPointToPoint 是标准的点对点 ICP
    result = o3d.pipelines.registration.registration_icp(
        src_pcd, 
        tgt_pcd, 
        threshold, 
        init_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter)
    )
    
    return np.asarray(result.transformation)


# ==============================================================
# �� Numpy 评估工具 (用于快速测试)
# (注意: 官方评估应使用 src/metrics/bop_eval.py)
# ==============================================================

def rotation_error_deg(R_pred: np.ndarray, R_gt: np.ndarray) -> float:
    """[Numpy版] 计算两个旋转矩阵之间的角度误差 (单位: 度)。"""
    R_pred = np.asarray(R_pred)
    R_gt = np.asarray(R_gt)
    
    R_diff = R_pred @ R_gt.T
    # 迹 (Trace)
    trace = np.trace(R_diff)
    # clip 保证 arccos 的输入在 [-1, 1] 范围内
    trace_clipped = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
    
    ang_rad = math.acos(trace_clipped)
    return float(math.degrees(ang_rad))


def translation_error_m(t_pred: np.ndarray, t_gt: np.ndarray) -> float:
    """[Numpy版] 计算两个平移向量之间的欧氏距离。"""
    return float(np.linalg.norm(np.asarray(t_pred) - np.asarray(t_gt)))


# ==============================================================
# �� 单元测试 (Sanity Check)
# ==============================================================
if __name__ == "__main__":
    print("--- 运行 geometry.py 完整性检查 ---")

    # 1. 准备 PnP 测试的合成数据
    np.random.seed(0)
    pts_3d = (np.random.rand(30, 3) - 0.5) * 0.2  # 3D 物体点
    
    # 2. 定义真值姿态 (R_gt, t_gt) 和相机内参 (K)
    angle = 20.0 * math.pi / 180.0
    R_gt = np.array([[math.cos(angle), -math.sin(angle), 0],
                     [math.sin(angle), math.cos(angle), 0],
                     [0, 0, 1]])
    t_gt = np.array([0.05, -0.02, 0.6]) # 60cm 远
    K = np.array([[600., 0., 320.], [0., 600., 240.], [0., 0., 1.]]) # 600mm 焦距

    # 3. 将 3D 点投影到 2D 图像
    pts_cam = (R_gt @ pts_3d.T).T + t_gt.reshape(1, 3)
    proj = (K @ pts_cam.T).T
    pts_2d = proj[:, :2] / proj[:, 2:3]

    # 4. 添加噪声
    pts_2d_noisy = pts_2d + np.random.randn(*pts_2d.shape) * 0.5 # 0.5 像素噪声

    # 5. 测试 PnP (非 RANSAC)
    if cv2 is not None:
        print("\n[测试] solve_pnp (ransac=False)")
        res = solve_pnp(pts_3d, pts_2d_noisy, K, ransac=False)
        if res is not None:
            R_est, t_est = res
            print(f"  PnP 旋转误差: {rotation_error_deg(R_est, R_gt):.3f} 度")
            print(f"  PnP 平移误差: {translation_error_m(t_est, t_gt):.4f} 米")
        else:
            print("  PnP 求解失败")

        # 6. 测试 PnP-RANSAC
        print("\n[测试] solve_pnp (ransac=True)")
        res2 = solve_pnp(pts_3d, pts_2d_noisy, K, ransac=True, reproj_thresh=6.0, num_iter=200)
        if res2 is not None:
            Rr, tr, inliers = res2
            print(f"  PnP-RANSAC 旋转误差: {rotation_error_deg(Rr, R_gt):.3f} 度")
            print(f"  PnP-RANSAC 平移误差: {translation_error_m(tr, t_gt):.4f} 米")
            print(f"  PnP-RANSAC 内点数: {int(inliers.sum())} / {len(pts_3d)}")
        else:
            print("  PnP-RANSAC 求解失败")
    else:
        print("\n[跳过] OpenCV 未安装，跳过 PnP 测试。")

    # 7. 测试 PyTorch 批量变换/投影
    print("\n[测试] PyTorch 批量变换和投影")
    device = torch.device("cpu") # 或 "cuda"
    B = 2 # 批量大小
    
    # 构造批量数据
    pts_t = torch.tensor(pts_3d, dtype=torch.float32, device=device).unsqueeze(0).expand(B, -1, -1)  # (B,N,3)
    R_t = torch.tensor(R_gt, dtype=torch.float32, device=device).unsqueeze(0).expand(B, -1, -1)   # (B,3,3)
    t_t = torch.tensor(t_gt, dtype=torch.float32, device=device).unsqueeze(0).expand(B, -1)     # (B,3)
    K_t = torch.tensor(K, dtype=torch.float32, device=device) # (3,3), 测试广播

    proj_t = project_points_torch(pts_t, R_t, t_t, K_t)
    print(f"  批量投影输出形状 (应为 {B}, {len(pts_3d)}, 2): {proj_t.shape}")
    
    # 8. 测试 PyTorch 矩阵/四元数转换
    print("\n[测试] PyTorch 矩阵 <-> 四元数转换")
    R_torch = torch.tensor(R_gt, dtype=torch.float32, device=device)
    q_torch = matrix_to_quaternion(R_torch)
    R_recon = quaternion_to_matrix(q_torch)
    
    # 计算重构误差
    R_diff = R_torch - R_recon
    print(f"  矩阵 -> 四元数 -> 矩阵 (重构误差): {torch.norm(R_diff).item():.2e}")

    print("\n--- 完整性检查结束 ---")