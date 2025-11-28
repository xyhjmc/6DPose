# src/utils/ransac_voting.py
import numpy as np
from typing import Tuple, Optional


def _intersect_rays_numpy(p1: np.ndarray, d1: np.ndarray, p2: np.ndarray, d2: np.ndarray) -> Optional[np.ndarray]:
    """求两条射线的交点（最小二乘意义）。"""
    A = np.stack([d1, -d2], axis=1)
    det = np.linalg.det(A)
    if abs(det) < 1e-6:
        return None
    try:
        sol = np.linalg.solve(A, p2 - p1)
    except np.linalg.LinAlgError:
        return None
    return p1 + sol[0] * d1


def _line_distance_numpy(points: np.ndarray, dirs: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    """计算点到射线的垂直距离，用于单位向量模式的内点计数。"""
    normals = np.stack([-dirs[:, 1], dirs[:, 0]], axis=1)
    numer = np.abs(np.sum(normals * (points - candidate[None, :]), axis=1))
    denom = np.linalg.norm(normals, axis=1)
    denom[denom < 1e-8] = 1.0
    return numer / denom


def _pixel_coords(h: int, w: int) -> np.ndarray:
    """
    生成 (H, W, 2) 的像素坐标网格。

    返回:
        coords: (H, W, 2) 数组，其中 coords[r, c] = [c, r] (即 x, y 坐标)
    """
    xs = np.arange(w, dtype=np.float32)
    ys = np.arange(h, dtype=np.float32)
    # xv 每一行都是 [0, 1, ..., w-1]
    # yv 每一列都是 [0, 1, ..., h-1]
    xv, yv = np.meshgrid(xs, ys)
    # 堆叠成 (H, W, 2)，最后一维是 (x, y) 坐标
    coords = np.stack([xv, yv], axis=-1)
    return coords


def ransac_voting_cpu(mask: np.ndarray,
                      vertex: np.ndarray,
                      num_votes: int = 512,
                      inlier_thresh: float = 2.0,
                      max_trials: int = 200,
                      seed: Optional[int] = None,
                      use_offset: bool = True) -> np.ndarray:
    """
    [CPU版] RANSAC 风格投票，用于从 PVNet 顶点场 (vertex field) 计算关键点位置。

    参数:
      mask: (H, W) 布尔或 0/1 数组，指示物体像素。
      vertex: (H, W, 2*K) 或 (2*K, H, W) 预测场。
              假设: (H, W, 2*K) 格式，(dx, dy) 向量。
      num_votes: (int) 用于投票的随机采样像素数。
      inlier_thresh: (float) 像素阈值，用于判断一个投票是否为 "内点" (inlier)。
      max_trials: (int) RANSAC 迭代次数。

    返回:
      kpt_2d: (K, 2) 数组，包含 K 个关键点的 (x, y) 坐标。
    """

    # --- 1. 检查和归一化输入形状 ---
    if vertex.ndim == 3 and vertex.shape[2] % 2 == 0:
        # 格式 (H, W, 2K)
        H, W, vn2 = vertex.shape
        K = vn2 // 2
    elif vertex.ndim == 3 and vertex.shape[0] % 2 == 0:
        # 格式 (2K, H, W) -> 转置为 (H, W, 2K)
        vertex = np.transpose(vertex, (1, 2, 0))
        H, W, vn2 = vertex.shape
        K = vn2 // 2
    else:
        raise ValueError("vertex 必须是 (H, W, 2K) 或 (2K, H, W) 格式")

    # --- 2. 准备采样点 ---
    # 获取掩码 (mask) 为 True 的所有像素的 (row, col) 索引
    mask_idx = np.argwhere(mask > 0)  # (M, 2)

    if mask_idx.shape[0] == 0:
        # 如果物体掩码为空，返回 K 个 (0, 0) 点
        return np.zeros((K, 2), dtype=np.float32)

    if seed is not None:
        np.random.seed(seed)

    # 从 M 个掩码点中，最多采样 num_votes 个
    n_sample = min(num_votes, mask_idx.shape[0])
    # 随机选择 n_sample 个索引
    samp_idx = mask_idx[np.random.choice(mask_idx.shape[0], n_sample, replace=False)]

    # --- 3. 逐个关键点 (K) 进行 RANSAC ---
    kpt_list = np.zeros((K, 2), dtype=np.float32)

    for ki in range(K):
        # --- 3a. 收集投票 (Voting) ---
        # 获取第 k 个关键点的 (dx, dy) 向量场
        vx = vertex[..., 2 * ki]  # (H, W)
        vy = vertex[..., 2 * ki + 1]  # (H, W)

        # 存储 n_sample 个投票的 (x, y) 结果
        base_xy = np.stack([samp_idx[:, 1], samp_idx[:, 0]], axis=1).astype(np.float32)
        if use_offset:
            votes = np.zeros((n_sample, 2), dtype=np.float32)
            for i, (r, c) in enumerate(samp_idx):
                votes[i, 0] = c + vx[r, c]  # 投票 x = c + dx
                votes[i, 1] = r + vy[r, c]  # 投票 y = r + dy
        else:
            votes = base_xy.copy()
        dirs = np.stack([vx[samp_idx[:, 0], samp_idx[:, 1]], vy[samp_idx[:, 0], samp_idx[:, 1]]], axis=1)

        # --- 3b. RANSAC ---
        best_count = 0
        # 使用所有投票的平均值作为回退（fallback）
        best_center = votes.mean(axis=0)

        # RANSAC 迭代
        n_trials = min(max_trials, n_sample)  # 试验次数不能超过样本数

        if use_offset:
            for _ in range(n_trials):
                idx = np.random.randint(0, n_sample)
                candidate = votes[idx]

                dists = np.linalg.norm(votes - candidate.reshape(1, 2), axis=1)

                inliers = votes[dists <= inlier_thresh]
                cnt = inliers.shape[0]

                if cnt > best_count:
                    best_count = cnt
                    best_center = inliers.mean(axis=0)
        else:
            if n_sample < 2:
                kpt_list[ki] = best_center
                continue
            for _ in range(n_trials):
                pair = np.random.choice(n_sample, 2, replace=False)
                p1, p2 = votes[pair[0]], votes[pair[1]]
                d1, d2 = dirs[pair[0]], dirs[pair[1]]
                inter = _intersect_rays_numpy(p1, d1, p2, d2)
                if inter is None:
                    continue

                dists = _line_distance_numpy(votes, dirs, inter)
                inlier_mask = dists <= inlier_thresh
                cnt = int(inlier_mask.sum())

                if cnt > best_count:
                    best_count = cnt
                    normals = np.stack([-dirs[inlier_mask, 1], dirs[inlier_mask, 0]], axis=1)
                    rhs = np.sum(normals * votes[inlier_mask], axis=1)
                    try:
                        refined, *_ = np.linalg.lstsq(normals, rhs, rcond=None)
                        best_center = refined
                    except np.linalg.LinAlgError:
                        best_center = inter

        kpt_list[ki] = best_center

    return kpt_list  # (K, 2)


def estimate_voting_distribution_with_mean(mask: np.ndarray,
                                           vertex: np.ndarray,
                                           seed_mean: np.ndarray):
    """
    [CPU版] 存根函数：计算围绕给定 'seed_mean' 的均值和方差。
    （这个函数只是一个简单的存根，可以被扩展）
    """
    # seed_mean: (K, 2)
    # 简单地返回传入的均值和零方差
    var = np.zeros_like(seed_mean[:, 0])
    return seed_mean, var



"""
PVNet 风格 RANSAC 投票（用于关键点定位）的 PyTorch 实现。

API:
    kpts2d, inlier_counts = ransac_voting_torch(mask, vertex, ...)

输入:
    mask:   (B, H, W) 或 (B, 1, H, W) 或 (H, W) - 二值掩码 (0/1) 或概率
    vertex: (B, 2K, H, W) 或 (2K, H, W)       - 预测的顶点偏移图
                                                (通道 [2k, 2k+1] 是 (dx, dy))
输出:
    kpts2d: (B, K, 2) - 估计的 2D 关键点坐标 (x, y)
    inlier_counts: (B, K) - 最佳假设的内点数量
"""
import torch
from typing import Tuple, Optional


def _ensure_batch_mask_vertex(mask: torch.Tensor, vertex: torch.Tensor
                              ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    工具函数：归一化输入的形状。
      mask -> (B, H, W)
      vertex -> (B, 2K, H, W)
    """
    # --- 归一化 Mask ---
    if mask.dim() == 2:
        # (H, W) -> (1, H, W)
        mask = mask.unsqueeze(0)
    if mask.dim() == 4 and mask.shape[1] == 1:
        # (B, 1, H, W) -> (B, H, W)
        mask = mask.squeeze(1)

    # --- 归一化 Vertex ---
    if vertex.dim() == 3:
        # (2K, H, W) -> (1, 2K, H, W)
        vertex = vertex.unsqueeze(0)

    # --- 最终检查与批次广播 ---
    if mask.dim() != 3:
        raise ValueError("mask 必须是 (H,W), (B,H,W) 或 (B,1,H,W)")
    if vertex.dim() != 4:
        raise ValueError("vertex 必须是 (2K,H,W) 或 (B,2K,H,W)")

    Bv = vertex.shape[0]  # Vertex 批次大小
    Bm = mask.shape[0]  # Mask 批次大小

    if Bm == 1 and Bv > 1:
        # 如果 Mask 批次为 1，Vertex 批次 > 1，广播 Mask
        mask = mask.expand(Bv, -1, -1)

    if Bm != Bv:
        raise ValueError(f"Mask (B={Bm}) 和 Vertex (B={Bv}) 的批次大小不匹配")

    return mask, vertex


def _intersect_rays_torch(p1: torch.Tensor, d1: torch.Tensor, p2: torch.Tensor, d2: torch.Tensor) -> Optional[torch.Tensor]:
    # torch.det 不支持 half 精度的 CUDA 张量，因此在此处显式转换到
    # float32 进行线性求解，最终再转换回原始 dtype。
    orig_dtype = p1.dtype
    A = torch.stack([d1, -d2], dim=1).float()
    det = torch.det(A)
    if torch.abs(det) < 1e-6:
        return None
    try:
        sol = torch.linalg.solve(A, (p2 - p1).float())
    except RuntimeError:
        return None
    return (p1.float() + sol[0] * d1.float()).to(orig_dtype)


def _line_distance_torch(points: torch.Tensor, dirs: torch.Tensor, candidate: torch.Tensor) -> torch.Tensor:
    normals = torch.stack([-dirs[:, 1], dirs[:, 0]], dim=1)
    numer = torch.abs(torch.sum(normals * (points - candidate.unsqueeze(0)), dim=1))
    denom = torch.norm(normals, dim=1).clamp(min=1e-8)
    return numer / denom


def ransac_voting_torch(mask: torch.Tensor,
                        vertex: torch.Tensor,
                        num_votes: int = 512,
                        inlier_thresh: float = 2.0,
                        max_trials: int = 200,
                        replace: bool = False,
                        seed: Optional[int] = None,
                        use_offset: bool = False,
                        vectorized_unit: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    [PyTorch/GPU版] RANSAC 投票 (向量化实现)。

    参数:
      mask: (B,H,W) 或 (B,1,H,W) 或 (H,W)
      vertex: (B,2K,H,W) 或 (2K,H,W)
      num_votes: (int) 每个关键点采样的前景像素数
      inlier_thresh: (float) 距离阈值 (像素)
      max_trials: (int) RANSAC 迭代次数
      replace: (bool) 采样时是否允许重复
      seed: (int) 随机种子
      vectorized_unit: (bool) 当 use_offset=False 时，使用向量化的 Unit-Vector RANSAC 版本

    返回:
      kpts2d: (B, K, 2) - 估计的关键点 (x, y)
      inlier_counts: (B, K) - 内点数量 (在 CPU 上)
    """
    if seed is not None:
        torch.manual_seed(seed)

    # 1. 确保输入具有 (B, ...) 格式
    mask, vertex = _ensure_batch_mask_vertex(mask, vertex)
    device, dtype = vertex.device, vertex.dtype
    B, C, H, W = vertex.shape

    if C % 2 != 0:
        raise ValueError("Vertex 通道数必须是偶数 (2*K)")
    K = C // 2

    # 将 vertex 格式从 (B, 2K, H, W) 变为 (B, H, W, 2K) 以方便索引
    vertex_hw = vertex.permute(0, 2, 3, 1)

    # 准备输出张量
    kpts2d = torch.zeros((B, K, 2), dtype=dtype, device=device)
    inlier_counts = torch.zeros((B, K), dtype=torch.int32, device='cpu')

    # 2. 创建像素坐标网格 (H, W, 2)
    #    (yy, xx) 使用 'ij' 索引确保形状为 (H, W)
    yy, xx = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing='ij'
    )
    coords = torch.stack([xx, yy], dim=-1)  # (H, W, 2)，值 (c, r)

    # 3. 按批次 (Batch) 循环处理
    # (注意：RANSAC 逻辑本身很难在批次上向量化，因此在批次上循环是常见做法)
    for b in range(B):
        # 3a. 获取前景点 (M, 2)，其中 M 是前景点数量
        fg_idx = torch.nonzero(mask[b] > 0.5, as_tuple=False)
        M = fg_idx.shape[0]

        if M == 0:
            continue  # 如果掩码为空，跳过

        # 3b. 采样 (Sampling)
        n_sample = min(num_votes, M)

        if replace or M < num_votes:
            # 有放回采样 (如果 M < n_sample 或明确要求)
            samp_choice = torch.randint(0, M, (n_sample,), device=device)
        else:
            # 无放回采样
            samp_choice = torch.randperm(M, device=device)[:n_sample]

        # 采样点索引 (n_sample, 2)，值为 (r, c)
        samp_coords = fg_idx[samp_choice].long()
        rows, cols = samp_coords[:, 0], samp_coords[:, 1]

        # 采样点的 (x, y) 坐标
        base_xy = torch.stack([cols, rows], dim=1).to(dtype)  # (n_sample, 2)

        # 3c. 收集投票 (Gather Votes)
        # 提取采样点的预测偏移向量
        vertex_sample = vertex_hw[b, rows, cols, :]  # (n_sample, 2K)
        vertex_sample = vertex_sample.view(n_sample, K, 2)  # (n_sample, K, 2)

        # 投票 = 像素坐标 + 偏移向量（offset 模式），或仅保留像素坐标（unit 向量模式）
        votes = base_xy.unsqueeze(1) + vertex_sample if use_offset else base_xy.unsqueeze(1).expand(-1, K, -1)

        # 3d. RANSAC 迭代
        # 随机选择 RANSAC 迭代的 "候选" 索引
        n_trials = min(max_trials, n_sample)
        rand_idx = torch.randint(0, n_sample, (n_trials,), device=device)

        # 候选的中心点 (n_trials, K, 2)
        candidates = votes[rand_idx]

        final_centers = torch.zeros((K, 2), dtype=dtype, device=device)
        best_counts = torch.zeros((K,), dtype=torch.int32, device=device)

        if use_offset:
            # 3e. [关键] 向量化距离计算
            # (K, 1, n_sample, 2) - (K, n_trials, 1, 2) -> (K, n_trials, n_sample, 2)
            v_exp = votes.permute(1, 0, 2).unsqueeze(1)  # (K, 1, n_sample, 2)
            c_exp = candidates.permute(1, 0, 2).unsqueeze(2)  # (K, n_trials, 1, 2)

            dists_sq = torch.sum((v_exp - c_exp) ** 2, dim=-1)  # (K, n_trials, n_sample)
            thresh_sq = inlier_thresh ** 2
            inlier_mask = dists_sq <= thresh_sq

            inlier_counts_trial = inlier_mask.sum(dim=-1)
            best_counts, best_idx = inlier_counts_trial.max(dim=1)  # (K,)
            best_counts = best_counts.to(torch.int32)

            for k in range(K):
                best_trial_k = best_idx[k]
                mask_k = inlier_mask[k, best_trial_k]  # (n_sample,)

                if mask_k.any():
                    inlier_votes = votes[mask_k, k]  # (N_inliers, 2)
                    final_centers[k] = inlier_votes.mean(dim=0)
                else:
                    final_centers[k] = candidates[best_trial_k, k]
        else:
            dirs_sample = vertex_sample  # (n_sample, K, 2) 方向
            if vectorized_unit:
                normals_all = torch.stack([-dirs_sample[:, :, 1], dirs_sample[:, :, 0]], dim=2)  # (n_sample, K, 2)
                norm_norm = torch.norm(normals_all, dim=2).clamp(min=1e-8)  # (n_sample, K)

                for k in range(K):
                    dirs_k = dirs_sample[:, k]
                    normals = normals_all[:, k]
                    norm_norm_k = norm_norm[:, k]

                    if n_sample < 2:
                        final_centers[k] = base_xy.mean(dim=0)
                        best_counts[k] = 0
                        continue

                    pair_idx = torch.randint(0, n_sample, (n_trials, 2), device=device)
                    p1 = base_xy[pair_idx[:, 0]]
                    p2 = base_xy[pair_idx[:, 1]]
                    d1 = dirs_k[pair_idx[:, 0]]
                    d2 = dirs_k[pair_idx[:, 1]]

                    A = torch.stack([d1, -d2], dim=1).float()  # (n_trials, 2, 2)
                    rhs = (p2 - p1).float()
                    det = torch.linalg.det(A)
                    valid = det.abs() >= 1e-6

                    best_center = base_xy.mean(dim=0)
                    best_count = torch.zeros((), device=device, dtype=torch.int32)

                    if valid.any():
                        try:
                            sol = torch.linalg.solve(A[valid], rhs[valid])
                            inter = (p1[valid].float() + sol[:, 0:1] * d1[valid].float()).to(dtype)

                            diff = base_xy.unsqueeze(0) - inter.unsqueeze(1)  # (n_valid, n_sample, 2)
                            numer = torch.abs(torch.sum(normals.unsqueeze(0) * diff, dim=2))
                            dists = numer / norm_norm_k.unsqueeze(0)
                            inlier_mask = dists <= inlier_thresh

                            counts = inlier_mask.sum(dim=1)
                            best_idx = torch.argmax(counts)
                            best_count = counts[best_idx]
                            best_mask = inlier_mask[best_idx]

                            if best_mask.any():
                                normals_in = normals[best_mask]
                                rhs_in = torch.sum(normals_in * base_xy[best_mask], dim=1)
                                refined_center = None
                                try:
                                    refined_center = torch.linalg.lstsq(normals_in, rhs_in).solution
                                except Exception:
                                    try:
                                        refined, _ = torch.lstsq(rhs_in.unsqueeze(1), normals_in)  # torch<=1.10 兼容
                                        refined_center = refined[:2, 0]
                                    except Exception:
                                        refined_center = None
                                best_center = refined_center if refined_center is not None else inter[best_idx]
                        except Exception:
                            pass

                    final_centers[k] = best_center
                    best_counts[k] = max(int(best_count.item()), 0)
            else:
                for k in range(K):
                    dirs_k = dirs_sample[:, k]
                    if n_sample < 2:
                        final_centers[k] = base_xy.mean(dim=0)
                        best_counts[k] = 0
                        continue

                    best_count = -1
                    best_center = base_xy.mean(dim=0)

                    for _ in range(n_trials):
                        pair = torch.randperm(n_sample, device=device)[:2]
                        inter = _intersect_rays_torch(base_xy[pair[0]], dirs_k[pair[0]], base_xy[pair[1]], dirs_k[pair[1]])
                        if inter is None:
                            continue

                        dists = _line_distance_torch(base_xy, dirs_k, inter)
                        inlier_mask = dists <= inlier_thresh
                        cnt = int(inlier_mask.sum().item())

                        if cnt > best_count:
                            best_count = cnt
                            normals = torch.stack([-dirs_k[inlier_mask, 1], dirs_k[inlier_mask, 0]], dim=1)
                            rhs = torch.sum(normals * base_xy[inlier_mask], dim=1)
                            refined_center = None
                            try:
                                refined_center = torch.linalg.lstsq(normals, rhs).solution
                            except Exception:
                                try:
                                    refined, _ = torch.lstsq(rhs.unsqueeze(1), normals)  # torch<=1.10 兼容
                                    refined_center = refined[:2, 0]
                                except Exception:
                                    refined_center = None
                            best_center = refined_center if refined_center is not None else inter

                    final_centers[k] = best_center
                    best_counts[k] = max(best_count, 0)

        kpts2d[b] = final_centers
        inlier_counts[b] = best_counts.to('cpu')

    return kpts2d, inlier_counts


def estimate_voting_distribution_with_mean_torch(
        mask: torch.Tensor,
        vertex: torch.Tensor,
        seed_mean: torch.Tensor,
        window: float = 5.0  # [FIXED] 修复了 window 参数的实现
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    [PyTorch/GPU版] 存根函数：围绕给定的 seed_mean 计算局部均值和方差。
    这是一个轻量级的 "投票分布" 近似。

    参数:
      mask: (B, H, W)
      vertex: (B, 2K, H, W)
      seed_mean: (B, K, 2) 种子点 (即 RANSAC 的输出)
      window: (float) 以 seed_mean 为中心的像素半径，用于收集统计数据

    返回:
      mean: (B, K, 2) (基本就是输入的 seed_mean)
      var:  (B, K)   (局部投票的平均方差)
    """
    mask, vertex = _ensure_batch_mask_vertex(mask, vertex)
    device, dtype = vertex.device, vertex.dtype
    B, C, H, W = vertex.shape
    K = C // 2
    seed = seed_mean.to(device=device, dtype=dtype)
    window_sq = window ** 2  # 使用平方距离

    var_out = torch.zeros((B, K), dtype=dtype, device=device)
    vertex_hw = vertex.permute(0, 2, 3, 1)  # (B, H, W, 2K)

    # (这仍然是一个慢速实现，因为它循环了 B 和 K)
    for b in range(B):
        # 获取所有前景点
        fg_idx = torch.nonzero(mask[b] > 0.5, as_tuple=False)
        if fg_idx.shape[0] == 0:
            continue
        rows, cols = fg_idx[:, 0], fg_idx[:, 1]
        base_xy = torch.stack([cols, rows], dim=1).to(dtype)  # (M, 2)

        for k in range(K):
            # 获取第 k 个关键点的所有前景投票
            # (M, 2K) -> (M, 2)
            dxdy_k = vertex_hw[b, rows, cols, 2 * k: 2 * k + 2]
            votes_k = base_xy + dxdy_k  # (M, 2)

            # 目标种子点
            seed_k = seed[b, k:k + 1, :]  # (1, 2)

            # [FIXED] 只考虑 window 范围内的点
            d2 = torch.sum((votes_k - seed_k) ** 2, dim=1)  # (M,)
            local_mask = d2 <= window_sq

            local_d2 = d2[local_mask]

            if local_d2.numel() > 0:
                # 计算局部投票的平均方差
                var_out[b, k] = local_d2.mean()
            else:
                var_out[b, k] = 0.0  # 窗口内没有投票

    return seed, var_out


# ==============================================================
# �� 统一接口：自动选择 CPU 或 GPU
# =================================0============================

def ransac_voting(mask,
                  vertex,
                  **kwargs):
    """
    统一的 RANSAC 投票 API。
    自动根据输入类型选择 CPU (Numpy) 或 GPU (PyTorch) 实现。
    """
    if torch.is_tensor(mask) or torch.is_tensor(vertex):
        # 如果输入是 PyTorch 张量
        return ransac_voting_torch(mask, vertex, **kwargs)
    else:
        # 否则假定是 Numpy 数组
        return ransac_voting_cpu(mask, vertex, **kwargs)


def estimate_voting_distribution_with_mean_auto(mask, vertex, seed_mean):
    """
    统一的投票分布估计 API (自动 CPU/GPU)。
    """
    if torch.is_tensor(mask) or torch.is_tensor(vertex):
        return estimate_voting_distribution_with_mean_torch(mask, vertex, seed_mean)
    else:
        return estimate_voting_distribution_with_mean(mask, vertex, seed_mean)


if __name__ == "__main__":
    import time

    # ======================================================
    # 测试 1: 微型测试 (B=1, K=2)
    # 目的: 验证在小负载下，CPU 启动开销更低
    # ======================================================
    print("--- 运行 RANSAC Voting [微型测试] (B=1, K=2) ---")

    # 1a. 准备数据
    B_MINI, K_MINI, H, W = 1, 2, 100, 100
    kpt_gt_mini = np.array([[50., 40.], [20., 80.]], dtype=np.float32)  # (K, 2)
    mask_np_mini = np.zeros((H, W), dtype=np.uint8)
    mask_np_mini[30:70, 30:70] = 1

    yy, xx = np.mgrid[0:H, 0:W]
    coords_np = np.stack([xx, yy], axis=-1).astype(np.float32)  # (H, W, 2)

    offsets_np = kpt_gt_mini.reshape(1, 1, K_MINI, 2) - coords_np.reshape(H, W, 1, 2)
    vertex_np_mini = offsets_np.reshape(H, W, 2 * K_MINI)
    vertex_np_mini[mask_np_mini == 0] = 0

    try:
        import torch

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        mask_torch_mini = torch.tensor(mask_np_mini, device=device)  # (H, W)
        vertex_torch_mini = torch.tensor(vertex_np_mini, device=device).permute(2, 0, 1)  # (2K, H, W)
        torch_available = True
    except ImportError:
        torch_available = False

    ransac_params = {
        'num_votes': 256,
        'inlier_thresh': 1.0,
        'max_trials': 50
    }

    # 1b. 测试 CPU (微型)
    print("\n[测试 1] CPU (Numpy) 实现 (B=1, K=2)...")
    start_time_cpu = time.time()
    kpt_cpu_out = ransac_voting(mask_np_mini, vertex_np_mini, **ransac_params)
    end_time_cpu = time.time()

    print(f"  CPU 耗时: {(end_time_cpu - start_time_cpu) * 1000:.2f} ms")
    assert np.linalg.norm(kpt_gt_mini - kpt_cpu_out) < 0.1

    # 1c. 测试 PyTorch (微型)
    if torch_available:
        print(f"\n[测试 1] PyTorch 实现 (设备: {device}, B=1, K=2)...")
        start_time_torch = time.time()
        if device.type == 'cuda':
            torch.cuda.synchronize()

        kpt_torch_out, _ = ransac_voting(mask_torch_mini, vertex_torch_mini, **ransac_params)

        if device.type == 'cuda':
            torch.cuda.synchronize()
        end_time_torch = time.time()

        kpt_torch_cpu = kpt_torch_out.cpu().numpy().squeeze(0)
        print(f"  PyTorch 耗时: {(end_time_torch - start_time_torch) * 1000:.2f} ms")
        assert np.linalg.norm(kpt_gt_mini - kpt_torch_cpu) < 0.1

    print("\n--- 微型测试结束 ---")

    # ======================================================
    # 测试 2: 压力测试 (B=16, K=32)
    # 目的: 验证在重负载下，GPU 并行优势胜出
    # ======================================================
    print("\n" + "=" * 50)
    print("--- 运行 RANSAC Voting [压力测试] (B=16, K=32) ---")
    print("=" * 50)

    B_HEAVY, K_HEAVY = 16, 32

    # 2a. 准备 "重" 数据
    # (B, K, 2) 真值
    kpt_gt_heavy_np = np.random.rand(B_HEAVY, K_HEAVY, 2) * np.array([W, H])
    kpt_gt_heavy_np = kpt_gt_heavy_np.astype(np.float32)

    # (B, H, W) 掩码
    mask_heavy_np = np.tile(mask_np_mini, (B_HEAVY, 1, 1))

    # (B, H, W, 2*K) 偏移场 (使用 Numpy 广播)
    offsets_heavy = kpt_gt_heavy_np.reshape(B_HEAVY, 1, 1, K_HEAVY, 2) - coords_np.reshape(1, H, W, 1, 2)
    vertex_heavy_np = offsets_heavy.reshape(B_HEAVY, H, W, 2 * K_HEAVY)
    vertex_heavy_np[mask_heavy_np == 0] = 0

    if torch_available:
        print(f"[Info] PyTorch 将使用设备: {device}")
        mask_heavy_torch = torch.tensor(mask_heavy_np, device=device)  # (B, H, W)
        # (B, H, W, 2K) -> (B, 2K, H, W)
        vertex_heavy_torch = torch.tensor(vertex_heavy_np, device=device).permute(0, 3, 1, 2)

    # 2b. 测试 "重" CPU (Numpy)
    # CPU 必须在 Python 中循环处理批量 (Batch)
    print("\n[测试 2] '重' CPU (Numpy) (B=16, K=32)...")
    start_time_cpu_heavy = time.time()

    cpu_heavy_results = []
    # Python 循环 B 次
    for b in range(B_HEAVY):
        # ransac_voting_cpu 内部循环 K 次
        kpt_out = ransac_voting(
            mask_heavy_np[b],
            vertex_heavy_np[b],
            **ransac_params
        )
        cpu_heavy_results.append(kpt_out)

    end_time_cpu_heavy = time.time()

    kpt_cpu_heavy_out = np.stack(cpu_heavy_results, axis=0)  # (B, K, 2)
    cpu_heavy_error = np.linalg.norm(kpt_gt_heavy_np - kpt_cpu_heavy_out, axis=-1).mean()

    print(f"  CPU 总耗时: {(end_time_cpu_heavy - start_time_cpu_heavy) * 1000:.2f} ms")
    print(f"  CPU 平均误差: {cpu_heavy_error:.4f} 像素")
    assert cpu_heavy_error < 0.1

    # 2c. 测试 "重" GPU (PyTorch)
    if torch_available and device.type == 'cuda':
        print("\n[测试 2] '重' PyTorch (GPU) (B=16, K=32)...")

        # 预热 (Warm-up): 运行一次以确保 CUDA 核已编译
        try:
            _, _ = ransac_voting(mask_heavy_torch, vertex_heavy_torch, **ransac_params)
            torch.cuda.synchronize()
        except Exception:
            pass

            # 正式计时 (包含同步)
        start_time_torch_heavy = time.time()
        torch.cuda.synchronize()

        # PyTorch 函数一次性处理所有 B 和 K (B 在 Python 循环, K 在内部并行)
        kpt_torch_heavy_out, _ = ransac_voting(
            mask_heavy_torch, vertex_heavy_torch, **ransac_params
        )

        torch.cuda.synchronize()
        end_time_torch_heavy = time.time()

        kpt_torch_heavy_cpu = kpt_torch_heavy_out.cpu().numpy()
        torch_heavy_error = np.linalg.norm(kpt_gt_heavy_np - kpt_torch_heavy_cpu, axis=-1).mean()

        print(f"  PyTorch 总耗时: {(end_time_torch_heavy - start_time_torch_heavy) * 1000:.2f} ms")
        print(f"  PyTorch 平均误差: {torch_heavy_error:.4f} 像素")
        assert torch_heavy_error < 0.1

        # 比较结果
        cpu_time = (end_time_cpu_heavy - start_time_cpu_heavy)
        gpu_time = (end_time_torch_heavy - start_time_torch_heavy)
        print("\n  [结论] 压力测试结果:")
        print(f"  CPU 耗时: {cpu_time * 1000:.2f} ms")
        print(f"  GPU 耗时: {gpu_time * 1000:.2f} ms")
        print(f"  GPU 加速比: {cpu_time / gpu_time:.2f} x")

    elif torch_available:
        print("\n[跳过] PyTorch 压力测试 (设备为 CPU，无法展示加速比)。")

    print("\n--- 所有测试结束 ---")