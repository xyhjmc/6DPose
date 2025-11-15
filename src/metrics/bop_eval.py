# src/metrics/bop_eval.py
"""
BOP 风格评估工具库 (ADD, ADD-S/ADI, 旋转/平移误差, VSD 接口, 批量评估)。
作者: (Generated based on user request)
说明:
 - 这个文件旨在作为一个实用、文档齐全的 BOP 指标计算骨架。
 - 对于繁重的计算任务（如大批量 ADD-S 计算），建议安装 trimesh 或 open3d 来加速 Mesh 操作，
   以及安装 scipy 来使用 cKDTree 进行快速最近邻搜索。
 - 如果没有这些库，本脚本也提供了纯 Numpy 的回退实现（但速度较慢）。
"""

from typing import Tuple, Dict, Any, List, Optional, Iterable, Union
import numpy as np
import json
import math
import os

# ---------------------------
# 依赖库检查与回退机制
# ---------------------------
# 尝试导入可选库以提升速度/便利性；如果不存在则优雅地回退。

try:
    from scipy.spatial import cKDTree as _cKDTree

    _HAVE_CKD = True
except Exception:
    _HAVE_CKD = False  # 如果没有 scipy，将使用暴力的 Numpy 广播计算距离

try:
    import trimesh

    _HAVE_TRIMESH = True
except Exception:
    _HAVE_TRIMESH = False

try:
    import open3d as _open3d  # noqa: F401

    _HAVE_OPEN3D = True
except Exception:
    _HAVE_OPEN3D = False


# ---------------------------
# 基础几何工具函数
# ---------------------------

def transform_points(points: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    对一组点进行刚体变换 (旋转 + 平移): p' = R @ p + t

    参数:
      points: (N, 3) 点云数据
      R: (3, 3) 旋转矩阵
      t: (3,) 或 (3, 1) 平移向量

    返回:
      变换后的点云 (N, 3)
    """
    # 确保输入格式正确
    points = np.asarray(points).reshape(-1, 3)
    R = np.asarray(R).reshape(3, 3)
    t = np.asarray(t).reshape(3, )

    # (R @ points.T).T 等价于 points @ R.T，最后加上平移广播
    return (R @ points.T).T + t.reshape(1, 3)


def quat_to_rotmat(q: Iterable[float]) -> np.ndarray:
    """
    将四元数转换为 3x3 旋转矩阵。
    支持 (x, y, z, w) 或 (w, x, y, z) 格式，并尝试自动检测。

    推荐输入格式为: (x, y, z, w) (这也是很多姿态估计网络的常见输出)。

    返回:
      3x3 旋转矩阵 (numpy array)
    """
    q = np.asarray(q).flatten()
    if q.size != 4:
        raise ValueError("四元数必须包含 4 个元素")

    # 启发式检测：通常 w 分量（实部）较大。
    # 这里我们默认假设输入是 (x, y, z, w)，这是 scipy/ROS 等常见格式。
    x, y, z, w = q[0], q[1], q[2], q[3]

    # 归一化四元数 (重要：网络输出通常未归一化)
    n = np.linalg.norm(q)
    if n == 0:
        raise ValueError("检测到零范数四元数 (Zero-norm quaternion)")
    x, y, z, w = x / n, y / n, z / n, w / n

    # 标准转换公式
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]
    ], dtype=float)
    return R


def rotmat_to_quat(R: np.ndarray) -> np.ndarray:
    """
    将 3x3 旋转矩阵转换为四元数 (x, y, z, w)。
    使用了数值稳定的转换算法。
    """
    R = np.asarray(R).reshape(3, 3)

    trace = np.trace(R)
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    else:
        # 处理 trace <= 0 的情况，寻找最大对角元素以保持数值稳定
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s

    q = np.array([x, y, z, w], dtype=float)
    q /= np.linalg.norm(q)
    return q


# ---------------------------
# Mesh 采样工具 (计算 ADD 指标的核心)
# ---------------------------

def sample_mesh_vertices(mesh: Union['trimesh.Trimesh', Tuple[np.ndarray, np.ndarray]],
                         n_points: int = 1000) -> np.ndarray:
    """
    在 Mesh 表面进行均匀采样。

    参数:
      mesh: 可以是 trimesh.Trimesh 实例，或者是元组 (vertices (V,3), faces (F,3))
      n_points: 采样点的数量 (BOP 标准通常不需要太多，除非做精细 ICP)

    返回:
      points: (n_points, 3) 的 Numpy 数组

    说明:
      - 优先使用 trimesh 或 open3d 的原生采样（如果已安装）。
      - 回退方案: 实现了简单的按面积加权的三角形采样算法。
    """
    # 1. 尝试 Trimesh
    if _HAVE_TRIMESH and isinstance(mesh, trimesh.Trimesh):
        return mesh.sample(n_points)

    # 2. 尝试 Open3D
    if _HAVE_OPEN3D and hasattr(mesh, "triangles") and hasattr(mesh, "vertices"):
        pcd = mesh.sample_points_uniformly(number_of_points=n_points)
        pts = np.asarray(pcd.points)
        return pts

    # 3. 手动实现的回退方案 (输入为顶点和面)
    if isinstance(mesh, tuple) or isinstance(mesh, list):
        verts = np.asarray(mesh[0], dtype=float)
        faces = np.asarray(mesh[1], dtype=int)
        if verts.ndim != 2 or faces.ndim != 2:
            raise ValueError("vertices must be (V,3), faces must be (F,3)")

        # 计算每个三角形的面积
        v0 = verts[faces[:, 0]]
        v1 = verts[faces[:, 1]]
        v2 = verts[faces[:, 2]]
        # 叉乘的一半即为面积
        tri_areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
        area_sum = tri_areas.sum()

        if area_sum <= 0:
            # 退化网格处理：直接随机重复顶点
            idx = np.random.choice(len(verts), size=n_points, replace=True)
            return verts[idx]

        # 根据面积大小作为概率进行抽样，面积大的三角形被选中的概率大
        probs = tri_areas / area_sum
        face_indices = np.random.choice(len(faces), size=n_points, p=probs)

        # 在选中的三角形内部采样重心坐标 (Barycentric coordinates)
        u = np.random.rand(n_points)
        v = np.random.rand(n_points)
        mask = u + v > 1.0
        u[mask] = 1 - u[mask]
        v[mask] = 1 - v[mask]
        w = 1 - (u + v)

        # 计算实际坐标
        sampled = (v0[face_indices] * w.reshape(-1, 1) +
                   v1[face_indices] * u.reshape(-1, 1) +
                   v2[face_indices] * v.reshape(-1, 1))
        return sampled

    raise ValueError("输入 mesh 必须是 trimesh.Trimesh, open3d 对象, 或 (verts, faces) 元组")


# ---------------------------
# 最近邻搜索辅助函数
# ---------------------------

def _nearest_neighbor_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    对于点云 a 中的每个点，找到点云 b 中距离最近的点的距离。

    逻辑:
      - 如果安装了 scipy，使用 cKDTree (O(N log N))，速度极快。
      - 否则，使用 Numpy 广播 (O(Na * Nb))，内存占用大且慢，仅作兜底。

    返回:
      长度为 N_a 的向量，包含每个点的最短距离。
    """
    a = np.asarray(a).reshape(-1, 3)
    b = np.asarray(b).reshape(-1, 3)
    if a.size == 0 or b.size == 0:
        return np.array([], dtype=float)

    if _HAVE_CKD:
        tree = _cKDTree(b)
        dists, _ = tree.query(a, k=1)  # k=1 表示找最近的一个点
        return dists

    # Numpy 回退实现 (不推荐用于大点云)
    # 计算所有点对的距离平方
    # a[:, None, :] 形状 (Na, 1, 3)
    # b[None, :, :] 形状 (1, Nb, 3)
    # diff 形状 (Na, Nb, 3)
    diff = a[:, None, :] - b[None, :, :]
    d2 = np.sum(diff * diff, axis=2)  # (Na, Nb)
    mins = np.min(d2, axis=1)
    return np.sqrt(mins)


# ---------------------------
# 核心指标: ADD / ADD-S (ADI)
# ---------------------------

def compute_add(pred_R: np.ndarray, pred_t: np.ndarray,
                gt_R: np.ndarray, gt_t: np.ndarray,
                model_points: np.ndarray) -> float:
    """
    计算 ADD (Average Distance of Model Points) 指标。
    适用于 **非对称** 物体。

    定义: 模型点在预测姿态下变换后，与在真值姿态下变换后的对应点之间的平均欧氏距离。

    参数:
      model_points: (M, 3) 模型表面采样点或顶点。

    返回:
      标量: 平均距离 (单位同 model_points/t，通常为米或毫米)。
    """
    model_points = np.asarray(model_points).reshape(-1, 3)

    pred_pts = transform_points(model_points, pred_R, pred_t)
    gt_pts = transform_points(model_points, gt_R, gt_t)

    # 点对点直接计算距离 (One-to-one mapping)
    d = np.linalg.norm(pred_pts - gt_pts, axis=1)
    return float(np.mean(d))


def compute_adds(pred_R: np.ndarray, pred_t: np.ndarray,
                 gt_R: np.ndarray, gt_t: np.ndarray,
                 model_points: np.ndarray) -> float:
    """
    计算 ADD-S (或称 ADI) 指标。
    适用于 **对称** 物体 (或者当你不确定物体是否对称时)。

    定义: 对于预测姿态下的每个点，在真值姿态下的点云中找到 **最近** 的那个点，计算平均距离。
    这解决了对称物体在旋转后点对应关系改变的问题。
    """
    model_points = np.asarray(model_points).reshape(-1, 3)

    pred_pts = transform_points(model_points, pred_R, pred_t)
    gt_pts = transform_points(model_points, gt_R, gt_t)

    # 最近邻搜索 (Many-to-one allowed)
    dists = _nearest_neighbor_distances(pred_pts, gt_pts)

    if dists.size == 0:
        return 0.0
    return float(np.mean(dists))


# 别名 ADI (Average Distance of Indistinguishable points)，文献中常用
compute_adi = compute_adds


# ---------------------------
# 基础指标: 旋转 / 平移误差
# ---------------------------

def rotation_error(R_pred: np.ndarray, R_gt: np.ndarray) -> float:
    """
    计算两个旋转矩阵之间的角度误差 (度)。
    公式: angle = arccos( (trace(R_diff) - 1) / 2 )
    """
    R_pred = np.asarray(R_pred).reshape(3, 3)
    R_gt = np.asarray(R_gt).reshape(3, 3)

    R_diff = R_pred @ R_gt.T
    trace = np.trace(R_diff)

    # 将数值限制在 [-1, 1] 之间，防止因为浮点误差导致 arccos 出现 NaN
    cos_angle = (trace - 1.0) / 2.0
    cos_angle = float(np.clip(cos_angle, -1.0, 1.0))

    angle_rad = math.acos(cos_angle)
    return float(math.degrees(angle_rad))


def translation_error(t_pred: np.ndarray, t_gt: np.ndarray) -> float:
    """
    计算平移向量之间的欧氏距离。
    """
    t_pred = np.asarray(t_pred).reshape(3, )
    t_gt = np.asarray(t_gt).reshape(3, )
    return float(np.linalg.norm(t_pred - t_gt))


# ---------------------------
# VSD (可见表面差异) - 简化接口
# ---------------------------

def compute_vsd_from_depth(pred_depth: np.ndarray,
                           gt_depth: np.ndarray,
                           valid_mask: Optional[np.ndarray] = None,
                           tau: float = 0.02) -> float:
    """
    基于已渲染好的深度图计算简化版 VSD。
    注意: 这不是完整的 BOP VSD 实现 (完整版需要实时渲染器计算可见性掩码)。

    参数:
      pred_depth, gt_depth: 相同形状的深度图 (单位: 米), 0 表示无效/背景。
      valid_mask: 可选的布尔掩码，指示只评估哪些像素 (例如只评估物体所在的区域)。
      tau: 容忍阈值 (米)。深度差小于此值的被认为是匹配的。

    返回:
      vsd_score: [0, 1], 越小越好。表示不匹配的可见像素比例。
    """
    if pred_depth.shape != gt_depth.shape:
        raise ValueError("pred_depth 和 gt_depth 形状必须相同")

    pd = np.asarray(pred_depth)
    gd = np.asarray(gt_depth)

    # 近似认为 GT 深度 > 0 的地方是物体可见区域
    gt_valid = (gd > 0)

    if valid_mask is not None:
        gt_valid = gt_valid & (valid_mask.astype(bool))

    if np.count_nonzero(gt_valid) == 0:
        # 没有可见像素 -> 结果不明确，返回 nan
        return float('nan')

    # 计算差异
    diff = np.abs(pd - gd)

    # 匹配条件: 深度差 <= tau 且 预测深度也有效 (>0)
    matched = (diff <= tau) & (pd > 0)

    # 统计: 在 GT 可见区域内，有多少像素是匹配的
    matched_visible = np.count_nonzero(matched & gt_valid)
    total_visible = np.count_nonzero(gt_valid)

    matched_fraction = matched_visible / float(total_visible)

    # VSD = 1 - 匹配比例 (即不匹配的比例)
    vsd = 1.0 - matched_fraction
    return float(vsd)


# ---------------------------
# 单样本评估包装器
# ---------------------------

def evaluate_single_prediction(pred: Dict[str, Any],
                               gt: Dict[str, Any],
                               model_points: np.ndarray,
                               thresholds: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """
    评估单个预测结果 vs 单个真值。
    这是一个常用的便利函数，自动计算 ADD, ADD-S, RE, TE 并判断是否通过阈值。

    参数:
      pred: 字典, 必须包含 'R', 't'. 可选 'depth_render'.
      gt: 字典, 必须包含 'R', 't'.
      model_points: 该物体的 3D 点云.
      thresholds: 阈值字典, 例如 {'add': 0.1, 're': 5.0}.

    返回:
      结果字典: 包含具体的误差数值和是否通过(pass_*)的布尔值。
    """
    thresholds = thresholds or {}
    pred_R = np.asarray(pred['R']).reshape(3, 3)
    pred_t = np.asarray(pred['t']).reshape(3, )
    gt_R = np.asarray(gt['R']).reshape(3, 3)
    gt_t = np.asarray(gt['t']).reshape(3, )

    # 计算核心指标
    add_val = compute_add(pred_R, pred_t, gt_R, gt_t, model_points)
    adds_val = compute_adds(pred_R, pred_t, gt_R, gt_t, model_points)
    re = rotation_error(pred_R, gt_R)
    te = translation_error(pred_t, gt_t)

    # 计算 VSD (如果提供了深度图)
    vsd_val = None
    if 'depth_render' in pred and 'depth_render' in gt:
        vsd_val = compute_vsd_from_depth(pred['depth_render'], gt['depth_render'],
                                         valid_mask=gt.get('mask', None),
                                         tau=thresholds.get('vsd_tau', 0.02))

    # 组装结果
    # 注意: BOP 标准中 ADD 阈值通常是物体直径的 10% (0.1 * diameter)
    # 这里的 thresholds['add'] 应该传入具体的数值 (比如 0.05m)，而不是百分比。
    res = {
        'add': add_val,
        'adds': adds_val,
        're': re,
        'te': te,
        'vsd': vsd_val,
        # 判断是否合格 (Pass/Fail)
        'pass_add': add_val <= float(thresholds.get('add', 0.1)),
        'pass_adds': adds_val <= float(thresholds.get('add', 0.1)),  # ADD-S 通常沿用 ADD 的阈值
        'pass_re': re <= float(thresholds.get('re', 5.0)),  # 角度阈值 (度)
        'pass_te': te <= float(thresholds.get('te', 0.05))  # 平移阈值 (米)
    }
    return res


# ---------------------------
# 批量评估与聚合
# ---------------------------

def evaluate_batch(predictions: Iterable[Dict[str, Any]],
                   gts: Iterable[Dict[str, Any]],
                   model_points_lookup: Dict[int, np.ndarray],
                   thresholds: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """
    批量评估一系列预测结果。

    参数:
      predictions: 预测列表 (List[Dict])。
      gts: 真值列表 (List[Dict])。需要与 predictions 按索引一一对应。
      model_points_lookup: 字典 {obj_id: model_points}，用于查找不同物体的点云。
      thresholds: 阈值配置。

    返回:
      汇总字典: 包含平均指标 (summary) 和每个物体的详细指标 (per_object)。
    """
    thresholds = thresholds or {}
    preds = list(predictions)
    gts = list(gts)

    if len(preds) != len(gts):
        raise ValueError("预测列表和真值列表的长度必须一致 (对于此简单批处理函数而言)")

    results = []
    per_obj = {}  # 用于按 obj_id 分组统计

    for p, g in zip(preds, gts):
        # 获取 obj_id
        obj_id = int(p.get('obj_id', g.get('obj_id', -1)))

        if obj_id not in model_points_lookup:
            raise KeyError(f"在 model_points_lookup 中找不到 obj_id {obj_id} 的模型点云")

        model_pts = np.asarray(model_points_lookup[obj_id])

        # 评估单个样本
        r = evaluate_single_prediction(p, g, model_pts, thresholds)

        # 注入 id 以便追踪
        r['scene_id'] = p.get('scene_id')
        r['im_id'] = p.get('im_id')
        r['obj_id'] = obj_id

        results.append(r)
        per_obj.setdefault(obj_id, []).append(r)

    # --- 统计聚合 (Aggregation) ---
    n = len(results)

    # 辅助函数: 安全求平均 (忽略 None)
    def safe_mean(key):
        vals = [r[key] for r in results if r.get(key) is not None]
        return float(np.mean(vals)) if len(vals) > 0 else float('nan')

    summary = {
        'n': n,
        'avg_add': safe_mean('add'),
        'avg_adds': safe_mean('adds'),
        'avg_re_deg': safe_mean('re'),
        'avg_te_m': safe_mean('te'),
        'vsd_mean': safe_mean('vsd'),
        # 召回率 (Recall / Accuracy): 通过阈值的比例
        'pass_rate_add': sum([1 for r in results if r['pass_add']]) / n if n > 0 else 0,
        'pass_rate_adds': sum([1 for r in results if r['pass_adds']]) / n if n > 0 else 0,
    }

    # 按物体统计 (Per-object summary)
    per_obj_summary = {}
    for obj_id, recs in per_obj.items():
        n_obj = len(recs)
        per_obj_summary[obj_id] = {
            'n': n_obj,
            'avg_add': float(np.mean([x['add'] for x in recs])),
            'pass_rate_add': float(sum([1 for x in recs if x['pass_add']]) / n_obj)
        }

    return {'summary': summary, 'per_object': per_obj_summary, 'all': results}


# ---------------------------
# IO 工具 (BOP 格式)
# ---------------------------

def save_bop_predictions(preds: Iterable[Dict[str, Any]], out_path: str):
    """
    将预测结果保存为 JSON 文件 (BOP 常用格式)。
    每个条目包含: scene_id, im_id, obj_id, score, R, t
    """
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    entries = []
    for p in preds:
        entry = {
            'scene_id': int(p.get('scene_id', -1)),
            'im_id': int(p.get('im_id', -1)),
            'obj_id': int(p.get('obj_id', -1)),
            'score': float(p.get('score', 1.0)),  # 置信度
            'R': np.asarray(p['R']).reshape(3, 3).tolist(),
            't': np.asarray(p['t']).reshape(3, ).tolist()
        }
        entries.append(entry)
    with open(out_path, 'w') as f:
        json.dump(entries, f, indent=2)


def load_bop_predictions(path: str) -> List[Dict[str, Any]]:
    """加载 save_bop_predictions 保存的 JSON 文件。"""
    with open(path, 'r') as f:
        data = json.load(f)
    return data


# ---------------------------
# 简单单元测试 / 使用示例
# ---------------------------
if __name__ == "__main__":
    print("正在运行 bop_eval 的基础完整性检查 (Sanity Checks)...")

    # 1. 定义一个微型模型: 以原点为中心的单位立方体的 8 个顶点
    cube_verts = np.array([
        [-0.5, -0.5, -0.5], [0.5, -0.5, -0.5],
        [-0.5, 0.5, -0.5], [0.5, 0.5, -0.5],
        [-0.5, -0.5, 0.5], [0.5, -0.5, 0.5],
        [-0.5, 0.5, 0.5], [0.5, 0.5, 0.5],
    ])
    model_points = cube_verts

    # 2. 模拟真值 (单位阵，原点)
    R_gt = np.eye(3)
    t_gt = np.zeros(3)

    # 3. 模拟预测:
    #    - 平移偏离 0.02 (2cm)
    #    - 旋转偏离 5度 (绕Z轴)
    angle_deg = 5.0
    angle_rad = math.radians(angle_deg)
    R_pred = np.array([
        [math.cos(angle_rad), -math.sin(angle_rad), 0],
        [math.sin(angle_rad), math.cos(angle_rad), 0],
        [0, 0, 1]
    ])
    t_pred = np.array([0.02, 0.0, 0.0])

    # 4. 运行计算
    add = compute_add(R_pred, t_pred, R_gt, t_gt, model_points)
    adds = compute_adds(R_pred, t_pred, R_gt, t_gt, model_points)
    re = rotation_error(R_pred, R_gt)
    te = translation_error(t_pred, t_gt)

    print(f"ADD Error: {add:.6f} m")
    print(f"ADD-S Error: {adds:.6f} m")
    print(f"Rotation Error: {re:.6f} deg (期望值 ~{angle_deg})")
    print(f"Translation Error: {te:.6f} m (期望值 0.02)")

    # 5. 测试 single eval 包装器
    pred_dict = {'R': R_pred, 't': t_pred}
    gt_dict = {'R': R_gt, 't': t_gt}
    # 阈值设定: ADD 允许 0.05m误差, 角度允许 10度
    res = evaluate_single_prediction(pred_dict, gt_dict, model_points,
                                     thresholds={'add': 0.05, 're': 10.0})
    print("Single Eval 结果:", res)

    print("检查完成。")