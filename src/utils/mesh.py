# src/utils/mesh.py
"""
通用 Mesh 加载与点云采样工具
---------------------------------------------------
为所有模块 (预处理器, 评估器) 提供统一的、带缓存的
3D 模型加载和点云采样功能。

特点:
- 优先使用 trimesh, 失败则回退到纯 Python/Numpy 实现。
- [核心] 全局缓存机制，避免重复I/O和重复采样。
"""

import os
import numpy as np
from tqdm import tqdm
from typing import Dict, Tuple, Optional, List, Union
from functools import lru_cache

# --- 可选依赖 (软依赖) ---
try:
    import trimesh

    _HAVE_TRIMESH = True
except Exception:
    _HAVE_TRIMESH = False

try:
    from scipy.spatial import cKDTree as _cKDTree

    _HAVE_CKD = True
except Exception:
    _HAVE_CKD = False

# -----------------------------------------------------
# 1. 缓存区 (全局)
# -----------------------------------------------------

# 缓存: { "mesh_path": (verts, faces) }
_MESH_CACHE: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

# 缓存: { ("mesh_path", n_points): sampled_points }
_POINTS_CACHE: Dict[Tuple[str, int], np.ndarray] = {}


# -----------------------------------------------------
# 2. 核心函数 (融合了我们之前的所有实现)
# -----------------------------------------------------

# [来自 tools/prepare_bop_dataset_pro.py]
# [改进] 添加 @lru_cache 以便在多进程中也能缓存
@lru_cache(maxsize=32)
def load_mesh(mesh_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    加载单个网格文件 (OBJ, PLY, STL 均支持)。
    优先使用 Trimesh。如果 Trimesh 不可用，则回退到简单的 .obj 解析器。

    返回:
      verts: (V, 3)
      faces: (F, 3)
    """
    if mesh_path in _MESH_CACHE:
        return _MESH_CACHE[mesh_path]

    if not os.path.exists(mesh_path):
        raise FileNotFoundError(f"Mesh 文件不存在: {mesh_path}")

    # --- 1. 优先尝试 Trimesh (功能最全) ---
    if _HAVE_TRIMESH:
        try:
            m = trimesh.load(mesh_path, force='mesh')
            verts = np.array(m.vertices, dtype=np.float32)
            faces = np.array(m.faces, dtype=np.int64)
            _MESH_CACHE[mesh_path] = (verts, faces)
            return verts, faces
        except Exception as e:
            print(f"Trimesh 加载失败 {mesh_path}: {e}。尝试 .obj 回退。")

    # --- 2. .obj 回退解析器 ---
    # (如果 Trimesh 失败或未安装)
    verts = []
    faces = []
    try:
        with open(mesh_path, 'r') as f:
            for line in f:
                if line.startswith('v '):
                    parts = line.strip().split()
                    verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
                elif line.startswith('f '):
                    parts = line.strip().split()
                    # OBJ 索引从 1 开始
                    # 假设是三角面，并处理 'f v/vt/vn' 格式
                    idxs = [int(p.split('/')[0]) - 1 for p in parts[1:4]]
                    faces.append(idxs)

        if not verts:
            raise RuntimeError("Trimesh 不可用，且 .obj 回退解析器未找到 'v' 顶点。")

        verts_np = np.array(verts, dtype=np.float32)
        faces_np = np.array(faces, dtype=np.int64)
        _MESH_CACHE[mesh_path] = (verts_np, faces_np)
        return verts_np, faces_np

    except Exception as e_obj:
        raise RuntimeError(f"加载模型 {mesh_path} 失败 (Trimesh 和 .obj 回退均失败): {e_obj}")


# [来自 src/metrics/bop_eval.py]
def sample_mesh_vertices(mesh: Tuple[np.ndarray, np.ndarray],
                         n_points: int = 1000) -> np.ndarray:
    """
    在 Mesh 表面进行均匀采样。

    参数:
      mesh: 元组 (vertices (V,3), faces (F,3))
      n_points: 采样点的数量

    返回:
      points: (n_points, 3) 的 Numpy 数组
    """

    verts, faces = mesh

    # --- 1. 优先使用 Trimesh 采样 (如果可用) ---
    if _HAVE_TRIMESH:
        try:
            # 创建一个临时的 Trimesh 对象 (不推荐，但功能最全)
            m_temp = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
            return m_temp.sample(n_points)
        except Exception:
            pass  # 回退到 Numpy 实现

    # --- 2. Numpy 回退实现 (按面积加权采样) ---
    if faces.shape[0] == 0:
        # 如果没有面 (例如只是一个点云)，则在顶点上采样
        indices = np.random.choice(verts.shape[0], n_points, replace=True)
        return verts[indices]

    v0 = verts[faces[:, 0]]
    v1 = verts[faces[:, 1]]
    v2 = verts[faces[:, 2]]

    # 计算所有三角形的面积
    tri_areas = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0), axis=1)
    area_sum = tri_areas.sum()

    if area_sum <= 1e-8:
        # 退化网格 (所有面积为0)，回退到顶点采样
        indices = np.random.choice(verts.shape[0], n_points, replace=True)
        return verts[indices]

    # 按面积比例选择 F 个面
    probs = tri_areas / area_sum
    face_indices = np.random.choice(faces.shape[0], size=n_points, p=probs)

    # --- 在选中的三角形内进行重心采样 ---
    # r1, r2 ~ Uniform(0, 1)
    r = np.random.rand(n_points, 2)
    sqrt_r1 = np.sqrt(r[:, 0])

    # (u, v, w) 是重心坐标
    u = 1.0 - sqrt_r1
    v = r[:, 1] * sqrt_r1
    w = 1.0 - u - v

    # p = w*v0 + u*v1 + v*v2
    sampled_points = (
            w[:, None] * v0[face_indices] +
            u[:, None] * v1[face_indices] +
            v[:, None] * v2[face_indices]
    )

    return sampled_points.astype(np.float32)


# [来自您的新脚本 mesh.py]
def get_model_points(mesh_path: str, n_points: int = 1000) -> np.ndarray:
    """
    获取采样的模型点云 (带缓存)。
    这是 Evaluator 和 Metric 计算应调用的主要函数。

    返回:
      (n_points, 3) Numpy 数组
    """
    cache_key = (mesh_path, n_points)

    # 1. 检查点云缓存
    if cache_key in _POINTS_CACHE:
        return _POINTS_CACHE[cache_key]

    # 2. 加载 Mesh (load_mesh 内部有自己的缓存)
    verts, faces = load_mesh(mesh_path)

    # 3. 采样
    pts = sample_mesh_vertices((verts, faces), n_points=n_points)

    # 4. 存入缓存并返回
    _POINTS_CACHE[cache_key] = pts
    return pts


# [来自您的新脚本 mesh.py]
def load_model_points_dict(model_dir: str,
                           obj_ids: List[int],
                           n_points: int = 2000) -> Dict[int, np.ndarray]:
    """
    [Evaluator 专用]
    为评估器 (Evaluator) 预加载所有需要的模型点云。

    参数:
      model_dir: BOP 的 'models' 或 'models_eval' 目录
      obj_ids: (list) 需要加载的物体 ID 列表
      n_points: (int) 为 ADD(-S) 采样的点数

    返回:
      { obj_id: (n_points, 3) 点云, ... }
    """
    result = {}
    print(f"[{__name__}] 正在预加载 {len(obj_ids)} 个模型的点云...")

    for oid in tqdm(obj_ids, desc="Pre-loading models"):
        # 自动查找 .ply (首选) 或 .obj
        path_ply = os.path.join(model_dir, f"obj_{oid:06d}.ply")
        path_obj = os.path.join(model_dir, f"obj_{oid:06d}.obj")

        mesh_path = None
        if os.path.exists(path_ply):
            mesh_path = path_ply
        elif os.path.exists(path_obj):
            mesh_path = path_obj
        else:
            print(f"警告: 找不到物体 {oid} 的模型文件 (既没有 .ply 也没有 .obj)")
            continue

        # get_model_points 内部有缓存，非常高效
        pts = get_model_points(mesh_path, n_points)
        result[oid] = pts

    print(f"[{__name__}] 成功加载 {len(result)} 个模型点云。")
    return result