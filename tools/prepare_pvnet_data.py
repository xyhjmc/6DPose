
"""
tools/prepare_pvnet_data.py

专业级 BOP -> PVNet 数据预处理脚本。
(BOP: http://bop.fel.cvut.cz/)

功能:
 - 适配 BOP 数据集标准格式 (例如 LINEMOD, T-LESS, YCB-V 等)。
 - [并行处理]: 使用 ProcessPoolExecutor 大幅加速数据处理。
 - [断点续跑]: 自动跳过已生成的 .npz 文件 (通过 --resume)。
 - [错误处理]: 捕获单个文件的处理异常，支持 --max-retries 重试。
 - [日志系统]: 将详细日志输出到控制台和文件 (prepare_bop.log)。
 - [结果报告]: 生成 index.json (汇总) 和 errors.csv (失败列表)。

输出 (每个实例一个 .npz 文件):
 - vertex: (2K, H, W) 顶点场 (像素偏移或单位向量)
 - mask: (H, W) 二值掩码
 - kp2d: (K, 2) 2D 关键点
 - kp3d: (K, 3) 3D 关键点
 - K: (3, 3) 相机内参
 - R, t: (3, 3), (3,) 物体姿态
 - rgb_path: (str) 原始 RGB 图像的绝对路径
"""

import os
import sys
import argparse
import json
import csv
import time
import traceback
from contextlib import contextmanager
from typing import Optional, Tuple, List, Dict, Any
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache  # [改进] 导入 lru_cache 用于进程内缓存
import logging

import numpy as np
import cv2
from tqdm import tqdm

# --- 可选导入 (软依赖) ---
try:
    import trimesh
except Exception:
    trimesh = None
try:
    import pyrender
except Exception:
    pyrender = None


# ---------------------
# 1. 日志设置
# ---------------------
def setup_logging(log_file: str, verbose: bool) -> logging.Logger:
    """
    配置日志记录器，同时输出到文件和控制台。
    """
    logger = logging.getLogger("prepare_bop")
    logger.setLevel(logging.DEBUG)  # 捕捉所有级别的日志

    # 避免重复添加 handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s %(levelname)-8s %(message)s", "%Y-%m-%d %H:%M:%S")

    # 1. 控制台处理器 (Console Handler)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG if verbose else logging.INFO)  # 根据 --verbose 设置级别
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # 2. 文件处理器 (File Handler)
    fh = logging.FileHandler(log_file, mode="a")  # 'a' = 追加模式
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


@contextmanager
def log_timing(logger: logging.Logger, message: str):
    start = time.time()
    logger.info(message)
    try:
        yield
    finally:
        elapsed = time.time() - start
        logger.info(f"{message} 用时: {elapsed:.2f}s")


# ---------------------
# 2. 核心辅助函数
# ---------------------

# [改进] 添加 lru_cache(maxsize=16)。
# 这将缓存最近 16 次调用的结果。在多进程中，
# 这意味着每个工作进程 (worker) 只需要从磁盘加载一次模型。
@lru_cache(maxsize=16)
def load_mesh_verts_faces(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    加载 3D 模型的顶点 (V, 3) 和面 (F, 3)。
    优先使用 Trimesh。如果 Trimesh 不可用，则回退到简单的 .obj 解析器。
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"3D 模型文件未找到: {path}")

    if trimesh is not None:
        try:
            # force='mesh' 确保加载的是几何体，而不是场景
            m = trimesh.load(path, force='mesh')
            verts = np.array(m.vertices, dtype=np.float32)
            faces = np.array(m.faces, dtype=np.int64)
            return verts, faces
        except Exception as e_trimesh:
            print(f"Trimesh 加载失败 {path}: {e_trimesh}。尝试 .obj 回退。")

    # .obj 回退解析器
    verts = []
    faces = []
    try:
        with open(path, 'r') as f:
            for line in f:
                if line.startswith('v '):
                    parts = line.strip().split()
                    verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
                elif line.startswith('f '):
                    parts = line.strip().split()
                    idxs = [int(p.split('/')[0]) - 1 for p in parts[1:4]]  # 索引-1
                    faces.append(idxs)
        return np.array(verts, dtype=np.float32), np.array(faces, dtype=np.int64)
    except Exception as e_obj:
        raise RuntimeError(f"加载模型 {path} 失败 (Trimesh 和 .obj 回退均失败): {e_obj}")


def farthest_point_sampling(pts: np.ndarray, k: int, seed: Optional[int] = 0) -> np.ndarray:
    """
    最远点采样 (FPS) (N, 3) -> (K, 3)。
    使用 O(N*K) 贪心算法，对于离线预处理足够快。
    """
    N = pts.shape[0]
    if k >= N:
        return pts.copy()[:k]

    np.random.seed(seed)
    chosen_indices = np.zeros((k,), dtype=np.int64)
    chosen_indices[0] = np.random.randint(0, N)
    min_dists = np.linalg.norm(pts - pts[chosen_indices[0]], axis=1)

    for i in range(1, k):
        idx = int(np.argmax(min_dists))  # 距离最远的点
        chosen_indices[i] = idx
        new_dists = np.linalg.norm(pts - pts[idx], axis=1)
        min_dists = np.minimum(min_dists, new_dists)  # 更新最小距离

    return pts[chosen_indices]


def read_pose_from_ann(ann: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    从 BOP 标注字典 (来自 scene_gt.json) 中读取 R 和 t。
    BOP 标准单位: t 是毫米 (mm)。
    """
    R = np.array(ann['cam_R_m2c'], dtype=np.float32).reshape(3, 3)
    t = np.array(ann['cam_t_m2c'], dtype=np.float32).reshape(3, )
    return R, t


def project_points(pts3d: np.ndarray, R: np.ndarray, t: np.ndarray, K: np.ndarray) -> np.ndarray:
    """
    [Numpy] 投影 3D 点 (N, 3) 到 2D 像素坐标 (N, 2)。
    """
    pts_cam = (R @ pts3d.T).T + t.reshape(1, 3)
    proj = (K @ pts_cam.T).T
    z = proj[:, 2:3].copy()
    z[z < 1e-8] = 1e-8  # 防止 z=0 或 z<0
    pts2d = proj[:, :2] / z
    return pts2d.astype(np.float32)


def make_vertex_map_from_kps(kp2d: np.ndarray, mask: np.ndarray, use_offset: bool = False) -> np.ndarray:
    """
    [核心] 构建顶点场 (Vertex Field)。

    参数:
      kp2d: (K, 2) 关键点 (x, y) 坐标
      mask: (H, W) 二值掩码 (0/1)
      use_offset: (bool)
                  - True:  返回 (kp - pixel) 像素偏移 (用于 SmoothL1Loss)
                  - False: 返回 (kp - pixel) 的单位方向向量 (默认)

    返回:
      vertex: (2K, H, W) 顶点场 (float32)
    """
    H, W = mask.shape
    K = kp2d.shape[0]

    # xs (H, W), ys (H, W)
    xs, ys = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))

    vertex = np.zeros((2 * K, H, W), dtype=np.float32)

    for i in range(K):
        kx, ky = float(kp2d[i, 0]), float(kp2d[i, 1])  # (x, y)
        dx, dy = kx - xs, ky - ys  # 向量场 (H, W)

        if use_offset:
            vx, vy = dx, dy
        else:
            dist = np.sqrt(dx * dx + dy * dy)
            nz = dist.copy()
            nz[nz < 1e-8] = 1.0  # 防止除零
            vx, vy = dx / nz, dy / nz

        # [修复] 必须在两种模式下都应用掩码
        mask_bool = (mask > 0)
        vertex[2 * i, :, :] = vx * mask_bool
        vertex[2 * i + 1, :, :] = vy * mask_bool

    return vertex


def render_mask_from_mesh(mesh_verts, mesh_faces, R, t, K, H, W) -> np.ndarray:
    """
    [可选] 使用 pyrender 渲染二值掩码。
    """
    if trimesh is None or pyrender is None:
        raise RuntimeError("必须安装 pyrender 和 trimesh 才能使用 --render-if-no-mask。")

    mesh = trimesh.Trimesh(vertices=mesh_verts, faces=mesh_faces, process=False)
    m = pyrender.Mesh.from_trimesh(mesh, smooth=False)
    scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0])  # 黑色背景

    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    camera = pyrender.IntrinsicsCamera(fx, fy, cx, cy, znear=0.05, zfar=100000.0)
    scene.add(camera, pose=np.eye(4))  # 相机在原点

    # 将模型放置在 R|t 姿态
    T_mesh = np.eye(4)
    T_mesh[:3, :3] = R
    T_mesh[:3, 3] = t
    scene.add(m, pose=T_mesh)

    r = pyrender.OffscreenRenderer(viewport_width=W, viewport_height=H)
    color, depth = r.render(scene)
    r.delete()

    mask = (depth > 0).astype(np.uint8)  # 深度 > 0 处即为物体
    return mask


# -------------------------
# 3. 并行工作函数 (Worker)
# -------------------------
def process_instance_task(task: Dict[str, Any]) -> Dict[str, Any]:
    """
    [并行工作单元]
    此函数在单独的进程中执行。它必须是健壮的。
    它接收一个 'task' 字典，返回一个结果字典。
    """
    try:
        # --- 1. 解包任务 ---
        rgb_path = task['rgb_path']
        mask_path = task.get('mask_path', None)  # mask_path 可能为 None
        R = task['R']
        t = task['t']
        K = task['K']
        kp3d = task['kp3d']
        out_path = task['out_path']
        use_offset = task['use_offset']
        render_if_no_mask = task['render_if_no_mask']
        # [改进] 只接收 model_path，而不是完整的 mesh 数据
        model_path = task.get('model_path', None)

        # [修复 1] 解包我们新添加的 ID
        obj_id = task['obj_id']
        scene_id = task['scene_id']
        im_id = task['im_id']

        # --- 2. 加载图像 ---
        img = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"无法读取图像: {rgb_path}")
        H, W = img.shape[:2]

        # --- 3. 加载或渲染掩码 ---
        mask = None
        if mask_path and os.path.exists(mask_path):
            # [改进] 使用 IMREAD_UNCHANGED 更健壮
            mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            if mask is None:
                raise RuntimeError(f"无法读取掩码 (文件可能损坏): {mask_path}")

            # 标准化为 (H, W) 和 0/1
            if mask.ndim == 3:
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            mask = (mask > 0).astype(np.uint8)  # [改进] 0/255 -> 0/1
        else:
            # 掩码未找到，尝试渲染
            if render_if_no_mask:
                if model_path is None:
                    raise RuntimeError("需要渲染掩码，但 model_path 未提供。")

                # [改进] 在工作进程中按需加载模型
                # load_mesh_verts_faces 被 @lru_cache 缓存
                mesh_verts, mesh_faces = load_mesh_verts_faces(model_path)

                mask = render_mask_from_mesh(mesh_verts, mesh_faces, R, t, K, H, W)
            else:
                raise RuntimeError(f"在 {mask_path} 找不到掩码，且 --render-if-no-mask=False。")

        # --- 4. 计算 2D 关键点 ---
        kp2d = project_points(kp3d, R, t, K)  # (K, 2)

        # --- 5. [核心] 计算顶点场 ---
        vertex = make_vertex_map_from_kps(kp2d, mask, use_offset=use_offset)

        # --- 6. 保存 .npz ---
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        np.savez_compressed(
            out_path,
            kp2d=kp2d.astype(np.float32),
            kp3d=kp3d.astype(np.float32),
            K=K.astype(np.float32),
            vertex=vertex.astype(np.float32),
            mask=mask.astype(np.uint8),
            rgb_path=rgb_path,  # 存储相对/绝对路径，供 DataLoader 使用
            R=R.astype(np.float32),
            t=t.astype(np.float32),
            # [修复 2] 将 ID 保存到 .npz 文件中
            obj_id = np.array(obj_id, dtype=np.int32),
            scene_id = np.array(scene_id, dtype=np.int32),
            im_id = np.array(im_id, dtype=np.int32)
        )

        # 成功返回
        return {'status': 'ok', 'out_path': out_path, 'task': task}

    except Exception as e:
        # [关键] 捕获所有异常，防止工作进程崩溃
        tb = traceback.format_exc()  # 获取详细的堆栈跟踪
        return {'status': 'error', 'error': f"{e}\n{tb}", 'task': task}


# -------------------------
# 4. 任务构建与执行
# -------------------------
def build_tasks(data_root: str, split: str, obj_id: int, out_dir: str,
                kp3d: np.ndarray, use_offset: bool, render_if_no_mask: bool,
                model_path: str, logger: logging.Logger) -> List[Dict[str, Any]]:
    """
    [任务构建器]
    遍历 BOP 目录结构，为*每一个*需要处理的物体实例构建一个 'task' 字典。
    """
    split_path = os.path.join(data_root, split)
    if not os.path.isdir(split_path):
        raise FileNotFoundError(f"数据集划分路径未找到: {split_path}")

    tasks = []
    # 遍历所有场景文件夹, e.g., '000001', '000002', ...
    scene_folders = sorted([d for d in os.listdir(split_path)
                            if os.path.isdir(os.path.join(split_path, d))])

    for scene in scene_folders:
        scene_path = os.path.join(split_path, scene)
        gt_path = os.path.join(scene_path, "scene_gt.json")
        cam_path = os.path.join(scene_path, "scene_camera.json")

        if not (os.path.exists(gt_path) and os.path.exists(cam_path)):
            logger.warning(f"场景 {scene} 缺少 scene_gt.json 或 scene_camera.json，跳过。")
            continue

        with open(gt_path, 'r') as f:
            gt_data = json.load(f)
        with open(cam_path, 'r') as f:
            cam_data = json.load(f)

        # 遍历场景中的所有图像, e.g., '0', '1', ...
        for img_id_str, anns in gt_data.items():
            img_id = int(img_id_str)

            # 检查相机数据是否存在
            if img_id_str not in cam_data:
                logger.warning(f"场景 {scene} 图像 {img_id} 缺少相机数据，跳过。")
                continue

            cam_K = np.array(cam_data[img_id_str]['cam_K'], dtype=np.float32).reshape(3, 3)

            # 遍历图像中的所有物体实例
            for inst_idx, ann in enumerate(anns):
                # 检查是否是我们感兴趣的物体
                if ann['obj_id'] != obj_id:
                    continue

                # --- 找到一个匹配的实例 ---

                # 1. 查找 RGB 路径 (BOP 可能存 .png 或 .jpg)
                rgb_png = os.path.join(scene_path, "rgb", f"{img_id:06d}.png")
                rgb_jpg = os.path.join(scene_path, "rgb", f"{img_id:06d}.jpg")
                rgb_path = rgb_png if os.path.exists(rgb_png) else (rgb_jpg if os.path.exists(rgb_jpg) else None)

                if rgb_path is None:
                    logger.warning(f"图像文件缺失：场景 {scene} 图像 {img_id}。跳过此实例。")
                    continue

                # 2. 查找掩码路径 (BOP 命名: {img_id}_{inst_idx})
                # 优先使用 'mask_visib' (可见掩码)，其次 'mask' (完整掩码)
                mask_visib = os.path.join(scene_path, "mask_visib", f"{img_id:06d}_{inst_idx:06d}.png")
                mask_full = os.path.join(scene_path, "mask", f"{img_id:06d}_{inst_idx:06d}.png")

                chosen_mask = mask_visib if os.path.exists(mask_visib) else (
                    mask_full if os.path.exists(mask_full) else None)

                if chosen_mask is None and not render_if_no_mask:
                    logger.warning(f"掩码文件缺失：{mask_visib} 或 {mask_full}，且未启用渲染。跳过此实例。")
                    continue

                # 3. 获取姿态
                R, t = read_pose_from_ann(ann)

                # 4. 定义输出路径
                out_name = f"obj_{obj_id:06d}_{scene}_{img_id:06d}_{inst_idx:06d}.npz"
                out_path = os.path.join(out_dir, out_name)

                # 5. [改进] 创建任务字典，不包含大型 mesh 数据
                task = {
                    'rgb_path': rgb_path,
                    'mask_path': chosen_mask,  # 可能是 None
                    'R': R, 't': t, 'K': cam_K,
                    'kp3d': kp3d,
                    'out_path': out_path,
                    'use_offset': use_offset,
                    'render_if_no_mask': render_if_no_mask,
                    'model_path': model_path,  # [改进] 只传递路径
                    # [修复] 添加这三个 ID
                    'obj_id': obj_id,
                    'scene_id': int(scene),
                    'im_id': img_id
                }
                tasks.append(task)

    return tasks


def run_parallel(tasks: List[Dict[str, Any]], num_workers: int, logger: logging.Logger,
                 max_retries: int = 1, resume: bool = True, overwrite: bool = False):
    """
    [并行执行器]
    使用 ProcessPoolExecutor 并行执行所有任务，并处理重试。

    返回:
      (List[Dict]) 最终失败的任务列表。
    """

    # --- 1. 过滤任务 (用于断点续跑) ---
    filtered_tasks = []
    if not resume:  # 如果不续跑 (即 --resume=False)，但也不覆盖
        logger.info("非续跑模式，将处理所有任务。")
        filtered_tasks = tasks
    else:
        for t in tasks:
            if os.path.exists(t['out_path']) and not overwrite:
                # 文件存在，且不覆盖 -> 跳过
                continue
            filtered_tasks.append(t)

    tasks = filtered_tasks
    total = len(tasks)
    if total == 0:
        logger.info("所有任务均已处理完毕。")
        return []

    logger.info(f"开始处理 {total} 个新任务 (使用 {num_workers} 个工作进程, {max_retries} 次重试)")

    # --- 2. 主循环 (包括重试) ---
    failed_tasks = tasks
    retries = 0

    while retries <= max_retries:
        if retries > 0:
            logger.info(f"--- 开始第 {retries}/{max_retries} 轮重试 ({len(failed_tasks)} 个任务) ---")

        # 重新设置任务列表
        tasks_to_run = failed_tasks
        failed_tasks = []  # 存储本轮失败的任务

        if not tasks_to_run:
            break  # 上一轮全部成功

        with ProcessPoolExecutor(max_workers=max(1, num_workers)) as ex:
            # 提交所有任务并记录起始时间
            future_to_task = {}
            start_time_map = {}
            for task in tasks_to_run:
                fut = ex.submit(process_instance_task, task)
                future_to_task[fut] = task
                start_time_map[fut] = time.time()

            pbar_desc = f"处理 (第 {retries + 1}/{max_retries + 1} 轮)"
            pbar = tqdm(total=len(future_to_task), desc=pbar_desc, unit=" 实例")

            success_count = 0
            failure_count = 0

            # 使用 as_completed 实时获取已完成的任务
            for fut in as_completed(future_to_task):
                task = future_to_task[fut]
                task_duration = time.time() - start_time_map.get(fut, time.time())
                try:
                    res = fut.result()  # 获取工作进程的返回字典
                except Exception as e:
                    # 这种情况 (e.g., worker 崩溃) 应该很少见，但仍需处理
                    logger.error(f"工作进程崩溃于任务 {task.get('out_path')}: {e}")
                    res = {'status': 'error', 'error': str(e), 'task': task}

                # 处理工作进程返回的结果
                if res['status'] == 'ok':
                    success_count += 1
                else:
                    failure_count += 1
                    logger.error(f"任务失败: {task.get('out_path')}\n  "
                                 f"错误: {res.get('error').splitlines()[0]} (耗时 {task_duration:.2f}s)")  # 只打印第一行错误
                    logger.debug(f"完整错误: {res.get('error')}\n任务耗时: {task_duration:.2f}s")  # 完整错误写入 debug 日志
                    failed_tasks.append(task)

                pbar.update(1)
                pbar.set_postfix({
                    "成功": success_count,
                    "失败": failure_count,
                    "耗时(s)": f"{task_duration:.2f}"
                })
            pbar.close()

        retries += 1

    logger.info(f"处理完成。最终失败任务数: {len(failed_tasks)}")
    return failed_tasks


ERROR_CSV_HEADER = ['out_path', 'rgb_path', 'mask_path', 'obj_id', 'scene', 'img_id', 'reason']


def _write_error_rows(rows: List[List[Any]], csv_path: str, logger: logging.Logger, mode: str = 'a'):
    dir_path = os.path.dirname(csv_path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    need_header = mode == 'w' or not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    with open(csv_path, mode, newline='') as f:
        writer = csv.writer(f)
        if need_header:
            writer.writerow(ERROR_CSV_HEADER)
        writer.writerows(rows)
    logger.info(f"错误记录已写入 {csv_path} ({len(rows)} 条)。")


def write_failure_csv(failures: List[Dict[str, Any]], csv_path: str, logger: logging.Logger):
    """
    将失败的任务列表写入 errors.csv。
    """
    logger.info(f"正在将 {len(failures)} 个失败记录写入: {csv_path}")
    rows = []
    for t in failures:
        name_parts = os.path.basename(t.get('out_path', 'unknown')).split('_')
        scene = name_parts[2] if len(name_parts) > 2 else 'unknown'
        img_id = name_parts[3] if len(name_parts) > 3 else 'unknown'
        obj_id = name_parts[1] if len(name_parts) > 1 else 'unknown'

        rows.append([
            t.get('out_path'),
            t.get('rgb_path'),
            t.get('mask_path', 'N/A'),
            obj_id,
            scene,
            img_id,
            'processing_error'
        ])

    _write_error_rows(rows, csv_path, logger, mode='w')


def write_index_file(out_dir: str, logger: logging.Logger) -> int:
    """
    [索引生成器]
    遍历输出目录，生成一个 index.json 文件，汇总所有 .npz 文件的元数据。
    这对于 Dataset 类快速构建文件列表非常有用。

    返回生成的条目数。
    """
    start_time = time.time()
    npz_entries = [e for e in os.scandir(out_dir) if e.is_file() and e.name.endswith('.npz')]
    total = len(npz_entries)
    logger.info(f"正在为 {out_dir} 生成 index.json，共 {total} 个 .npz")
    idx = []
    for entry in tqdm(npz_entries, desc="生成索引", unit="文件"):
        p = entry.path
        try:
            # 只加载键 (keys)，不加载数据，速度极快
            with np.load(p) as d:
                item = {
                    'file': entry.name,
                    'rgb_path': str(d['rgb_path']),
                    'kp3d_shape': d['kp3d'].shape,
                    'vertex_shape': d['vertex'].shape,
                    'mask_shape': d['mask'].shape
                }
                idx.append(item)
        except Exception as e:
            logger.warning(f"索引器: 无法读取 {p} (可能已损坏): {e}")

    # 按文件名排序
    idx.sort(key=lambda x: x['file'])

    with open(os.path.join(out_dir, 'index.json'), 'w') as f:
        json.dump(idx, f, indent=2)
    elapsed = time.time() - start_time
    logger.info(f"成功写入 index.json，包含 {len(idx)} 个条目。耗时 {elapsed:.2f}s")
    if elapsed > 10:
        logger.info("索引生成耗时较长，如在网络盘上运行可考虑并行或本地磁盘生成。")
    return len(idx)


def clean_index_file(out_dir: str, errors_path: str, logger: logging.Logger,
                     enable_clean: bool = True, min_foreground: int = 1) -> Tuple[int, int]:
    """
    清理 index.json，剔除 mask 前景像素过少的样本。

    返回 (清理前数量, 清理后数量)。
    """
    if not enable_clean:
        logger.info("已跳过 index.json 清理 (--disable-clean-index)")
        return 0, 0

    index_path = os.path.join(out_dir, 'index.json')
    if not os.path.exists(index_path):
        logger.warning(f"未找到 {index_path}，跳过清理。")
        return 0, 0

    with open(index_path, 'r') as f:
        index = json.load(f)

    total_before = len(index)
    logger.info(f"开始清理 index.json，共 {total_before} 条。前景阈值: {min_foreground} 像素")

    clean_index = []
    removed_records = []
    for rec in tqdm(index, desc="清理 index", unit="条目"):
        npz_rel = rec.get('file')
        if npz_rel is None:
            logger.warning(f"记录缺少 file 字段: {rec}")
            continue

        npz_path = os.path.join(out_dir, npz_rel)
        file_name = os.path.basename(npz_rel)
        parts = file_name.split('_')
        obj_id = parts[1] if len(parts) > 1 else 'unknown'
        scene = parts[2] if len(parts) > 2 else 'unknown'
        img_id = parts[3] if len(parts) > 3 else 'unknown'
        if not os.path.exists(npz_path):
            logger.warning(f"[缺失文件] {npz_path}")
            removed_records.append([npz_path, rec.get('rgb_path', 'N/A'), 'N/A', obj_id, scene, img_id, 'missing_npz'])
            continue

        try:
            with np.load(npz_path) as data:
                mask = data['mask']
        except Exception as e:
            logger.warning(f"读取 {npz_path} 失败: {e}")
            removed_records.append([npz_path, rec.get('rgb_path', 'N/A'), 'N/A', obj_id, scene, img_id, 'corrupted_npz'])
            continue

        if mask.sum() < min_foreground:
            removed_records.append([
                npz_path,
                rec.get('rgb_path', 'N/A'),
                rec.get('file', 'N/A'),
                obj_id,
                scene,
                img_id,
                'mask_empty'
            ])
            continue

        clean_index.append(rec)

    backup_path = os.path.join(out_dir, 'index_raw.json')
    if not os.path.exists(backup_path):
        os.rename(index_path, backup_path)
        logger.info(f"已备份原始索引到 {backup_path}")
    else:
        logger.info("备份 index_raw.json 已存在，将直接覆盖 index.json")

    with open(index_path, 'w') as f:
        json.dump(clean_index, f, indent=2)

    if removed_records:
        _write_error_rows(removed_records, errors_path, logger, mode='a')

    total_after = len(clean_index)
    logger.info(f"清理完成: 原始 {total_before} 条，保留 {total_after} 条，移除 {len(removed_records)} 条。")
    return total_before, total_after



def parse_args():
    p = argparse.ArgumentParser(description="Prepare BOP dataset for PVNet (professional)")
    p.add_argument("--data-root", required=False, default="/home/xyh/datasets/LM(BOP)/lm",help="BOP dataset root (contains models/, scenes/)")
    p.add_argument("--dataset-split", required=False,default="test" ,help="split folder (e.g. train_pbr)")
    p.add_argument("--obj-id", required=False,default=8 ,type=int, help="object ID (int)")
    p.add_argument("--out-dir", required=False, default="/home/xyh/PycharmProjects/6DPose/data/linemod_pvnet/driller_test",help="output directory for .npz files")
    p.add_argument("--num-kp", type=int, default=9, help="number of keypoints K")
    p.add_argument("--kp3d-path", default=None, help="optional .npy with (K,3) keypoints")
    p.add_argument("--use-offset", default=False,action='store_true', help="generate offset maps (pixel units) instead of unit vectors")
    p.add_argument("--render-if-no-mask", action='store_true', help="use pyrender to render mask if not found")
    p.add_argument("--model-file", default=None, help="override model file path (obj_xxxxxx.ply) if needed")
    p.add_argument("--num-workers", type=int, default=8, help="number of worker processes")
    p.add_argument("--max-retries", type=int, default=1, help="retry failed tasks up to this many times")
    p.add_argument("--resume", default=False,action='store_true', help="skip existing outputs")
    p.add_argument("--overwrite", action='store_true', help="overwrite existing outputs")
    p.add_argument("--log-file", default="prepare_bop_dataset.log", help="log file path")
    p.add_argument("--errors-csv", default="errors.csv", help="errors CSV path")
    p.add_argument("--disable-clean-index", action='store_true', help="skip cleaning index.json after generation")
    p.add_argument("--clean-min-foreground", type=int, default=1, help="minimum foreground pixels to keep an entry during cleaning")
    p.add_argument("--verbose", action='store_true', help="verbose logging to console")
    return p.parse_args()

# -------------------------
# 5. 主入口
# -------------------------

def main():
    args = parse_args()

    # --- 1. 设置 ---
    # 确保输出目录存在
    os.makedirs(args.out_dir, exist_ok=True)
    # 设置日志
    log_path = os.path.join(args.out_dir, args.log_file)
    errors_path = os.path.join(args.out_dir, args.errors_csv)
    logger = setup_logging(log_path, verbose=args.verbose)

    logger.info(f"--- BOP -> PVNet 预处理开始 (专业版) ---")
    logger.info(f"日志文件: {log_path}")
    logger.info(f"数据根目录: {args.data_root}")
    logger.info(f"处理划分: {args.dataset_split}")
    logger.info(f"处理物体 ID: {args.obj_id}")
    logger.info(f"输出目录: {args.out_dir}")
    logger.info(f"并行工作进程: {args.num_workers}")
    logger.info(f"使用偏移向量 (SmoothL1): {args.use_offset}")

    # --- 2. 查找 3D 模型 ---
    model_dir = os.path.join(args.data_root, "models")
    model_path = args.model_file
    if model_path is None:
        model_path_ply = os.path.join(model_dir, f"obj_{args.obj_id:06d}.ply")
        if os.path.exists(model_path_ply):
            model_path = model_path_ply
        else:
            # 自动查找其他扩展名
            for ext in ('.obj', '.stl', '.off'):
                cand = os.path.join(model_dir, f"obj_{args.obj_id:06d}{ext}")
                if os.path.exists(cand):
                    model_path = cand
                    break
    if model_path is None:
        logger.error(f"在 {model_dir} 中找不到 obj_{args.obj_id:06d} 的模型文件。")
        if args.render_if_no_mask:
            logger.error("启用了渲染，但没有模型，即将退出。")
            return
    else:
        logger.info(f"使用 3D 模型: {model_path}")

    # --- 3. [修复] 加载或计算 3D 关键点 (kp3d) ---
    kp3d = None
    if args.kp3d_path:
        try:
            kp3d = np.load(args.kp3d_path).astype(np.float32)
            if kp3d.ndim != 2 or kp3d.shape[1] != 3:
                logger.error(f"--kp3d_path ({args.kp3d_path}) 必须是 (N, 3) numpy 数组。")
                return
            logger.info(f"从 {args.kp3d_path} 加载了 {kp3d.shape[0]} 个 3D 关键点。")
        except Exception as e:
            logger.error(f"加载 --kp3d_path ({args.kp3d_path}) 失败: {e}")
            return
    else:
        # Fallback: 自动计算 8 角点 + 质心
        logger.info("未提供 --kp3d_path，将自动计算 8 个角点 + 质心 (需要 Trimesh)。")
        if trimesh is None:
            logger.error("Trimesh 未安装，无法自动计算关键点。请 `pip install trimesh` 或提供 --kp3d_path。")
            return
        if model_path is None:
            logger.error("无法计算关键点，因为找不到 3D 模型。")
            return
        try:
            m = trimesh.load(model_path)
            corners = m.bounding_box.vertices  # (8, 3)
            centroid = m.centroid  # (3,)
            kp3d = np.vstack([corners, centroid])
            logger.info(f"自动计算了 {kp3d.shape[0]} 个 3D 关键点 (8 角点 + 质心)。")
        except Exception as e:
            logger.error(f"使用 Trimesh 计算关键点失败: {e}")
            return

    # 确保关键点数量符合要求
    if kp3d.shape[0] < args.num_kp:
        raise RuntimeError(f"加载的 3D 关键点数 ({kp3d.shape[0]}) 少于 --num_kp ({args.num_kp})")
    kp3d = kp3d[:args.num_kp]  # 取前 K 个
    logger.info(f"最终使用 {kp3d.shape[0]} 个 3D 关键点。")

    # --- 4. 构建任务列表 ---
    try:
        with log_timing(logger, "构建任务列表"):
            tasks = build_tasks(args.data_root, args.dataset_split, args.obj_id, args.out_dir,
                                kp3d=kp3d, use_offset=args.use_offset,
                                render_if_no_mask=args.render_if_no_mask,
                                model_path=model_path, logger=logger)
    except Exception as e:
        logger.exception("构建任务列表时发生致命错误，退出。")
        return

    logger.info(f"总共构建了 {len(tasks)} 个处理任务。")

    # --- 5. 执行并行处理 ---
    start_time = time.time()
    failed_tasks = run_parallel(tasks, num_workers=args.num_workers, logger=logger,
                                max_retries=args.max_retries, resume=args.resume,
                                overwrite=args.overwrite)
    end_time = time.time()

    # --- 6. 报告 ---
    if failed_tasks:
        write_failure_csv(failed_tasks, errors_path, logger)

    try:
        with log_timing(logger, "生成 index.json"):
            write_index_file(args.out_dir, logger)
    except Exception:
        logger.exception("生成 index.json 失败。")

    try:
        clean_index_file(args.out_dir, errors_path, logger,
                         enable_clean=not args.disable_clean_index,
                         min_foreground=args.clean_min_foreground)
    except Exception:
        logger.exception("清理 index.json 失败。")

    logger.info(f"--- 预处理完成 ---")
    logger.info(f"总耗时: {(end_time - start_time):.2f} 秒")
    logger.info(f"成功任务: {len(tasks) - len(failed_tasks)}")
    logger.info(f"失败任务: {len(failed_tasks)} (详情见: {errors_path})")
    logger.info(f"索引文件: {os.path.join(args.out_dir, 'index.json')}")


if __name__ == "__main__":
    main()
