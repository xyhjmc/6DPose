# src/engines/evaluator.py
"""
评估引擎 (Evaluator Engine) - [专业融合版]

设计哲学:
- [灵活] 自动检测模型输出 ('R'/'t', 或 'kp_2d', 或 'vertex'/'seg')。
- [健壮] 动态导入项目工具 (utils/metrics)，如果导入失败，则优雅地回退 (Fallback)
         到本脚本内实现的轻量级 PnP 和度量计算。
- [解耦] 优先使用 src.metrics.bop_eval.evaluate_batch (BOP 标准) 进行评估。
- [BOP 兼容] 自动加载 BOP 的 models_info.json 和 3D 模型点云。
"""

import os
import json
import time
from typing import Any, Dict, List, Optional, Tuple
import torch.amp
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# -----------------------------------------------------------------
# 4. 轻量级回退 (Fallback) 度量函数
# (当 src.metrics.bop_eval 导入失败时使用)
# -----------------------------------------------------------------

# 尝试导入 scipy 以加速 ADD-S，如果失败则使用纯 Numpy
try:
    from scipy.spatial import cKDTree

    _HAVE_SCIPY_CKDTREE = True
except ImportError:
    _HAVE_SCIPY_CKDTREE = False
    print("[Evaluator 回退警告] 未找到 'scipy'。ADD-S 度量将使用较慢的 O(N^2) Numpy 实现。")


def _fallback_rotation_error_deg(R_pred: np.ndarray, R_gt: np.ndarray) -> float:
    """[Fallback] 计算两个 3x3 旋转矩阵之间的角度误差 (度)。"""
    R_pred = np.asarray(R_pred)
    R_gt = np.asarray(R_gt)

    # 相对旋转: R_rel = R_pred.T @ R_gt
    # (或者 R_pred @ R_gt.T 也可以，结果相同)
    R_rel = R_pred.T @ R_gt

    # 计算迹 (Trace)
    trace = np.trace(R_rel)

    # 保证 arccos 的输入在 [-1, 1] 范围内
    trace_clipped = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)

    # 角度 (弧度)
    angle_rad = np.arccos(trace_clipped)

    return float(np.degrees(angle_rad))


def _fallback_translation_error(t_pred: np.ndarray, t_gt: np.ndarray) -> float:
    """[Fallback] 计算两个平移向量之间的欧氏距离。"""
    return float(np.linalg.norm(t_pred.reshape(3, ) - t_gt.reshape(3, )))


def _fallback_add_metric(R_pred: np.ndarray, t_pred: np.ndarray,
                         R_gt: np.ndarray, t_gt: np.ndarray,
                         model_points: np.ndarray) -> float:
    """
    [Fallback] ADD 度量 (用于非对称物体)。
    计算变换后的模型点之间的平均欧氏距离。

    参数:
      model_points: (N, 3) 3D 模型点云
    """
    # (N, 3)
    pts_pred = (R_pred @ model_points.T).T + t_pred.reshape(1, 3)
    pts_gt = (R_gt @ model_points.T).T + t_gt.reshape(1, 3)

    dists = np.linalg.norm(pts_pred - pts_gt, axis=1)
    return float(dists.mean())


def _fallback_adds_metric(R_pred: np.ndarray, t_pred: np.ndarray,
                          R_gt: np.ndarray, t_gt: np.ndarray,
                          model_points: np.ndarray) -> float:
    """
    [Fallback] ADD-S 度量 (用于对称物体)。
    计算每个预测点到最近的真值点之间的平均距离。
    """
    pts_pred = (R_pred @ model_points.T).T + t_pred.reshape(1, 3)
    pts_gt = (R_gt @ model_points.T).T + t_gt.reshape(1, 3)

    if _HAVE_SCIPY_CKDTREE:
        # [首选] 使用 cKDTree (O(N log N))
        tree = cKDTree(pts_gt)
        dists, _ = tree.query(pts_pred, k=1)
        return float(dists.mean())
    else:
        # [回退] 使用纯 Numpy (O(N^2))
        # (N_pred, 1, 3) - (1, N_gt, 3) -> (N_pred, N_gt, 3)
        dists_matrix = np.linalg.norm(pts_pred[:, None, :] - pts_gt[None, :, :], axis=2)
        # (N_pred, N_gt) -> (N_pred,)
        min_dists = dists_matrix.min(axis=1)
        return float(min_dists.mean())


# --- 1. 动态导入项目标准工具 ---
# 这一步使评估器非常灵活，即使某些 utils 缺失也能运行
def _import_optional(name: str):
    """尝试导入 src.utils.* 或 src.metrics.* 下的模块"""
    try:
        full_name = f"src.{name}" if not name.startswith("src.") else name
        module = __import__(full_name, fromlist=['*'])
        print(f"[Evaluator] 成功导入: {full_name}")
        return module
    except ImportError:
        print(f"[Evaluator 警告] 未找到模块: {name}。依赖此模块的功能将回退。")
        return None


# 优先使用我们项目中的标准工具
ransac_voting_mod = _import_optional('utils.ransac_voting')
geometry_mod = _import_optional('utils.geometry')
bop_eval_mod = _import_optional('metrics.bop_eval')
torch_utils_mod = _import_optional('utils.torch_utils')
mesh_utils_mod = _import_optional('utils.mesh')  # 导入我们新的 mesh 工具

# 尝试导入 OpenCV (作为 PnP 的最终回退)
try:
    import cv2

    _HAS_CV2 = True
except ImportError:
    cv2 = None
    _HAS_CV2 = False


# ---------------------------
# 2. 轻量级回退 (Fallback) 函数
# (当导入失败时使用)
# ---------------------------

def _fallback_farthest_point_sampling(pts: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    """简单的CPU FPS回退实现。"""
    N = pts.shape[0]
    if k >= N: return pts.copy()[:k]
    np.random.seed(seed)
    chosen = np.zeros(k, dtype=np.int64)
    chosen[0] = np.random.randint(0, N)
    dists = np.linalg.norm(pts - pts[chosen[0]], axis=1)
    for i in range(1, k):
        idx = int(np.argmax(dists))
        chosen[i] = idx
        d_new = np.linalg.norm(pts - pts[idx], axis=1)
        dists = np.minimum(dists, d_new)
    return pts[chosen]


def _fallback_load_mesh_verts_faces(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """最简 .ply/.obj 读取回退（仅支持 v 和 f）。"""
    verts, faces = [], []
    try:
        with open(path, 'r') as f:
            for line in f:
                if line.startswith('v '):
                    parts = line.strip().split()
                    verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
                elif line.startswith('f '):
                    parts = line.strip().split()
                    idxs = [int(p.split('/')[0]) - 1 for p in parts[1:4]]
                    faces.append(idxs)
    except Exception as e:
        raise RuntimeError(f"Fallback mesh loader 失败: {e}")
    if not verts:
        raise RuntimeError(f"Fallback mesh loader 无法解析: {path}")
    return np.array(verts, dtype=np.float32), np.array(faces, dtype=np.int64)


def _fallback_move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """将batch中的Tensor字段移动到device（简单非递归版）。"""
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


# ---------------------------
# 3. 评估器 (Evaluator) 主类
# ---------------------------

class Evaluator:

    def __init__(self,
                 model: nn.Module,
                 dataloader: DataLoader,
                 device: torch.device,
                 cfg: Any,  # 完整的配置命名空间
                 out_dir: Optional[str] = None,
                 verbose: bool = True):
        """
        初始化评估器。

        参数:
          model: 训练好的模型 (将设为 .eval() 模式)。
          dataloader: 验证数据加载器。
          device: torch.device
          cfg: (SimpleNamespace) 完整的配置对象 (用于获取所有参数)。
          out_dir: (可选) 保存详细 .json 结果的目录。
          verbose: (bool) 是否显示 tqdm 进度条。
        """
        self.model = model.to(device).eval()
        self.dataloader = dataloader
        self.device = device
        self.cfg = cfg
        self.out_dir = out_dir
        self.verbose = verbose

        if self.out_dir:
            os.makedirs(self.out_dir, exist_ok=True)

        # --- 1. [关键] 加载评估所需的 3D 模型数据 ---

        # 1a. 加载 models_info.json (用于获取直径、对称性)
        # [修复] 路径基于 cfg.dataset.data_root (BOP 根目录)
        model_root = os.path.join(cfg.dataset.data_root, "models_eval")
        if not os.path.exists(model_root):
            model_root = os.path.join(cfg.dataset.data_root, "models")  # 备用路径

        model_info_path = os.path.join(model_root, "models_info.json")
        if not os.path.exists(model_info_path):
            raise FileNotFoundError(f"models_info.json 未找到: {model_info_path}。")

        with open(model_info_path, 'r') as f:
            model_info_str_keys = json.load(f)
        self.model_info = {int(k): v for k, v in model_info_str_keys.items()}

        # 1b. 预加载 3D 点云 (用于 ADD-S 计算)
        self.model_points_lookup = self._preload_model_points(model_root)

    def _preload_model_points(self, model_root: str, sample_num: int = 2000) -> Dict[int, np.ndarray]:
        """为所有在 model_info.json 中定义的物体加载 3D 点云。"""

        # [改进] 优先使用我们 src.utils.mesh.py 中的工具
        if mesh_utils_mod is not None:
            print(f"[{type(self).__name__}] 正在使用 'src.utils.mesh' 预加载模型点云...")
            obj_ids = list(self.model_info.keys())
            # [接口对齐] 调用 mesh_utils_mod.load_model_points_dict
            return mesh_utils_mod.load_model_points_dict(model_root, obj_ids, n_points=sample_num)

        # --- 回退逻辑 (如果 mesh_utils_mod 导入失败) ---
        print(f"[{type(self).__name__} 警告] 'src.utils.mesh' 未找到。回退到本地 mesh 加载器。")
        lookup = {}
        for obj_id in self.model_info.keys():
            ply_path = os.path.join(model_root, f"obj_{obj_id:06d}.ply")
            if not os.path.exists(ply_path):
                print(f"[Evaluator 警告] 未找到 {ply_path}，跳过。")
                continue
            try:
                verts, _ = _fallback_load_mesh_verts_faces(ply_path)
                pts = _fallback_farthest_point_sampling(verts, sample_num, seed=0)
                lookup[obj_id] = pts
            except Exception as e:
                print(f"[Evaluator 警告] 加载/采样模型 {ply_path} 失败: {e}")

        print(f"[Evaluator] 成功为 {len(lookup)} 个模型加载了点云。")
        return lookup

    def _decode_kp2d_from_output(self, output_gpu: Dict[str, Any]) -> Optional[np.ndarray]:
        """
        只使用 PVNet 正统的 vertex + seg + RANSAC 解码。
        禁用辅助 kpt_2d 预测（它没有监督，质量极差）
        """

        # 1. PVNet vertex + seg → RANSAC (主路径)
        if 'vertex' in output_gpu and 'seg' in output_gpu:
            if ransac_voting_mod is None:
                print("[Evaluator 警告] 'ransac_voting' 未找到")
                return None

            vt, seg = output_gpu['vertex'], output_gpu['seg']
            scale = getattr(self.cfg.model, 'vertex_scale', 1.0)
            vt = vt * scale

            # mask decode
            if seg.shape[1] > 1:
                mask_pred = (torch.softmax(seg, dim=1)[:, 1] > 0.5).float()
            else:
                mask_pred = (torch.sigmoid(seg[:, 0]) > 0.5).float()

            try:
                kp2d, _ = ransac_voting_mod.ransac_voting(
                    mask=mask_pred,
                    vertex=vt,
                    num_votes=self.cfg.model.ransac_voting.vote_num,
                    inlier_thresh=self.cfg.model.ransac_voting.inlier_thresh,
                    max_trials=self.cfg.model.ransac_voting.max_trials
                )
                return np.squeeze(kp2d.detach().cpu().numpy()).astype(np.float32)
            except Exception as e:
                print(f"[Evaluator 错误] RANSAC 解码失败: {e}")
                return None

        # 2. 禁用辅助 kpt_2d（避免 PN P 炸掉）
        # for key in ['kpt_2d', 'kp_2d', 'kp2d']:
        #     if key in output_gpu:
        #         return np.squeeze(output_gpu[key].detach().cpu().numpy()).astype(np.float32)

        return None

    def _solve_pnp(self, kp3d: np.ndarray, kp2d: np.ndarray, K: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        [智能推理链 2] 从 (kp3d, kp2d, K) 解算 6D 姿态 (R, t)。
        """
        if kp3d is None or kp2d is None or K is None:
            return None

        # 1. [首选] 尝试我们项目中的 'src.utils.geometry.solve_pnp'
        if geometry_mod is not None:
            try:
                # [接口对齐] 调用我们统一的 solve_pnp 接口
                # 我们的 solve_pnp 接受 (obj_pts, img_pts, K, ransac=True, **kwargs)
                out = geometry_mod.solve_pnp(
                    kp3d,
                    kp2d,
                    K,
                    ransac=True,
                    # [接口对齐] 从 cfg.pnp 读取
                    reproj_thresh=self.cfg.pnp.reproj_error_thresh
                )
                if out is None: return None
                R, t = out[0], out[1]  # (R, t, inliers_mask)
                return R.astype(np.float32), t.astype(np.float32)
            except Exception as e:
                print(f"[Evaluator 警告] 'src.utils.geometry.solve_pnp' 失败: {e}。回退到本地 OpenCV。")

        # 2. [回退] 尝试本地 OpenCV (如果 solve_pnp 不可用或失败)
        if _HAS_CV2:
            try:
                obj, imgp, Kf = kp3d.astype(np.float64), kp2d.astype(np.float64), K.astype(np.float64)
                success, rvec, tvec, _ = cv2.solvePnPRansac(
                    obj, imgp, Kf, None,
                    reprojectionError=float(self.cfg.pnp.reproj_error_thresh),
                    flags=cv2.SOLVEPNP_EPNP
                )
                if not success: return None
                R, _ = cv2.Rodrigues(rvec)
                return R.astype(np.float32), tvec.reshape(3, ).astype(np.float32)
            except Exception as e:
                print(f"[Evaluator 错误] 本地 OpenCV PnP 回退失败: {e}")

        return None

    def _get_pose_from_output(self,
                              output_gpu: Dict[str, Any],
                              gt_data: Dict[str, Any]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        [改进] 封装“智能推理链” (您的 Problem 3)。

        参数:
          output_gpu: 模型在 GPU 上的单一样本输出
          gt_data: GT 数据 (Numpy 格式)，用于 PnP
        """

        # 1. 检查模型是否直接输出了 R, t
        if 'R' in output_gpu and 't' in output_gpu:
            R_pred = output_gpu['R'].detach().cpu().numpy().squeeze()
            t_pred = output_gpu['t'].detach().cpu().numpy().squeeze()
            return R_pred, t_pred

        # 2. 尝试从 2D 关键点解算 (PnP)
        #    (内部会处理 PVNet 'vertex'/'seg' 的情况)
        kp2d_pred_np = self._decode_kp2d_from_output(output_gpu)

        if kp2d_pred_np is not None:
            # 3. 解算 PnP
            # [修复] 先将结果存入变量
            pnp_result = self._solve_pnp(
                gt_data['kp3d'],
                kp2d_pred_np,
                gt_data['K']
            )

            # [修复] 关键检查：在解包前检查 PnP 是否成功
            if pnp_result is not None:
                R_pred, t_pred = pnp_result  # 成功，现在可以安全地解包
                return R_pred, t_pred
            else:
                # PnP 失败 (e.g., RANSAC inliers 不足)
                return None, None  # 返回 None, None

            # 所有方法都失败
        return None, None

    def evaluate(self) -> Dict[str, Any]:
        """
        [主循环] 运行评估并返回最终的指标概要。
        """
        self.model.eval()

        all_pred_dicts_for_bop = []  # 用于 bop_eval.py
        all_gt_dicts_for_bop = []  # 用于 bop_eval.py

        loader_desc = f"[Evaluate] 评估 {self.cfg.dataset.val_data_dir.split('/')[-1]}"
        it = tqdm(self.dataloader, desc=loader_desc) if self.verbose else self.dataloader

        # [改进] 优先使用我们项目中的 'move_batch_to_device'
        move_to_device = torch_utils_mod.move_batch_to_device if torch_utils_mod else _fallback_move_batch_to_device

        for batch in it:
            # 1. 将数据移动到 GPU
            batch_gpu = move_to_device(batch, self.device)

            with torch.no_grad():
                with torch.amp.autocast(device_type='cuda', enabled=torch.cuda.is_available()):
                    # 2. 模型前向传播 (B,C,H,W) -> Dict[str, Tensor]
                    outputs_gpu = self.model(batch_gpu['inp'])

            B = batch['inp'].shape[0]  # 批量大小

            # 3. 逐个样本处理 (PnP 是非批量的)
            for i in range(B):
                # [接口对齐] 提取第 i 个样本的 GT (在 CPU 上)
                gt = {
                    'K': batch['K'][i].cpu().numpy(),
                    'R': batch['R'][i].cpu().numpy(),
                    't': batch['t'][i].cpu().numpy(),
                    'kp3d': batch['kp3d'][i].cpu().numpy(),
                    'obj_id': int(batch['meta'][i]['obj_id']),
                    'scene_id': int(batch['meta'][i]['scene_id']),
                    'im_id': int(batch['meta'][i]['im_id']),
                }

                # 提取第 i 个样本的预测 (在 GPU 上，保持批次维度 [1, ...])
                pred_gpu = {k: v[i:i + 1] for k, v in outputs_gpu.items()
                            if isinstance(v, torch.Tensor)}

                # 4. [改进] 调用重构后的“智能推理链”
                # --- DEBUG START: 仅打印前 3 个样本 ---
                if len(all_pred_dicts_for_bop) < 3:
                    print(f"\n\n[DEBUG] 正在检查第 {len(all_pred_dicts_for_bop)} 个样本:")

                    # 1. 检查 GT 3D 点 (kp3d)
                    k3 = gt['kp3d']
                    print(f"  > GT kp3d (前3个): \n{k3[:3]}")
                    print(f"  > GT kp3d 范围: min={k3.min():.2f}, max={k3.max():.2f}")
                    # 正常情况: LINEMOD 钻头直径约 200mm，所以值应该在 -100 到 +100 左右

                    # 2. 检查预测的 2D 点 (kp2d_pred)
                    # 我们需要手动调用一次解码来看看中间结果
                    kp2d_debug = self._decode_kp2d_from_output(pred_gpu)
                    if kp2d_debug is not None:
                        print(f"  > Pred kp2d (前3个): \n{kp2d_debug[:3]}")
                        print(f"  > Pred kp2d 范围: min={kp2d_debug.min():.2f}, max={kp2d_debug.max():.2f}")
                        # 正常情况: 应该在 0 到 640/480 之间。如果出现 -1000 或 +10000 就是 RANSAC 炸了
                    else:
                        print("  > Pred kp2d: 解码失败 (None)")

                    # 3. 检查内参 (K)
                    k_mat = gt['K']
                    print(f"  > K (0,0)={k_mat[0, 0]:.2f}, (0,2)={k_mat[0, 2]:.2f}")

                    # 4. 检查最终解算的姿态 (R, t)
                    R_pred, t_pred = self._get_pose_from_output(pred_gpu, gt)
                    print(f"  > 结果 R_pred:\n{R_pred}")
                    print(f"  > 结果 t_pred: {t_pred}")
                    # 正常情况: t_pred 应该是 [x, y, z]，z 大约在 500~1500 (mm) 之间
                    # 异常情况: 如果 z 是 3.0e+14，那就是 PnP 炸了
                else:
                    # 正常运行
                    R_pred, t_pred = self._get_pose_from_output(pred_gpu, gt)
                # --- DEBUG END ---

                # 5. 组装 BOP 格式的 pred 和 gt 字典
                pred_dict = {
                    'obj_id': gt['obj_id'],
                    'scene_id': gt['scene_id'],
                    'im_id': gt['im_id'],
                    'R': R_pred if R_pred is not None else np.eye(3),
                    't': t_pred if t_pred is not None else np.zeros(3),
                    'score': 1.0 if R_pred is not None else 0.0
                }
                gt_dict = {k: v for k, v in gt.items() if k in ['obj_id', 'scene_id', 'im_id', 'R', 't']}

                all_pred_dicts_for_bop.append(pred_dict)
                all_gt_dicts_for_bop.append(gt_dict)

        # 循环结束

        # --- 8. [核心] 聚合与报告 ---

        # [首选] 尝试使用我们项目的 BOP 评估器
        if bop_eval_mod is not None:
            print(f"评估完成。收集了 {len(all_pred_dicts_for_bop)} 个预测。正在调用 BOP 指标计算...")

            # 假设评估器只针对一个物体 ID (来自 config)
            obj_id = int(self.cfg.model.obj_id)
            if obj_id not in self.model_info:
                raise ValueError(f"物体 ID {obj_id} 不在 models_info.json 中。")

            diameter = self.model_info[obj_id]['diameter']
            is_symmetric = self.model_info[obj_id].get('symmetries_discrete') or \
                           self.model_info[obj_id].get('symmetries_continuous', False)

            thresholds = {'add': diameter * 0.1}  # 0.1d

            # [修复] 调用我们 bop_eval.py 的正确接口
            # 它不接受 'use_adds_for_symmetric'
            results = bop_eval_mod.evaluate_batch(
                predictions=all_pred_dicts_for_bop,
                gts=all_gt_dicts_for_bop,
                model_points_lookup=self.model_points_lookup,
                thresholds=thresholds
                # (use_adds_for_symmetric 参数已被移除)
            )
            summary = results.get('summary', {})

            # [修复] 手动选择正确的指标进行报告
            # bop_eval.py 返回了 avg_add 和 avg_adds
            if is_symmetric:
                final_metric_key = 'pass_rate_adds'
                final_avg_err_key = 'avg_adds'
                metric_name = "ADD-S@0.1d"
            else:
                final_metric_key = 'pass_rate_add'
                final_avg_err_key = 'avg_add'
                metric_name = "ADD@0.1d"

            final_summary = {
                "obj_id": obj_id,
                "is_symmetric": is_symmetric,
                "metric": metric_name,
                "threshold_mm": thresholds['add'],
                "recall": summary.get(final_metric_key, 0.0),
                "avg_error_mm": summary.get(final_avg_err_key, 0.0),
                "total_instances": summary.get('n', 0)
            }

        else:
            # [回退] 使用本地的轻量级度量
            print("[Evaluator 警告] 'src.metrics.bop_eval' 未找到。回退到轻量级本地摘要。")
            final_summary = self._fallback_summarize(all_pred_dicts_for_bop, all_gt_dicts_for_bop)

        print("\n--- 评估结果 (Summary) ---")
        print(json.dumps(final_summary, indent=2))
        return final_summary

    def _fallback_summarize(self, predictions: List[Dict], gts: List[Dict]) -> Dict[str, Any]:
        """ [回退摘要] (如果 bop_eval.py 不可用) """
        re_list, te_list, add_list, adds_list = [], [], [], []

        for p, g in zip(predictions, gts):
            if g['R'] is None or g['t'] is None or p['score'] == 0.0:
                continue
            R_pred, t_pred = p['R'], p['t']
            R_gt, t_gt = g['R'], g['t']

            re_list.append(_fallback_rotation_error_deg(R_pred, R_gt))
            te_list.append(_fallback_translation_error(t_pred, t_gt))

            obj_id = p['obj_id']
            if obj_id in self.model_points_lookup:
                pts = self.model_points_lookup[obj_id]
                is_symmetric = self.model_info[obj_id].get('symmetries_discrete')
                if is_symmetric:
                    adds_list.append(_fallback_adds_metric(R_pred, t_pred, R_gt, t_gt, pts))
                else:
                    add_list.append(_fallback_add_metric(R_pred, t_pred, R_gt, t_gt, pts))

        summary = {'n_total': len(predictions), 'n_valid_gt': len(re_list)}
        if re_list:
            summary['re_mean'] = float(np.mean(re_list))
            summary['te_mean'] = float(np.mean(te_list))
        if add_list:
            summary['add_mean'] = float(np.mean(add_list))
        if adds_list:
            summary['adds_mean'] = float(np.mean(adds_list))

        return summary