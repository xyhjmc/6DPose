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
                 verbose: bool = True,
                 enable_debug: bool = False,
                                  ):
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
        self.use_amp = (self.device.type == 'cuda') and torch.cuda.is_available()
        self.cfg = cfg
        self.out_dir = out_dir
        self.verbose = verbose
        self.enable_debug = enable_debug
        self.vertex_scale = cfg.model.vertex_scale
        self.use_offset = getattr(cfg.model, "use_offset", True)

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

    def _decode_kp2d_from_output(self, output_gpu: Dict[str, Any], keep_batch_dim: bool = False) -> Optional[np.ndarray]:
        """
        使用 PVNet 正统的 vertex + seg + RANSAC 解码 2D 关键点。

        注意：
        - vertex 在 NormalizeAndToTensor 中被 / vertex_scale 过，
          这里要乘回 self.vertex_scale，恢复到“像素偏移”尺度；
        - mask 解码与 PVNet.forward / 实验 E 保持一致：
          * seg_dim == 1: sigmoid > 0.5
          * seg_dim >  1: argmax 通道作为前景
        """

        # 1. 必须有 'vertex' 和 'seg'
        if 'vertex' not in output_gpu or 'seg' not in output_gpu:
            return None
        if ransac_voting_mod is None:
            print("[Evaluator 警告] 'ransac_voting' 未找到，无法从 vertex/seg 解码 kp2d")
            return None

        # 2. 取出预测的顶点场和分割 logits
        vertex_pred = output_gpu['vertex']  # (B, 2K, H, W)，此时是 /vertex_scale 的“归一化空间”
        seg_pred = output_gpu['seg']  # (B, C_seg, H, W)

        # 3. 还原 vertex 到“像素偏移”尺度
        scale = float(getattr(self, "vertex_scale", 1.0)) if self.use_offset else 1.0
        vertex_px = vertex_pred * scale  # (B, 2K, H, W)

        # 4. 解码二值 mask（与 PVNet.decode_keypoint 逻辑对齐）
        if seg_pred.shape[1] == 1:
            # 单通道：BCE 情况，用 sigmoid > 0.5
            # seg_pred: (B, 1, H, W)
            mask_bin = (torch.sigmoid(seg_pred) > 0.5).float()  # (B, 1, H, W)
        else:
            # 多通道：CrossEntropy 情况，直接取 argmax
            # seg_pred: (B, C_seg, H, W)
            # argmax 结果是 (B, H, W)，1 代表前景、0 代表背景
            mask_bin = torch.argmax(seg_pred, dim=1, keepdim=True).float()  # (B, 1, H, W)

        try:
            # 5. 调用统一 ransac_voting 接口（torch 版，跑在 GPU）
            #   mask_bin:   (B, 1, H, W)
            #   vertex_px:  (B, 2K, H, W)
            kpts2d_t, inlier_counts = ransac_voting_mod.ransac_voting(
                mask=mask_bin,
                vertex=vertex_px,
                num_votes=self.cfg.model.ransac_voting.vote_num,
                inlier_thresh=self.cfg.model.ransac_voting.inlier_thresh,
                max_trials=self.cfg.model.ransac_voting.max_trials,
                use_offset=self.use_offset
            )

            # 6. 转为 numpy (默认去掉 batch 维度，与旧接口兼容)
            kp2d_np = kpts2d_t.detach().cpu().numpy().astype(np.float32)  # (B, K, 2)
            if not keep_batch_dim and kp2d_np.shape[0] == 1:
                kp2d_np = kp2d_np[0]

            return kp2d_np

        except Exception as e:
            print(f"[Evaluator 错误] RANSAC 解码失败: {e}")
            return None

    def _solve_pnp(self, kp3d: np.ndarray, kp2d: np.ndarray, K: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        if kp3d is None or kp2d is None or K is None:
            return None

        R, t = None, None

        # 1. 首选 geometry_mod
        if geometry_mod is not None:
            try:
                out = geometry_mod.solve_pnp(
                    kp3d,
                    kp2d,
                    K,
                    ransac=True,
                    reproj_thresh=self.cfg.pnp.reproj_error_thresh
                )
                if out is not None:
                    R, t = out[0], out[1]
            except Exception as e:
                print(f"[Evaluator 警告] 'src.utils.geometry.solve_pnp' 失败: {e}。回退到本地 OpenCV。")

        # 2. 回退到 OpenCV
        if R is None or t is None:
            if _HAS_CV2:
                try:
                    obj, imgp, Kf = kp3d.astype(np.float64), kp2d.astype(np.float64), K.astype(np.float64)
                    success, rvec, tvec, _ = cv2.solvePnPRansac(
                        obj, imgp, Kf, None,
                        reprojectionError=float(self.cfg.pnp.reproj_error_thresh),
                        flags=cv2.SOLVEPNP_EPNP
                    )
                    if not success:
                        return None
                    R, _ = cv2.Rodrigues(rvec)
                    t = tvec.reshape(3, )
                except Exception as e:
                    print(f"[Evaluator 错误] 本地 OpenCV PnP 回退失败: {e}")
                    return None
            else:
                return None

        # 3. ---- 关键：PnP 结果 sanity check ----
        t = np.asarray(t).reshape(3, )
        R = np.asarray(R).reshape(3, 3)

        t_norm = float(np.linalg.norm(t))
        z = float(t[2])

        # ⚠️ 这里的阈值可以按你实际场景微调
        #   - z: 物体深度，假设在 0.1m ~ 5m 之间（100 ~ 5000 mm）
        #   - |t|: 平移范数，给个稍小的硬上限防止数值飞起
        if not (50.0 < z < 5000.0) or t_norm > 6000.0:
            # 前几次调试时可以打印，确认一下触发情况：
            # print(f"[PnP WARNING] t 过于异常，视为失败: t={t}, |t|={t_norm}, z={z}")
            return None

        return R.astype(np.float32), t.astype(np.float32)

    def _get_pose_from_output(self,
                              output_gpu: Dict[str, Any],
                              gt_data: Dict[str, Any],
                              kp2d_pred_np: Optional[np.ndarray] = None) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
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
        if kp2d_pred_np is None:
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

        first_batch = True  # 放到 evaluate() 函数最上面（for batch 之前）

        with torch.inference_mode():
            for batch in it:

                if first_batch:
                    print("[DEBUG_BATCH] batch keys:", list(batch.keys()))
                    # 如果 meta 里还有子字段，也可以看一下
                    if 'meta' in batch:
                        print("[DEBUG_BATCH] meta[0] keys:", batch['meta'][0].keys())
                    first_batch = False

                # 1. 将数据移动到 GPU
                batch_gpu = move_to_device(batch, self.device)

                with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                    # 2. 模型前向传播 (B,C,H,W) -> Dict[str, Tensor]
                    outputs_gpu = self.model(batch_gpu['inp'])

                B = batch['inp'].shape[0]  # 批量大小

                # 2.5. 先对整个 batch 解码一次 kp2d，避免单样本重复触发 RANSAC GPU kernel
                kp2d_batch = self._decode_kp2d_from_output(outputs_gpu, keep_batch_dim=True)

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
                kp2d_np = None
                for k2d_key in ['kp2d', 'kp_2d', 'kpt_2d', 'kpt2d', 'corner_2d', 'corners_2d']:
                    if k2d_key in batch:
                        kp2d_np = batch[k2d_key][i].cpu().numpy()
                        #print(f"[DEBUG_PNP_GT] 使用 GT 2D 字段 '{k2d_key}', 形状={kp2d_np.shape}")
                        break

                if kp2d_np is not None:
                    gt['kp2d'] = kp2d_np
                # 提取第 i 个样本的预测 (在 GPU 上，保持批次维度 [1, ...])
                pred_gpu = {k: v[i:i + 1] for k, v in outputs_gpu.items()
                            if isinstance(v, torch.Tensor)}
                # 解码预测的 2D 关键点（用于后续 PnP debug）
                kp2d_pred_np = None
                if kp2d_batch is not None and i < len(kp2d_batch):
                    # 直接复用批量结果，避免重复 GPU/CPU 往返
                    kp2d_pred_np = kp2d_batch[i]
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
                    if kp2d_pred_np is not None:
                        print(f"  > Pred kp2d (前3个): \n{kp2d_pred_np[:3]}")
                        print(f"  > Pred kp2d 范围: min={kp2d_pred_np.min():.2f}, max={kp2d_pred_np.max():.2f}")
                    else:
                        print("  > Pred kp2d: 解码失败 (None)")

                    # 3. 检查内参 (K)
                    k_mat = gt['K']
                    print(f"  > K (0,0)={k_mat[0, 0]:.2f}, (0,2)={k_mat[0, 2]:.2f}")

                    # 4. 检查最终解算的姿态 (R, t)
                    R_pred, t_pred = self._get_pose_from_output(pred_gpu, gt, kp2d_pred_np)
                    print(f"  > 结果 R_pred:\n{R_pred}")
                    print(f"  > 结果 t_pred: {t_pred}")
                    # 正常情况: t_pred 应该是 [x, y, z]，z 大约在 500~1500 (mm) 之间
                    # 异常情况: 如果 z 是 3.0e+14，那就是 PnP 炸了
                else:
                    # 正常运行
                    R_pred, t_pred = self._get_pose_from_output(pred_gpu, gt, kp2d_pred_np)
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
                if kp2d_pred_np is not None:
                    pred_dict['kp2d_pred'] = kp2d_pred_np
                gt_dict = {
                    k: v for k, v in gt.items()
                    if k in ['obj_id', 'scene_id', 'im_id', 'R', 't', 'kp3d', 'kp2d', 'K']
                }

                all_pred_dicts_for_bop.append(pred_dict)
                all_gt_dicts_for_bop.append(gt_dict)

        # 循环结束

        # --- 8. [核心] 聚合与报告 ---

        # [首选] 尝试使用我们项目的 BOP 评估器
        # if bop_eval_mod is not None:
        #     # --- 8. 在调用 bop_eval 之前，先做一次本地 ADD/统计 Debug ---
        #     print("\n[DEBUG] ---- 本地 ADD 统计（不依赖 bop_eval） ----")
        #     obj_id = int(self.cfg.model.obj_id)
        #     if obj_id not in self.model_points_lookup:
        #         print(f"[DEBUG] model_points_lookup 中找不到 obj_id={obj_id}")
        #     else:
        #         pts = self.model_points_lookup[obj_id]  # (N,3)
        #         print(f"[DEBUG] 模型点云统计: N={pts.shape[0]}, "
        #               f"min={pts.min():.4f}, max={pts.max():.4f}, "
        #               f"mean_norm={np.linalg.norm(pts, axis=1).mean():.4f}")
        #
        #         # 取前 20 个样本做一下简易 ADD 统计
        #         add_list = []
        #         t_norm_list = []
        #         for p, g in zip(all_pred_dicts_for_bop[:20], all_gt_dicts_for_bop[:20]):
        #             R_pred, t_pred = p['R'], p['t']
        #             R_gt, t_gt = g['R'], g['t']
        #
        #             # 计算平移范数
        #             t_norm_list.append(float(np.linalg.norm(t_pred)))
        #
        #             # 计算 ADD
        #             add_val = _fallback_add_metric(R_pred, t_pred, R_gt, t_gt, pts)
        #             add_list.append(add_val)
        #
        #         if add_list:
        #             add_arr = np.array(add_list)
        #             t_arr = np.array(t_norm_list)
        #             print(f"[DEBUG] 前 20 个样本 ADD 统计: "
        #                   f"min={add_arr.min():.4f}, max={add_arr.max():.4f}, mean={add_arr.mean():.4f}")
        #             print(f"[DEBUG] 前 20 个样本 |t_pred| 范数统计: "
        #                   f"min={t_arr.min():.4f}, max={t_arr.max():.4f}, mean={t_arr.mean():.4f}")
        #         print("[DEBUG] ---- 本地 ADD 统计结束 ----\n")
        #
        #     print(f"评估完成。收集了 {len(all_pred_dicts_for_bop)} 个预测。正在调用 BOP 指标计算...")
        #
        #     # 假设评估器只针对一个物体 ID (来自 config)
        #     obj_id = int(self.cfg.model.obj_id)
        #     if obj_id not in self.model_info:
        #         raise ValueError(f"物体 ID {obj_id} 不在 models_info.json 中。")
        #
        #     diameter = self.model_info[obj_id]['diameter']
        #     is_symmetric = self.model_info[obj_id].get('symmetries_discrete') or \
        #                    self.model_info[obj_id].get('symmetries_continuous', False)
        #
        #     thresholds = {'add': diameter * 0.1}  # 0.1d
        #
        #     # [修复] 调用我们 bop_eval.py 的正确接口
        #     # 它不接受 'use_adds_for_symmetric'
        #     results = bop_eval_mod.evaluate_batch(
        #         predictions=all_pred_dicts_for_bop,
        #         gts=all_gt_dicts_for_bop,
        #         model_points_lookup=self.model_points_lookup,
        #         thresholds=thresholds
        #         # (use_adds_for_symmetric 参数已被移除)
        #     )
        #     summary = results.get('summary', {})
        #
        #     # [修复] 手动选择正确的指标进行报告
        #     # bop_eval.py 返回了 avg_add 和 avg_adds
        #     if is_symmetric:
        #         final_metric_key = 'pass_rate_adds'
        #         final_avg_err_key = 'avg_adds'
        #         metric_name = "ADD-S@0.1d"
        #     else:
        #         final_metric_key = 'pass_rate_add'
        #         final_avg_err_key = 'avg_add'
        #         metric_name = "ADD@0.1d"
        #
        #     final_summary = {
        #         "obj_id": obj_id,
        #         "is_symmetric": is_symmetric,
        #         "metric": metric_name,
        #         "threshold_mm": thresholds['add'],
        #         "recall": summary.get(final_metric_key, 0.0),
        #         "avg_error_mm": summary.get(final_avg_err_key, 0.0),
        #         "total_instances": summary.get('n', 0)
        #     }
        #
        # else:
        #     # [回退] 使用本地的轻量级度量
        #     print("[Evaluator 警告] 'src.metrics.bop_eval' 未找到。回退到轻量级本地摘要。")
        #     final_summary = self._fallback_summarize(all_pred_dicts_for_bop, all_gt_dicts_for_bop)
                # 实验 B：GT kp2d + PnP
        if self.enable_debug:
            self._debug_pnp_with_gt_kp2d(all_gt_dicts_for_bop, max_samples=200)

            self._debug_add_gt_vs_gt(all_gt_dicts_for_bop, max_samples=50)

            self._debug_pnp_with_pred_kp2d(all_pred_dicts_for_bop, all_gt_dicts_for_bop, max_samples=200)
            self._debug_kp2d_pred_vs_gt(all_pred_dicts_for_bop, all_gt_dicts_for_bop, max_samples=200)
            self._debug_ransac_with_gt_vertex(max_samples=100)  # ← 新增的实验 E
            self._debug_ransac_with_pred_vertex(max_samples=100)
        final_summary = self._local_bop_like_eval(all_pred_dicts_for_bop, all_gt_dicts_for_bop)

        print("\n--- 评估结果 (Summary) ---")
        print(json.dumps(final_summary, indent=2))
        return final_summary
    def _debug_add_gt_vs_gt(self, gts: List[Dict[str, Any]], max_samples: int = 50):
        """
        实验 A：用 GT 的 (R_gt, t_gt) 和自己算 ADD，理论上应该接近 0。
        用来检查：model_points 的尺度 / 坐标系 / ADD 实现 是否一致。
        """
        obj_id = int(self.cfg.model.obj_id)
        if obj_id not in self.model_points_lookup:
            print(f"[DEBUG_GT] model_points_lookup 中找不到 obj_id={obj_id}")
            return

        pts = self.model_points_lookup[obj_id]   # (N,3)

        add_vals = []
        t_norms = []

        for g in gts[:max_samples]:
            if g['obj_id'] != obj_id:
                continue

            R_gt = g['R']
            t_gt = g['t']

            # 用同一个 GT 姿态做 ADD
            add_val = _fallback_add_metric(R_gt, t_gt, R_gt, t_gt, pts)
            add_vals.append(add_val)

            t_norms.append(float(np.linalg.norm(t_gt)))

        if not add_vals:
            print("[DEBUG_GT] 没有匹配 obj_id 的 GT 样本，无法做 GT vs GT 检查。")
            return

        add_arr = np.array(add_vals)
        t_arr = np.array(t_norms)

        print("\n[DEBUG_GT] ==== 实验 A：GT vs GT ADD 检查 ====")
        print(f"[DEBUG_GT] 使用样本数: {len(add_arr)}")
        print(f"[DEBUG_GT] GT |t_gt| 统计: min={t_arr.min():.4f}, "
              f"max={t_arr.max():.4f}, mean={t_arr.mean():.4f}")
        print(f"[DEBUG_GT] ADD(R_gt,t_gt; R_gt,t_gt) 统计: "
              f"min={add_arr.min():.6e}, max={add_arr.max():.6e}, "
              f"mean={add_arr.mean():.6e}")
        print("[DEBUG_GT] 理论上这三个值都应该非常接近 0，如果不是，则说明 ADD/坐标系存在问题。\n")

    def _debug_ransac_with_gt_vertex(self, max_samples: int = 100):
        """
        实验 E：用 GT 的 (mask, vertex) 通过 ransac_voting 解一次 kp2d，
        看能恢复多接近 GT 的 kp2d。

        注意：
        - batch['vertex'] 已经被 NormalizeAndToTensor 除以 vertex_scale，
          在这里要乘回去，恢复像素级偏移。
        - 强制走 PyTorch 版 ransac_voting，绕开 CPU 版 shape 误判的问题。
        """
        if ransac_voting_mod is None:
            print("[DEBUG_E] ransac_voting 模块不可用，跳过实验 E。")
            return

        l2_all = []
        img_mean = []

        n = 0
        for batch in self.dataloader:
            B = batch['inp'].shape[0]
            for i in range(B):
                if n >= max_samples:
                    break

                # 1) 取 GT 数据（都是 Tensor）
                mask_gt = batch['mask'][i]           # (H, W)
                vertex_gt = batch['vertex'][i]       # (2K, H, W)，此时已被 / vertex_scale
                kp2d_gt = batch['kp2d'][i].cpu().numpy()  # (K, 2)

                # 2) 恢复 vertex 到像素偏移尺度
                scale = float(getattr(self.cfg.model, "vertex_scale", 1.0)) if self.use_offset else 1.0
                vertex_gt_px = vertex_gt * scale     # (2K, H, W)

                # 3) 拼成 batch 维度，并放到 device 上
                mask_t = mask_gt.unsqueeze(0).to(self.device)           # (1, H, W)
                vertex_t = vertex_gt_px.unsqueeze(0).to(self.device)    # (1, 2K, H, W)

                # 4) 调用统一接口 ransac_voting（会自动走 torch 版）
                #    torch 版返回 (kpts2d, inlier_counts)
                with torch.no_grad():
                    kpts2d_t, inlier_counts = ransac_voting_mod.ransac_voting(
                        mask_t,
                        vertex_t,
                        num_votes=self.cfg.model.ransac_voting.vote_num,
                        inlier_thresh=self.cfg.model.ransac_voting.inlier_thresh,
                        max_trials=self.cfg.model.ransac_voting.max_trials,
                        use_offset=self.use_offset
                    )

                # 5) 取出 (K,2) 关键点坐标，转 numpy
                kp2d_pred = kpts2d_t[0].detach().cpu().numpy()  # (K, 2)

                # 6) 计算像素 L2 误差
                diff = kp2d_pred - kp2d_gt
                err = np.linalg.norm(diff, axis=1)  # (K,)
                l2_all.append(err)
                img_mean.append(err.mean())
                n += 1

            if n >= max_samples:
                break

        if not l2_all:
            print("[DEBUG_E] 没有样本，无法执行实验 E。")
            return

        l2_all = np.concatenate(l2_all, axis=0)
        img_mean = np.array(img_mean)

        print("\n[DEBUG_E] ==== 实验 E：GT vertex + RANSAC → kp2d =====")
        print(f"[DEBUG_E] 有效样本数: {n}")
        print(f"[DEBUG_E] 所有关键点 L2 像素误差统计："
              f"min={l2_all.min():.2f}, max={l2_all.max():.2f}, mean={l2_all.mean():.2f}")
        print(f"[DEBUG_E] 按图像平均的关键点误差："
              f"min={img_mean.min():.2f}, max={img_mean.max():.2f}, mean={img_mean.mean():.2f}")
        print("[DEBUG_E] 理论预期：这里应该是接近 0～几像素，否则说明 vertex GT 或关键点顺序有问题。")

    def _debug_ransac_with_pred_vertex(self, max_samples: int = 100):
        """
        实验 F：
          F.1: Pred vertex + GT mask + RANSAC → kp2d
          F.2: Pred vertex + Pred mask + RANSAC → kp2d

        用来拆分：
          - vertex head 本身的预测质量
          - seg head 错误对 RANSAC 的影响
        """
        if ransac_voting_mod is None:
            print("[DEBUG_F] ransac_voting 模块不可用，跳过实验 F。")
            return

        # 和 evaluate() 里一样的 move_to_device 策略
        move_to_device = torch_utils_mod.move_batch_to_device if torch_utils_mod else _fallback_move_batch_to_device

        # 统计量
        l2_all_gtmask = []       # Pred vertex + GT mask
        img_mean_gtmask = []

        l2_all_predmask = []     # Pred vertex + Pred mask
        img_mean_predmask = []

        n = 0
        scale = float(getattr(self.cfg.model, "vertex_scale", 1.0)) if self.use_offset else 1.0

        # 注意：这里会再跑一遍整个 dataloader 的前向
        # 但验证集不大，开 no_grad + amp 问题不大
        for batch in self.dataloader:
            if n >= max_samples:
                break

            batch_gpu = move_to_device(batch, self.device)

            with torch.no_grad():
                with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                    outputs_gpu = self.model(batch_gpu['inp'])

            if 'vertex' not in outputs_gpu or 'seg' not in outputs_gpu:
                print("[DEBUG_F] 模型输出中缺少 'vertex' 或 'seg'，无法执行实验 F。")
                return

            vertex_pred = outputs_gpu['vertex']          # (B, 2K, H, W)
            seg_pred = outputs_gpu['seg']                # (B, C, H, W)

            B = batch['inp'].shape[0]
            for i in range(B):
                if n >= max_samples:
                    break

                # GT kp2d
                kp2d_gt = batch['kp2d'][i].cpu().numpy()     # (K, 2)

                # ---- F.1: Pred vertex + GT mask ----
                try:
                    mask_gt_t = batch['mask'][i].unsqueeze(0).to(self.device)      # (1, H, W)
                    vertex_i = (vertex_pred[i:i+1] * scale)                       # (1, 2K, H, W)

                    with torch.no_grad():
                        kpts2d_gtmask_t, _ = ransac_voting_mod.ransac_voting(
                            mask=mask_gt_t,
                            vertex=vertex_i,
                            num_votes=self.cfg.model.ransac_voting.vote_num,
                            inlier_thresh=self.cfg.model.ransac_voting.inlier_thresh,
                            max_trials=self.cfg.model.ransac_voting.max_trials,
                            use_offset=self.use_offset
                        )

                    kp2d_pred_gtmask = kpts2d_gtmask_t[0].detach().cpu().numpy()  # (K, 2)

                    diff_gt = kp2d_pred_gtmask - kp2d_gt
                    err_gt = np.linalg.norm(diff_gt, axis=1)  # (K,)
                    l2_all_gtmask.append(err_gt)
                    img_mean_gtmask.append(err_gt.mean())
                except Exception as e:
                    print(f"[DEBUG_F] F.1 (Pred vertex + GT mask) 第 {n} 个样本失败: {e}")
                    # 出错的话就只统计 Pred mask 那支，继续往下

                # ---- F.2: Pred vertex + Pred mask ----
                try:
                    seg_i = seg_pred[i:i+1]  # (1, C, H, W)
                    if seg_i.shape[1] > 1:
                        prob_fg = torch.softmax(seg_i, dim=1)[:, 1]  # (1, H, W)
                    else:
                        # 单通道情况下，视为 logits，走 sigmoid
                        prob_fg = torch.sigmoid(seg_i[:, 0:1]).squeeze(1)  # (1, H, W)

                    mask_pred_t = (prob_fg > 0.5).float()  # (1, H, W)

                    vertex_i = (vertex_pred[i:i+1] * scale)

                    with torch.no_grad():
                        kpts2d_predmask_t, _ = ransac_voting_mod.ransac_voting(
                            mask=mask_pred_t,
                            vertex=vertex_i,
                            num_votes=self.cfg.model.ransac_voting.vote_num,
                            inlier_thresh=self.cfg.model.ransac_voting.inlier_thresh,
                            max_trials=self.cfg.model.ransac_voting.max_trials,
                            use_offset=self.use_offset
                        )

                    kp2d_pred_predmask = kpts2d_predmask_t[0].detach().cpu().numpy()  # (K, 2)

                    diff_pm = kp2d_pred_predmask - kp2d_gt
                    err_pm = np.linalg.norm(diff_pm, axis=1)  # (K,)
                    l2_all_predmask.append(err_pm)
                    img_mean_predmask.append(err_pm.mean())
                except Exception as e:
                    print(f"[DEBUG_F] F.2 (Pred vertex + Pred mask) 第 {n} 个样本失败: {e}")
                    # 继续下一个样本

                n += 1

        if not l2_all_gtmask and not l2_all_predmask:
            print("[DEBUG_F] 没有有效样本，实验 F 失败。")
            return

        print("\n[DEBUG_F] ==== 实验 F：Pred vertex + GT/Pred mask → kp2d ====")
        print(f"[DEBUG_F] 有效样本数: {n}")

        # 汇总统计：GT mask 分支
        if l2_all_gtmask:
            l2_all_gtmask = np.concatenate(l2_all_gtmask, axis=0)
            img_mean_gtmask = np.array(img_mean_gtmask)
            print(f"[DEBUG_F][GT mask] 所有关键点 L2 像素误差统计："
                  f"min={l2_all_gtmask.min():.2f}, "
                  f"max={l2_all_gtmask.max():.2f}, "
                  f"mean={l2_all_gtmask.mean():.2f}")
            print(f"[DEBUG_F][GT mask] 按图像平均的关键点误差："
                  f"min={img_mean_gtmask.min():.2f}, "
                  f"max={img_mean_gtmask.max():.2f}, "
                  f"mean={img_mean_gtmask.mean():.2f}")
        else:
            print("[DEBUG_F][GT mask] 没有成功样本。")

        # 汇总统计：Pred mask 分支
        if l2_all_predmask:
            l2_all_predmask = np.concatenate(l2_all_predmask, axis=0)
            img_mean_predmask = np.array(img_mean_predmask)
            print(f"[DEBUG_F][Pred mask] 所有关键点 L2 像素误差统计："
                  f"min={l2_all_predmask.min():.2f}, "
                  f"max={l2_all_predmask.max():.2f}, "
                  f"mean={l2_all_predmask.mean():.2f}")
            print(f"[DEBUG_F][Pred mask] 按图像平均的关键点误差："
                  f"min={img_mean_predmask.min():.2f}, "
                  f"max={img_mean_predmask.max():.2f}, "
                  f"mean={img_mean_predmask.mean():.2f}")
        else:
            print("[DEBUG_F][Pred mask] 没有成功样本。")

        print("[DEBUG_F] 对比结论：")
        print("  - 如果 [GT mask] 分支的误差已经很大 → 主要问题在 vertex head；")
        print("  - 如果 [GT mask] 分支还好，但 [Pred mask] 分支很差 → seg/mask 质量在拖后腿。")

    def _debug_pnp_with_gt_kp2d(self, gts: List[Dict[str, Any]], max_samples: int = 100):
        """
        实验 B：用 GT 的 (kp3d, kp2d, K) 通过当前的 `_solve_pnp` 解一次，
        看能恢复多接近 GT 的 (R_gt, t_gt)。

        如果这个实验的 ADD 很小 → PnP 没问题，问题主要在关键点预测 / RANSAC。
        如果这个实验的 ADD 也很大 → PnP 管线有问题（坐标系 / 单位 / sanity check 等）。
        """
        obj_id = int(self.cfg.model.obj_id)
        if obj_id not in self.model_points_lookup:
            print(f"[DEBUG_PNP_GT] model_points_lookup 中找不到 obj_id={obj_id}")
            return

        pts = self.model_points_lookup[obj_id]
        diameter = self.model_info[obj_id]['diameter']
        thr = 0.1 * diameter

        add_list = []
        re_list = []
        te_list = []
        n_total = 0
        n_success = 0

        for g in gts[:max_samples]:
            if g['obj_id'] != obj_id:
                continue
            if 'kp2d' not in g:
                continue

            n_total += 1

            kp3d = g['kp3d']
            kp2d_gt = g['kp2d']
            K = g['K']
            R_gt = g['R']
            t_gt = g['t']

            # 用和预测时完全一样的 PnP 函数
            out = self._solve_pnp(kp3d, kp2d_gt, K)
            if out is None:
                continue
            R_pnp, t_pnp = out
            n_success += 1

            # 旋转 / 平移误差
            re = _fallback_rotation_error_deg(R_pnp, R_gt)
            te = _fallback_translation_error(t_pnp, t_gt)
            re_list.append(re)
            te_list.append(te)

            # ADD 误差
            add_val = _fallback_add_metric(R_pnp, t_pnp, R_gt, t_gt, pts)
            add_list.append(add_val)

        print("\n[DEBUG_PNP_GT] ==== 实验 B：GT kp2d + PnP 检查 ====")
        print(f"[DEBUG_PNP_GT] 总共尝试样本数(前 {max_samples}): {n_total}")
        print(f"[DEBUG_PNP_GT] PnP 成功解出的样本数: {n_success}")

        if not add_list:
            print("[DEBUG_PNP_GT] 没有成功的样本，说明 PnP 经常被 sanity check 判为失败，"
                  "可以考虑暂时放宽 _solve_pnp 里的 z 范围 / t_norm 上限再试一次。")
            return

        add_arr = np.array(add_list)
        re_arr = np.array(re_list)
        te_arr = np.array(te_list)

        print(f"[DEBUG_PNP_GT] 旋转误差 re(deg): min={re_arr.min():.4f}, "
              f"max={re_arr.max():.4f}, mean={re_arr.mean():.4f}")
        print(f"[DEBUG_PNP_GT] 平移误差 te(mm): min={te_arr.min():.4f}, "
              f"max={te_arr.max():.4f}, mean={te_arr.mean():.4f}")
        print(f"[DEBUG_PNP_GT] ADD(R_pnp,t_pnp; R_gt,t_gt): "
              f"min={add_arr.min():.4f}, max={add_arr.max():.4f}, mean={add_arr.mean():.4f}")
        print(f"[DEBUG_PNP_GT] 其中 ADD<{thr:.3f} 的比例 (理论上类似 ADD-0.1d 上限): "
              f"{(add_arr < thr).mean():.4f}")
        print("[DEBUG_PNP_GT] 如果这里的 ADD 很小，说明 PnP 管线是健康的；"
              "反之则需要优先检查 PnP / 坐标系 / 单位。")
    def _debug_pnp_with_pred_kp2d(self,
                                  preds: List[Dict[str, Any]],
                                  gts: List[Dict[str, Any]],
                                  max_samples: int = 100):
        """
        实验 C：用 预测的 kp2d_pred + GT kp3d + K 跑一次 PnP，
        直接度量“关键点预测 + 当前 PnP 管线”的误差。

        如果：
          - 实验 B (GT kp2d + PnP) 很好；
          - 实验 C (pred kp2d + PnP) 很烂；
        则说明问题几乎完全在“关键点预测 / RANSAC 解码”上。
        """
        obj_id = int(self.cfg.model.obj_id)
        if obj_id not in self.model_points_lookup:
            print(f"[DEBUG_PNP_PRED] model_points_lookup 中找不到 obj_id={obj_id}")
            return

        pts = self.model_points_lookup[obj_id]
        diameter = self.model_info[obj_id]['diameter']
        thr = 0.1 * diameter

        add_list = []
        re_list = []
        te_list = []
        n_total = 0
        n_success = 0

        for p, g in zip(preds[:max_samples], gts[:max_samples]):
            if p['obj_id'] != obj_id:
                continue
            if 'kp2d_pred' not in p:
                continue

            n_total += 1

            kp2d_pred = p['kp2d_pred']
            kp3d = g['kp3d']
            K = g['K']
            R_gt = g['R']
            t_gt = g['t']

            out = self._solve_pnp(kp3d, kp2d_pred, K)
            if out is None:
                continue
            R_pnp, t_pnp = out
            n_success += 1

            re = _fallback_rotation_error_deg(R_pnp, R_gt)
            te = _fallback_translation_error(t_pnp, t_gt)
            re_list.append(re)
            te_list.append(te)

            add_val = _fallback_add_metric(R_pnp, t_pnp, R_gt, t_gt, pts)
            add_list.append(add_val)

        print("\n[DEBUG_PNP_PRED] ==== 实验 C：Pred kp2d + PnP 检查 ====")
        print(f"[DEBUG_PNP_PRED] 总共尝试样本数(前 {max_samples}): {n_total}")
        print(f"[DEBUG_PNP_PRED] PnP 成功解出的样本数: {n_success}")

        if not add_list:
            print("[DEBUG_PNP_PRED] 没有成功的样本，说明用预测关键点跑 PnP 也经常被 sanity check 删掉。")
            return

        add_arr = np.array(add_list)
        re_arr = np.array(re_list)
        te_arr = np.array(te_list)

        print(f"[DEBUG_PNP_PRED] 旋转误差 re(deg): min={re_arr.min():.4f}, "
              f"max={re_arr.max():.4f}, mean={re_arr.mean():.4f}")
        print(f"[DEBUG_PNP_PRED] 平移误差 te(mm): min={te_arr.min():.4f}, "
              f"max={te_arr.max():.4f}, mean={te_arr.mean():.4f}")
        print(f"[DEBUG_PNP_PRED] ADD(R_pnp,t_pnp; R_gt,t_gt): "
              f"min={add_arr.min():.4f}, max={add_arr.max():.4f}, mean={add_arr.mean():.4f}")
        print(f"[DEBUG_PNP_PRED] 其中 ADD<{thr:.3f} 的比例: {(add_arr < thr).mean():.4f}")
        print("[DEBUG_PNP_PRED] 这里如果也非常差，就可以确认问题在关键点预测 / RANSAC。")

    def _local_bop_like_eval(self, predictions: List[Dict], gts: List[Dict]) -> Dict[str, Any]:
        obj_id = int(self.cfg.model.obj_id)
        if obj_id not in self.model_points_lookup:
            raise ValueError(f"model_points_lookup 中找不到 obj_id={obj_id}")

        pts = self.model_points_lookup[obj_id]
        diameter = self.model_info[obj_id]['diameter']
        thr = 0.1 * diameter

        adds = []
        bad_indices = []  # 存储疑似炸掉的样本 index
        t_norms = []

        for idx, (p, g) in enumerate(zip(predictions, gts)):
            if p['obj_id'] != obj_id:
                continue
            if p['score'] == 0.0:
                continue

            R_pred, t_pred = p['R'], p['t']
            R_gt, t_gt = g['R'], g['t']

            add_val = _fallback_add_metric(R_pred, t_pred, R_gt, t_gt, pts)
            adds.append(add_val)

            t_norm = float(np.linalg.norm(t_pred))
            t_norms.append(t_norm)

            # 如果 ADD 或 |t| 特别夸张，记录下来
            if add_val > 1e6 or t_norm > 1e6:
                bad_indices.append((idx, add_val, t_norm))

        # ---- Debug 打印 ----
        if adds:
            adds_arr = np.array(adds)
            t_arr = np.array(t_norms)
            print(f"[DEBUG_LOCAL] 全部有效样本 ADD 统计: "
                  f"min={adds_arr.min():.4f}, max={adds_arr.max():.4e}, mean={adds_arr.mean():.4e}")
            print(f"[DEBUG_LOCAL] 全部有效样本 |t_pred| 统计: "
                  f"min={t_arr.min():.4f}, max={t_arr.max():.4e}, mean={t_arr.mean():.4e}")

            if bad_indices:
                print(f"[DEBUG_LOCAL] 发现 {len(bad_indices)} 个疑似炸掉的样本 (ADD>1e6 或 |t|>1e6)，列出前 5 个:")
                for i, (idx, add_val, t_norm) in enumerate(bad_indices[:5]):
                    p = predictions[idx]
                    g = gts[idx]
                    print(f"  - idx={idx}, ADD={add_val:.4e}, |t_pred|={t_norm:.4e}, "
                          f"scene_id={p['scene_id']}, im_id={p['im_id']}")
                    print(f"    R_pred:\n{p['R']}")
                    print(f"    t_pred: {p['t']}")
                    print(f"    R_gt:\n{g['R']}")
                    print(f"    t_gt: {g['t']}")
        else:
            print("[DEBUG_LOCAL] 没有有效预测 (adds 为空)。")

        # ---- 正常的统计逻辑 ----
        if not adds:
            return {
                "obj_id": obj_id,
                "metric": "ADD@0.1d_local",
                "threshold_mm": thr,
                "recall": 0.0,
                "avg_error_mm": float('nan'),
                "total_instances": 0
            }

        adds_arr = np.array(adds)
        recall = float((adds_arr < thr).mean())
        avg_err = float(adds_arr.mean())

        return {
            "obj_id": obj_id,
            "metric": "ADD@0.1d_local",
            "threshold_mm": thr,
            "recall": recall,
            "avg_error_mm": avg_err,
            "total_instances": int(len(adds_arr))
        }
    def _debug_kp2d_pred_vs_gt(self,
                               preds: List[Dict[str, Any]],
                               gts: List[Dict[str, Any]],
                               max_samples: int = 100):
        """
        实验 D：直接比较 kp2d_pred 和 GT kp2d 的像素误差。

        输出：
          - 每个关键点的 L2 像素误差统计（min/max/mean）
          - 每个样本的平均关键点误差统计
        """

        obj_id = int(self.cfg.model.obj_id)

        all_pt_errors = []     # 所有关键点的 L2 误差（逐点）
        per_img_mean_errors = []  # 每张图的平均关键点误差

        n_samples = 0

        for p, g in zip(preds[:max_samples], gts[:max_samples]):
            if p['obj_id'] != obj_id:
                continue
            if 'kp2d_pred' not in p:
                continue
            if 'kp2d' not in g:
                continue

            kp2d_pred = np.asarray(p['kp2d_pred'], dtype=np.float32)
            kp2d_gt   = np.asarray(g['kp2d'], dtype=np.float32)

            if kp2d_pred.shape != kp2d_gt.shape:
                print(f"[DEBUG_KP2D] 形状不一致: pred={kp2d_pred.shape}, gt={kp2d_gt.shape}")
                continue

            # (N,2) → 每个关键点的 L2 像素误差
            diffs = kp2d_pred - kp2d_gt
            dists = np.linalg.norm(diffs, axis=1)  # (N,)

            all_pt_errors.extend(dists.tolist())
            per_img_mean_errors.append(float(dists.mean()))
            n_samples += 1

        print("\n[DEBUG_KP2D] ==== 实验 D：Pred kp2d vs GT kp2d 像素误差 ====")
        print(f"[DEBUG_KP2D] 有效样本数: {n_samples}")

        if not all_pt_errors:
            print("[DEBUG_KP2D] 没有可用的关键点误差（可能缺少 kp2d 或 kp2d_pred）")
            return

        all_pt_errors = np.array(all_pt_errors)
        per_img_mean_errors = np.array(per_img_mean_errors)

        print(f"[DEBUG_KP2D] 所有关键点 L2 像素误差统计："
              f"min={all_pt_errors.min():.2f}, "
              f"max={all_pt_errors.max():.2f}, "
              f"mean={all_pt_errors.mean():.2f}")
        print(f"[DEBUG_KP2D] 按图像平均的关键点误差："
              f"min={per_img_mean_errors.min():.2f}, "
              f"max={per_img_mean_errors.max():.2f}, "
              f"mean={per_img_mean_errors.mean():.2f}")
        print("[DEBUG_KP2D] 参考：一般 <5px 很好，10~20px 勉强能用，>50px 基本就很难出好姿态了。")


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