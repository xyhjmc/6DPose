# src/engines/evaluator.py
"""
评估引擎 (Evaluator Engine) - 专业版 (单文件)

说明:
- 这个文件是完整的 evaluator 实现：自动检测模型输出，回退到
  本地实现（如果项目工具缺失），并优先调用 src.metrics.bop_eval
  进行标准 BOP 评估。
- 兼容 PVNet 风格输出 (vertex + seg)、直接关键点输出 (kpt_2d)、
  或网络直接输出 R/t。1
- 程序尽量保守：遇到个别样本错误只会记录并跳过，不会中断整个评估过程。

使用:
    from src.engines.evaluator import Evaluator
    evaluator = Evaluator(model, dataloader, device, cfg, out_dir=..., verbose=True)
    summary = evaluator.evaluate()
"""

import os
import json
import logging
import traceback
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# --- optional imports (may or may not exist in your project) ---
def _import_optional(name: str):
    """
    尝试导入 src.<name>，如果失败返回 None。
    name 可以是 "utils.ransac_voting" 或 "metrics.bop_eval" 等。
    """
    try:
        full = f"src.{name}" if not name.startswith("src.") else name
        module = __import__(full, fromlist=['*'])
        print(f"[Evaluator] 成功导入: {full}")
        return module
    except Exception:
        print(f"[Evaluator 警告] 未能导入: src.{name}，将回退到本地实现（如可用）。")
        return None


# try to import project modules
ransac_voting_mod = _import_optional("utils.ransac_voting")  # 必需时用于 vertex->kpt2d
geometry_mod = _import_optional("utils.geometry")            # 优先用于 solve_pnp
bop_eval_mod = _import_optional("metrics.bop_eval")         # 优先用于 BOP 评价
torch_utils_mod = _import_optional("utils.torch_utils")     # 优先用于 move_batch_to_device
mesh_utils_mod = _import_optional("utils.mesh")             # 优先用于加载/采样模型点云

# optional dependency: cv2 for fallback PnP
try:
    import cv2
    _HAS_CV2 = True
except Exception:
    cv2 = None
    _HAS_CV2 = False
    print("[Evaluator 警告] OpenCV (cv2) 未安装，无法使用本地 PnP 回退。")

# optional dependency: scipy cKDTree for fast ADD-S
try:
    from scipy.spatial import cKDTree  # type: ignore
    _HAVE_CKD = True
except Exception:
    cKDTree = None
    _HAVE_CKD = False
    print("[Evaluator 警告] scipy.spatial.cKDTree 未安装，ADD-S 将使用较慢的 Numpy 实现。")

# ------------------------
# Logging helper
# ------------------------
logger = logging.getLogger("evaluator")
if not logger.handlers:
    # 基本 console logger（用户项目可能会设置更复杂的 logger）
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(ch)
logger.setLevel(logging.INFO)


# ------------------------
# Fallback helpers (local/simple implementations)
# ------------------------
def _fallback_move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """将 batch 中的 Tensor 移动到 device（非递归的简单实现）。"""
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def _fallback_rotation_error_deg(R_pred: np.ndarray, R_gt: np.ndarray) -> float:
    """两个 3x3 旋转矩阵之间的角度误差（度）。"""
    R_pred = np.asarray(R_pred).reshape(3, 3)
    R_gt = np.asarray(R_gt).reshape(3, 3)
    R_rel = R_pred.T @ R_gt
    cos_val = np.clip((np.trace(R_rel) - 1.0) / 2.0, -1.0, 1.0)
    ang = float(np.degrees(np.arccos(cos_val)))
    return ang


def _fallback_translation_error(t_pred: np.ndarray, t_gt: np.ndarray) -> float:
    t_pred = np.asarray(t_pred).reshape(3,)
    t_gt = np.asarray(t_gt).reshape(3,)
    return float(np.linalg.norm(t_pred - t_gt))


def _fallback_add(Rp, tp, Rg, tg, model_points: np.ndarray) -> float:
    """简单 ADD（非对称）"""
    pts = np.asarray(model_points).reshape(-1, 3)
    pred_pts = (Rp @ pts.T).T + tp.reshape(1,3)
    gt_pts = (Rg @ pts.T).T + tg.reshape(1,3)
    d = np.linalg.norm(pred_pts - gt_pts, axis=1)
    return float(np.mean(d))


def _fallback_adds(Rp, tp, Rg, tg, model_points: np.ndarray) -> float:
    """ADD-S (对称) 回退实现（优先 cKDTree）"""
    pts = np.asarray(model_points).reshape(-1, 3)
    pred_pts = (Rp @ pts.T).T + tp.reshape(1,3)
    gt_pts = (Rg @ pts.T).T + tg.reshape(1,3)
    if _HAVE_CKD:
        tree = cKDTree(gt_pts)
        dists, _ = tree.query(pred_pts, k=1)
        return float(np.mean(dists))
    else:
        # O(N^2) 回退
        d2 = np.linalg.norm(pred_pts[:, None, :] - gt_pts[None, :, :], axis=2)
        return float(np.mean(np.min(d2, axis=1)))


# ------------------------
# Mesh utilities fallback (very small parser)
# ------------------------
def _fallback_load_mesh_verts_faces(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    最简回退的 mesh loader: 支持 ascii .obj/.ply 中的 'v ' 和 'f ' 行。
    仅作为最后回退方案。推荐安装 trimesh 并使用项目的 mesh 工具。
    """
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
                    # 支持 f v v v 或 f v/t/vn ...
                    idxs = [int(p.split('/')[0]) - 1 for p in parts[1:4]]
                    faces.append(idxs)
    except Exception as e:
        raise RuntimeError(f"fallback mesh loader failed for {path}: {e}")
    if not verts:
        raise RuntimeError(f"fallback mesh loader could not parse any vertices: {path}")
    return np.array(verts, dtype=np.float32), np.array(faces, dtype=np.int64)


def _fallback_farthest_point_sampling(verts: np.ndarray, k: int, seed: int = 0) -> np.ndarray:
    """简单的 CPU FPS 回退实现。"""
    N = verts.shape[0]
    if k >= N:
        return verts.copy()[:k]
    np.random.seed(seed)
    chosen = np.zeros(k, dtype=np.int64)
    chosen[0] = np.random.randint(0, N)
    dists = np.linalg.norm(verts - verts[chosen[0]], axis=1)
    for i in range(1, k):
        idx = int(np.argmax(dists))
        chosen[i] = idx
        d_new = np.linalg.norm(verts - verts[idx], axis=1)
        dists = np.minimum(dists, d_new)
    return verts[chosen]


# ------------------------
# Evaluator class
# ------------------------
class Evaluator:
    def __init__(self,
                 model: nn.Module,
                 dataloader: DataLoader,
                 device: torch.device,
                 cfg: Any,
                 out_dir: Optional[str] = None,
                 verbose: bool = True):
        """
        model: 已加载并切换到 eval 的模型（或会在 evaluator 中设置）
        dataloader: 验证集 DataLoader
        device: torch.device
        cfg: SimpleNamespace（你的配置）
        out_dir: 可选，保存 per-sample JSON 的目录
        verbose: 是否显示进度条
        """
        self.model = model.to(device).eval()
        self.dataloader = dataloader
        self.device = device
        self.cfg = cfg
        self.out_dir = out_dir
        self.verbose = verbose
        self.vertex_scale = getattr(cfg.model, "vertex_scale", 1.0)
        self.use_offset = getattr(cfg.model, "use_offset", True)

        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        # determine move-to-device function
        self.move_batch_to_device = getattr(torch_utils_mod, "move_batch_to_device", None) if torch_utils_mod else None
        if self.move_batch_to_device is None:
            self.move_batch_to_device = _fallback_move_batch_to_device

        # load BOP models_info.json
        self.model_info = self._load_models_info()

        # preload model points (for ADD/ADD-S)
        self.model_points_lookup = self._preload_model_points()

    def _load_models_info(self) -> Dict[int, Dict[str, Any]]:
        """
        根据 cfg.dataset.data_root 加载 models_info.json（BOP）
        返回: {obj_id: {...}}
        """
        data_root = getattr(self.cfg.dataset, "data_root", None)
        if data_root is None:
            raise ValueError("cfg.dataset.data_root 未设置")
        # try "models_eval" then "models"
        model_root = os.path.join(data_root, "models_eval")
        if not os.path.exists(model_root):
            model_root = os.path.join(data_root, "models")
        model_info_path = os.path.join(model_root, "models_info.json")
        if not os.path.exists(model_info_path):
            raise FileNotFoundError(f"models_info.json not found at {model_info_path}")
        with open(model_info_path, 'r') as f:
            d = json.load(f)
        # keys to int
        return {int(k): v for k, v in d.items()}

    def _preload_model_points(self, n_points: int = 2000) -> Dict[int, np.ndarray]:
        """
        预加载每个物体的点云（均匀采样或 FPS 采样），以便计算 ADD/ADD-S。
        优先使用 src.utils.mesh 提供的接口，否则使用回退实现。
        """
        lookup: Dict[int, np.ndarray] = {}
        # choose model_root from cfg
        data_root = getattr(self.cfg.dataset, "data_root", None)
        model_root = os.path.join(data_root, "models_eval")
        if not os.path.exists(model_root):
            model_root = os.path.join(data_root, "models")

        # if mesh_utils_mod present, try to use its helper
        if mesh_utils_mod is not None and hasattr(mesh_utils_mod, "load_model_points_dict"):
            try:
                obj_ids = list(self.model_info.keys())
                # mesh_utils_mod.load_model_points_dict should accept (model_root, obj_ids, n_points)
                lookup = mesh_utils_mod.load_model_points_dict(model_root, obj_ids, n_points=n_points)
                logger.info(f"[Evaluator] 使用 src.utils.mesh 加载并采样 {len(lookup)} 个模型点云")
                return lookup
            except Exception as e:
                logger.warning(f"[Evaluator] src.utils.mesh.load_model_points_dict 失败: {e}. 将回退到本地实现.")

        # fallback: iterate each model file, load vertices and sample
        for obj_id in sorted(self.model_info.keys()):
            ply_path = os.path.join(model_root, f"obj_{obj_id:06d}.ply")
            if not os.path.exists(ply_path):
                # try other extensions
                found = False
                for ext in (".obj", ".ply", ".stl", ".off"):
                    cand = os.path.join(model_root, f"obj_{obj_id:06d}{ext}")
                    if os.path.exists(cand):
                        ply_path = cand
                        found = True
                        break
                if not found:
                    logger.warning(f"[Evaluator] 未找到模型文件 for obj {obj_id} in {model_root}, skip ADD-S")
                    continue
            try:
                if mesh_utils_mod is not None and hasattr(mesh_utils_mod, "load_mesh_verts_faces"):
                    verts, faces = mesh_utils_mod.load_mesh_verts_faces(ply_path)
                else:
                    verts, faces = _fallback_load_mesh_verts_faces(ply_path)
                # sample points (prefer a project's fps util if present)
                if mesh_utils_mod is not None and hasattr(mesh_utils_mod, "farthest_point_sampling"):
                    pts = mesh_utils_mod.farthest_point_sampling(verts, n_points, seed=0)
                else:
                    pts = _fallback_farthest_point_sampling(verts, n_points, seed=0)
                lookup[obj_id] = np.asarray(pts, dtype=np.float32)
            except Exception as e:
                logger.warning(f"[Evaluator] 加载/采样模型 {ply_path} 失败: {e}")
        logger.info(f"[Evaluator] 预加载完成：{len(lookup)} 模型点云")
        return lookup

    # ------------------------
    # Decoding helpers
    # ------------------------
    def _decode_kp2d_from_output(self, output_gpu: Dict[str, Any]) -> Optional[np.ndarray]:
        """
        尝试从模型输出解码 2D keypoints (K,2)：
         - 直接 kpt_2d / kp2d / kp_2d
         - PVNet 风格: 'vertex' + 'seg' -> ransac_voting -> kpts2d
        返回 numpy (K,2) 或 None
        """
        # direct keys
        for key in ("kpt_2d", "kp_2d", "kp2d"):
            if key in output_gpu:
                arr = output_gpu[key]
                if isinstance(arr, torch.Tensor):
                    arr = arr.detach().cpu().numpy()
                return np.squeeze(np.array(arr)).astype(np.float32)

        # PVNet style
        if "vertex" in output_gpu and "seg" in output_gpu:
            if ransac_voting_mod is None:
                logger.warning("[Evaluator] 输出包含 'vertex'，但未找到 ransac_voting 模块，无法解码 kp2d")
                return None
            vt = output_gpu["vertex"]   # Tensor (B, 2K, H, W)
            if self.use_offset:
                vt = vt * float(self.vertex_scale)
            seg = output_gpu["seg"]     # Tensor (B, C_seg, H, W) or (B,1,H,W)

            # produce mask_pred in shape (B, 1, H, W) expected by ransac_voting
            try:
                if seg.dim() == 4 and seg.shape[1] > 1:
                    # multiclass logits -> foreground is class 1 (assumption)
                    mask_pred = (torch.softmax(seg, dim=1)[:, 1:2, :, :] > 0.5).float()
                else:
                    # single logits or single channel
                    if seg.dim() == 4:
                        mask_pred = (torch.sigmoid(seg[:, 0:1, :, :]) > 0.5).float()
                    else:
                        mask_pred = (torch.sigmoid(seg).unsqueeze(1) > 0.5).float()
            except Exception as e:
                logger.warning(f"[Evaluator] 生成 mask_pred 失败: {e}")
                return None

            # call ransac_voting; interface may vary so attempt several common signatures
            try:
                # most implementations accept (mask, vertex, num_votes=..., inlier_thresh=..., max_trials=...)
                out = ransac_voting_mod.ransac_voting(
                    mask=mask_pred,
                    vertex=vt,
                    num_votes=getattr(self.cfg.model.ransac_voting, "vote_num", 512),
                    inlier_thresh=getattr(self.cfg.model.ransac_voting, "inlier_thresh", 2.0),
                    max_trials=getattr(self.cfg.model.ransac_voting, "max_trials", 200),
                    use_offset=self.use_offset,
                )
                # expect out[0] = kpts2d (B, K, 2)
                kpts2d = out[0] if isinstance(out, (list, tuple)) else out
                if isinstance(kpts2d, torch.Tensor):
                    kpts2d = kpts2d.detach().cpu().numpy()
                return np.squeeze(np.array(kpts2d)).astype(np.float32)
            except Exception as e:
                logger.warning(f"[Evaluator] 调用 ransac_voting 失败: {e}")
                return None

        return None

    def _solve_pnp(self, kp3d: np.ndarray, kp2d: np.ndarray, K: np.ndarray) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        使用优先级顺序解算 PnP:
          1) geometry_mod.solve_pnp (项目实现)
          2) OpenCV solvePnPRansac 回退
        返回 (R (3x3), t (3,)) 或 None
        """
        if kp3d is None or kp2d is None or K is None:
            return None

        # ensure shapes
        kp3d = np.asarray(kp3d).reshape(-1, 3)
        kp2d = np.asarray(kp2d).reshape(-1, 2)
        K = np.asarray(K).reshape(3, 3)

        # try geometry_mod if available
        if geometry_mod is not None and hasattr(geometry_mod, "solve_pnp"):
            try:
                out = geometry_mod.solve_pnp(kp3d, kp2d, K,
                                             ransac=True,
                                             reproj_thresh=float(getattr(self.cfg.pnp, "reproj_error_thresh", 3.0)))
                if out is None:
                    return None
                # expect out like (R, t, inliers)
                R_pred, t_pred = out[0], out[1]
                return np.asarray(R_pred, dtype=np.float32), np.asarray(t_pred, dtype=np.float32)
            except Exception as e:
                logger.warning(f"[Evaluator] geometry.solve_pnp 调用失败，回退到 OpenCV: {e}")

        # fallback to OpenCV
        if _HAS_CV2:
            try:
                # OpenCV expects float64
                obj_pts = kp3d.astype(np.float64)
                img_pts = kp2d.astype(np.float64)
                Kf = K.astype(np.float64)
                dist_coeffs = None
                # Use solvePnPRansac
                success, rvec, tvec, inliers = cv2.solvePnPRansac(
                    objectPoints=obj_pts,
                    imagePoints=img_pts,
                    cameraMatrix=Kf,
                    distCoeffs=dist_coeffs,
                    reprojectionError=float(getattr(self.cfg.pnp, "reproj_error_thresh", 3.0)),
                    iterationsCount=int(getattr(self.cfg.pnp, "max_iter", 100)),
                    flags=cv2.SOLVEPNP_ITERATIVE
                )
                if not success:
                    return None
                R_mat, _ = cv2.Rodrigues(rvec)
                return np.asarray(R_mat, dtype=np.float32), np.asarray(tvec.reshape(3,), dtype=np.float32)
            except Exception as e:
                logger.warning(f"[Evaluator] OpenCV solvePnPRansac 失败: {e}")
                return None

        logger.error("[Evaluator] 无法解 PnP：既没有 geometry_mod，也没有安装 OpenCV（cv2）")
        return None

    def _get_pose_from_output(self, output_gpu: Dict[str, Any], gt_item: Dict[str, Any]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        智能推理链：
        - 若模型直接输出 R/t -> 返回
        - 否则尝试解码 kp2d (可能含 vertex+seg) -> PnP -> R/t
        """
        # direct R/t
        if "R" in output_gpu and "t" in output_gpu:
            try:
                R = output_gpu["R"].detach().cpu().numpy().squeeze()
                t = output_gpu["t"].detach().cpu().numpy().squeeze()
                return np.asarray(R, dtype=np.float32), np.asarray(t, dtype=np.float32)
            except Exception as e:
                logger.warning(f"[Evaluator] 直接读取 R/t 失败: {e}")

        # decode kp2d
        kp2d = self._decode_kp2d_from_output(output_gpu)
        if kp2d is None:
            return None, None

        # PnP using kp3d from gt_item
        kp3d = gt_item.get("kp3d", None)
        K = gt_item.get("K", None)
        if kp3d is None or K is None:
            logger.warning("[Evaluator] 无法进行 PnP，因为缺少 kp3d 或 K")
            return None, None

        pnp_res = self._solve_pnp(kp3d, kp2d, K)
        if pnp_res is None:
            return None, None
        R_pred, t_pred = pnp_res
        return R_pred, t_pred

    # ------------------------
    # Main evaluation loop
    # ------------------------
    def evaluate(self) -> Dict[str, Any]:
        """
        运行评估循环并返回 summary。
        """
        self.model.eval()
        all_preds_for_bop: List[Dict[str, Any]] = []
        all_gts_for_bop: List[Dict[str, Any]] = []

        # iterate dataloader
        desc = f"[Evaluate] {os.path.basename(getattr(self.cfg.dataset, 'val_data_dir', 'val'))}"
        iterator = tqdm(self.dataloader, desc=desc) if self.verbose else self.dataloader

        # loop
        for batch in iterator:
            try:
                # move to device
                batch_gpu = self.move_batch_to_device(batch, self.device)

                with torch.no_grad():
                    # amp autocast if cuda available
                    with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
                        outputs_gpu = self.model(batch_gpu['inp'])
            except Exception as e:
                logger.error(f"[Evaluator] 模型推理失败于一个 batch：{e}\n{traceback.format_exc()}")
                # skip this batch
                continue

            # ensure outputs_gpu is dict
            if not isinstance(outputs_gpu, dict):
                logger.error("[Evaluator] model(...) did not return a dict. Skipping batch.")
                continue

            # batch size (use CPU batch)
            B = int(batch['inp'].shape[0])

            for i in range(B):
                try:
                    # build gt item (numpy)
                    meta = batch.get('meta', [None]*B)[i]
                    gt_item = {
                        'K': batch['K'][i].cpu().numpy(),
                        'R': batch['R'][i].cpu().numpy(),
                        't': batch['t'][i].cpu().numpy(),
                        'kp3d': batch['kp3d'][i].cpu().numpy(),
                        'obj_id': int(meta.get('obj_id', -1)) if meta else -1,
                        'scene_id': int(meta.get('scene_id', -1)) if meta else -1,
                        'im_id': int(meta.get('im_id', -1)) if meta else -1,
                    }

                    # extract per-sample GPU outputs (keep dims)
                    per_pred_gpu = {k: v[i:i+1] for k, v in outputs_gpu.items() if isinstance(v, torch.Tensor)}

                    # get pose
                    R_pred, t_pred = self._get_pose_from_output(per_pred_gpu, gt_item)

                    pred_dict = {
                        'obj_id': gt_item['obj_id'],
                        'scene_id': gt_item['scene_id'],
                        'im_id': gt_item['im_id'],
                        'R': R_pred if R_pred is not None else np.eye(3, dtype=np.float32),
                        't': t_pred if t_pred is not None else np.zeros(3, dtype=np.float32),
                        'score': 1.0 if R_pred is not None else 0.0
                    }
                    gt_dict = {
                        'obj_id': gt_item['obj_id'],
                        'scene_id': gt_item['scene_id'],
                        'im_id': gt_item['im_id'],
                        'R': gt_item['R'],
                        't': gt_item['t']
                    }

                    # append
                    all_preds_for_bop.append(pred_dict)
                    all_gts_for_bop.append(gt_dict)

                    # optionally save per-sample json
                    if self.out_dir:
                        fname = f"{gt_item['scene_id']:06d}_{gt_item['im_id']:06d}_{gt_item['obj_id']:06d}.json"
                        out_path = os.path.join(self.out_dir, fname)
                        try:
                            save_obj = {
                                'pred': {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in pred_dict.items()},
                                'gt': {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in gt_dict.items()}
                            }
                            with open(out_path, 'w') as f:
                                json.dump(save_obj, f, indent=2)
                        except Exception as e:
                            logger.warning(f"[Evaluator] 无法保存 {out_path}: {e}")

                except Exception as e:
                    logger.error(f"[Evaluator] 处理单样本失败: {e}\n{traceback.format_exc()}")
                    continue

        # evaluation collected, now compute metrics
        if bop_eval_mod is not None and hasattr(bop_eval_mod, 'evaluate_batch'):
            try:
                obj_id_cfg = int(getattr(self.cfg.model, "obj_id", -1))
                if obj_id_cfg not in self.model_info:
                    logger.warning(f"[Evaluator] cfg.model.obj_id={obj_id_cfg} not in models_info.json keys")
                diameter = self.model_info.get(obj_id_cfg, {}).get('diameter', None)

                # threshold is diameter * 0.1 (BOP), keep unit same as model_points (assumed)
                thresholds = {}
                if diameter is not None:
                    thresholds['add'] = float(diameter) * 0.1

                # call evaluate_batch - interface varies in different projects,
                # our bop_eval.py implemented evaluate_batch(predictions, gts, model_points_lookup, thresholds)
                results = bop_eval_mod.evaluate_batch(
                    predictions=all_preds_for_bop,
                    gts=all_gts_for_bop,
                    model_points_lookup=self.model_points_lookup,
                    thresholds=thresholds
                )
                summary = results.get('summary', results)
                # build a stable human-readable final summary
                is_symmetric = False
                if diameter is not None:
                    mi = self.model_info.get(obj_id_cfg, {})
                    is_symmetric = bool(mi.get('symmetries_discrete') or mi.get('symmetries_continuous', False))

                if is_symmetric:
                    metric_name = "ADD-S@0.1d"
                    recall = summary.get('pass_rate_adds', 0.0)
                    avg_err = summary.get('avg_adds', float('nan'))
                else:
                    metric_name = "ADD@0.1d"
                    recall = summary.get('pass_rate_add', 0.0)
                    avg_err = summary.get('avg_add', float('nan'))

                final_summary = {
                    "obj_id": obj_id_cfg,
                    "is_symmetric": is_symmetric,
                    "metric": metric_name,
                    "threshold": thresholds.get('add', None),
                    "recall": float(recall),
                    "avg_error": float(avg_err),
                    "total_instances": int(summary.get('n', len(all_preds_for_bop)))
                }
                logger.info("[Evaluator] BOP evaluation done.")
                print(json.dumps(final_summary, indent=2))
                return final_summary

            except Exception as e:
                logger.error(f"[Evaluator] 调用 bop_eval.evaluate_batch 失败: {e}\n{traceback.format_exc()}")
                # fall through to fallback summary

        # fallback summary if bop_eval unavailable or failed
        logger.warning("[Evaluator] 使用本地回退统计 (bop_eval 不可用或失败)。")
        return self._fallback_summarize(all_preds_for_bop, all_gts_for_bop)

    def _fallback_summarize(self, preds: List[Dict[str, Any]], gts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        如果没有 bop_eval 模块，则用本地度量（RE/TE 平均值，若有点云则计算 ADD/ADD-S）。
        """
        re_list = []
        te_list = []
        add_list = []
        adds_list = []
        n_total = len(preds)
        n_valid = 0

        for p, g in zip(preds, gts):
            try:
                if p['score'] == 0.0:
                    continue
                R_p, t_p = p['R'], p['t']
                R_g, t_g = g['R'], g['t']
                re_list.append(_fallback_rotation_error_deg(R_p, R_g))
                te_list.append(_fallback_translation_error(t_p, t_g))
                n_valid += 1

                obj_id = p.get('obj_id', -1)
                if obj_id in self.model_points_lookup:
                    pts = self.model_points_lookup[obj_id]
                    mi = self.model_info.get(obj_id, {})
                    is_sym = bool(mi.get('symmetries_discrete') or mi.get('symmetries_continuous', False))
                    if is_sym:
                        adds_list.append(_fallback_adds(R_p, t_p, R_g, t_g, pts))
                    else:
                        add_list.append(_fallback_add(R_p, t_p, R_g, t_g, pts))
            except Exception as e:
                logger.debug(f"[Evaluator] 回退度量单样本失败: {e}")

        summary = {"n_total": n_total, "n_valid_with_pred": n_valid}
        if re_list:
            summary['re_mean_deg'] = float(np.mean(re_list))
            summary['te_mean'] = float(np.mean(te_list))
        if add_list:
            summary['add_mean'] = float(np.mean(add_list))
        if adds_list:
            summary['adds_mean'] = float(np.mean(adds_list))
        logger.info(f"[Evaluator] 回退统计完成: {summary}")
        return summary
