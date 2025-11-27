# src/losses//pvnet/pvnet_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict  # ⬅️ [修正] 添加了这里的导入


class PVNetLoss(nn.Module):
    """
    PVNet 复合损失函数。

    计算两部分损失:
    1. 分割损失 (Segmentation Loss): 使用交叉熵 (Cross-Entropy)。
    2. 顶点损失 (Vertex/Vote Loss): 使用 Smooth L1，但仅在前景掩码区域内计算。
    """

    def __init__(self,
                 seg_loss_type: str = 'focal',
                 vote_loss_type: str = 'smooth_l1',
                 vote_weight: float = 1.0,
                 seg_weight: float = 1.0,
                 seg_focal_gamma: float = 2.0,
                 seg_focal_alpha=None,
                 seg_class_weights=None,
                 vote_normalize_eps: float = 1e-6,
                 vote_smooth_l1_beta: float = 0.1,
                 skip_vote_if_no_fg: bool = False):
        """
        初始化 PVNet 损失。

        参数:
          seg_loss_type: 分割损失类型 (例如 'cross_entropy' 或 'focal')，默认使用 focal 以缓解前景/背景不平衡。
          vote_loss_type: 顶点损失类型 (例如 'smooth_l1')。
          vote_weight: 顶点损失的权重。
          seg_weight: 分割损失的权重。
          vote_smooth_l1_beta: SmoothL1Loss 的 beta，缩小二次区间以防梯度过小。
        """
        super().__init__()

        self.vote_weight = vote_weight
        self.seg_weight = seg_weight
        self.seg_loss_type = seg_loss_type
        self.seg_focal_gamma = seg_focal_gamma
        self.skip_vote_if_no_fg = skip_vote_if_no_fg
        self.vote_normalize_eps = vote_normalize_eps
        self.vote_smooth_l1_beta = vote_smooth_l1_beta

        if seg_class_weights is not None:
            weight_tensor = torch.tensor(seg_class_weights, dtype=torch.float32)
            self.register_buffer("seg_class_weights", weight_tensor)
        else:
            self.seg_class_weights = None

        if seg_focal_alpha is not None:
            alpha_tensor = torch.tensor(seg_focal_alpha, dtype=torch.float32)
            self.register_buffer("seg_focal_alpha", alpha_tensor)
        else:
            self.seg_focal_alpha = None

        # --- 初始化分割损失 ---
        if seg_loss_type == 'cross_entropy':
            # PyTorch 的 CrossEntropyLoss 自动处理 LogSoftmax
            # 假设 seg_dim > 1 (例如 [背景, 前景])
            self.seg_crit = nn.CrossEntropyLoss(weight=self.seg_class_weights)
        elif seg_loss_type == 'focal':
            self.seg_crit = None  # 使用自定义 focal 计算
        else:
            raise NotImplementedError(f"不支持的分割损失: {seg_loss_type}")

        # --- 初始化顶点损失 ---
        if vote_loss_type == 'smooth_l1':
            # 我们使用 'sum' 归约，然后手动进行掩码归一化
            self.vote_crit = nn.SmoothL1Loss(beta=self.vote_smooth_l1_beta,
                                            reduction='sum')
        else:
            raise NotImplementedError(f"不支持的顶点损失: {vote_loss_type}")

    def forward(self,
                output: Dict[str, torch.Tensor],
                batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        计算总损失。

        参数:
          output: (dict) 模型的输出, 必须包含:
                  - 'seg': (B, C_seg, H, W) 分割 logits
                  - 'vertex': (B, 2K, H, W) 顶点场
          batch: (dict) 数据加载器提供的真值 (Ground Truth), 必须包含:
                 - 'mask': (B, H, W) 真值掩码 (0 或 1)
                 - 'vertex': (B, 2K, H, W) 真值顶点场

        返回:
          total_loss: (torch.Tensor) 标量，可用于反向传播。
          loss_dict: (dict) 包含各分项损失，用于日志记录。
        """

        # --- 1. 准备输入 ---
        seg_pred = output['seg']
        vertex_pred = output['vertex']

        mask_gt = batch['mask']
        vertex_gt = batch['vertex']

        # --- 2. 分割损失 (Segmentation Loss) ---
        # (B, C_seg, H, W) vs (B, H, W)
        if self.seg_loss_type == 'cross_entropy':
            seg_loss = self.seg_crit(seg_pred, mask_gt.long())
        else:
            # focal loss
            ce_per_pixel = F.cross_entropy(
                seg_pred,
                mask_gt.long(),
                reduction='none',
                weight=self.seg_class_weights
            )
            probs = F.softmax(seg_pred, dim=1)
            p_t = probs.gather(1, mask_gt.unsqueeze(1)).squeeze(1)

            if self.seg_focal_alpha is None:
                alpha_t = 1.0
            elif self.seg_focal_alpha.numel() == 1:
                # 对于二分类，前景使用 alpha，背景使用 (1 - alpha)
                alpha_t = torch.where(
                    mask_gt.bool(),
                    self.seg_focal_alpha,
                    1.0 - self.seg_focal_alpha
                )
            else:
                alpha_t = self.seg_focal_alpha[mask_gt]

            focal_factor = torch.pow(1.0 - p_t, self.seg_focal_gamma)
            seg_loss = (alpha_t * focal_factor * ce_per_pixel).mean()

        # --- 3. 顶点损失 (Vertex/Vote Loss) ---

        # (B, H, W) -> (B, 1, H, W)
        # 复制权重，使其与 (B, 2K, H, W) 兼容
        weight = mask_gt.unsqueeze(1).float()

        # 计算总共有多少个前景像素
        num_fg_pixels = weight.sum()

        # 计算带权重的 L1 损失 (只在前景像素上)
        # (B, 2K, H, W) * (B, 1, H, W)
        vote_loss_sum = self.vote_crit(vertex_pred * weight, vertex_gt * weight)

        if self.skip_vote_if_no_fg and num_fg_pixels.item() < 1e-6:
            vote_loss = vote_loss_sum.new_zeros([])
        else:
            denom = num_fg_pixels.clamp(min=1.0) + self.vote_normalize_eps
            # 归一化: (总损失 / 前景像素数)
            vote_loss = vote_loss_sum / denom

        # --- 4. 总损失 ---
        total_loss = (self.seg_weight * seg_loss) + (self.vote_weight * vote_loss)

        # 准备日志
        loss_dict = {
            'total_loss': total_loss,
            'seg_loss': seg_loss,
            'vote_loss': vote_loss
        }

        return total_loss, loss_dict
