# src/losses/alternate_losses.py
"""
来自 clean-pvnet/lib/utils/net_utils.py 的备选损失函数。

包含:
- FocalLoss (用于 CenterNet 风格的检测)
- SmoothL1Loss (带掩码的平滑L1损失的手动实现)
- AELoss (关联嵌入损失，用于实例分割)
- PolyMatchingLoss
- AttentionLoss
- ... 以及它们的辅助函数
"""

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


__all__ = [
    "FocalLoss",
    "SmoothL1Loss",
    "AELoss",
    "PolyMatchingLoss",
    "AttentionLoss",
    "Ind2dRegL1Loss",
    "IndL1Loss1d",
    "GeoCrossEntropyLoss",
]



# --- Focal Loss (及其辅助函数) ---

def sigmoid(x):
    y = torch.clamp(x.sigmoid(), min=1e-4, max=1 - 1e-4)
    return y


def _neg_loss(pred, gt):
    """
    修改版的 Focal Loss (来自 CornerNet)。

    参数:
      pred (B, C, H, W) 预测的热图 (已 sigmoid)
      gt (B, C, H, W) 真值热图
    """
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, 4)

    loss = 0

    # 正样本损失
    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    # 负样本损失
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss


class FocalLoss(nn.Module):
    """Focal Loss 的 nn.Module 封装。"""

    def __init__(self):
        super(FocalLoss, self).__init__()
        self.neg_loss = _neg_loss

    def forward(self, out, target):
        # 注意: 原始实现要求 out 必须先经过 sigmoid
        return self.neg_loss(out, target)


# --- Smooth L1 (及其辅助函数) ---

def smooth_l1_loss(vertex_pred, vertex_targets, vertex_weights, sigma=1.0, normalize=True, reduce=True):
    """
    [手动实现的 Smooth L1 Loss]
    这在功能上与我们 `pvnet_loss.py` 中的实现类似，但提供了更手动的归一化控制。

    参数:
      vertex_pred:     [B, C, H, W] 预测值
      vertex_targets:  [B, C, H, W] 真值
      vertex_weights:  [B, 1, H, W] 权重掩码 (例如前景掩码)
      sigma: L1 和 L2 切换的阈值
      normalize: 是否按权重和通道数归一化
      reduce: 是否计算均值
    """
    b, ver_dim, _, _ = vertex_pred.shape
    sigma_2 = sigma ** 2
    vertex_diff = vertex_pred - vertex_targets

    # 乘以权重 (掩码)
    diff = vertex_weights * vertex_diff
    abs_diff = torch.abs(diff)

    # Smooth L1 的数学定义
    smoothL1_sign = (abs_diff < 1. / sigma_2).detach().float()
    in_loss = torch.pow(diff, 2) * (sigma_2 / 2.) * smoothL1_sign \
              + (abs_diff - (0.5 / sigma_2)) * (1. - smoothL1_sign)

    if normalize:
        # 归一化: (B, C*H*W) -> (B,)
        in_loss = torch.sum(in_loss.view(b, -1), 1)
        # 除以 (通道数 * 像素权重总和)
        norm = (ver_dim * torch.sum(vertex_weights.view(b, -1), 1) + 1e-3)
        in_loss = in_loss / norm

    if reduce:
        in_loss = torch.mean(in_loss)

    return in_loss


class SmoothL1Loss(nn.Module):
    """SmoothL1Loss 的 nn.Module 封装。"""

    def __init__(self):
        super(SmoothL1Loss, self).__init__()
        self.smooth_l1_loss = smooth_l1_loss

    def forward(self, preds, targets, weights, sigma=1.0, normalize=True, reduce=True):
        return self.smooth_l1_loss(preds, targets, weights, sigma, normalize, reduce)


# --- AE Loss (关联嵌入) ---

class AELoss(nn.Module):
    """关联嵌入损失 (Associative Embedding Loss)，用于实例分割。"""

    def __init__(self):
        super(AELoss, self).__init__()

    def forward(self, ae, ind, ind_mask):
        """
        参数:
          ae: [B, 1, H, W] 预测的 "tag" (嵌入) 图
          ind: [B, max_objs, max_parts] 实例中每个部分的像素索引
          ind_mask: [B, max_objs, max_parts] 掩码
        """
        b, _, h, w = ae.shape
        b, max_objs, max_parts = ind.shape
        obj_mask = torch.sum(ind_mask, dim=2) != 0

        ae = ae.view(b, h * w, 1)
        seed_ind = ind.view(b, max_objs * max_parts, 1)

        # 收集每个实例部分的 "tag"
        tag = ae.gather(1, seed_ind).view(b, max_objs, max_parts)

        # 计算每个实例的 "tag" 均值
        tag_mean = tag * ind_mask
        tag_mean = tag_mean.sum(2) / (ind_mask.sum(2) + 1e-4)

        # 1. "Pull" 损失: 将同一实例的 tag 拉向它们的均值
        pull_dist = (tag - tag_mean.unsqueeze(2)).pow(2) * ind_mask
        obj_num = obj_mask.sum(dim=1).float()
        pull = (pull_dist.sum(dim=(1, 2)) / (obj_num + 1e-4)).sum()
        pull /= b

        # 2. "Push" 损失: 将不同实例的均值推开
        push_dist = torch.abs(tag_mean.unsqueeze(1) - tag_mean.unsqueeze(2))
        push_dist = 1 - push_dist
        push_dist = nn.functional.relu(push_dist, inplace=True)
        obj_mask_pairs = (obj_mask.unsqueeze(1) + obj_mask.unsqueeze(2)) == 2
        push_dist = push_dist * obj_mask_pairs.float()
        push = ((push_dist.sum(dim=(1, 2)) - obj_num) / (obj_num * (obj_num - 1) + 1e-4)).sum()
        push /= b

        return pull, push


# --- 其他损失 (PolyMatching, Attention, IndReg) ---

class PolyMatchingLoss(nn.Module):
    """多边形匹配损失 (用于多边形回归)。"""

    def __init__(self, pnum):
        super(PolyMatchingLoss, self).__init__()
        self.pnum = pnum
        batch_size = 1
        pidxall = np.zeros(shape=(batch_size, pnum, pnum), dtype=np.int32)
        for b in range(batch_size):
            for i in range(pnum):
                pidx = (np.arange(pnum) + i) % pnum
                pidxall[b, i] = pidx

        # [警告] 原始代码硬编码了 'cuda'
        # device = torch.device('cuda')
        # 我们将其改为可注册的缓冲区 (buffer)，以便 .to(device) 生效
        pidxall = torch.from_numpy(np.reshape(pidxall, newshape=(batch_size, -1)))
        feature_id = pidxall.unsqueeze_(2).long().expand(pidxall.size(0), pidxall.size(1), 2)
        self.register_buffer('feature_id', feature_id, persistent=False)

    def forward(self, pred, gt, loss_type="L2"):
        pnum = self.pnum
        batch_size = pred.size()[0]
        # 使用 self.feature_id (它会自动跟随 .to(device))
        feature_id = self.feature_id.expand(batch_size, self.feature_id.size(1), 2)

        gt_expand = torch.gather(gt, 1, feature_id).view(batch_size, pnum, pnum, 2)
        pred_expand = pred.unsqueeze(1)
        dis = pred_expand - gt_expand

        if loss_type == "L2":
            dis = (dis ** 2).sum(3).sqrt().sum(2)
        elif loss_type == "L1":
            dis = torch.abs(dis).sum(3).sum(2)

        min_dis, min_id = torch.min(dis, dim=1, keepdim=True)
        return torch.mean(min_dis)


class AttentionLoss(nn.Module):
    """注意力损失 (用于边缘检测等)。"""

    def __init__(self, beta=4, gamma=0.5):
        super(AttentionLoss, self).__init__()
        self.beta = beta
        self.gamma = gamma

    def forward(self, pred, gt):
        num_pos = torch.sum(gt)
        num_neg = torch.sum(1 - gt)
        alpha = num_neg / (num_pos + num_neg)

        # 根据预测值调整权重
        edge_beta = torch.pow(self.beta, torch.pow(1 - pred, self.gamma))
        bg_beta = torch.pow(self.beta, torch.pow(pred, self.gamma))

        loss = 0
        loss = loss - alpha * edge_beta * torch.log(torch.clamp(pred, 1e-4)) * gt
        loss = loss - (1 - alpha) * bg_beta * torch.log(torch.clamp(1 - pred, 1e-4)) * (1 - gt)
        return torch.mean(loss)


# --- 索引采集 (Index Gathering) 辅助函数 ---

def _gather_feat(feat, ind, mask=None):
    """根据索引 (ind) 从特征图 (feat) 中采集特征。"""
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def _tranpose_and_gather_feat(feat, ind):
    """
    辅助函数：(B, C, H, W) -> (B, H*W, C)，然后采集。
    """
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat


class Ind2dRegL1Loss(nn.Module):
    """
    (用于 CenterNet)
    在特定索引 (ind) 处计算 L1/SmoothL1 回归损失 (2D 偏移)。
    """

    def __init__(self, type='l1'):
        super(Ind2dRegL1Loss, self).__init__()
        if type == 'l1':
            self.loss = F.l1_loss
        elif type == 'smooth_l1':
            self.loss = F.smooth_l1_loss

    def forward(self, output, target, ind, ind_mask):
        """ind: [B, max_objs, max_parts]"""
        b, max_objs, max_parts = ind.shape
        ind = ind.view(b, max_objs * max_parts)
        # 采集预测值
        pred = _tranpose_and_gather_feat(output, ind).view(b, max_objs, max_parts, output.size(1))

        mask = ind_mask.unsqueeze(3).expand_as(pred)
        loss = self.loss(pred * mask, target * mask, reduction='sum')
        loss = loss / (mask.sum() + 1e-4)
        return loss


class IndL1Loss1d(nn.Module):
    """
    (用于 CenterNet)
    在特定索引 (ind) 处计算 L1/SmoothL1 回归损失 (1D 值，例如深度)。
    """

    def __init__(self, type='l1'):
        super(IndL1Loss1d, self).__init__()
        if type == 'l1':
            self.loss = F.l1_loss
        elif type == 'smooth_l1':
            self.loss = F.smooth_l1_loss

    def forward(self, output, target, ind, weight):
        """ind: [B, N]"""
        output = _tranpose_and_gather_feat(output, ind)
        weight = weight.unsqueeze(2)
        loss = self.loss(output * weight, target * weight, reduction='sum')
        loss = loss / (weight.sum() * output.size(2) + 1e-4)
        return loss


class GeoCrossEntropyLoss(nn.Module):
    """几何交叉熵损失 (GeoCrossEntropyLoss)"""

    def __init__(self):
        super(GeoCrossEntropyLoss, self).__init__()

    def forward(self, output, target, poly):
        output = F.softmax(output, dim=1)
        output = torch.log(torch.clamp(output, min=1e-4))
        poly = poly.view(poly.size(0), 4, poly.size(1) // 4, 2)
        target = target[..., None, None].expand(poly.size(0), poly.size(1), 1, poly.size(3))
        target_poly = torch.gather(poly, 2, target)
        sigma = (poly[:, :, 0] - poly[:, :, 1]).pow(2).sum(2, keepdim=True)
        kernel = torch.exp(-(poly - target_poly).pow(2).sum(3) / (sigma / 3))
        loss = -(output * kernel.transpose(2, 1)).sum(1).mean()
        return loss