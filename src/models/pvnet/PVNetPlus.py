"""
PVNetPlus: 更高容量的 PVNet 变体
--------------------------------
设计目标：
- 采用更强的 backbone (ResNet34/50) 与更宽的解码通道数，增强特征表达能力。
- 通过轻量级通道注意力 (SE) 与上下文卷积，提升顶点矢量预测的细节一致性。
- 保持与现有 PVNet 推理/训练接口兼容：输入/输出键值与 `PVNet` 相同，
  可直接被现有 Trainer/Evaluator 调用。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.pvnet.resnet import resnet18, resnet34, resnet50
from src.utils.ransac_voting import ransac_voting


class SEBlock(nn.Module):
    """Squeeze-and-Excitation 通道注意力。"""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.pool(x)
        weight = self.fc(weight)
        return x * weight


class ContextBlock(nn.Module):
    """轻量级上下文扩张卷积块，用于增大感受野。"""

    def __init__(self, channels: int, dilation: int = 3):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class PVNetPlus(nn.Module):
    def __init__(
        self,
        ver_dim: int,
        seg_dim: int,
        backbone: str = "resnet34",
        fcdim: int = 512,
        s8dim: int = 256,
        s4dim: int = 128,
        s2dim: int = 64,
        raw_dim: int = 64,
        ctx_dilation: int = 3,
        use_un_pnp: bool = False,
        vote_num: int = 768,
        inlier_thresh: float = 2.0,
        max_trials: int = 300,
        vertex_scale: float = 1.0,
        dropout: float = 0.1,
        use_offset: bool = True,
    ):
        super().__init__()
        self.ver_dim = ver_dim
        self.seg_dim = seg_dim
        self.use_un_pnp = use_un_pnp
        self.vote_num = vote_num
        self.inlier_thresh = inlier_thresh
        self.max_trials = max_trials
        self.vertex_scale = vertex_scale
        self.use_offset = use_offset

        backbone = backbone.lower()
        if backbone == "resnet18":
            base = resnet18
            block_expansion = 1
        elif backbone == "resnet34":
            base = resnet34
            block_expansion = 1
        elif backbone == "resnet50":
            base = resnet50
            block_expansion = 4
        else:
            raise ValueError(f"Unsupported backbone for PVNetPlus: {backbone}")

        encoder = base(
            fully_conv=True,
            pretrained=True,
            output_stride=8,
            remove_avg_pool_layer=True,
        )
        encoder.fc = nn.Sequential(
            nn.Conv2d(encoder.inplanes, fcdim, 3, padding=1, bias=False),
            nn.BatchNorm2d(fcdim),
            nn.ReLU(inplace=True),
        )
        self.encoder = encoder

        # 跳连特征通道数随 backbone 变化（ResNet18/34 使用 BasicBlock，ResNet50 使用 Bottleneck）
        x4s_channels = 64 * block_expansion
        x8s_channels = 128 * block_expansion

        # FPN 解码器 + 注意力
        self.conv8s = nn.Sequential(
            nn.Conv2d(x8s_channels + fcdim, s8dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(s8dim),
            nn.LeakyReLU(0.1, inplace=True),
            SEBlock(s8dim),
        )
        self.conv4s = nn.Sequential(
            nn.Conv2d(x4s_channels + s8dim, s4dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(s4dim),
            nn.LeakyReLU(0.1, inplace=True),
            ContextBlock(s4dim, dilation=ctx_dilation),
        )
        self.conv2s = nn.Sequential(
            nn.Conv2d(64 + s4dim, s2dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(s2dim),
            nn.LeakyReLU(0.1, inplace=True),
            SEBlock(s2dim),
        )
        self.convraw = nn.Sequential(
            nn.Conv2d(3 + s2dim, raw_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(raw_dim),
            nn.LeakyReLU(0.1, inplace=True),
            ContextBlock(raw_dim, dilation=ctx_dilation),
            nn.Dropout2d(dropout),
            nn.Conv2d(raw_dim, seg_dim + ver_dim, 1, bias=True),
        )

    @torch.no_grad()
    def decode_keypoint(self, seg_pred: torch.Tensor, vertex_pred: torch.Tensor) -> dict:
        if self.seg_dim == 1:
            mask_bin = (torch.sigmoid(seg_pred) > 0.5).float()
        else:
            mask_bin = torch.argmax(seg_pred, dim=1, keepdim=True).float()

        vertex_for_voting = vertex_pred
        if getattr(self, "vertex_scale", 1.0) != 1.0 and self.use_offset:
            vertex_for_voting = vertex_pred * self.vertex_scale

            kpt_2d, inlier_counts = ransac_voting(
                mask=mask_bin,
                vertex=vertex_for_voting,
                num_votes=self.vote_num,
                inlier_thresh=self.inlier_thresh,
                max_trials=self.max_trials,
                use_offset=self.use_offset,
        )
        return {"kpt_2d": kpt_2d, "inlier_counts": inlier_counts}

    def forward(self, x: torch.Tensor) -> dict:
        x2s, x4s, x8s, x16s, x32s, xfc = self.encoder(x)

        fm = self.conv8s(torch.cat([xfc, x8s], dim=1))
        fm = F.interpolate(fm, size=x4s.shape[2:], mode="bilinear", align_corners=False)
        fm = self.conv4s(torch.cat([fm, x4s], dim=1))
        fm = F.interpolate(fm, size=x2s.shape[2:], mode="bilinear", align_corners=False)
        fm = self.conv2s(torch.cat([fm, x2s], dim=1))
        fm = F.interpolate(fm, size=x.shape[2:], mode="bilinear", align_corners=False)
        out = self.convraw(torch.cat([fm, x], dim=1))

        seg_pred = out[:, : self.seg_dim, :, :]
        ver_pred = out[:, self.seg_dim :, :, :]
        ret = {"seg": seg_pred, "vertex": ver_pred}

        if not self.training:
            ret.update(self.decode_keypoint(seg_pred, ver_pred))
        return ret


def get_pvnet_plus(ver_dim: int, seg_dim: int, **kwargs) -> PVNetPlus:
    return PVNetPlus(ver_dim=ver_dim, seg_dim=seg_dim, **kwargs)