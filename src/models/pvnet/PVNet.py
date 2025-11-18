# src/models/pvnet/PVNet.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# 假设 resnet.py 在同一 models 文件夹或上级目录
#  注意: 请确保您的 resnet.py 路径正确
from src.models.pvnet.resnet import resnet18
# 导入我们之前编写的、支持 CPU/GPU 自动切换的 ransac_voting 统一接口
from src.utils.ransac_voting import ransac_voting

"""
PVNet (PyTorch版)
----------------
核心思想:
1. Backbone (ResNet) + FPN (特征金字塔) 提取特征。
2. 预测两个头 (Head):
   - seg_pred: 语义分割掩码 (物体在哪里)
   - ver_pred: 顶点场 (Vertex Field)，一个 (H, W, 2*K) 的张量，
               每个像素 (r, c) 存储 K 个指向 (x, y) 关键点的 2D 向量。
3. 推理 (Inference):
   - 使用 `ransac_voting` 对 `ver_pred` 进行投票，
     从 `seg_pred` 掩码内的像素中稳健地找出 K 个关键点。
"""


class PVNet(nn.Module):
    def __init__(self, ver_dim, seg_dim, fcdim=256, s8dim=128, s4dim=64,
                 s2dim=32, raw_dim=32, use_un_pnp=False,
                 vote_num=512, inlier_thresh=2.0, max_trials=200,vertex_scale:float = 1.0):
        """
        初始化 PVNet 模型。

        参数:
          ver_dim: (int) 顶点场的通道数 (K*2, K=关键点数)
          seg_dim: (int) 语义分割的通道数 (例如 2, [背景, 前景])
          fcdim...raw_dim: (int) 解码器各阶段的特征维度
          use_un_pnp: (bool) [未使用] 是否启用分布式PnP

          --- RANSAC 解码器参数 ---
          vote_num: (int) 每个关键点采样的投票数
          inlier_thresh: (float) RANSAC 内点阈值 (像素)
          max_trials: (int) 每个关键点 RANSAC 迭代次数
        """
        super().__init__()
        self.ver_dim = ver_dim
        self.seg_dim = seg_dim
        self.use_un_pnp = use_un_pnp  # 注意: 此参数在当前代码中未被使用

        # --- RANSAC 投票参数 (用于推理) ---
        self.vote_num = vote_num
        self.inlier_thresh = inlier_thresh
        self.max_trials = max_trials
        self.vertex_scale = vertex_scale
        # -------------------------------
        # 1. 骨干网 (Backbone): ResNet18
        # -------------------------------
        # fully_conv=True, output_stride=8:
        #   使 ResNet 成为全卷积网络，并设置最大下采样步长为 8
        resnet18_8s = resnet18(fully_conv=True, pretrained=True,
                               output_stride=8, remove_avg_pool_layer=True)

        # 替换 ResNet 末尾的 FC 层为 1x1 卷积，以保持空间维度
        resnet18_8s.fc = nn.Sequential(
            nn.Conv2d(resnet18_8s.inplanes, fcdim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(fcdim),
            nn.ReLU(True)
        )
        self.resnet18_8s = resnet18_8s

        # -------------------------------
        # 2. 解码器 (Decoder / FPN)
        # -------------------------------
        # 8s -> 8s (融合 xfc 和 x8s)
        self.conv8s = nn.Sequential(
            nn.Conv2d(128 + fcdim, s8dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(s8dim),
            nn.LeakyReLU(0.1, True)
        )
        # 8s -> 4s (融合 x4s)
        self.conv4s = nn.Sequential(
            nn.Conv2d(64 + s8dim, s4dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(s4dim),
            nn.LeakyReLU(0.1, True)
        )
        # 4s -> 2s (融合 x2s)
        self.conv2s = nn.Sequential(
            nn.Conv2d(64 + s4dim, s2dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(s2dim),
            nn.LeakyReLU(0.1, True)
        )
        # 2s -> 1s (融合原始图像 x)
        self.convraw = nn.Sequential(
            nn.Conv2d(3 + s2dim, raw_dim, 3, 1, 1, bias=False),
            nn.BatchNorm2d(raw_dim),
            nn.LeakyReLU(0.1, True),
            # 最终输出层: 1x1 卷积生成 (seg_dim + ver_dim) 个通道
            nn.Conv2d(raw_dim, seg_dim + ver_dim, 1, 1)
        )

        #  改进: 删除了 __init__ 中的 nn.UpsamplingBilinear2d 层。
        #       我们将在 forward 中使用 F.interpolate 进行动态上采样。

    # ----------------------------------------------------------
    # 3. 推理解码器 (RANSAC 投票)
    # ----------------------------------------------------------
    #  修复: 将 decode_keypoint 从 __init__ 移到类级别，并设为方法。
    @torch.no_grad()  # 推理时不需要梯度
    def decode_keypoint(self, seg_pred: torch.Tensor, vertex_pred: torch.Tensor
                        ) -> dict:
        """
        [推理时调用] 从网络输出 (mask, vertex) 中解码出 2D 关键点。

        使用统一的 ransac_voting() 工具 (自动选择 CPU/GPU)。

        参数:
          seg_pred:    (B, C_seg, H, W) 分割logits张量
          vertex_pred: (B, 2K, H, W) 顶点场张量

        返回:
          output: (dict) 包含解码结果
            {
                'kpt_2d': (B, K, 2) 关键点 (x, y) 坐标,
                'inlier_counts': (B, K) 内点数量
            }
        """
        output = {}

        # --- 1. 二值化 Mask ---
        # (B, C_seg, H, W) -> (B, 1, H, W)
        if self.seg_dim == 1:
            # 如果是单通道 (BCE Loss)，使用 sigmoid > 0.5
            mask_bin = (torch.sigmoid(seg_pred) > 0.5).float()
        else:
            # 如果是多通道 (CrossEntropy Loss)，使用 argmax
            # (B, C_seg, H, W) -> (B, 1, H, W)
            mask_bin = torch.argmax(seg_pred, dim=1, keepdim=True).float()

        vertex_for_voting = vertex_pred
        if getattr(self, "vertex_scale", 1.0) != 1.0:
            vertex_for_voting = vertex_pred * self.vertex_scale

        # --- 2. 调用 RANSAC 投票 ---
        # ransac_voting 会自动处理 (B, 1, H, W) 的 mask
        output['kpt_2d'], output['inlier_counts'] = ransac_voting(
            mask=mask_bin,
            vertex=vertex_for_voting,
            num_votes=self.vote_num,
            inlier_thresh=self.inlier_thresh,
            max_trials=self.max_trials
        )

        # --- 3. 返回结果 ---
        return output

    # ----------------------------------------------------------
    # 4. 前向传播
    # ----------------------------------------------------------
    def forward(self, x: torch.Tensor) -> dict:
        """
        PVNet 的前向传播。

        参数:
          x: (B, 3, H, W) 输入图像

        返回:
          ret: (dict)
            - 训练时: {'seg': (B, C_seg, H, W), 'vertex': (B, 2K, H, W)}
            - 推理时: {'seg': ..., 'vertex': ..., 'kpt_2d': (B, K, 2), ...}
        """

        # --- 1. ResNet 编码器 (Encoder) ---
        # x2s, x4s... 是不同尺度的跳层连接特征
        x2s, x4s, x8s, x16s, x32s, xfc = self.resnet18_8s(x)

        # --- 2. FPN 解码器 (Decoder) ---
        # 融合 8s (xfc + x8s)
        fm = self.conv8s(torch.cat([xfc, x8s], 1))

        #  改进: 动态上采样 (8s -> 4s)，确保与 x4s 尺寸匹配
        fm = F.interpolate(fm, size=x4s.shape[2:], mode='bilinear', align_corners=False)
        # 融合 4s
        fm = self.conv4s(torch.cat([fm, x4s], 1))

        #  改进: 动态上采样 (4s -> 2s)，确保与 x2s 尺寸匹配
        fm = F.interpolate(fm, size=x2s.shape[2:], mode='bilinear', align_corners=False)
        # 融合 2s
        fm = self.conv2s(torch.cat([fm, x2s], 1))

        #  改进: 动态上采样 (2s -> 1s)，确保与 x (原始图像) 尺寸匹配
        fm = F.interpolate(fm, size=x.shape[2:], mode='bilinear', align_corners=False)
        # 融合 1s (原始图像)
        out = self.convraw(torch.cat([fm, x], 1))

        # --- 3. 拆分输出头 ---
        # out: (B, C_seg + 2K, H, W)
        seg_pred = out[:, :self.seg_dim, :, :]
        ver_pred = out[:, self.seg_dim:, :, :]

        # 组装输出字典
        ret = {'seg': seg_pred, 'vertex': ver_pred}

        # --- 4. 推理时解码 ---
        if not self.training:
            #  修复: 正确调用 decode_keypoint 并更新 (update) 字典
            decoded_output = self.decode_keypoint(seg_pred, ver_pred)
            ret.update(decoded_output)

        return ret


# ======================================================
#  工厂函数 (Factory Function)
# ======================================================
def get_pvnet(ver_dim, seg_dim, **kwargs):
    """
    用于创建 PVNet 模型的工厂函数。
    """
    return PVNet(ver_dim=ver_dim, seg_dim=seg_dim, **kwargs)