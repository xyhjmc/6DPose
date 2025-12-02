"""YOLO-backed PVNet variant.

This module builds a lightweight YOLO11-style backbone (from ``yolo11.yaml`` and
``yoloblocks.py``) and attaches a PVNet-style decoder head that predicts
segmentation masks and vertex fields.

The goal is API compatibility with the existing PVNet family while providing an
incremental path to experiment with YOLO backbones.
"""
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from src.models.yolo.yoloblocks import (
    C2PSA,
    C3k2,
    Concat,
    Conv,
    SPPF,
)
from src.utils.ransac_voting import ransac_voting

LayerDef = Sequence[Any]


def _make_divisible(x: float, divisor: int = 8) -> int:
    """Round channel dimensions following YOLO's width multiplier logic."""

    return int(math.ceil(x / divisor) * divisor)


class YOLO11Backbone(nn.Module):
    """
    Parse and build a YOLO11-style backbone + neck from a YAML spec.

    Only the backbone and upsampling neck layers are materialized; the Detect
    head is skipped because PVNet provides its own prediction head.
    """

    def __init__(self, cfg_path: Union[str, Path], variant: str = "n") -> None:
        super().__init__()
        self.cfg_path = Path(cfg_path)
        with self.cfg_path.open("r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

        self.variant = variant
        self.depth_gain, self.width_gain, self.max_channels = self._get_scale()
        (self.model, self.froms, self.channels, self.feature_indices,) = self._build_model()

    def _get_scale(self) -> Tuple[float, float, int]:
        scale_cfg = self.cfg.get("scales", {})
        if self.variant not in scale_cfg:
            raise ValueError(f"Unknown YOLO11 scale '{self.variant}' in {self.cfg_path}")
        depth, width, max_ch = scale_cfg[self.variant]
        return depth, width, max_ch

    def _depth(self, n: int) -> int:
        return max(int(round(n * self.depth_gain)), 1) if n > 1 else n

    def _width(self, ch: int) -> int:
        ch *= self.width_gain
        ch = _make_divisible(ch, 8)
        return min(int(ch), self.max_channels)

    def _resolve_arg(self, arg: Any) -> Any:
        if isinstance(arg, str) and arg == "nc":
            return self.cfg.get("nc", 80)
        return arg

    def _build_model(self) -> Tuple[nn.ModuleList, List[Union[int, List[int]]], List[int], List[int]]:
        backbone_defs: List[LayerDef] = self.cfg.get("backbone", [])
        head_defs: List[LayerDef] = self.cfg.get("head", [])
        layers: List[nn.Module] = []
        froms: List[Union[int, List[int]]] = []
        ch: List[int] = [3]  # input image channels

        layer_defs = backbone_defs + head_defs
        # Track outputs we want to expose (P3, P4, P5 indices from YAML)
        feature_indices: List[int] = [16, 19, 22]

        for i, (f, n, m, args) in enumerate(layer_defs):
            # Resolve module class; skip detection heads
            m = self._resolve_module(m)
            if m is None:
                # Skip Detect layer; PVNet provides its own heads
                continue
            n = self._depth(n)
            args = [self._resolve_arg(a) for a in args]

            # Compute input channels
            if isinstance(f, int):
                c1 = ch[f + 1] if f != -1 else ch[-1]
            else:
                c1 = sum(ch[j + 1] if j != -1 else ch[-1] for j in f)

            module, c2 = self._make_module(m, c1, args, n)
            layers.append(module)
            froms.append(f)
            ch.append(c2)

        return nn.ModuleList(layers), froms, ch[1:], feature_indices

    def _make_module(
        self,
        module_cls: type,
        c1: int,
        args: List[Any],
        n: int,
    ) -> Tuple[nn.Module, int]:
        # Modules with (c1, c2, ...)
        if module_cls in {Conv, C3k2, SPPF, C2PSA}:
            c2 = self._width(args[0])
            new_args = [c1, c2] + args[1:]
            mod = nn.Sequential(*[module_cls(*new_args) for _ in range(n)]) if n > 1 else module_cls(*new_args)
            return mod, c2

        if module_cls is Concat:
            dim = args[0] if args else 1
            mod = Concat(dimension=dim)
            c2 = c1
            return mod, c2

        if module_cls is nn.Upsample:
            # YAML stores upsample as [size, scale_factor, mode]. Guard against
            # simultaneously passing ``size`` and ``scale_factor`` (PyTorch raises)
            # by preferring scale_factor-driven interpolation when both are
            # provided.
            size = args[0] if len(args) > 0 else None
            scale = args[1] if len(args) > 1 else None
            mode = args[2] if len(args) > 2 else "nearest"

            if size is not None and scale is not None:
                size = None

            mod = module_cls(size=size, scale_factor=scale, mode=mode)
            return mod, c1

        raise ValueError(f"Unsupported module type in YOLO11 builder: {module_cls}")

    def _resolve_module(self, name: Union[str, Any]) -> Union[type, None]:
        if name in {Conv, C3k2, SPPF, C2PSA, Concat, nn.Upsample}:
            return name  # already a class
        if isinstance(name, str):
            lookup = {
                "Conv": Conv,
                "C3k2": C3k2,
                "SPPF": SPPF,
                "C2PSA": C2PSA,
                "Concat": Concat,
                "nn.Upsample": nn.Upsample,
            }
            if name == "Detect":
                return None
            if name not in lookup:
                raise ValueError(f"Unknown module '{name}' in YOLO11 config")
            return lookup[name]
        raise TypeError(f"Unsupported module identifier: {name}")

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        outputs: List[torch.Tensor] = []
        for f, m in zip(self.froms, self.model):
            if isinstance(f, int):
                inp = x if f == -1 else outputs[f]
            else:
                inp = [x if j == -1 else outputs[j] for j in f]
            out = m(inp) if isinstance(inp, list) else m(inp)
            outputs.append(out)
            x = out
        return outputs


class YoloPVNet(nn.Module):
    """PVNet variant that uses a YOLO11 backbone/neck for feature extraction."""

    def __init__(
        self,
        ver_dim: int,
        seg_dim: int,
        backbone_cfg: Union[str, Path] = Path("src/models/yolo/yolo11.yaml"),
        variant: str = "n",
        vote_num: int = 512,
        inlier_thresh: float = 2.0,
        max_trials: int = 200,
        vertex_scale: float = 1.0,
        use_offset: bool = True,
        decoder_dims: Tuple[int, int, int] = (256, 192, 160),
    ) -> None:
        super().__init__()
        self.ver_dim = ver_dim
        self.seg_dim = seg_dim
        self.vote_num = vote_num
        self.inlier_thresh = inlier_thresh
        self.max_trials = max_trials
        self.vertex_scale = vertex_scale
        self.use_offset = use_offset
        self.decode_in_eval: bool = True

        # Backbone + neck
        self.backbone = YOLO11Backbone(cfg_path=backbone_cfg, variant=variant)
        f_indices = self.backbone.feature_indices
        c3, c4, c5 = (self.backbone.channels[idx] for idx in f_indices)
        d8, d4, d2 = decoder_dims

        self.reduce_p5 = nn.Sequential(
            nn.Conv2d(c5, d8, 3, padding=1, bias=False),
            nn.BatchNorm2d(d8),
            nn.ReLU(inplace=True),
        )
        self.fuse_p4 = nn.Sequential(
            nn.Conv2d(c4 + d8, d4, 3, padding=1, bias=False),
            nn.BatchNorm2d(d4),
            nn.ReLU(inplace=True),
        )
        self.fuse_p3 = nn.Sequential(
            nn.Conv2d(c3 + d4, d2, 3, padding=1, bias=False),
            nn.BatchNorm2d(d2),
            nn.ReLU(inplace=True),
        )
        self.pred_head = nn.Sequential(
            nn.Conv2d(d2 + 3, 96, 3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, seg_dim + ver_dim, 1),
        )
        self._init_vertex_head()
        self.feature_indices = f_indices

    @torch.no_grad()
    def decode_keypoint(self, seg_pred: torch.Tensor, vertex_pred: torch.Tensor) -> Dict[str, torch.Tensor]:
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

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feats = self.backbone(x)
        p3, p4, p5 = (feats[idx] for idx in self.feature_indices)

        p5 = self.reduce_p5(p5)
        p5_up = F.interpolate(p5, size=p4.shape[2:], mode="bilinear", align_corners=False)
        p4 = self.fuse_p4(torch.cat([p5_up, p4], dim=1))

        p4_up = F.interpolate(p4, size=p3.shape[2:], mode="bilinear", align_corners=False)
        p3 = self.fuse_p3(torch.cat([p4_up, p3], dim=1))

        fm = F.interpolate(p3, size=x.shape[2:], mode="bilinear", align_corners=False)
        out = self.pred_head(torch.cat([fm, x], dim=1))

        seg_pred = out[:, : self.seg_dim, :, :]
        ver_pred = out[:, self.seg_dim :, :, :]
        ret = {"seg": seg_pred, "vertex": ver_pred}

        if (not self.training) and getattr(self, "decode_in_eval", True):
            ret.update(self.decode_keypoint(seg_pred, ver_pred))
        return ret

    def _init_vertex_head(self) -> None:
        head: nn.Conv2d = self.pred_head[-1]
        with torch.no_grad():
            vertex_weight = head.weight[self.seg_dim :, ...]
            vertex_bias = head.bias[self.seg_dim :]
            vertex_weight.normal_(mean=0.0, std=0.001)
            vertex_bias.zero_()


def get_yolo_pvnet(ver_dim: int, seg_dim: int, **kwargs: Any) -> YoloPVNet:
    return YoloPVNet(ver_dim=ver_dim, seg_dim=seg_dim, **kwargs)
