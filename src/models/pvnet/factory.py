"""
Model factory for PVNet family.
Provides a single entry point to construct PVNet or PVNetPlus based on config.
"""
from types import SimpleNamespace
from typing import Any

from src.models.pvnet.PVNet import PVNet
from src.models.pvnet.PVNetPlus import PVNetPlus


def _get_attr(ns: Any, name: str, default: Any):
    return getattr(ns, name, default) if ns is not None else default


def build_model_from_cfg(cfg) -> PVNet:
    model_cfg: SimpleNamespace = cfg.model
    name = getattr(model_cfg, "name", "PVNet").lower()

    common_kwargs = dict(
        ver_dim=model_cfg.ver_dim,
        seg_dim=model_cfg.seg_dim,
        vote_num=model_cfg.ransac_voting.vote_num,
        inlier_thresh=model_cfg.ransac_voting.inlier_thresh,
        max_trials=model_cfg.ransac_voting.max_trials,
        vertex_scale=getattr(model_cfg, "vertex_scale", 1.0),
        use_offset=getattr(model_cfg, "use_offset", True),
    )

    if name == "pvnetplus":
        plus_cfg = getattr(model_cfg, "plus", None)
        plus_kwargs = dict(
            backbone=_get_attr(model_cfg, "backbone", _get_attr(plus_cfg, "backbone", "resnet34")),
            fcdim=_get_attr(plus_cfg, "fcdim", 512),
            s8dim=_get_attr(plus_cfg, "s8dim", 256),
            s4dim=_get_attr(plus_cfg, "s4dim", 128),
            s2dim=_get_attr(plus_cfg, "s2dim", 64),
            raw_dim=_get_attr(plus_cfg, "raw_dim", 64),
            ctx_dilation=_get_attr(plus_cfg, "ctx_dilation", 3),
            dropout=_get_attr(plus_cfg, "dropout", 0.1),
        )
        return PVNetPlus(**common_kwargs, **plus_kwargs)

    if name == "pvnet":
        return PVNet(**common_kwargs)

    raise ValueError(f"Unsupported model name: {model_cfg.name}")
