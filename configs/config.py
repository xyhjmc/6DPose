# config/config.py
"""
配置文件加载器（增强版）

特性：
- 支持在实验配置中使用 `defaults: [ "default", "other" ]` 或 `_base_: ["default.yaml"]`
  来引用一个或多个 base 配置文件（按顺序合并，后者覆盖前者）。
- 递归合并（实验配置覆盖 base 配置）。
- 将 dict 转换为 Namespace（递归），并且会把 list 中的 dict 元素也转换成 Namespace。
- 对若干关键路径进行稳健检查并报出友好错误。
- 兼容相对路径（在 configs 目录下搜索 base 文件）。
"""

import os
import yaml
from types import SimpleNamespace
from typing import Dict, Any, List, Optional


def _load_yaml_file(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"YAML 文件不存在: {path}")
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
        return data if data is not None else {}


def _recursive_merge(base: Dict[str, Any], new: Dict[str, Any]) -> Dict[str, Any]:
    """
    将 new 覆盖合并到 base（就地修改 base 并返回）。
    对于两个均为 dict 的同名键，递归合并；否则 new 覆盖 base。
    """
    for k, v in new.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _recursive_merge(base[k], v)
        else:
            base[k] = v
    return base


def _dict_to_namespace(d: Dict[str, Any]) -> SimpleNamespace:
    """
    递归将 dict -> SimpleNamespace。
    同时把 list 中的 dict 元素也转换为 Namespace，以便使用 cfg.xxx.yyy 访问。
    """
    ns = SimpleNamespace()
    for k, v in d.items():
        if isinstance(v, dict):
            setattr(ns, k, _dict_to_namespace(v))
        elif isinstance(v, list):
            # 如果列表元素是 dict，则转换
            new_list = []
            for item in v:
                if isinstance(item, dict):
                    new_list.append(_dict_to_namespace(item))
                else:
                    new_list.append(item)
            setattr(ns, k, new_list)
        else:
            setattr(ns, k, v)
    return ns


def _find_base_paths(exp_cfg: Dict[str, Any], config_path: str, default_path: Optional[str]) -> List[str]:
    """
    确定需要加载的 base 配置文件路径（按顺序）。
    支持:
      - exp_cfg 包含 'defaults': ['default', 'other'] （常见 Hydra 风格）
      - exp_cfg 包含 '_base_' 或 'base'：字符串或列表
      - fallback 使用 default_path（如果给了）
    返回文件路径列表（可以为空）
    """
    configs_dir = os.path.dirname(default_path) if default_path else os.path.dirname(config_path)
    base_paths = []

    # 1) 首先处理 explicit keys：'_base_', 'base', 'defaults'
    for key in ['_base_', 'base', 'defaults']:
        if key in exp_cfg and exp_cfg[key] is not None:
            entries = exp_cfg.pop(key)
            if isinstance(entries, str):
                entries = [entries]
            if not isinstance(entries, list):
                continue
            for e in entries:
                # e 可能是 'default' 或 'path/to/file.yaml' 或 {'some': 'mapping'}（ignore mapping case)
                if isinstance(e, dict):
                    # 如果是 mapping（Hydra 支持复杂 defaults），跳过复杂解析，用户不应该用 mapping 形式
                    continue
                # 尝试原样作为文件路径
                cand = e
                # 如果没有扩展名，尝试添加 .yaml
                if not os.path.splitext(cand)[1]:
                    cand_try = cand + ".yaml"
                else:
                    cand_try = cand
                # 先相对于 configs_dir 查找
                p1 = os.path.join(configs_dir, cand_try)
                if os.path.exists(p1):
                    base_paths.append(p1)
                    continue
                # 再尝试相对于 config_path 的同级目录
                p2 = os.path.join(os.path.dirname(config_path), cand_try)
                if os.path.exists(p2):
                    base_paths.append(p2)
                    continue
                # 再尝试 cand as absolute/relative path
                if os.path.exists(cand):
                    base_paths.append(os.path.abspath(cand))
                    continue
                # 未找到则抛出（显式告诉用户）
                raise FileNotFoundError(f"未找到 base 配置文件: '{e}' (尝试过: {p1}, {p2}, {cand})")

    # 2) 如果没有任何 base 指定但提供了 default_path，则把 default_path 作为首选 base
    if not base_paths and default_path:
        if os.path.exists(default_path):
            base_paths.append(default_path)

    return base_paths


def _validate_required_keys(cfg_dict: Dict[str, Any], required_paths: Optional[List[str]] = None):
    """
    required_paths: list of dotted-paths, 比如 ['run.log_dir', 'dataset.train_data_dir']
    抛出 ValueError 当某个路径不存在或为 None。
    """
    if not required_paths:
        return
    for p in required_paths:
        parts = p.split('.')
        cur = cfg_dict
        missing = False
        for part in parts:
            if not isinstance(cur, dict) or part not in cur:
                missing = True
                break
            cur = cur[part]
        if missing or cur is None:
            raise ValueError(f"配置缺失或为空: '{p}'. 请在 experiment config 中覆盖对应项.")


def load_config(config_path: str, default_path: Optional[str] = "configs/default.yaml",
                required: Optional[List[str]] = None) -> SimpleNamespace:
    """
    主入口：加载并合并配置文件。

    - config_path: 实验配置路径 (必需)
    - default_path: 默认配置 (可选)
    - required: 可选的必需字段（以 dotted path 表示），例如:
        ['run.log_dir', 'run.checkpoint_dir', 'dataset.train_data_dir', 'dataset.val_data_dir']
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"实验配置文件不存在: {config_path}")

    # 加载实验 cfg
    exp_cfg = _load_yaml_file(config_path)

    # 解析并加载 base 文件（如果有 defaults/_base_）
    base_paths = _find_base_paths(exp_cfg, config_path, default_path)

    merged: Dict[str, Any] = {}
    for bp in base_paths:
        base_cfg = _load_yaml_file(bp)
        _recursive_merge(merged, base_cfg)

    # 最后把实验 cfg 合并进来（覆盖 base）
    _recursive_merge(merged, exp_cfg)

    # 验证必需键（在字典层）
    if required is None:
        required = ['run.log_dir', 'run.checkpoint_dir', 'dataset.train_data_dir', 'dataset.val_data_dir']
    try:
        _validate_required_keys(merged, required)
    except ValueError as e:
        # 给出友好信息并重新抛出
        raise ValueError(f"配置验证失败: {e}")

    # 转 Namespace
    cfg_ns = _dict_to_namespace(merged)
    return cfg_ns


# -------------------------
# 简易单元演示（仅在直接运行此模块时）
# -------------------------
if __name__ == "__main__":
    # 演示用例：假设 project_root/configs 下有 default.yaml 和 一个 experiment.yaml
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg_dir = os.path.join(here, "configs")
    example_default = os.path.join(cfg_dir, "default.yaml")
    example_exp = os.path.join(cfg_dir, "pvnet_linemod_driller.yaml")

    print("示例 config 目录:", cfg_dir)
    if os.path.exists(example_default) and os.path.exists(example_exp):
        cfg = load_config(example_exp, default_path=example_default)
        print("加载成功，部分字段预览:")
        print(" run.log_dir:", getattr(cfg.run, "log_dir", None))
        print(" dataset.train_data_dir:", getattr(cfg.dataset, "train_data_dir", None))
        print(" model.name:", getattr(cfg.model, "name", None))
    else:
        print("示例文件不存在，请把 configs 放在正确位置再试。")
