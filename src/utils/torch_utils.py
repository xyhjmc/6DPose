# src/utils/torch_utils.py
"""
存放 PyTorch 相关的、可重用的通用工具函数。
"""
import torch
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Any, List, Tuple  # 导入 List 和 Tuple


def move_batch_to_device(batch: Any,  # [修正]：将类型从 Dict[str, Any] 改为 Any
                         device: torch.device) -> Any:  # [修正]：返回类型也改为 Any
    """
    将一个批次 (batch) 中的所有张量递归地移动到指定 device。

    参数:
      batch: 包含张量或其他数据的字典, 列表, 或单个张量。
      device: 目标设备 (例如 torch.device('cuda'))。

    返回:
      与输入结构相同，但张量已移动到设备上的新批次。
    """
    # non_blocking=True 可以在 CPU 和 GPU 之间异步传输数据，提高效率

    # 基础情况：如果它是一个张量，移动它
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=True)

    # 递归情况 1：如果是字典
    elif isinstance(batch, dict):
        return {
            k: move_batch_to_device(v, device)
            for k, v in batch.items()
        }

    # 递归情况 2：如果是列表或元组
    elif isinstance(batch, (list, tuple)):
        # [注]：保持原始类型（list 或 tuple）
        return type(batch)(move_batch_to_device(v, device) for v in batch)

    # 基础情况 2：保持非张量/非容器类型（如 int, str）不变
    else:
        return batch


def log_scalars_to_writer(writer: SummaryWriter,
                          log_dict: Dict[str, Any],
                          prefix: str,
                          step: int):
    """
    将一个字典中的标量记录到 TensorBoard。
    它会自动处理 .item()。

    参数:
      writer: TensorBoard 的 SummaryWriter 实例。
      log_dict: 包含标量损失的字典 (例如 {'total_loss': tensor(0.5)})。
      prefix: TensorBoard 标签的前缀 (例如 'train' 或 'val')。
      step: 当前的全局步骤或 epoch。
    """
    for k, v in log_dict.items():
        if isinstance(v, torch.Tensor):
            # 确保张量是标量并获取 .item()
            v = v.item()

        if isinstance(v, (float, int)):
            writer.add_scalar(f"{prefix}/{k}", v, step)
        else:
            print(f"警告: 无法记录非标量值 {k} (类型: {type(v)}) 到 TensorBoard。")