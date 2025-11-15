# src/utils/train_model_utils.py
"""
模型训练相关的工具函数，主要负责：
1. 保存和加载完整的训练状态 (模型、优化器、调度器、epoch)。
2. 加载模型权重 (用于推理或迁移)。
3. 操作 state_dict 的键名 (例如添加/删除 'module.' 前缀)。
"""

import torch
import torch.nn as nn
import os
from collections import OrderedDict
from typing import Any,Union


def save_model(net: nn.Module,
               optim: torch.optim.Optimizer,
               scheduler: torch.optim.lr_scheduler._LRScheduler,
               recorder,  # recorder 可能是您自定义的日志/状态记录器
               epoch: int,
               model_dir: str):
    """
    保存一个完整的训练检查点 (checkpoint)。

    参数:
      net: 模型 (nn.Module)
      optim: 优化器
      scheduler: 学习率调度器
      recorder: 状态记录器 (例如用于保存 mAP, loss 历史等)
      epoch: 当前的 epoch 索引
      model_dir: 保存检查点的目录
    """
    # 确保目录存在
    os.makedirs(model_dir, exist_ok=True)

    # 组合保存内容
    save_dict = {
        'net': net.state_dict(),
        'optim': optim.state_dict(),
        'scheduler': scheduler.state_dict(),
        'recorder': recorder,
        'epoch': epoch
    }

    # 保存到文件
    torch.save(save_dict, os.path.join(model_dir, '{}.pth'.format(epoch)))

    # --- (可选) 清理旧的检查点 ---
    # 原始代码包含一个清理逻辑，保留最新的 N 个检查点
    # [修复后的代码]
    # [修复后的代码 - 单行版]
    pths = [int(pth.split('.')[0]) for pth in os.listdir(model_dir)
            if pth.endswith('.pth') and pth.split('.')[0].isdigit()]
    if len(pths) <= 20:  # 保留最近 20 个
        return

    # 找到最旧的检查点并删除
    try:
        oldest_pth = min(pths)
        os.remove(os.path.join(model_dir, '{}.pth'.format(oldest_pth)))
    except Exception as e:
        print(f"警告: 清理旧检查点 {oldest_pth}.pth 失败: {e}")


def load_model(net: nn.Module,
               optim: torch.optim.Optimizer,
               scheduler: torch.optim.lr_scheduler._LRScheduler,
               recorder,
               model_dir: str,
               resume: bool = True,
               epoch: int = -1) -> int:
    """
    加载一个完整的训练检查点 (用于恢复训练)。

    参数:
      ... (与 save_model 相同) ...
      resume: (bool) 是否加载
      epoch: (int) 要加载的特定 epoch (-1 表示加载最新的)

    返回:
      (int) 加载后的下一个 epoch 索引 (例如, 加载了 epoch 99, 返回 100)
    """
    if not resume:
        return 0

    if not os.path.exists(model_dir):
        return 0

    # 查找所有 .pth 文件
    pths = [int(pth.split('.')[0]) for pth in os.listdir(model_dir)
            if pth.endswith('.pth') and pth.split('.')[0].isdigit()]
    if len(pths) == 0:
        return 0

    # 确定要加载哪个 epoch
    if epoch == -1:
        load_epoch = max(pths)
    else:
        load_epoch = epoch

    load_path = os.path.join(model_dir, '{}.pth'.format(load_epoch))

    if not os.path.exists(load_path):
        print(f"警告: 找不到检查点文件 {load_path}")
        return 0

    print(f'加载检查点: {load_path}')

    try:
        pretrained_model = torch.load(load_path)

        net.load_state_dict(pretrained_model['net'])
        optim.load_state_dict(pretrained_model['optim'])
        scheduler.load_state_dict(pretrained_model['scheduler'])
        # [修复] recorder 是一个字典，使用 .update() 来加载键值
        recorder.update(pretrained_model['recorder'])

        return pretrained_model['epoch'] + 1
    except Exception as e:
        print(f"加载检查点失败: {e}。从 epoch 0 开始。")
        return 0


# [修复后的代码]
def load_network(net: nn.Module,
                 model_dir: str,
                 resume: bool = True,
                 epoch: Union[int, str] = -1,  # [修复] 接受 int (-1) 或 str ('best')
                 strict: bool = True):
    """
    [仅加载模型权重] (用于推理或迁移学习)。
    """
    if not resume:
        return 0

    if not os.path.exists(model_dir):
        return 0

    load_epoch_name = None  # 将是 'best' 或一个数字

    if epoch == -1:
        # --- 自动查找最新的 epoch ---
        # [修复] 过滤掉 'best.pth'
        pths = [int(pth.split('.')[0]) for pth in os.listdir(model_dir)
                if pth.endswith('.pth') and pth.split('.')[0].isdigit()]
        if len(pths) == 0:
            print(f"警告: 在 {model_dir} 中找不到任何数字 epoch 的 .pth 文件。")
            return 0
        load_epoch_name = max(pths)  # (例如 49)
    else:
        # --- 使用指定的 epoch (例如 'best') ---
        load_epoch_name = epoch  # (例如 "best")

    load_path = os.path.join(model_dir, f"{load_epoch_name}.pth")

    if not os.path.exists(load_path):
        print(f"警告: 找不到检查点文件 {load_path}")
        return 0

    print(f"加载模型权重: {load_path}")

    try:
        pretrained_model = torch.load(load_path)

        # 仅加载 'net' 部分
        net.load_state_dict(pretrained_model['net'], strict=strict)

        # [修复] 检查 'epoch' 键的类型
        loaded_epoch = pretrained_model['epoch']
        if isinstance(loaded_epoch, int):
            # 如果是数字 (例如 49)，返回下一个 epoch (50)
            return loaded_epoch + 1
        else:
            # 如果是字符串 (例如 "best")，返回 0 表示加载成功
            return 0

    except Exception as e:
        print(f"加载模型权重失败: {e}。")
        return 0
#
# def load_network(net: nn.Module,
#                  model_dir: str,
#                  resume: bool = True,
#                  epoch: int = -1,
#                  strict: bool = True):
#     """
#     [仅加载模型权重] (用于推理或迁移学习)。
#
#     参数:
#       net: 你的模型实例
#       model_dir: 权重目录
#       strict: (bool) 是否严格匹配 state_dict 的键名
#
#     返回:
#       (int) 加载的 epoch 编号 + 1
#     """
#     if not resume:
#         return 0
#
#     if not os.path.exists(model_dir):
#         return 0
#
#     pths = [int(pth.split('.')[0]) for pth in os.listdir(model_dir)
#             if pth.endswith('.pth') and pth.split('.')[0].isdigit()]
#     if len(pths) == 0:
#         return 0
#
#     if epoch == -1:
#         load_epoch = max(pths)
#     else:
#         load_epoch = epoch
#
#     load_path = os.path.join(model_dir, '{}.pth'.format(load_epoch))
#     load_path = os.path.join(model_dir, '{}.pth'.format(pths))
#
#     if not os.path.exists(load_path):
#         print(f"警告: 找不到检查点文件 {load_path}")
#         return 0
#
#     print('加载模型权重: {}'.format(load_path))
#
#     try:
#         pretrained_model = torch.load(load_path)
#
#         # 仅加载 'net' 部分
#         net.load_state_dict(pretrained_model['net'], strict=strict)
#
#         # [修复] 检查 'epoch' 键的类型
#         loaded_epoch = pretrained_model['epoch']
#         if isinstance(loaded_epoch, int):
#             # 如果是数字 (例如 49)，返回下一个 epoch (50)
#             return loaded_epoch + 1
#         else:
#             # 如果是字符串 (例如 "best")，返回 0 或 -1 表示加载成功
#             return 0
#
#     except Exception as e:
#         print(f"加载模型权重失败: {e}。")
#         return 0

# --- state_dict 键名操作 ---

def remove_net_prefix(net_state_dict: OrderedDict, prefix: str) -> OrderedDict:
    """
    移除 state_dict 键名中的前缀 (例如 'module.')。
    这在加载 DataParallel 训练的模型时很有用。
    """
    net_ = OrderedDict()
    for k in net_state_dict.keys():
        if k.startswith(prefix):
            net_[k[len(prefix):]] = net_state_dict[k]
        else:
            net_[k] = net_state_dict[k]
    return net_


def add_net_prefix(net_state_dict: OrderedDict, prefix: str) -> OrderedDict:
    """
    为 state_dict 键名添加前缀。
    """
    net_ = OrderedDict()
    for k in net_state_dict.keys():
        net_[prefix + k] = net_state_dict[k]
    return net_


def replace_net_prefix(net_state_dict: OrderedDict, orig_prefix: str, prefix: str) -> OrderedDict:
    """
    替换 state_dict 键名中的前缀。
    """
    net_ = OrderedDict()
    for k in net_state_dict.keys():
        if k.startswith(orig_prefix):
            net_[prefix + k[len(orig_prefix):]] = net_state_dict[k]
        else:
            net_[k] = net_state_dict[k]
    return net_


def remove_net_layer(net_state_dict: OrderedDict, layers_to_remove: list) -> OrderedDict:
    """
    从 state_dict 中删除指定的层 (按层名的前缀匹配)。
    """
    keys = list(net_state_dict.keys())
    for k in keys:
        for layer in layers_to_remove:
            if k.startswith(layer):
                del net_state_dict[k]
    return net_state_dict