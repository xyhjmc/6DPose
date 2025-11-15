# src/engines/trainer.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
import time
from typing import Dict, Any


# [修复] 导入新的 AMP 模块
import torch.amp
# [修改] 导入我们新的工具函数
from src.utils.train_model_utils import save_model, load_model
from src.utils.torch_utils import move_batch_to_device, log_scalars_to_writer
from src.losses.pvnet.pvnet_loss import PVNetLoss  # 类型提示

# 定义数据批次 (Batch) 的类型 (字典)
BatchType = Dict[str, torch.Tensor]


class Trainer:
    """
    通用的训练引擎 (Trainer)。
    """

    def __init__(self,
                 model: nn.Module,
                 loss_fn: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler._LRScheduler,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 device: torch.device,
                 log_dir: str,
                 checkpoint_dir: str,
                 max_epochs: int,
                 log_interval: int = 10,
                 use_amp: bool = True,
                 resume: bool = True):

        # ( ... __init__ 的其他部分保持不变 ... )
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.log_dir = log_dir
        self.checkpoint_dir = checkpoint_dir
        self.max_epochs = max_epochs
        self.log_interval = log_interval

        self.use_amp = use_amp and torch.cuda.is_available()
        # [修复] 使用 torch.amp.GradScaler 并指定 device_type
        self.scaler = torch.amp.GradScaler( enabled=self.use_amp)

        self.writer = SummaryWriter(log_dir=self.log_dir)
        self.start_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')

        self.recorder = {'epoch': 0, 'best_val_loss': self.best_val_loss}
        if resume:
            self.load_checkpoint()

    # [已删除] _to_device(self, batch) 方法
    # (我们现在使用 src.utils.torch_utils 中的 move_batch_to_device)

    # [已删除] _log_scalars(self, loss_dict, ...) 方法
    # (我们现在使用 src.utils.torch_utils 中的 log_scalars_to_writer)

    def train_epoch(self, epoch: int):
        """
        执行一个完整的训练 epoch。
        """
        self.model.train()
        progress_bar = tqdm(self.train_loader,
                            desc=f"Epoch {epoch}/{self.max_epochs} [Train]")
        epoch_losses = {}

        for batch in progress_bar:
            # 1. [修改] 使用外部工具函数
            batch = move_batch_to_device(batch, self.device)

            # [修复] 使用 torch.amp.autocast 并指定 device_type
            with torch.amp.autocast(device_type='cuda', enabled=self.use_amp):
                # 3. 前向传播
                output = self.model(batch['inp'])

                # 4. 计算损失
                total_loss, loss_dict = self.loss_fn(output, batch)

            # 5. 反向传播
            self.optimizer.zero_grad()

            # 6. AMP 梯度缩放
            self.scaler.scale(total_loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # 7. 日志记录
            for k, v in loss_dict.items():
                epoch_losses[k] = epoch_losses.get(k, 0.0) + v.item()

            # [改进] 显示所有损失
            postfix_dict = {
                'total': f"{total_loss.item():.4f}",
                'seg': f"{loss_dict['seg_loss'].item():.4f}",
                'vote': f"{loss_dict['vote_loss'].item():.4f}"
            }
            progress_bar.set_postfix(**postfix_dict)

            if self.global_step % self.log_interval == 0:
                # [修改] 使用外部工具函数
                log_scalars_to_writer(self.writer, loss_dict, "train_step", self.global_step)

            self.global_step += 1

        # --- Epoch 结束 ---
        avg_losses = {k: v / len(self.train_loader) for k, v in epoch_losses.items()}
        # [修改] 使用外部工具函数
        log_scalars_to_writer(self.writer, avg_losses, "train_epoch", epoch)
        return avg_losses

    def val_epoch(self, epoch: int):
        """
        执行一个完整的验证 epoch。
        """
        self.model.eval()
        progress_bar = tqdm(self.val_loader,
                            desc=f"Epoch {epoch}/{self.max_epochs} [Val]")
        epoch_losses = {}

        with torch.no_grad():
            for batch in progress_bar:
                # 1. [修改] 使用外部工具函数
                batch = move_batch_to_device(batch, self.device)

                # [修复] 使用 torch.amp.autocast 并指定 device_type
                with torch.amp.autocast(device_type='cuda', enabled=self.use_amp):
                    # 3. 前向传播
                    output = self.model(batch['inp'])
                    # 4. 计算损失
                    total_loss, loss_dict = self.loss_fn(output, batch)

                # 5. 累积损失
                for k, v in loss_dict.items():
                    epoch_losses[k] = epoch_losses.get(k, 0.0) + v.item()

                progress_bar.set_postfix(val_loss=f"{total_loss.item():.4f}")

        # --- Epoch 结束 ---
        avg_losses = {k: v / len(self.val_loader) for k, v in epoch_losses.items()}
        # [修改] 使用外部工具函数
        log_scalars_to_writer(self.writer, avg_losses, "val_epoch", epoch)
        return avg_losses

    def save_checkpoint(self, epoch: int, is_best: bool):
        """
        保存检查点 (此方法保持不变，因为它管理内部状态)。
        """
        self.recorder['epoch'] = epoch
        self.recorder['best_val_loss'] = self.best_val_loss

        # 保存 "latest.pth"
        save_model(self.model, self.optimizer, self.scheduler,
                   self.recorder, epoch, self.checkpoint_dir)

        if is_best:
            # 另存为 "best.pth"
            save_model(self.model, self.optimizer, self.scheduler,
                       self.recorder, "best", self.checkpoint_dir)

    def load_checkpoint(self):
        """
        加载检查点 (此方法保持不变，因为它管理内部状态)。
        """
        print(f"尝试从 {self.checkpoint_dir} 恢复训练...")
        next_epoch = load_model(self.model, self.optimizer, self.scheduler,
                                self.recorder, self.checkpoint_dir,
                                resume=True, epoch=-1)

        self.start_epoch = next_epoch
        self.global_step = self.start_epoch * len(self.train_loader)
        self.best_val_loss = self.recorder.get('best_val_loss', float('inf'))

        if next_epoch > 0:
            print(f"成功恢复训练，将从 Epoch {self.start_epoch} 开始。")
        else:
            print("未找到检查点，将从 Epoch 0 开始。")

    def run(self):
        """
        启动并运行整个训练流程 (此方法保持不变)。
        """
        print(f"开始训练，总共 {self.max_epochs} 个 Epoch。")
        print(f"设备: {self.device}")
        print(f"使用自动混合精度 (AMP): {self.use_amp}")

        for epoch in range(self.start_epoch, self.max_epochs):
            start_time = time.time()
            train_loss_dict = self.train_epoch(epoch)
            train_time = time.time() - start_time

            start_time = time.time()
            val_loss_dict = self.val_epoch(epoch)
            val_time = time.time() - start_time

            self.scheduler.step()

            lr = self.optimizer.param_groups[0]['lr']
            self.writer.add_scalar("train_epoch/learning_rate", lr, epoch)
            print(f"[Epoch {epoch}] Train Loss: {train_loss_dict['total_loss']:.4f} (T: {train_time:.2f}s) | "
                  f"Val Loss: {val_loss_dict['total_loss']:.4f} (T: {val_time:.2f}s) | LR: {lr:.1e}")

            is_best = val_loss_dict['total_loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss_dict['total_loss']
                print(f"  ✨ 新的最佳验证损失: {self.best_val_loss:.4f}。保存 'best.pth'。")

            self.save_checkpoint(epoch, is_best)

        self.writer.close()
        print("训练完成。")