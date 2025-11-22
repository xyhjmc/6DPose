import torch
import numpy as np
import cv2
import argparse
import matplotlib.pyplot as plt
from configs.config import load_config
from src.models.pvnet.PVNet import PVNet
from datasets.bop_pvnet_dataset import BopPvnetDataset, pvnet_collate_fn
from datasets.transforms import Compose, Resize, NormalizeAndToTensor
from src.utils.ransac_voting import ransac_voting


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=False,default="configs/pvnet_linemod_driller_less.yaml")
    parser.add_argument("--checkpoint", required=False,default="checkpoints/pvnet_linemod_driller_less/best.pth")
    args = parser.parse_args()

    # 1. 加载配置和模型
    cfg = load_config(args.config)
    if not cfg.transforms.use_offset and getattr(cfg.model, "vertex_scale", 1.0) != 1.0:
        print("[配置提示] use_offset=False，debug_viz 将 vertex_scale 重置为 1.0。")
        cfg.model.vertex_scale = 1.0
    cfg.model.use_offset = cfg.transforms.use_offset
    device = torch.device(cfg.device)

    # 验证集遵循配置的 offset 模式
    val_transforms = Compose([
        Resize(output_size_hw=cfg.transforms.input_size_hw, use_offset=cfg.transforms.use_offset),
        NormalizeAndToTensor(
            mean=np.array(cfg.transforms.mean),
            std=np.array(cfg.transforms.std),
            vertex_scale=getattr(cfg.model, "vertex_scale", 1.0),
            use_offset=cfg.transforms.use_offset,
        )
    ])

    val_dataset = BopPvnetDataset(data_dir=cfg.dataset.val_data_dir, transforms=val_transforms, split_name="val")
    loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=True, collate_fn=pvnet_collate_fn)

    model = PVNet(
        ver_dim=cfg.model.ver_dim, seg_dim=cfg.model.seg_dim,
        vote_num=cfg.model.ransac_voting.vote_num, inlier_thresh=10.0, max_trials=200,
        vertex_scale=getattr(cfg.model, "vertex_scale", 1.0),
        use_offset=getattr(cfg.model, "use_offset", True),
    ).to(device)

    print(f"加载权重: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['net'])
    model.eval()

    print("开始可视化... (请关闭弹窗以查看下一张)")

    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= 5: break

            img = batch['inp'].to(device)
            # 预测
            out = model(img)

            # RANSAC 投票 (使用宽阈值测试)
            mask_pred = (torch.argmax(out['seg'], dim=1) > 0).float()
            kpt_pred, _ = ransac_voting(
                mask_pred,
                out['vertex'],
                num_votes=512,
                inlier_thresh=10.0,
                max_trials=500,
                use_offset=getattr(cfg.model, "use_offset", True),
            )
            kpt_pred = kpt_pred[0].cpu().numpy()

            # 准备可视化
            img_np = batch['inp'][0].permute(1, 2, 0).cpu().numpy()
            # 反归一化
            mean = np.array(cfg.transforms.mean);
            std = np.array(cfg.transforms.std)
            img_np = (img_np * std + mean) * 255
            img_np = np.clip(img_np, 0, 255).astype(np.uint8).copy()
            img_gt = img_np.copy()

            # 画真值 (绿)
            kp_gt = batch['kp2d'][0].cpu().numpy()
            for k in kp_gt:
                cv2.circle(img_gt, (int(k[0]), int(k[1])), 5, (0, 255, 0), -1)

            # 画预测 (红)
            for k in kpt_pred:
                cv2.circle(img_np, (int(k[0]), int(k[1])), 5, (255, 0, 0), -1)

            # 画顶点场 (取第0个关键点的 x 分量)
            vertex_pred = out['vertex'][0, 0, :, :].cpu().numpy()

            plt.figure(figsize=(12, 4))
            plt.subplot(1, 3, 1);
            plt.imshow(img_gt);
            plt.title("GT Keypoints (Green)")
            plt.subplot(1, 3, 2);
            plt.imshow(img_np);
            plt.title("Pred Keypoints (Red)")
            plt.subplot(1, 3, 3);
            plt.imshow(vertex_pred, cmap='jet');
            plt.title("Pred Vertex (dx)")
            plt.show()


if __name__ == "__main__":
    main()