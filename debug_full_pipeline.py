import torch
import numpy as np

# --------- 自动导入你的项目模块 ---------
try:
    import src.utils.ransac_voting as ransac_voting_mod
    import src.utils.geometry as geometry_mod
    print("[DEBUG] 成功导入 ransac_voting_mod 与 geometry_mod")
except Exception as e:
    print("[DEBUG ERROR] 无法导入 ransac_voting / geometry:", e)
    raise


def print_matrix(name, M):
    print(f"{name}:\n{np.array2string(M, precision=4, floatmode='fixed')}\n")


def run_full_debug(model, dataset, cfg):
    """
    model     : 训练好的 PVNet
    dataset   : 验证集 dataset（不是 dataloader）
    cfg       : 配置信息
    """
    print("====================================================")
    print("�� FULL PIPELINE DEBUG START")
    print("====================================================\n")

    # ---------- 1. 加载一个样本 ----------
    sample = dataset.__getitem__(0)

    print("============== 1. GT 数据（transform 后）==============")
    print_matrix("GT K", sample['K'].numpy())
    print_matrix("GT kp3d", sample['kp3d'].numpy())
    print_matrix("GT kp2d", sample['kp2d'].numpy())
    print("GT img size:", sample['inp'].shape)

    # ---------- 2. 前向推理 ----------
    print("\n============== 2. 模型前向输出 ==============")
    model.eval()
    inp_gpu = sample['inp'].unsqueeze(0).cuda()
    with torch.no_grad():
        out = model(inp_gpu)

    print("模型输出 key:", out.keys())
    if 'vertex' in out:
        print("vertex shape:", list(out['vertex'].shape))
    if 'seg' in out:
        print("seg shape:", list(out['seg'].shape))

    # ---------- 3. 检查 vertex 缩放 ----------
    print("\n============== 3. vertex scale 检查 ==============")
    if 'vertex' in out:
        vertex_pred = out['vertex'][0].cpu().numpy()
        print("vertex_pred raw min/max:", vertex_pred.min(), vertex_pred.max())

        scale = getattr(cfg.model, "vertex_scale", 1.0)
        print("cfg.model.vertex_scale =", scale)
        print("vertex_pred * scale min/max:",
              (vertex_pred * scale).min(),
              (vertex_pred * scale).max())

    # ---------- 4. RANSAC decode ----------
    print("\n============== 4. RANSAC 解算 kp2d ==============")

    if 'vertex' in out and 'seg' in out:
        vt = out['vertex'] * cfg.model.vertex_scale
        seg = out['seg']

        # derive mask
        if seg.shape[1] > 1:
            mask_pred = (torch.softmax(seg, dim=1)[:, 1] > 0.5).float()
        else:
            mask_pred = (torch.sigmoid(seg[:, 0]) > 0.5).float()

        kp2d_pred, _ = ransac_voting_mod.ransac_voting(
            mask=mask_pred,
            vertex=vt,
            num_votes=cfg.model.ransac_voting.vote_num,
            inlier_thresh=cfg.model.ransac_voting.inlier_thresh,
            max_trials=cfg.model.ransac_voting.max_trials
        )

        kp2d_pred = kp2d_pred[0].cpu().numpy()

        print_matrix("RANSAC kp2d_pred", kp2d_pred)
        print("range:", kp2d_pred.min(), kp2d_pred.max())
    else:
        print("❌ 模型未输出 vertex+seg")


    # ---------- 5. PnP 输入检查 ----------
    print("\n============== 5. PnP 输入检查 ==============")

    kp3d_np = sample['kp3d'].numpy()
    K_np = sample['K'].numpy()

    print_matrix("kp3d", kp3d_np)
    print_matrix("kp2d_pred", kp2d_pred)
    print_matrix("K", K_np)

    # ---------- 6. PnP 求解 ----------
    print("\n============== 6. PnP 求解 ==============")

    R_pred, t_pred = geometry_mod.solve_pnp(
        kp3d_np, kp2d_pred, K_np,
        ransac=True,
        reproj_thresh=cfg.pnp.reproj_error_thresh
    )[:2]

    print_matrix("R_pred", R_pred)
    print("t_pred =", t_pred)

    print("\n====================================================")
    print("�� FULL PIPELINE DEBUG END")
    print("====================================================\n")
