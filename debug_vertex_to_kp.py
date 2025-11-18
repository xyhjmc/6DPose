import numpy as np
import glob
import os

def inspect_vertex_kp2d(dir_path, name, max_files=50):
    files = sorted(glob.glob(os.path.join(dir_path, "*.npz")))[:max_files]
    if not files:
        print(f"[{name}] 目录里没有 npz 文件: {dir_path}")
        return

    all_errs = []  # 所有关键点的误差
    per_img_errs = []  # 每张图平均误差

    print(f"\n===== 实验 G：{name} ({dir_path})，前 {len(files)} 个样本 =====")

    for f in files:
        d = np.load(f)
        kp2d = d["kp2d"].astype(np.float32)      # (K, 2)
        vertex = d["vertex"].astype(np.float32)  # (2K, H, W)
        mask = (d["mask"] > 0)                   # (H, W)

        ys, xs = np.where(mask)
        if len(xs) == 0:
            print(f"[{name}] {os.path.basename(f)}: 前景像素数 = 0，跳过")
            continue

        K = kp2d.shape[0]
        err_img = []

        for i in range(K):
            vx = vertex[2*i,   ys, xs]  # 这一关键点在前景像素上的 vx
            vy = vertex[2*i+1, ys, xs]

            # 按“偏移”假设反推关键点位置:
            kp_est_x = np.mean(xs + vx)
            kp_est_y = np.mean(ys + vy)

            kx, ky = kp2d[i]
            err = float(np.linalg.norm([kp_est_x - kx, kp_est_y - ky]))
            all_errs.append(err)
            err_img.append(err)

        if err_img:
            per_img_errs.append(float(np.mean(err_img)))

    if not all_errs:
        print(f"[{name}] 没有有效样本（可能全是前景=0），无法统计误差。")
        return

    all_errs = np.array(all_errs)
    per_img_errs = np.array(per_img_errs)

    print(f"[{name}] 所有关键点像素误差：min={all_errs.min():.2f}, "
          f"max={all_errs.max():.2f}, mean={all_errs.mean():.2f}")
    print(f"[{name}] 按图像平均误差：min={per_img_errs.min():.2f}, "
          f"max={per_img_errs.max():.2f}, mean={per_img_errs.mean():.2f}")


if __name__ == "__main__":
    inspect_vertex_kp2d(
        "/home/xyh/PycharmProjects/6DPose/data/linemod_pvnet/driller_test",
        "driller_test"
    )
    inspect_vertex_kp2d(
        "/home/xyh/PycharmProjects/6DPose/data/linemod_pvnet/driller_mini",
        "driller_mini"
    )
