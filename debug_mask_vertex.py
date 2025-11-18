import numpy as np
import glob
import os

def inspect_dir(dir_path, name, max_files=500):
    files = sorted(glob.glob(os.path.join(dir_path, "*.npz")))[:max_files]
    if not files:
        print(f"[{name}] 目录里没有 npz 文件: {dir_path}")
        return

    fg_counts = []
    norms = []

    print(f"\n===== 检查 {name} ({dir_path})，前 {len(files)} 个样本 =====")

    for f in files:
        d = np.load(f)
        v = d["vertex"].astype(np.float32)   # (2K, H, W)
        mask = (d["mask"] > 0)               # (H, W)

        fg = int(mask.sum())
        fg_counts.append(fg)

        if fg == 0:
            print(f"[{name}] {os.path.basename(f)}: 前景像素数 = 0")
            continue

        # 只看第一个关键点的向量 (vx, vy)
        vx = v[0]
        vy = v[1]
        n = np.sqrt(vx[mask]**2 + vy[mask]**2)
        norms.append(float(n.mean()))

    valid_fg = sum(c > 0 for c in fg_counts)
    print(f"[{name}] 样本总数: {len(files)}, 有前景的样本数: {valid_fg}")

    if norms:
        print(f"[{name}] vertex 范数前景均值的平均 ≈ {np.mean(norms):.4f}")
    else:
        print(f"[{name}] 因为没有任何前景像素，无法计算 vertex 范数均值。")


if __name__ == "__main__":
    inspect_dir(
        "/home/xyh/PycharmProjects/6DPose/data/linemod_pvnet/driller_test",
        "driller_test"
    )
    inspect_dir(
        "/home/xyh/PycharmProjects/6DPose/data/linemod_pvnet/driller_mini",
        "driller_mini"
    )
