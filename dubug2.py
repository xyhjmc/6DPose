import numpy as np
import glob

def inspect_vertex_stats(dir_path, name, max_files=20):
    files = sorted(glob.glob(f"{dir_path}/*.npz"))[:max_files]
    norms = []
    for f in files:
        d = np.load(f)
        v = d["vertex"]          # (2K, H, W)
        mask = d["mask"] > 0     # (H, W)

        K2 = v.shape[0] // 2
        # 只看第一个关键点的 (vx, vy) 分量
        vx = v[0]
        vy = v[1]
        fg = mask

        # 只统计前景像素
        n = np.sqrt(vx[fg]**2 + vy[fg]**2)
        norms.append(n.mean())

    print(f"[{name}] 平均范数 (前景均值的平均):", np.mean(norms))

inspect_vertex_stats(
    "/home/xyh/PycharmProjects/6DPose/data/linemod_pvnet/driller_test",
    "driller_test"
)
inspect_vertex_stats(
    "/home/xyh/PycharmProjects/6DPose/data/linemod_pvnet/driller_mini",
    "driller_mini"
)
