import os
import json
import numpy as np

data_dir = "/home/xyh/PycharmProjects/6DPose/data/linemod_pvnet/driller_all"
index_path = os.path.join(data_dir, "index.json")

with open(index_path, "r") as f:
    index_list = json.load(f)

clean_list = []
num_total = len(index_list)
num_empty = 0

for meta in index_list:
    npz_path = os.path.join(data_dir, meta["file"])
    data = np.load(npz_path, allow_pickle=True)
    mask = data["mask"]
    fg = (mask > 0).sum()
    if fg == 0:
        num_empty += 1
    else:
        clean_list.append(meta)

print(f"总样本数: {num_total}")
print(f"前景为 0 的样本数: {num_empty}")
print(f"保留的样本数: {len(clean_list)}")

out_path = os.path.join(data_dir, "index_clean.json")
with open(out_path, "w") as f:
    json.dump(clean_list, f, indent=2)
print(f"已写入清理后的 index: {out_path}")
