import os
import json
import numpy as np
from tqdm import tqdm

root = "/home/xyh/PycharmProjects/6DPose/data/linemod_pvnet/driller_test"
index_path = os.path.join(root, "index.json")

with open(index_path, "r") as f:
    index = json.load(f)

print("原始样本数:", len(index))

clean_index = []
bad_index = []

for rec in tqdm(index, desc="检查前景像素"):
    # 根据你 index.json 的字段名改这里，比如 "npz_file" / "npz_path"
    npz_rel = rec["file"]   # 或 rec["npz_path"]
    npz_path = os.path.join(root, npz_rel)

    if not os.path.exists(npz_path):
        print("[缺失文件]", npz_path)
        continue

    data = np.load(npz_path)
    # 根据你 npz 里的键改这里，可能是 "mask" 或 "mask_visib"
    mask = data["mask"]

    if mask.sum() == 0:
        bad_index.append(rec)
    else:
        clean_index.append(rec)

print("清理结果：")
print("  原始总数:", len(index))
print("  保留样本:", len(clean_index))
print("  空前景样本:", len(bad_index))

# 先备份原来的 index.json
os.rename(index_path, os.path.join(root, "index_raw.json"))

# 再写入新的干净 index.json
with open(index_path, "w") as f:
    json.dump(clean_index, f, indent=2)

print("已写入清理后的 index.json，并备份为 index_raw.json")
