import numpy as np
import glob

files = glob.glob("/home/xyh/PycharmProjects/6DPose/data/linemod_pvnet/driller_less/*.npz")

f = files[0]
data = np.load(f)
print("*** Checking NPZ:", f)
print("vertex shape:", data['vertex'].shape)
print("msak shape:", data['mask'].shape)
print("kp3d:", data['kp3d'].shape)
print("mean vertex:", data['vertex'].mean())
print("max vertex:", data['vertex'].max())
print("min vertex:", data['vertex'].min())
