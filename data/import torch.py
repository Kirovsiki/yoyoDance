import torch

# 检查是否有可用的 GPU
if torch.cuda.is_available():
    device = torch.device("cuda")          # 如果有 GPU，选择 GPU
    print("GPU 可用")
else:
    device = torch.device("cpu")           # 如果没有 GPU，选择 CPU
    print("GPU 不可用")

# 创建一个随机张量并将其移动到所选设备上
x = torch.rand(5, 3).to(device)

# 打印张量
print("Tensor:", x)


import os
import glob

# 删除缓存的pkl文件
backup_path = "C:\\Users\\Administrator\\Desktop\\yoyoDance\\checkpoint"
cached_files = glob.glob(os.path.join(backup_path, "*.pkl"))
for file in cached_files:
    os.remove(file)
print("缓存文件已删除")


import os

data_path = r"C:\Users\Administrator\Desktop\yoyoDance\data\train"
motions_path = os.path.join(data_path, "motions_sliced")
features_path = os.path.join(data_path, "jukebox_feats")
wavs_path = os.path.join(data_path, "wavs_sliced")

print(f"motions_path: {motions_path}")
print(f"features_path: {features_path}")
print(f"wavs_path: {wavs_path}")

print(f"Motion files: {len(os.listdir(motions_path))}")
print(f"Feature files: {len(os.listdir(features_path))}")
print(f"Wav files: {len(os.listdir(wavs_path))}")
import pickle
import numpy as np

def check_sliced_data(data_path):
    motion_path = os.path.join(data_path, "motions_sliced")
    motions = sorted(glob.glob(os.path.join(motion_path, "*.pkl")))

    for motion in motions[:5]:  # 仅检查前五个文件
        data = pickle.load(open(motion, "rb"))
        pos = data["pos"]
        q = data["q"]
        print(f"File: {motion}, pos shape: {pos.shape}, q shape: {q.shape}")

# 执行检查
check_sliced_data("C:/Users/Administrator/Desktop/yoyoDance/data/train")
import os

cache_file = os.path.join(backup_path, pickle_name)
if os.path.exists(cache_file):
    os.remove(cache_file)
