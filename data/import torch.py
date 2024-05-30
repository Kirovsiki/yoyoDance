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
