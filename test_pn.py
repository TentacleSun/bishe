import torch
import torch.nn as nn

# 定义一个批量正则化层
batch_norm = nn.BatchNorm1d(5)  # 对 5 维特征进行正则化

# 输入一个 3x5 的批次（3 个样本，每个样本有 5 个特征）
input_data = torch.tensor([[10.0, 20.0, 30.0, 40.0, 50.0],
                           [15.0, 25.0, 35.0, 45.0, 55.0],
                           [12.0, 22.0, 32.0, 42.0, 52.0]])

# 模拟输入为浮点数并打印原始数据
input_data = input_data.float()
print("原始输入数据：")
print(input_data)

# 经过批量正则化
output_data = batch_norm(input_data)
print("\n批量正则化后的数据：")
print(output_data)
