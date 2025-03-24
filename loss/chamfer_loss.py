import torch
import torch.nn as nn

def chamfer_loss(point_cloud1, point_cloud2):
    """
    Args:
        point_cloud1: (B, N, 3) 或 (N, 3)
        point_cloud2: (B, N, 3) 或 (N, 3)
    Returns:
        loss: 标量
    """
    # 计算前向距离（A -> B）
    dist_a_to_b = torch.cdist(point_cloud1, point_cloud2, p=2.0)  # 欧氏距离矩阵
    min_dist_a_to_b, _ = torch.min(dist_a_to_b, dim=-1)  # 取每个点的最小距离 (B, N)
    loss_a_to_b = torch.mean(min_dist_a_to_b)  # 对所有样本和点取均值 → 标量

    # 计算反向距离（B -> A）
    dist_b_to_a = torch.cdist(point_cloud2, point_cloud1, p=2.0)
    min_dist_b_to_a, _ = torch.min(dist_b_to_a, dim=-1)
    loss_b_to_a = torch.mean(min_dist_b_to_a)  # 对所有样本和点取均值 → 标量

    # 总损失为双向损失之和（标量）
    return loss_a_to_b + loss_b_to_a

class ChamferLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, template, source):
        return chamfer_loss(template, source)