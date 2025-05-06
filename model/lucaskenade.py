import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================================
# 1. PointNet 特征提取器
# =============================================
class PointNetFeature(nn.Module):
    def __init__(self, feature_dim=1024):
        super(PointNetFeature, self).__init__()
        self.mlp1 = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, feature_dim, 1),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU()
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )

    def forward(self, x):
        # x: (B, N, 3)
        x = x.transpose(1, 2)  # (B, 3, N)
        x = self.mlp1(x)       # (B, feature_dim, N)
        x = F.max_pool1d(x, x.size(2))  # (B, feature_dim, 1)
        x = x.squeeze(2)       # (B, feature_dim)
        x = self.mlp2(x)       # (B, feature_dim)
        return x

# =============================================
# 2. 点云变换函数
# =============================================
def apply_transform(points, T):
    """
    Apply transformation matrix T to point cloud points.
    points: (B, N, 3)
    T: (B, 4, 4)
    return: transformed_points: (B, N, 3)
    """
    B, N, _ = points.shape
    points_homogeneous = torch.cat([points, torch.ones(B, N, 1).to(points)], dim=-1)  # (B, N, 4)
    transformed_points_homogeneous = torch.bmm(points_homogeneous, T.transpose(1, 2))  # (B, N, 4)
    transformed_points = transformed_points_homogeneous[:, :, :3]  # (B, N, 3)
    return transformed_points

# =============================================
# 3. PointNetLK 接口类
# =============================================
class PointNetLKInterface(nn.Module):
    def __init__(self, feature_dim=1024):
        super(PointNetLKInterface, self).__init__()
        self.feature_extractor = PointNetFeature(feature_dim)
        self.feature_dim = feature_dim

    def forward(self, source, target, T):
        """
        Compute transformed source and error based on current T.
        source: (B, N, 3)
        target: (B, N, 3)
        T: (B, 4, 4)  # 用户传入的当前变换矩阵
        return:
            transformed_source: (B, N, 3)
            error: (B, feature_dim)
        """
        # 应用变换
        transformed_source = apply_transform(source, T)

        # 提取特征
        source_features = self.feature_extractor(transformed_source)
        target_features = self.feature_extractor(target)

        # 计算特征误差
        error = source_features - target_features  # (B, feature_dim)

        return transformed_source, error

    def get_initial_T(self, B, device):
        """
        返回单位变换矩阵作为初始猜测。
        B: batch size
        device: 'cuda' or 'cpu'
        return: T (B, 4, 4)
        """
        return torch.eye(4, device=device).unsqueeze(0).repeat(B, 1, 1)

# =============================================
# 4. 示例用法（外部优化逻辑）
# =============================================
if __name__ == "__main__":
    # 模拟输入
    B, N = 16, 1024
    source = torch.rand(B, N, 3)
    target = source + torch.randn(B, N, 3) * 0.1

    # 初始化模型
    model = PointNetLKInterface()
    T = model.get_initial_T(B, device=source.device)

    # 外部优化逻辑（示例：简单梯度下降）
    T.requires_grad = True
    optimizer = torch.optim.Adam([T], lr=0.01)

    for i in range(10):  # 假设优化10步
        transformed_source, error = model(source, target, T)
        loss = error.pow(2).sum()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Iteration {i}, Loss: {loss.item()}")