import torch
import torch.nn as nn
import torch.nn.functional as F
class LucasKenade(nn.Module):
    def __init__(self, mlp, feature_dim=1024):
        super(LucasKenade, self).__init__()
        self.feature_extractor = mlp
        self.fc = nn.Sequential(
            nn.Linear(feature_dim * 2, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 6)  # 输出 [轴角旋转, 平移]
        )
    def _axis_angle_to_matrix(self, axis_angle):
        """将轴角转换为旋转矩阵 (Rodrigues公式近似)"""
        theta = torch.norm(axis_angle, dim=1, keepdim=True)
        axis = axis_angle / (theta + 1e-8)
        zeros = torch.zeros_like(theta)
        K = torch.stack([
            zeros, -axis[:, 2], axis[:, 1],
            axis[:, 2], zeros, -axis[:, 0],
            -axis[:, 1], axis[:, 0], zeros
        ], dim=1).view(-1, 3, 3)
        
        I = torch.eye(3, device=axis_angle.device).unsqueeze(0)
        R = I + torch.sin(theta.unsqueeze(2)) * K + (1 - torch.cos(theta.unsqueeze(2))) * torch.bmm(K, K)
        return R
    def forward(self, source, template):
            """
            Args:
                source: 源点云 (B, N, 3)
                template: 目标点云 (B, N, 3)
            Returns:
                transformed_source: 对齐后的源点云 (B, N, 3)
                R: 旋转矩阵 (B, 3, 3)
                t: 平移向量 (B, 3)
            """
            # 提取全局特征
            feat_source = self.feature_extractor(source)
            feat_template = self.feature_extractor(template)
            
            # 拼接特征并预测变换参数
            feat = torch.cat([feat_source, feat_template], dim=1)
            delta = self.fc(feat)  # (B, 6)
            
            # 将轴角转换为旋转矩阵
            R = self._axis_angle_to_matrix(delta[:, :3])
            t = delta[:, 3:]  # (B, 3)
            
            # 应用变换
            transformed_source = torch.bmm(source, R.transpose(1, 2)) + t.unsqueeze(1)
            return  R, t,transformed_source,