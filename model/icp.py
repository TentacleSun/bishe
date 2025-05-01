import torch
import torch.nn as nn

class ICPRegistration(nn.Module):
    def __init__(self, num_iterations=10):
        super(ICPRegistration, self).__init__()
        self.num_iterations = num_iterations

    def nearest_neighbor(self, src, tgt):
        """
        寻找源点云中每个点在目标点云中的最近邻点。
        """
        inner = -2 * torch.bmm(src, tgt.transpose(1, 2))  # BxNxM
        xx = torch.sum(src ** 2, dim=2, keepdim=True)    # BxNx1
        yy = torch.sum(tgt ** 2, dim=2, keepdim=True).transpose(1, 2)  # Bx1xM
        distances = xx + inner + yy  # BxNxM
        _, indices = distances.min(dim=2)  # BxN
        return indices

    def compute_transformation(self, src, tgt):
        """
        计算刚体变换（旋转和平移）。
        """
        batch_size, N, _ = src.shape

        # 计算质心
        centroid_src = torch.mean(src, dim=1, keepdim=True)  # Bx1x3
        centroid_tgt = torch.mean(tgt, dim=1, keepdim=True)  # Bx1x3

        # 去中心化
        src_centered = src - centroid_src
        tgt_centered = tgt - centroid_tgt

        # 协方差矩阵
        H = torch.bmm(src_centered.transpose(1, 2), tgt_centered) / N  # Bx3x3

        # SVD 分解
        U, S, V = torch.svd(H)

        # 构造旋转矩阵
        R = torch.bmm(V, U.transpose(1, 2))

        # 处理反射情况（行列式为负）
        det = torch.det(R)
        mask = (det < 0).float()
        V_scaled = V.clone()
        V_scaled[:, :, 2] *= (1 - 2 * mask).unsqueeze(-1)
        R = torch.bmm(V_scaled, U.transpose(1, 2))

        # 计算平移向量
        t = centroid_tgt - torch.bmm(R, centroid_src.transpose(1, 2)).transpose(1, 2)
        t = t.squeeze(1)  # 确保形状为 (B, 3)

        return R, t

    def forward(self, src, tgt):
        """
        执行 ICP 配准。
        """
        transformed_src = src.clone()
        rotation_matrices_final = None
        translations_final = None

        for _ in range(self.num_iterations):
            indices = self.nearest_neighbor(transformed_src, tgt)
            matched_tgt = tgt.gather(1, indices.unsqueeze(-1).expand(-1, -1, 3))

            R, t = self.compute_transformation(transformed_src, matched_tgt)
            transformed_src = torch.bmm(transformed_src, R) + t.unsqueeze(1)

            rotation_matrices_final = R
            translations_final = t

        # 确保输出形状为 (B, 3)
        assert translations_final.dim() == 2 and translations_final.shape[1] == 3, \
            "Translation must be of shape (B, 3)"

        return {'est_R': rotation_matrices_final,				# source -> template
				  'est_t': translations_final,				# source -> template
				#   'est_T': Rigidtransform.convert2transformation(est_R, est_t),			# source -> template
				#   'r': template_features - self.source_features,
				  'transformed_source': transformed_src}