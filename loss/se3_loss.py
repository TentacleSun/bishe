import torch
import torch.nn as nn
import torch.nn.functional as F
def normalize_rotation_matrix(R):
    """
    将任意 3x3 矩阵正交化为合法的旋转矩阵（SO(3)）
    输入: R - [B, 3, 3]
    输出: R_normalized - [B, 3, 3]
    """
    # 使用奇异值分解（SVD）
    U, _, V = torch.svd(R)

    # 构造正交矩阵：U * V^T
    R_normalized = torch.bmm(U, V.transpose(1, 2))

    # 确保行列式为 +1（避免镜像变换）
    det = torch.det(R_normalized).unsqueeze(-1).unsqueeze(-1)
    sign = torch.sign(det)
    U = U * sign
    R_normalized = torch.bmm(U, V.transpose(1, 2))

    return R_normalized
class SE3Loss(nn.Module):
    def __init__(self):
        super(SE3Loss, self).__init__()
    def forward(self, pred_R, pred_t, gt_transform:torch.Tensor):
        # 提取真值四元数和位移 [[1]]
        gt_transform = gt_transform.squeeze()
        gt_quat = gt_transform[:, :4]

        igt_t = gt_transform[:, 4:]
  
		# 创建一个空的张量用于存储所有旋转矩阵，形状为 (20, 3, 3)
        igt_R = torch.zeros(gt_transform.size(0), 3, 3).cuda()
        for j, rotate_tensor in enumerate(gt_quat):
            w, x, y, z = rotate_tensor[..., 0], rotate_tensor[..., 1], rotate_tensor[..., 2], rotate_tensor[..., 3]
            single_igt_R = torch.stack([
                1 - 2 * (y**2 + z**2),     2 * (x*y - w*z),         2 * (x*z + w*y),
                2 * (x*y + w*z),          1 - 2 * (x**2 + z**2),   2 * (y*z - w*x),
                2 * (x*z - w*y),          2 * (y*z + w*x),         1 - 2 * (x**2 + y**2)
            ], dim=-1).view(3, 3)
            # 将旋转矩阵存入 igt_R
            igt_R[j] = single_igt_R
        igt_R = normalize_rotation_matrix(igt_R).cuda()
        pred_R = normalize_rotation_matrix(pred_R).cuda()
        total_loss = F.mse_loss(torch.matmul(pred_R.transpose(2, 1), igt_R), torch.eye(3).cuda().repeat(gt_transform.size(0),1,1)) + F.mse_loss(pred_t, igt_t.unsqueeze(1))


        return total_loss
    

    