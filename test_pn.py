import torch
import torch.nn as nn
from data_utils import *
from model import DCP, PCRNet, PointNet
from types import SimpleNamespace
from loss import ChamferLoss

# 定义一个批量正则化层
# batch_norm = nn.BatchNorm1d(5)  # 对 5 维特征进行正则化

# # 输入一个 3x5 的批次（3 个样本，每个样本有 5 个特征）
# input_data = torch.tensor([[10.0, 20.0, 30.0, 40.0, 50.0],
#                            [15.0, 25.0, 35.0, 45.0, 55.0],
#                            [12.0, 22.0, 32.0, 42.0, 52.0]])

# # 模拟输入为浮点数并打印原始数据
# input_data = input_data.float()
# print("原始输入数据：")
# print(input_data)

# # 经过批量正则化
# output_data = batch_norm(input_data)
# print("\n批量正则化后的数据：")
# print(output_data)

# trainset = RegistrationData(ModelNet40Data(train=True))
# train_loader = DataLoader(trainset,batch_size=100, shuffle=True, drop_last=True, num_workers=4)
# pass

# template, source = torch.rand(10,1024,3), torch.rand(10,1024,3)
# pn = PointNet(input_shape='bnc')

# net = PCRNet(pn)
# #net = ICPRegistration()
# # args = {
# #     'emb_dims':1024,
# #     'emb_nn': 'pointnet',
# #     'head': 'mlp',
# #     'n_blocks':1,
# #     'n_heads':4,
# #     'ff_dims':1024,
# #     'dropout':0.0,
    
# # }
# # net = DCP(args=SimpleNamespace(**args))
# result = net(template, source)
# pass


def quat2mat(quat):
    """
    将四元数转换为旋转矩阵
    quat: [B, 1, 4] -> [B, 3, 3]
    quat 的格式是 [qx, qy, qz, qw]
    """
    quat = quat[:,:4]
    qx, qy, qz, qw = quat.unbind(-1)
    B = quat.shape[0]

    # 构造旋转矩阵
    R_xx = 1 - 2 * (qy**2 + qz**2)
    R_xy = 2 * (qx*qy - qw*qz)
    R_xz = 2 * (qx*qz + qw*qy)

    R_yx = 2 * (qx*qy + qw*qz)
    R_yy = 1 - 2 * (qx**2 + qz**2)
    R_yz = 2 * (qy*qz - qw*qx)

    R_zx = 2 * (qx*qz - qw*qy)
    R_zy = 2 * (qy*qz + qw*qx)
    R_zz = 1 - 2 * (qx**2 + qy**2)

    R = torch.stack([
        R_xx, R_xy, R_xz,
        R_yx, R_yy, R_yz,
        R_zx, R_zy, R_zz
    ], dim=-1).view(B, 3, 3)

    return R
def forward(gt_transform:torch.Tensor):
        # 提取真值四元数和位移 [[1]]
        gt_transform = gt_transform.squeeze()
        gt_quat = gt_transform[:, :4]
        gt_t = gt_transform[:, 4:]

        # 归一化四元数 [[4]]
        gt_quat_normalized = torch.nn.functional.normalize(gt_quat, p=2, dim=1)

        # 四元数转旋转矩阵 [[4]]
        w, x, y, z = torch.unbind(gt_quat_normalized, dim=1)
        R_xx = 1 - 2 * (y**2 + z**2)
        R_xy = 2 * (x*y - w*z)
        R_xz = 2 * (x*z + w*y)
        R_yx = 2 * (x*y + w*z)
        R_yy = 1 - 2 * (x**2 + z**2)
        R_yz = 2 * (y*z - w*x)
        R_zx = 2 * (x*z - w*y)
        R_zy = 2 * (y*z + w*x)
        R_zz = 1 - 2 * (x**2 + y**2)
        gt_R = torch.stack([
            R_xx, R_xy, R_xz,
            R_yx, R_yy, R_yz,
            R_zx, R_zy, R_zz
        ], dim=1).view(-1, 3, 3)
        return gt_R
if __name__ =="__main__":
    q = torch.rand(10,7)
    quat2mat(q)
    forward(q)
# def apply_transform(src, transform):
#     """
#     src: [B, N, 3]       - 源点云
#     transform: [B, 1, 7]  - [qx, qy, qz, qw, tx, ty, tz]
#     返回：[B, N, 3] - 变换后的点云
#     """
#     B, N = src.shape[0], src.shape[1]

#     # 分离四元数和平移向量
#     quat = transform[:, :, :4]            # [B, 1, 4] → [qx, qy, qz, qw]
#     trans = transform[:, :, 4:]           # [B, 1, 3]

#     # 归一化四元数（非常重要）
#     quat = F.normalize(quat, p=2, dim=-1)  # 避免非单位四元数带来的误差

#     # 转换为旋转矩阵
#     R = quat2mat(quat)                    # [B, 3, 3]

#     # 应用旋转和平移
#     src_rotated = torch.matmul(src, R.transpose(1, 2))  # [B, N, 3] × [B, 3, 3]^T → [B, N, 3]
#     src_transformed = src_rotated + trans               # 加上平移

#     return src_transformed
# trainset = RegistrationData(noise_level=0,data_class=ModelNet40Data(train=True))
# train_loader = DataLoader(trainset, batch_size=20,shuffle=True, drop_last=True, num_workers=4)
# for i, data in enumerate(train_loader):
# 	template, source, igt = data
# 	test1 = apply_transform(source,igt)

# 	test2 = apply_transform(template,igt)
# 	loss1 = ChamferLoss()(test1,template)
# 	loss2 = ChamferLoss()(test2,source)
# 	print(loss1, loss2)
#	pass