import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def get_graph_feature(x:torch.Tensor, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(1)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k)
    if torch.cuda.is_available():    
        device = torch.device('cuda')
    elif torch.mps.is_available():
        device = torch.device('mps')
    else: device = torch.device('cpu')
    # idx_base = torch.arange(0, batch_size,device=device).view(-1, 1, 1)*num_points
    # # idx_base = torch.arange(0, batch_size).view(-1, 1, 1) * num_points
    # idx = idx + idx_base

    # idx = idx.view(-1)
 
    # _, num_dims, _ = x.size()

    # x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size, num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    # feature = x.view(batch_size*num_points, -1)[idx, :]
    # feature = feature.view(batch_size, num_points, k, num_dims) 
    # x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    # feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
  
    return idx

def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx
import torch

def combine_point_cloud_and_knn(point_cloud, knn_indices):
    """
    根据 KNN 下标矩阵从点云数据中提取邻居特征。
    
    参数:
        point_cloud (torch.Tensor): 点云数据，形状为 (batch_size, num_points, channels)。
        knn_indices (torch.Tensor): KNN 下标矩阵，形状为 (batch_size, num_points, topk)。
    
    返回:
        torch.Tensor: 邻居特征，形状为 (batch_size, num_points, topk, channels)。
    """
    batch_size, num_points, channels = point_cloud.size()
    _, _, topk = knn_indices.size()

    # Step 1: 展平点云数据
    point_cloud_flat = point_cloud.view(batch_size * num_points, channels)

    # Step 2: 展平 KNN 下标矩阵
    knn_indices_flat = knn_indices.view(batch_size * num_points * topk)

    # Step 3: 提取邻居特征
    neighbor_features_flat = point_cloud_flat[knn_indices_flat]

    # Step 4: 恢复形状
    neighbor_features = neighbor_features_flat.view(batch_size, num_points, topk, channels)

    return neighbor_features


class DGCNN(nn.Module):
    def __init__(self ,input_shape ='bnc' ,emb_dim=1024, k=20, negative_slope = 0.2):
        super(DGCNN, self).__init__()
        if input_shape not in ["bcn", "bnc"]:
            raise ValueError("Allowed shapes are 'bcn' (batch * channels * num_points), 'bnc' ")
        
        self.input_shape = input_shape
        self.k = k
        self.emb_dim = emb_dim
        self.negative_slope = negative_slope

        # self.ch1_output_num = int(self.emb_dim/8)
        # self.ch2_output_num = int(self.emb_dim/8)
        # self.ch3_output_num = int(self.emb_dim/4)
        # self.ch4_output_num = int(self.emb_dim/2)
        # self.bn1 = nn.BatchNorm2d(self.ch1_output_num)
        # self.bn2 = nn.BatchNorm2d(self.ch2_output_num)
        # self.bn3 = nn.BatchNorm2d(self.ch3_output_num)
        # self.bn4 = nn.BatchNorm2d(self.ch4_output_num)
        # # self.bn5 = nn.BatchNorm1d(self.emb_dim)

        # self.conv1 = nn.Sequential(nn.Conv2d(6, self.ch1_output_num, kernel_size=1, bias=False),
        #                            self.bn1,
        #                            nn.LeakyReLU(self.negative_slope))
        # self.conv2 = nn.Sequential(nn.Conv2d(self.ch1_output_num*2, self.ch2_output_num, kernel_size=1, bias=False),
        #                            self.bn2,
        #                            nn.LeakyReLU(self.negative_slope))
        # self.conv3 = nn.Sequential(nn.Conv2d(self.ch2_output_num*2, self.ch3_output_num, kernel_size=1, bias=False),
        #                            self.bn3,
        #                            nn.LeakyReLU(self.negative_slope))
        # self.conv4 = nn.Sequential(nn.Conv2d(self.ch3_output_num*2, self.ch4_output_num, kernel_size=1, bias=False),
        #                            self.bn4,
        #                            nn.LeakyReLU(self.negative_slope))
        # self.conv5 = nn.Sequential(nn.Conv1d(512, self.emb_dim, kernel_size=1, bias=False),
        #                            self.bn5,
        #                            nn.LeakyReLU(self.negative_slope))
        # self.linear1 = nn.Linear(emb_dim*2, emb_dim, bias=False)
        # self.bn6 = nn.BatchNorm1d(512)
        # self.dp1 = nn.Dropout(p=args.dropout)
        # self.linear2 = nn.Linear(512, 256)
        # self.bn7 = nn.BatchNorm1d(256)
        # self.dp2 = nn.Dropout(p=args.dropout)
        # self.linear3 = nn.Linear(256, output_channels)
        self.mlp1 = nn.Sequential(
            nn.Linear(3,64,True),
            nn.ReLU(),
            nn.Linear(64, 64,True),
            nn.ReLU(),
            nn.Linear(64, 64,True)
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(64,128,True)
        )
        self.mlp3 = nn.Sequential(
            nn.Linear(128,self.emb_dim,True)
        )

    def forward(self, x):

        if self.input_shape == "bcn":
            x = x.permute(0, 2, 1)
        channel = x.size(2)
        batch_size = x.size(0)
        number = x.size(1)

        topk = get_graph_feature(x, k=self.k)
        topk_feature = combine_point_cloud_and_knn(x,topk).view(batch_size*number*self.k,-1)

        x = self.mlp1(topk_feature)
        x = x.view(batch_size,number,self.k,-1)
        x = x.max(dim=2, keepdim=False)[0]

        topk = get_graph_feature(x, k=self.k)
        topk_feature = combine_point_cloud_and_knn(x,topk).view(batch_size*number*self.k,-1)
        x = self.mlp2(topk_feature)
        x = x.view(batch_size,number,self.k,-1)
        x = x.max(dim=2, keepdim=False)[0]
        x = self.mlp3(x)
        return x


#测试函数
if __name__ == '__main__':
    # Test the code.
    x = torch.rand((10,1024,3)).to(torch.device('cuda')).float()
    # if torch.cuda.is_available():
    #     device = torch.device('cuda')
    # x = x.to(device)
    pn = DGCNN(input_shape="bnc").to(torch.device('cuda'))
    y = pn(x)
    print("Network Architecture: ")
    print(pn)
    print("Input Shape of DGCNN: ", x.shape, "\nOutput Shape of DGCNN: ", y.shape)
    get_graph_feature(x, 20)