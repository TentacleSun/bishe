import torch
import torch.nn as nn
import torch.nn.functional as F

# 提取特征函数Pointnet
class PointNet(nn.Module):
    def __init__(self, emb_dim=1024, channels=3, input_shape="bcn", use_bn=False):
        # emb_dim:输出的embedding维度
        # input_shape:输入点云批次的格式

        super(PointNet, self).__init__()
        if input_shape not in ["bcn", "bnc"]:
            raise ValueError("Allowed shapes are 'bcn' (batch * channels * num_points), 'bnc' ")
        self.input_shape = input_shape
        self.emb_dim = emb_dim
        self.input_channels = channels

        self.conv1 = torch.nn.Conv1d(self.input_channels, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 64, 1)
        self.conv3 = torch.nn.Conv1d(64, 64, 1)
        self.conv4 = torch.nn.Conv1d(64, 128, 1)
        self.conv5 = torch.nn.Conv1d(128, self.emb_dim, 1)
        self.relu = torch.nn.ReLU()

        if use_bn:  # 如果启用了批量正则化
            self.bn1 = nn.BatchNorm1d(64)
            self.bn2 = nn.BatchNorm1d(64)
            self.bn3 = nn.BatchNorm1d(64)
            self.bn4 = nn.BatchNorm1d(128)
            self.bn5 = nn.BatchNorm1d(emb_dim)
            
            # 将每个卷积层和对应的批量正则化层组合在一起
            self.conv1 = nn.Sequential(self.conv1, self.bn1)
            self.conv2 = nn.Sequential(self.conv2, self.bn2)
            self.conv3 = nn.Sequential(self.conv3, self.bn3)
            self.conv4 = nn.Sequential(self.conv4, self.bn4)
            self.conv5 = nn.Sequential(self.conv5, self.bn5)

        self.layers = [self.conv1, self.relu,
                    self.conv2, self.relu, 
                    self.conv3, self.relu,
                    self.conv4, self.relu,
                    self.conv5, self.relu]

    def forward(self, input_data):
        # 输入：点云批次，以及点云批次类型
        # 输出：生成的PointNet高维度特征向量
        if self.input_shape == "bnc":
            input_data = input_data.permute(0, 2, 1)
        num_points = input_data.shape[2]

        output = input_data
        for idx, layer in enumerate(self.layers):
            output = layer(output)
        return output

#测试函数
if __name__ == '__main__':
	# Test the code.
	x = torch.rand((10,1024,3))

	pn = PointNet(use_bn=True, input_shape="bnc")
	y = pn(x)
	print("Network Architecture: ")
	print(pn)
	print("Input Shape of PointNet: ", x.shape, "\nOutput Shape of PointNet: ", y.shape)