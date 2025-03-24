import torch
import torch.nn as nn
import torch.nn.functional as F
from geomloss import SamplesLoss


class EMDLoss(nn.Module):
	def __init__(self):
		super(EMDLoss, self).__init__()
		self.emd = SamplesLoss(loss='sinkhorn', p=2,blur=0.05)

	def forward(self, template:torch.Tensor, source:torch.Tensor):
		loss = self.emd(template, source) / template.size(1)
		return loss.mean()

# 使用示例
if __name__ == "__main__":
    # 随机生成两个点云批次 (B=8, N=1024, C=3)
    template = torch.randn(8, 1024, 3).cuda()
    source = torch.randn(8, 1024, 3).cuda()

    # 初始化 EMD Loss
    emd_loss = EMDLoss().cuda()

    # 计算损失
    loss = emd_loss(template, source)
    print(f"EMD Loss: {loss.item():.4f}")
