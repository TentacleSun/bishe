import torch
import torch.nn as nn
import torch.nn.functional as F

# 池化层，统一使用bnc模式
class Pooling(torch.nn.Module):
	def __init__(self, pool_type='max'):
		self.pool_type = pool_type
		super(Pooling, self).__init__()

	def forward(self, input):
		if self.pool_type == 'max':
			return torch.max(input, 2)[0].contiguous()
		elif self.pool_type == 'avg':
			return torch.mean(input, 2).contiguous()
		
if __name__=="__main__":
	x = torch.rand((10,1024,3))
	poolint_test = Pooling(pool_type='max')
	print("input data")
	print(x.shape)
	print("output data")
	print(poolint_test(x).shape)