import torch
import torch.nn as nn
import torch.nn.functional as F

def emd_loss(template: torch.Tensor, source: torch.Tensor):
	from loss.emd_loss import EMDLoss
	emd_loss = torch.mean(self.emd(template, source))/(template.size()[1])
	return emd_loss


class EMDLoss(nn.Module):
	def __init__(self):
		super(EMDLoss, self).__init__()

	def forward(self, template, source):
		return emd_loss(template, source)