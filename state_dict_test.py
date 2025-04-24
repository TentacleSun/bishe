import torch
print("\nPretrained weights' state_dict:")
pretrained_dict = torch.load('/Users/sunjunyang/Desktop/bishe/checkpoints/exp_ipcrnetdg/models/best_model.t7', map_location='cpu')
for param_tensor in pretrained_dict:
    print(param_tensor, "\t", pretrained_dict[param_tensor].size())