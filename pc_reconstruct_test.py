import torch
import torch.nn as nn 
from tqdm import tqdm
from loss import ChamferLoss
from model import Autoencoder, DGCNN
import argparse
import os
import numpy as np
from data_utils import *

#全局参数
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#运行参数
def setArguments():
    argsParser = argparse.ArgumentParser(description="PointCloud registration network trainning")
    # 基础设置
    argsParser.add_argument('--exp_name', type=str, default='exp_dgcnn', metavar='N',
                        help='Name of the experiment')
    argsParser.add_argument('--eval', type=bool, default=False, help='Train or Evaluate the network.')

    # 输入数据设置
    argsParser.add_argument('--dataset_type', default='modelnet40', choices=['modelnet40', 'custom'],
                        metavar='DATASET', help='dataset type (default: modelnet40)')
    argsParser.add_argument('--num_points', default=1024, type=int,
                        metavar='N', help='points in point-cloud (default: 1024)')
    argsParser.add_argument('--dataset_path', type=str , default=BASE_DIR+'/dataset', metavar='PATH', help='path to the dataset')

    # 对称函数与特征函数
    argsParser.add_argument('--featfn', default='pointnet', type=str, choices=['pointnet', 'dgcnn'],
                        help='feature extraction function choice(default: dgcnn)')
    argsParser.add_argument('--emb_dims', default=1024, type=int,
                        metavar='K', help='dim. of the feature vector (default: 1024)')

    # 训练设置
    argsParser.add_argument('--deterministic', type=bool,default=False,
                            help='use cudnn deterministic (default: false), accompany with --seed to repeate experiment')
    argsParser.add_argument('--seed', type=int, default=1234)

    argsParser.add_argument('-j', '--workers', default=4, type=int,
                        metavar='N', help='number of data loading workers (default: 4)')
    argsParser.add_argument('--batch_size', default=12, type=int,
                        metavar='N', help='mini-batch size (default: 32)')
    argsParser.add_argument('--epochs', default=200, type=int,
                        metavar='N', help='number of total epochs to run')
    argsParser.add_argument('--optimizer', default='Adam', choices=['Adam', 'SGD'],
                        metavar='METHOD', help='name of an optimizer (default: Adam)')
    argsParser.add_argument('--resume', default='', type=str,
                        metavar='PATH', help='path to latest checkpoint (default: null (no-use)), used with --start_epoch argument')
    argsParser.add_argument('--start_epoch', default=0, type=int,
                        metavar='N', help='manual epoch number (useful on restarts)')
    argsParser.add_argument('--pretrained', default='', type=str,
                        metavar='PATH', help='path to pretrained model file (default: null (no-use))')
    argsParser.add_argument('--device', default='cuda', type=str,
                        metavar='DEVICE', help='use CUDA if available')
    args = argsParser.parse_args()

    return args

class Trainunit(nn.Module):
    def __init__(self, feature, reconstructor):
        super(Trainunit, self).__init__()
        self.feature = feature
        self.reconstructor = reconstructor

    def forward(self, x: torch.Tensor):
        # x = torch.transpose(x,1,2)
        # Step 1: Extract features
        features = self.feature(x)

        # Step 2: Reconstruct point cloud
        reconstructed = self.reconstructor(features)

        return reconstructed

def display_open3d(template, source):
	template_ = o3d.geometry.PointCloud()
	source_ = o3d.geometry.PointCloud()
	template_.points = o3d.utility.Vector3dVector(template)
	source_.points = o3d.utility.Vector3dVector(source + np.array([0,0,0]))
	template_.paint_uniform_color([1, 0, 0])
	source_.paint_uniform_color([0, 1, 0])
	o3d.visualization.draw_geometries([template_, source_])

def test(args, trainunit, test_loader, device):
    trainunit.eval()
    test_loss = 0.0
    pred  = 0.0
    count = 0
    errors = []
    for i, data in enumerate(tqdm(test_loader)):
        template, source, igt = data
        source = source if torch.rand(1).item() < 0.5 else template
        source = source.to(device)

        source = source - torch.mean(source, dim=1, keepdim=True)
        template = trainunit(source)
        loss_val = ChamferLoss()(template,source)
        display_open3d(template, source)
        test_loss += loss_val.item()
        count += 1
    test_loss = float(test_loss)/count
    errors = np.mean(np.array(errors), axis=0)

    print("Test Loss: {}".format(test_loss))

def main():
    args = setArguments()
    if args.deterministic == True:
        torch.backends.cudnn.deterministic = True
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)

    testset = RegistrationData(data_class=ModelNet40Data(False))
    test_loader = DataLoader(testset,batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.workers)

    #判断设备
    if torch.cuda.is_available():
        args.device = 'cuda'
    elif torch.mps.is_available():
        args.device = 'mps'
    else:
        args.device = 'cpu'

    ae = Autoencoder(args.emb_dims,args.num_points)
    dgcnn = DGCNN('bnc',args.emb_dims,20)
    trainunit = Trainunit(dgcnn,ae)

    checkpoint = None
    if args.resume:
        assert os.path.isfile(args.resume)
        checkpoint = torch.load(args.resume)
        args.start_epoch = checkpoint['epoch']
        ae.load_state_dict(checkpoint['model'])
    
    if args.pretrained:
        assert os.path.isfile(args.pretrained)
        ae.load_state_dict(torch.load(args.pretrained, map_location=args.device))
    
    device =torch.device(args.device)
    trainunit = trainunit.to(device)

    test(args, trainunit, test_loader, device)
if __name__=='__main__':
    main()