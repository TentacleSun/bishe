import argparse
import os 
import torch
import numpy as np
from tensorboardX import SummaryWriter
from data_utils import *
from model import dgcnn, pointnet
#全局参数
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

#运行参数
def setArguments():
    argsParser = argparse.ArgumentParser(description="PointCloud registration network trainning")
    # 基础设置
    argsParser.add_argument('--exp_name', type=str, default='exp_ipcrnet', metavar='N',
                        help='Name of the experiment')
    argsParser.add_argument('--eval', type=bool, default=False, help='Train or Evaluate the network.')

    # 输入数据设置
    argsParser.add_argument('--dataset_type', default='modelnet40', choices=['modelnet40', 'custom'],
                        metavar='DATASET', help='dataset type (default: modelnet40)')
    argsParser.add_argument('--num_points', default=1024, type=int,
                        metavar='N', help='points in point-cloud (default: 1024)')
    argsParser.add_argument('--dataset_path', type=str , default=BASE_DIR+'/dataset', metavar='PATH', help='path to the dataset')

    # 对称函数与特征函数
    argsParser.add_argument('--featfn', default='dgcnn', type=str, choices=['pointnet', 'dgcnn'],
                        help='feature extraction function choice(default: dgcnn)')
    argsParser.add_argument('--emb_dims', default=1024, type=int,
                        metavar='K', help='dim. of the feature vector (default: 1024)')
    argsParser.add_argument('--symfn', default='max', choices=['max', 'avg'],
                        help='symmetric function (default: max)')
    # 训练设置
    argsParser.add_argument('--deterministic', type=bool,default=False,
                            help='use cudnn deterministic (default: false), accompany with --seed to repeate experiment')
    argsParser.add_argument('--seed', type=int, default=1234)

    argsParser.add_argument('-j', '--workers', default=4, type=int,
                        metavar='N', help='number of data loading workers (default: 4)')
    argsParser.add_argument('--batch_size', default=20, type=int,
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
def main():
    args = setArguments()
    if args.deterministic == True:
        torch.backends.cudnn.deterministic = True
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)
    #TODO 查看日志相关模块并完善
    boardio = SummaryWriter(log_dir='checkpoints/' + args.exp_name)

    trainset = RegistrationData(data_class=ModelNet40Data(train=True))
    train_loader = DataLoader(trainset, batch_size=args.batch_size,shuffle=True, drop_last=True, num_workers=args.workers)
    testset = RegistrationData(data_class=ModelNet40Data(False))
    test_loader = DataLoader(testset,batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.workers)
    
    #判断设备
    if torch.cuda.is_available():
        args.device = 'cuda'
    elif torch.mps.is_available():
        args.device = 'mps'
    else:
        args.device = 'cpu'
        
    if args.featfn == 'dgcnn':
        featfn = dgcnn.DGCNN(emb_dim=args.emb_dims)
    elif args.featfn == 'pointnet':
        featfn = pointnet.PointNet(emb_dim=args.emb_dims)
    model = PC
        
    train(args, model)
        
    return 
if __name__=="__main__":
    main()