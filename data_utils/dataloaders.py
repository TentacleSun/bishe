import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
import sys
import glob
import h5py
import numpy as np
import open3d as o3d
import transform_utils
import random
            

class ModelNet40Data(Dataset):

    def __init__(self, train=True, num_points=1024, randomize_data=False):
        super(ModelNet40Data, self).__init__()
        self.data, self.labels = self.load_data(train)
        self.num_points = num_points
        self.randomize_data = randomize_data

    def load_data(self, train):
        if train: partition = 'train'
        else: partition = 'test'
        if sys.platform == 'linux':
            BASE_DIR = os.path.dirname(os.path.abspath(__file__))
            #BASE_DIR = os.path.dirname('/home/sunjunyang/bishe')
            DATA_DIR = os.path.join(BASE_DIR, os.pardir, 'data')
            #DATA_DIR = os.path.dirname('/home/sunjunyang/bishe/data/')
        elif sys.platform == 'darwin':
            BASE_DIR = os.path.dirname(os.path.abspath(__file__))
            #BASE_DIR = os.path.dirname('/Users/sunjunyang/Desktop/bishe/')
            DATA_DIR = os.path.join(BASE_DIR, os.pardir, 'data')
            #DATA_DIR = os.path.dirname('/Users/sunjunyang/Desktop/bishe/data')
        else:
            raise OSError('your system is not capable of this program')

        all_data = []
        all_label = []
        
        # 遍历 HDF5 文件
        for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5' % partition)):
            f = h5py.File(h5_name)
            data = f['data'][:].astype('float32')
            label = f['label'][:].astype('int64')
            f.close()
            all_data.append(data)
            all_label.append(label)
            
            # for i, point_cloud in enumerate(data):
            #     # 创建 Open3D 点云对象
            #     pcd = o3d.geometry.PointCloud()
            #     pcd.points = o3d.utility.Vector3dVector(point_cloud)  # 设置点云坐标

            #     # 可选：为点云着色（例如随机颜色）
            #     pcd.paint_uniform_color([np.random.rand(), np.random.rand(), np.random.rand()])

            #     # 显示点云
            #     print(f"Displaying point cloud {i + 1} from file {h5_name}")
            #     o3d.visualization.draw_geometries([pcd])

        all_data = np.concatenate(all_data, axis=0)
        all_label = np.concatenate(all_label, axis=0)
        return all_data, all_label

    def __getitem__(self, idx):
        if self.randomize_data: 
            current_points = self.randomize(idx)
        else: current_points = self.data[idx].copy()
        current_points = torch.from_numpy(current_points).float()
        label = torch.from_numpy(self.labels[idx]).type(torch.LongTensor)
        return current_points, label
    
    def __len__(self):
        return self.data.shape[0]

    # def get_shape(self, label):
    #     return self.shapes[label]

    def randomize(self, idx):
        pt_idxs = np.arange(0, self.num_points)
        np.random.shuffle(pt_idxs)
        return self.data[idx, pt_idxs].copy()

class Match3D(Dataset):
    def __init__(self, train=True, num_points=1024, transform_algorithm='rigid', is_testing=False):
        super(Match3D, self).__init__()
        # self.data, self.labels = self.load_data(train)
        self.num_points = num_points

        if train: partition = 'train'
        else: partition = 'test'
        self.root = '/home/sunjunyang/bishe/data/3DMatch/threedmatch'
        train_ids = self.read_ids('./data_utils/train.txt')
        files = sorted(os.listdir(self.root))
        self.suffixes, self.clss = {}, {}
        for file in files:
            suffix = file.split('.')[-1]
            self.suffixes[suffix] = self.suffixes.get(suffix, 0) + 1
            cls = file.split('@')[0]
            if cls not in train_ids:
                continue

            if '0.30' in file:
                seq_id = file.split('@')[1][:6]
                pairs = self.read_correspondence_pairs(os.path.join(self.root, file))
                self.clss[(cls, seq_id)] = self.clss.get((cls, seq_id), 0) + len(pairs)
        print(self.suffixes)
        for cls, num in self.clss.items():
            print(cls, num)
        ids = set([cls[0] for cls, num in self.clss.items()])
        if transform_algorithm=='rigid':
            self.transform_class = Rigidtransform(data_size=len(self), angle_range=45, trans_range=1)
        print('Total class: {}, npairs: {}'.format(len(ids), sum(self.clss.values())))

    def __getitem__(self, idx):

        id, seq = random.choice(list(self.clss.keys()))
        #id, seq = '7-scenes-chess', 'seq-01'
        id_seq_path = os.path.join(self.root, '{}@{}-0.30.txt'.format(id, seq))
        pairs = self.read_correspondence_pairs(id_seq_path)
        pair = random.choice(pairs)
        pcd1,pcd2 = self.load_data(pair)
        pcd1,pcd2 = torch.from_numpy(pcd1).float(),torch.from_numpy(pcd2).float()
        source, igt = self.transform_class.get_transformation(idx, pcd2)
        return pcd1, source, igt
    def load_data(self, pair):
        path1, path2 = os.path.join(self.root, pair[0]), os.path.join(self.root, pair[1])
        pc1, pc2 = np.load(path1), np.load(path2)
        pcd1, pcd2 = np.asarray(pc1['pcd']), np.asarray(pc2['pcd'])
        pcd1, pcd2 = self.downsample(pcd1), self.downsample(pcd2)
        return pcd1, pcd2

    def downsample(self, pcd):
        
        indices= np.random.choice(pcd.shape[0], size=2048, replace=False)
        return pcd[indices]

    def __len__(self):
        return 7960
    def read_correspondence_pairs(self,path):
        pairs = []
        with open(path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                pairs.append(line.split())
        return pairs
    def read_ids(self, path):
        ids = []
        with open(path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                ids.append(line.strip())
        return ids  
#给定数据集大小，创建随机刚性变换的配准数据
class Rigidtransform:
    def __init__(self, data_size, angle_range=45, trans_range=1, data_type=torch.float32):
        self.angle_range = angle_range
        self.translation_range = trans_range
        self.dtype = data_type
        self.data_size = data_size
        self.transformations = [self.create_random_transformation(self, self.dtype,self.angle_range,self.translation_range) for _ in range(self.data_size)]
    @staticmethod
    def deg_to_rad(deg):
        return np.pi / 180 * deg
    
    @staticmethod
    def create_random_transformation(self, dtype, angle_range, translation_range):
        #转弧度制
        angle_range = self.deg_to_rad(angle_range)
        rot_vec = np.random.uniform(-angle_range,angle_range,[1, 3])
        trans_vec = np.random.uniform(-translation_range, translation_range, [1, 3])
        quat = transform_utils.euler_to_quaternion(rot_vec, 'xyz')
        
        vec = np.concatenate([quat, trans_vec], axis=1)
        return torch.tensor(vec, dtype=dtype)
    
    @staticmethod
    def create_pose_7d(vector: torch.Tensor):
        # Normalize the quaternion.
        pre_normalized_quaternion = vector[:, 0:4]
        normalized_quaternion = F.normalize(pre_normalized_quaternion, dim=1)

        # B x 7 vector of 4 quaternions and 3 translation parameters
        translation = vector[:, 4:]
        vector = torch.cat([normalized_quaternion, translation], dim=1)
        return vector.view([-1, 7])
    
    @staticmethod
    def quaternion_rotate(point_cloud: torch.Tensor, pose_7d: torch.Tensor):
        ndim = point_cloud.dim()
        if ndim == 2:
            N, _ = point_cloud.shape
            assert pose_7d.shape[0] == 1
            # repeat transformation vector for each point in shape
            quat = pose_7d[:, 0:4].expand([N, -1])
            rotated_point_cloud = transform_utils.qrot(quat, point_cloud)

        elif ndim == 3:
            B, N, _ = point_cloud.shape
            quat = pose_7d[:, 0:4].unsqueeze(1).expand([-1, N, -1]).contiguous()
            rotated_point_cloud = transform_utils.qrot(quat, point_cloud)

        return rotated_point_cloud
    @staticmethod
    def get_quaternion(pose_7d: torch.Tensor):
        return pose_7d[:, 0:4]

    @staticmethod
    def get_translation(pose_7d: torch.Tensor):
        return pose_7d[:, 4:]
    
    @staticmethod
    def quaternion_transform(point_cloud: torch.Tensor, pose_7d: torch.Tensor):
        transformed_point_cloud = Rigidtransform.quaternion_rotate(point_cloud, pose_7d) + Rigidtransform.get_translation(pose_7d).view(-1, 1, 3).repeat(1, point_cloud.shape[1], 1)      # Ps' = R*Ps + t
        return transformed_point_cloud
    @staticmethod
    def convert2transformation(rotation_matrix: torch.Tensor, translation_vector: torch.Tensor):
        one_ = torch.tensor([[[0.0, 0.0, 0.0, 1.0]]]).repeat(rotation_matrix.shape[0], 1, 1).to(rotation_matrix)    # (Bx1x4)
        transformation_matrix = torch.cat([rotation_matrix, translation_vector[:,0,:].unsqueeze(-1)], dim=2)                        # (Bx3x4)
        transformation_matrix = torch.cat([transformation_matrix, one_], dim=1)                                     # (Bx4x4)
        return transformation_matrix
    
    def get_transformation(self, index, source):
        igt = self.create_pose_7d(self.transformations[index])
        target= self.quaternion_rotate(source, igt) + igt[:, 4:]
        
        return target, igt
    
#获取配准数据对的数据对
# TODO 加入生成点云配准的转换算法
class RegistrationData(Dataset):
    def __init__(self, data_class=ModelNet40Data(), transform_algorithm='rigid', is_testing=False):
        super(RegistrationData, self).__init__()
        self.is_testing = is_testing
        self.data_class = data_class
        if transform_algorithm=='rigid':
            self.transform_class = Rigidtransform(data_size=len(data_class), angle_range=45, trans_range=1)
    
    def __len__(self):
        return len(self.data_class)
    
    def __getitem__(self, index):
        template = self.data_class[index][0]
        source, igt = self.transform_class.get_transformation(index, template)
        
        return template,source,igt