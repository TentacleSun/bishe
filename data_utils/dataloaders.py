import torch.nn as nn
import torch
import torch.nn.functional as functional
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
import glob
import h5py
import numpy as np
import open3d as o3d
            
# 读取点云数据
def load_data(train):
	if train: partition = 'train'
	else: partition = 'test'
	#BASE_DIR = os.path.dirname(os.path.abspath(__file__))
	BASE_DIR = os.path.dirname('/Users/sunjunyang/Desktop/bishe/')
	#DATA_DIR = os.path.join(BASE_DIR, os.pardir, 'data')
	DATA_DIR = os.path.dirname('/Users/sunjunyang/Desktop/bishe/data/')
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
		# 	# 创建 Open3D 点云对象
		# 	pcd = o3d.geometry.PointCloud()
		# 	pcd.points = o3d.utility.Vector3dVector(point_cloud)  # 设置点云坐标

		# 	# 可选：为点云着色（例如随机颜色）
		# 	pcd.paint_uniform_color([np.random.rand(), np.random.rand(), np.random.rand()])

		# 	# 显示点云
		# 	print(f"Displaying point cloud {i + 1} from file {h5_name}")
		# 	o3d.visualization.draw_geometries([pcd])

	all_data = np.concatenate(all_data, axis=0)
	all_label = np.concatenate(all_label, axis=0)
	return all_data, all_label

def deg_to_rad(deg):
	return np.pi / 180 * deg
		
class ModelNet40Data(Dataset):
    def __init__(self, train=True, num_points=1024, randomize_data=False):
        super(ModelNet40Data, self).__init__()
        self.data, self.labels = load_data(train)
        if not train: self.shapes = self.read_classes_ModelNet40()
        self.num_points = num_points
        self.randomize_data = randomize_data

    def __getitem__(self, idx):
        if self.randomize_data: 
            current_points = self.randomize(idx)
        else: current_points = self.data[idx].copy()
        current_points = torch.from_numpy(current_points).float()
        label = torch.from_numpy(self.labels[idx]).type(torch.LongTensor)
        return current_points, label
    
    def __len__(self):
        return self.data.shape[0]

    def get_shape(self, label):
        return self.shapes[label]

    def randomize(self, idx):
        pt_idxs = np.arange(0, self.num_points)
        np.random.shuffle(pt_idxs)
        return self.data[idx, pt_idxs].copy()
    # TODO弄懂这一步是在干什么
    # def read_classes_ModelNet40(self):
    #     BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    #     DATA_DIR = os.path.join(BASE_DIR, os.pardir, 'data')
    #     file = open(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'shape_names.txt'), 'r')
    #     shape_names = file.read()
    #     shape_names = np.array(shape_names.split('\n')[:-1])
    #     return shape_names
    


#获取配准数据对的数据对
# TODO 加入生成点云配准的转换算法
class RegistrationData(Dataset):
    def __init__(self, data_class=ModelNet40Data(), transform_algorithm='rigid', is_testing=False):
        super(RegistrationData, self).__init__()
        self.is_testing = is_testing
        self.data_class = data_class
        self.trans_algo = transform_algorithm
    
    def __len__(self):
         return len(self.data_class)
    
    def __getitem__(self, index):
         template, label = self.data_class[index]
         
        
if __name__=='__main__':
    load_data(False)