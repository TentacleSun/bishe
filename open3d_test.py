import open3d as o3d
import numpy as np

def generate_random_point_cloud(num_points=1000):
    """生成随机点云"""
    print(f"Generating random point cloud with {num_points} points...")
    # 生成随机点 (x, y, z) 坐标
    points = np.random.rand(num_points, 3)
    # 创建 Open3D 点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    print("Random point cloud generated.")
    return pcd

def visualize_point_cloud(pcd, window_name="Point Cloud"):
    """可视化点云"""
    print("Visualizing point cloud...")
    o3d.visualization.draw_geometries([pcd], window_name=window_name)

def downsample_point_cloud(pcd, voxel_size=0.05):
    """对点云进行体素降采样"""
    print(f"Downsampling point cloud with voxel size {voxel_size}...")
    downsampled_pcd = pcd.voxel_down_sample(voxel_size)
    print(f"After downsampling: {len(downsampled_pcd.points)} points.")
    return downsampled_pcd

def estimate_normals(pcd, radius=0.1, max_nn=30):
    """估计点云法向量"""
    print("Estimating normals...")
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
    )
    print("Normals estimated.")
    return pcd

def main():
    # 生成随机点云
    pcd = generate_random_point_cloud(num_points=1000)

    # 可视化原始点云
    visualize_point_cloud(pcd, "Original Random Point Cloud")

    # 降采样点云
    downsampled_pcd = downsample_point_cloud(pcd, voxel_size=0.05)

    # 可视化降采样后的点云
    visualize_point_cloud(downsampled_pcd, "Downsampled Point Cloud")

    # 估计法向量
    pcd_with_normals = estimate_normals(downsampled_pcd, radius=0.1, max_nn=30)

    # 可视化带有法向量的点云
    visualize_point_cloud(pcd_with_normals, "Point Cloud with Normals")

if __name__ == "__main__":
    main()