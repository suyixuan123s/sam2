import numpy as np
from PIL import Image  # 用于读取图像

def get_depth_from_points(depth_image, points, camera_matrix):
    """
    从深度图和相机内参获取点的深度值 (纯 NumPy 实现)。

    Args:
        depth_image: 深度图 (NumPy 数组)。
        points: 一个包含 (x, y) 坐标的列表或 NumPy 数组。
        camera_matrix: 3x3 的相机内参矩阵 (NumPy 数组)。

    Returns:
        一个包含对应点深度值的列表。如果某个点超出图像范围，则返回 None。
    """
    depth_values = []
    for x, y in points:
        x = int(round(x))
        y = int(round(y))

        if 0 <= x < depth_image.shape[1] and 0 <= y < depth_image.shape[0]:
            depth = depth_image[y, x]
            depth_values.append(depth)
        else:
            depth_values.append(None)
            print(f"Warning: Point ({x}, {y}) is out of bounds.")

    return depth_values

def get_3d_coordinates(depth_image, points, camera_matrix):
    """
    从深度图、点坐标和相机内参计算 3D 坐标 (纯 NumPy 实现)。
    """
    depth_values = get_depth_from_points(depth_image, points, camera_matrix)
    coordinates_3d = []

    for (x, y), depth in zip(points, depth_values):
        if depth is not None:
            # 反投影公式
            x_3d = (x - camera_matrix[0, 2]) * depth / camera_matrix[0, 0]
            y_3d = (y - camera_matrix[1, 2]) * depth / camera_matrix[1, 1]
            z_3d = depth
            coordinates_3d.append((x_3d, y_3d, z_3d))
        else:
             coordinates_3d.append(None)
    return coordinates_3d


# 示例用法
depth_image_path = "/home/suyixuan/AI/Pose_Estimation/sam2/suyixuan/depth_image_20250224-221415-802002.png"  # 替换为你的深度图路径
depth_image = np.array(Image.open(depth_image_path))
points = [(100, 200), (300, 400),
          (500, 600)]  # 替换为你要查询的点的坐标
camera_matrix = np.array([
    [9.099396972656250000e+02, 0.0, 6.464209594726562500e+02],
    [0.0, 9.099850463867187500e+02, 3.617229309082031250e+02],
    [0.0, 0.0, 1.0]

])  # 替换为你的相机内参矩阵



depth_values = get_depth_from_points(depth_image, points, camera_matrix)
print(f"Depth values: {depth_values}")

coordinates_3d = get_3d_coordinates(depth_image, points, camera_matrix)
print(f"3D Coordinates: {coordinates_3d}")

