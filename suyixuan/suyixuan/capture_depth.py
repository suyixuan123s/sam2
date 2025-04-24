#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：sam2 
@File    ：capture_depth.py
@IDE     ：PyCharm 
@Author  ：suyixuan
@Date    ：2025-02-26 11:48:48 
'''
import cv2
import numpy as np

def get_depth_from_points_cv2(depth_image_path, points, camera_matrix):
    """
    使用 OpenCV 从深度图和相机内参获取点的深度值。

    Args:
        depth_image_path: 深度图的路径 (例如，.png 或 .tif)。
        points: 一个包含 (x, y) 坐标的列表或 NumPy 数组。
                例如：[(x1, y1), (x2, y2), ...] 或 np.array([[x1, y1], [x2, y2], ...])
        camera_matrix: 3x3 的相机内参矩阵 (NumPy 数组)。

    Returns:
        一个包含对应点深度值的列表。如果某个点超出图像范围，则返回 None。
    """

    # 读取深度图
    depth_image = cv2.imread(depth_image_path, cv2.IMREAD_ANYDEPTH)
    if depth_image is None:
        raise ValueError(f"无法读取深度图: {depth_image_path}")

    depth_values = []
    for x, y in points:
        # 确保坐标是整数
        x = int(round(x))
        y = int(round(y))

        # 检查点是否在图像范围内
        if 0 <= x < depth_image.shape[1] and 0 <= y < depth_image.shape[0]:
            depth = depth_image[y, x]  # 注意 OpenCV 中图像坐标是 (y, x)
            depth_values.append(depth)
        else:
            depth_values.append(None)  # 点超出范围
            print(f"Warning: Point ({x}, {y}) is out of bounds.")

    return depth_values


# 示例用法
depth_image_path = "/home/suyixuan/AI/Pose_Estimation/sam2/suyixuan/depth_image_20250224-221415-802002.png"  # 替换为你的深度图路径
points = [(100, 200), (300, 400), (500, 600)]  # 替换为你要查询的点的坐标
camera_matrix = np.array([
    [9.099396972656250000e+02, 0.0, 6.464209594726562500e+02],
    [0.0, 9.099850463867187500e+02, 3.617229309082031250e+02],
    [0.0, 0.0, 1.0]

])  # 替换为你的相机内参矩阵

depth_values = get_depth_from_points_cv2(depth_image_path, points, camera_matrix)
print(f"Depth values: {depth_values}")

# 如果你想获取 3D 坐标 (而不仅仅是深度)，可以使用反投影：
def get_3d_coordinates_cv2(depth_image_path, points, camera_matrix):
    depth_values = get_depth_from_points_cv2(depth_image_path, points, camera_matrix)
    coordinates_3d = []

    for (x,y), depth in zip(points, depth_values):
        if depth is not None:
            # 反投影公式
            x_3d = (x - camera_matrix[0, 2]) * depth / camera_matrix[0, 0]
            y_3d = (y - camera_matrix[1, 2]) * depth / camera_matrix[1, 1]
            z_3d = depth
            coordinates_3d.append((x_3d, y_3d, z_3d))
        else:
            coordinates_3d.append(None)

    return coordinates_3d

coordinates_3d = get_3d_coordinates_cv2(depth_image_path, points, camera_matrix)
print(f"3D Coordinates: {coordinates_3d}")


