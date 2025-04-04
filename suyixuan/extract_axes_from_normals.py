#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：sam2 
@File    ：extract_axes_from_normals.py
@IDE     ：PyCharm 
@Author  ：suyixuan
@Date    ：2025-04-03 10:23:05 
'''
import numpy as np


def extract_axes_from_normals(normals):
    """
    通过 .obj 文件的法向量估算物体的 XYZ 方向
    :param normals: 法向量数组 (N, 3)
    :return: 3x3 方向矩阵 obb_axes（每列分别是 X, Y, Z 轴方向）
    """
    # 计算所有法向量的均值，作为物体主要方向
    mean_normal = np.mean(normals, axis=0)
    mean_normal /= np.linalg.norm(mean_normal)  # 归一化

    # 寻找与 mean_normal 垂直的一个向量作为第二个轴
    up_vector = np.array([0, 1, 0])  # 以世界坐标系的Y轴作为参考
    if np.allclose(mean_normal, up_vector):  # 避免平行
        up_vector = np.array([1, 0, 0])

    # 使用叉乘计算另两个轴
    x_axis = np.cross(up_vector, mean_normal)
    x_axis /= np.linalg.norm(x_axis)

    y_axis = np.cross(mean_normal, x_axis)
    y_axis /= np.linalg.norm(y_axis)

    return np.stack([x_axis, y_axis, mean_normal], axis=1)


def read_normals_from_obj(file_path):
    """
    从 .obj 文件中读取法向量
    :param file_path: .obj 文件路径
    :return: 法向量数组
    """
    normals = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('vn '):  # 只读取法向量行
                parts = line.strip().split()
                normal = list(map(float, parts[1:4]))  # 提取法向量的三个分量
                normals.append(normal)
    return np.array(normals)


# 假设你有一个 .obj 文件路径
obj_file_path = 'mesh/textured_mesh1.obj'  # 替换为你的 .obj 文件路径

# 从 .obj 文件中读取法向量
normals = read_normals_from_obj(obj_file_path)

# 估算物体的坐标轴方向
obb_axes = extract_axes_from_normals(normals)
print("估算的物体坐标轴方向:\n", obb_axes)



# /home/suyixuan/anaconda3/envs/sam2/bin/python /home/suyixuan/AI/Pose_Estimation/sam2/suyixuan/extract_axes_from_normals.py
# 估算的物体坐标轴方向:
#  [[-0.80444482  0.44957728  0.38826383]
#  [ 0.          0.65361267 -0.75682922]
#  [-0.59402738 -0.60882735 -0.52579533]]
#
# Process finished with exit code 0