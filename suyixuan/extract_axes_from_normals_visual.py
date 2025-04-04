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
import pyvista as pv

def load_mesh(file_path):
    """
    加载 .obj 网格文件
    :param file_path: .obj 文件路径
    :return: PyVista 3D 网格对象, 顶点数组
    """
    mesh = pv.read(file_path)  # 读取网格
    points = np.array(mesh.points)  # 提取顶点坐标
    return mesh, points

def compute_center(points):
    """
    计算物体的质心（作为坐标轴原点）
    :param points: 顶点坐标 (N, 3)
    :return: 质心坐标 (3,)
    """
    return np.mean(points, axis=0)

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



def visualize_obj_with_axes(mesh, center, axes):
    """
    可视化 .obj 物体及其坐标轴
    :param mesh: 物体网格
    :param center: 质心坐标 (3,)
    :param axes: XYZ 轴方向 3x3 矩阵（每列分别是 X, Y, Z 轴）
    """
    plotter = pv.Plotter()

    # 添加物体模型
    plotter.add_mesh(mesh, color='lightblue', opacity=0.7, label='Mesh')

    # 颜色定义
    axis_colors = ['r', 'g', 'b']  # 红 X，绿 Y，蓝 Z

    # 绘制坐标轴
    for i in range(3):  # 遍历 XYZ 轴
        start = center
        end = center + axes[:, i] * 0.2  # 轴的方向和长度
        plotter.add_arrows(start, end - start, color=axis_colors[i], label=f'Axis {i + 1}')

    plotter.set_background('white')
    plotter.add_legend()
    plotter.show()


# 假设你有一个 .obj 文件路径
obj_file_path = 'mesh/textured_mesh1.obj'  # 替换为你的 .obj 文件路径

# 从 .obj 文件中读取法向量
normals = read_normals_from_obj(obj_file_path)

# 1. 加载物体
mesh, points = load_mesh(obj_file_path)

# 估算物体的坐标轴方向
obb_axes = extract_axes_from_normals(normals)
print("估算的物体坐标轴方向:\n", obb_axes)

# 2. 计算质心（作为坐标轴原点）
center = compute_center(points)

# 4. 可视化物体 + 其坐标轴
visualize_obj_with_axes(mesh, center, obb_axes)



# /home/suyixuan/anaconda3/envs/sam2/bin/python /home/suyixuan/AI/Pose_Estimation/sam2/suyixuan/extract_axes_from_normals.py
# 估算的物体坐标轴方向:
#  [[-0.80444482  0.44957728  0.38826383]
#  [ 0.          0.65361267 -0.75682922]
#  [-0.59402738 -0.60882735 -0.52579533]]
#
# Process finished with exit code 0