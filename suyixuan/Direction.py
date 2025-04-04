#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：sam2 
@File    ：Direction.py
@IDE     ：PyCharm 
@Author  ：suyixuan
@Date    ：2025-03-31 17:04:23 
'''
import numpy as np


def load_obj(file_path):
    """加载 OBJ 文件并返回顶点坐标"""
    vertices = []

    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('v '):  # 顶点行
                parts = line.strip().split()
                vertex = list(map(float, parts[1:4]))  # 提取 x, y, z 坐标
                vertices.append(vertex)

    return np.array(vertices)


def calculate_direction(vertices):
    """计算模型的方向"""
    # 计算模型的中心
    center = np.mean(vertices, axis=0)

    # 计算 X, Y, Z 方向
    x_direction = vertices[:, 0] - center[0]
    y_direction = vertices[:, 1] - center[1]
    z_direction = vertices[:, 2] - center[2]

    return x_direction, y_direction, z_direction


# 使用示例
input_file_path = 'mesh/textured_mesh.obj'  # 替换为您的 OBJ 文件路径
vertices = load_obj(input_file_path)
x_dir, y_dir, z_dir = calculate_direction(vertices)

print("X Direction:", x_dir)
print("Y Direction:", y_dir)
print("Z Direction:", z_dir)
