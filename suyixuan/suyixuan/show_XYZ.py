#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：sam2
@File    ：show_XYZ.py
@IDE     ：PyCharm
@Author  ：suyixuan
@Date    ：2025-03-31 16:41:39
'''
import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# 加载 .obj 文件
mesh = trimesh.load_mesh('mesh/textured_mesh.obj')

# 获取网格的顶点和面信息
vertices = mesh.vertices
faces = mesh.faces

# 设置 3D 绘图
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制网格表面
ax.plot_trisurf(vertices[:, 0], vertices[:, 1], faces, vertices[:, 2], cmap='gray', edgecolor='none')

# 绘制坐标轴（X轴、Y轴、Z轴）
# 原点坐标为 (0, 0, 0)
ax.quiver(0, 0, 0, 1, 0, 0, length=1, color='r', label='X axis')
ax.quiver(0, 0, 0, 0, 1, 0, length=1, color='g', label='Y axis')
ax.quiver(0, 0, 0, 0, 0, 1, length=1, color='b', label='Z axis')

# 绘制原点（可以用一个小球表示）
ax.scatter(0, 0, 0, color='k', s=50, label='Origin')

# 设置标签
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# 添加图例
ax.legend()

# 显示图形
plt.show()
