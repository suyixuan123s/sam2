#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：sam2 
@File    ：visualization_mesh.py
@IDE     ：PyCharm 
@Author  ：suyixuan
@Date    ：2025-03-31 17:31:40 
'''
import open3d as o3d



mesh = o3d.io.read_triangle_mesh('mesh/textured_mesh.obj')

# 可视化网格
o3d.visualization.draw_geometries([mesh])