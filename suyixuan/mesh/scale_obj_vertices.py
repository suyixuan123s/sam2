#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：sam2 
@File    ：scale_obj_vertices.py
@IDE     ：PyCharm 
@Author  ：suyixuan
@Date    ：2025-04-03 11:00:08 
'''

def scale_obj_vertices(input_file, output_file, scale_factor):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            # 处理顶点行
            if line.startswith('v '):
                parts = line.split()
                # 将 x, y, z 坐标缩小 scale_factor 倍
                scaled_vertices = [parts[0]] + [str(float(coord) / scale_factor) for coord in parts[1:4]]
                # 写入新的顶点行
                outfile.write(f"{' '.join(scaled_vertices)}\n")
            else:
                # 其他行直接写入
                outfile.write(line)

# 使用示例
input_obj_file = 'mesh_cleaned1.obj'  # 输入文件
output_obj_file = 'scaled_textured_mesh.obj'  # 输出文件
scale_factor = 10.0  # 缩小比例

scale_obj_vertices(input_obj_file, output_obj_file, scale_factor)
print(f"已将 {input_obj_file} 中的顶点坐标缩小 {scale_factor} 倍，并保存为 {output_obj_file}.")



