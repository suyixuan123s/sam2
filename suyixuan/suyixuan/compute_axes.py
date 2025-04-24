import pyvista as pv
import numpy as np


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


def extract_axes_from_obj(points):
    """
    直接从 .obj 文件的数据获取 XYZ 方向
    方法：基于物体的顶点分布计算其 X、Y、Z 轴的方向
    :param points: 顶点坐标 (N, 3)
    :return: 3x3 方向矩阵 obb_axes (每列分别是 X, Y, Z 轴方向)
    """
    # 计算顶点的协方差矩阵
    cov_matrix = np.cov(points, rowvar=False)

    # 计算特征值和特征向量
    eigvals, eigvecs = np.linalg.eigh(cov_matrix)

    # 按特征值从大到小排序（确保方向是主轴方向）
    sort_idx = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, sort_idx]

    return eigvecs  # 返回 XYZ 轴的方向


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

def main(file_path):
    """
    主函数：加载 .obj，计算质心，获取 XYZ 方向，并显示
    :param file_path: .obj 文件路径
    """
    # 1. 加载物体
    mesh, points = load_mesh(file_path)

    # 2. 计算质心（作为坐标轴原点）
    center = compute_center(points)

    # 3. 从 .obj 文件直接提取 XYZ 轴方向
    axes = extract_axes_from_obj(points)
    print("XYZ 轴方向（从.obj 文件提取）：")
    print(axes)

    # 4. 可视化物体 + 其坐标轴
    visualize_obj_with_axes(mesh, center, axes)



file_path = "/home/suyixuan/AI/Pose_Estimation/FoundationPose/demo_data/10ml_rack1/mesh/textured_mesh.obj"
main(file_path)
