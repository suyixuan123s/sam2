import pyvista as pv
import numpy as np
from scipy.spatial import ConvexHull


def compute_obb(mesh):
    """
    计算网格的外接包围盒（OBB）
    :param mesh: 输入网格对象
    :return: OBB的中心、方向和大小（长宽高）
    """
    # 获取网格的顶点坐标
    points = mesh.points

    # 计算网格的边界框
    bounds = mesh.bounds  # [xmin, xmax, ymin, ymax, zmin, zmax]
    print("Bounds of the mesh:", bounds)

    # 获取网格的主要轴方向
    hull = ConvexHull(points)
    volume_points = points[hull.vertices]

    # 计算主要轴
    covariance_matrix = np.cov(volume_points.T)
    eigvals, eigvecs = np.linalg.eigh(covariance_matrix)
    obb_axes = eigvecs  # 主轴方向

    # OBB的中心就是网格的质心
    obb_center = np.mean(points, axis=0)

    # 计算OBB的长宽高（由主轴的特征值决定）
    obb_lengths = np.sqrt(eigvals)

    return obb_center, obb_axes, obb_lengths


def visualize_obb(mesh, obb_center, obb_axes, obb_lengths):
    """
    可视化网格和外接包围盒（OBB）
    :param mesh: 输入网格对象
    :param obb_center: OBB中心
    :param obb_axes: OBB主轴方向
    :param obb_lengths: OBB长宽高
    """
    # 创建可视化场景
    plotter = pv.Plotter()

    # 可视化网格
    plotter.add_mesh(mesh, color='lightblue', opacity=0.7, label='Mesh')

    # 颜色定义
    axis_colors = ['r', 'g', 'b']  # 红 X，绿 Y，蓝 Z

    # 可视化OBB的长宽高
    for i in range(3):
        start = obb_center
        end = obb_center + obb_axes[:, i] * obb_lengths[i]
        plotter.add_arrows(start, end, color=axis_colors[i], label=f'Axis {i + 1}')

    # 设置视图
    plotter.set_background('white')
    plotter.add_legend()
    plotter.show()


def main(file_path):
    # 加载网格文件
    mesh = pv.read(file_path)

    # 计算OBB信息
    obb_center, obb_axes, obb_lengths = compute_obb(mesh)

    # 打印OBB的中心、方向和大小
    print(f"OBB Center: {obb_center}")
    print(f"OBB Axes:\n{obb_axes}")
    print(f"OBB Lengths: {obb_lengths}")

    # 可视化网格和OBB
    visualize_obb(mesh, obb_center, obb_axes, obb_lengths)


if __name__ == "__main__":
    file_path = '/home/suyixuan/AI/Pose_Estimation/BundleSDF/data/rack/out_rack/textured_mesh.obj'  # 请替换为你的网格文件路径
    main(file_path)


# /home/suyixuan/anaconda3/envs/sam2/bin/python /home/suyixuan/AI/Pose_Estimation/sam2/suyixuan/compute_obb.py
# Bounds of the mesh: (-0.69287115, 0.88664269, -0.67462403, 0.67322731, -0.44992203, 0.98930019)
# OBB Center: [0.08705164 0.02011    0.31653731]
# OBB Axes:
# [[-0.02585829  0.034276   -0.99907783]
#  [ 0.81269678  0.58268589 -0.00104377]
#  [-0.58211278  0.81197432  0.04292326]]
# OBB Lengths: [0.40408754 0.44876996 0.56172821]



# /home/suyixuan/anaconda3/envs/sam2/bin/python /home/suyixuan/AI/Pose_Estimation/sam2/suyixuan/compute_obb.py
# Bounds of the mesh: (-0.07361943, 0.08812231, -0.06570107, 0.07281199, -0.03535114, 0.11334069)
# OBB Center: [0.0061917  0.00495135 0.04282256]
# OBB Axes:
# [[-0.06859279  0.00393422 -0.99763698]
#  [ 0.5633101   0.82548364 -0.0354752 ]
#  [-0.82339344  0.56441234  0.05883841]]
# OBB Leng
# ths: [0.04142935 0.04598986 0.05914746]



# Bounds of the mesh: (-0.00069865304, 0.00087625116, -0.00066857767, 0.0006801453200000001, -0.00045171875, 0.0009961165199999999)
# OBB Center: [7.58236054e-05 3.09561926e-05 3.15194386e-04]
# OBB Axes:
# [[-0.06859278  0.0039342  -0.99763698]
#  [ 0.56331014  0.82548361 -0.03547522]
#  [-0.82339341  0.56441238  0.0588384 ]]
# OBB Lengths: [0.0004034  0.00044781 0.00057593]


# /home/suyixuan/anaconda3/envs/sam2/bin/python /home/suyixuan/AI/Pose_Estimation/sam2/suyixuan/compute_obb.py
# Bounds of the mesh: (-0.07361943, 0.08812231, -0.06570107, 0.07281199, -0.03535114, 0.11334069)
# OBB Center: [0.0061917  0.00495135 0.04282256]
# OBB Axes:
# [[-0.06859279  0.00393422 -0.99763698]
#  [ 0.5633101   0.82548364 -0.0354752 ]
#  [-0.82339344  0.56441234  0.05883841]]
# OBB Lengths: [0.04142935 0.04598986 0.05914746]