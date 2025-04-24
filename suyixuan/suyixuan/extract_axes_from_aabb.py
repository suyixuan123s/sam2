import numpy as np
from scipy.spatial import ConvexHull

def extract_axes_from_aabb(points):
    """
    通过 AABB（轴对齐包围盒）估计物体的 XYZ 方向
    :param points: 物体顶点 (N, 3)
    :return: 3x3 方向矩阵 obb_axes（每列分别是 X, Y, Z 轴方向）
    """
    # 计算最小和最大边界
    min_bound = np.min(points, axis=0)
    max_bound = np.max(points, axis=0)

    # 计算 AABB 轴方向（近似于原始坐标系）
    x_axis = np.array([1, 0, 0]) if max_bound[0] - min_bound[0] > 0 else np.array([0, 0, 0])
    y_axis = np.array([0, 1, 0]) if max_bound[1] - min_bound[1] > 0 else np.array([0, 0, 0])
    z_axis = np.array([0, 0, 1]) if max_bound[2] - min_bound[2] > 0 else np.array([0, 0, 0])

    return np.stack([x_axis, y_axis, z_axis], axis=1)

# 假设 points 是物体的顶点数据
points = np.array([
    [-1, -1, 0], [1, -1, 0], [-1, 1, 0], [1, 1, 0],
    [-1, -1, 2], [1, -1, 2], [-1, 1, 2], [1, 1, 2]
])  # 示例数据

obb_axes = extract_axes_from_aabb(points)
print("AABB 估算的物体坐标轴方向:\n", obb_axes)
