import numpy as np
import pythreejs as p3


def load_obj(file_path):
    """加载 OBJ 文件并返回顶点和面"""
    vertices = []
    faces = []

    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('v '):  # 顶点
                parts = line.strip().split()
                vertex = list(map(float, parts[1:4]))
                vertices.append(vertex)
            elif line.startswith('f '):  # 面
                parts = line.strip().split()
                face = [int(p.split('/')[0]) - 1 for p in parts[1:]]  # OBJ 索引从 1 开始
                faces.append(face)

    return np.array(vertices), np.array(faces)


def create_axes(length=1.0):
    """创建 XYZ 坐标轴"""
    axes = p3.Group()

    # X 轴
    x_axis = p3.Line(
        geometry=p3.BufferGeometry(
            attributes={
                'position': p3.BufferAttribute(
                    array=np.array([[0, 0, 0], [length, 0, 0]]).flatten().astype(np.float32),
                    normalized=False
                )
            }
        ),
        material=p3.LineBasicMaterial(color='red')
    )

    # Y 轴
    y_axis = p3.Line(
        geometry=p3.BufferGeometry(
            attributes={
                'position': p3.BufferAttribute(
                    array=np.array([[0, 0, 0], [0, length, 0]]).flatten().astype(np.float32),
                    normalized=False
                )
            }
        ),
        material=p3.LineBasicMaterial(color='green')
    )

    # Z 轴
    z_axis = p3.Line(
        geometry=p3.BufferGeometry(
            attributes={
                'position': p3.BufferAttribute(
                    array=np.array([[0, 0, 0], [0, 0, length]]).flatten().astype(np.float32),
                    normalized=False
                )
            }
        ),
        material=p3.LineBasicMaterial(color='blue')
    )

    axes.add(x_axis)
    axes.add(y_axis)
    axes.add(z_axis)
    return axes


# 使用示例
input_file_path = 'mesh/textured_mesh.obj'  # 输入的 OBJ 文件路径

# 加载原始网格
vertices, faces = load_obj(input_file_path)

# 创建网格
geometry = p3.BufferGeometry(
    attributes={
        'position': p3.BufferAttribute(
            array=vertices.flatten().astype(np.float32),
            normalized=False
        )
    }
)

# 创建面
indices = np.array(faces.flatten(), dtype=np.uint32)
geometry.setIndex(p3.BufferAttribute(array=indices, normalized=False))

material = p3.MeshBasicMaterial(color='lightgrey', side=p3.DoubleSide)
mesh = p3.Mesh(geometry=geometry, material=material)

# 创建场景
scene = p3.Scene(children=[mesh, create_axes()])

# 创建相机
camera = p3.PerspectiveCamera(position=[3, 3, 3], fov=75)

# 创建渲染器
renderer = p3.Renderer(camera=camera, scene=scene, controls=[p3.OrbitControls(controlling=camera)], width=800,
                       height=600)

# 显示
renderer
