import cv2
import numpy as np

def draw_bounding_box(image_path, box_coordinates, color=(0, 255, 0), thickness=2):
    """
    在图像上绘制一个指定颜色的边界框。

    Args:
        image_path (str): 图像文件的路径。
        box_coordinates (tuple): 包含边界框四个点的坐标的元组 (x1, y1, x2, y2)。
                                 (x1, y1) 是左上角坐标，(x2, y2) 是右下角坐标。
        color (tuple): 边界框的颜色，默认为绿色 (0, 255, 0)。
                       颜色格式为 BGR (Blue, Green, Red)。
        thickness (int): 边界框的线条粗细，默认为 2 像素。
    """
    try:
        # 读取图像
        img = cv2.imread(image_path)

        # 检查图像是否成功读取
        if img is None:
            print(f"无法读取图像文件: {image_path}")
            return

        # 提取边界框坐标
        x1, y1, x2, y2 = box_coordinates

        # 在图像上绘制矩形
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

        # 显示图像
        cv2.imshow("Image with Bounding Box", img)
        cv2.waitKey(0)  # 等待按键按下
        cv2.destroyAllWindows()  # 关闭所有窗口

    except FileNotFoundError:
        print(f"文件未找到: {image_path}")
    except Exception as e:
        print(f"发生错误: {e}")

# 示例用法
if __name__ == '__main__':
    image_file = "color_image_20250415-145851.jpg"  # 替换为你的图像文件路径
    # bounding_box = (140, 0, 550, 430)  # 替换为你的边界框坐标
    # bounding_box = (743, 595, 909, 443)  # 替换为你的边界框坐标 10ml_rack1.mp4
    # bounding_box = (900, 426, 1065, 215)  # 替换为你的边界框坐标
    # bounding_box = (890, 435, 1070, 220)  # 替换为你的边界框坐标 10ml_metal_rack.mp4
    bounding_box = (720, 552, 875, 391)  # 替换为你的边界框坐标 10ml_rack_0408.mp4
    box_color = (1, 0, 0)  # 蓝色
    line_thickness = 2

    draw_bounding_box(image_file, bounding_box, box_color, line_thickness)
