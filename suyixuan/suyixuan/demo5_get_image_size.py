import os
from PIL import Image  # 导入 Pillow 库

def get_image_size(image_path):
    """
    计算一张图像的大小（宽度和高度）。

    Args:
        image_path (str): 图像文件的路径。

    Returns:
        tuple: 包含图像宽度和高度的元组 (width, height)。
               如果无法打开图像文件，则返回 None。
    """
    try:
        # 使用 Pillow 库打开图像文件
        img = Image.open(image_path)

        # 获取图像的宽度和高度
        width, height = img.size

        return (width, height)

    except FileNotFoundError:
        print(f"文件未找到: {image_path}")
        return None
    except Exception as e:
        print(f"发生错误: {e}")
        return None

# 示例用法
if __name__ == '__main__':
    image_file = "/home/suyixuan/AI/Pose_Estimation/sam2/suyixuan/color_image_20250325-184212-010998.jpg"  # 替换为你的图像文件路径

    frame_size = get_image_size(image_file)

    if frame_size:
        width, height = frame_size
        print(f"图像大小: 宽度 = {width}, 高度 = {height}")
    else:
        print("无法获取图像大小。")
