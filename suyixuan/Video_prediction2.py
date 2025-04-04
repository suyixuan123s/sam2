"""
# Time： 2025/2/21 19:53
# Author： Yixuan Su
# File： demo3_highlighted_frames.py
# IDE： PyCharm
# Motto：ABC(Never give up!)
# Description：
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2

from sam2.build_sam import build_sam2_video_predictor
import os


# 加载模型的检查点和配置文件
checkpoint = "../checkpoints/sam2.1_hiera_large.pt"
model_cfg = "../sam2/configs/sam2.1/sam2.1_hiera_l.yaml"
predictor = build_sam2_video_predictor(model_cfg, checkpoint)

# 1. 可视化掩码
# 使用 matplotlib 或 OpenCV 将掩码可视化。


def highlight_masked_area(frame, masks, threshold=0.5, color=(128, 0, 128), alpha=0.5):
    """
    将掩码区域高亮显示
    :param frame: 原始图像
    :param masks: 掩码张量 (shape: [1, 1, H, W])
    :param threshold: 掩码值的阈值，用于确定哪些区域需要高亮
    :param color: 高亮颜色 (BGR)
    :param alpha: 掩码区域与原图的透明度混合比例
    """
    # 将掩码转换为 numpy 数组
    mask_np = masks.squeeze().cpu().numpy()  # 去除多余的维度并转换为 numpy 数组

    # 创建一个与原图大小相同的背景图像
    highlighted_frame = frame.copy()


    # 找到掩码区域，并将其覆盖为指定颜色
    highlighted_frame[mask_np > threshold] = np.array(color)  # 高亮区域

    # 混合图像，将掩码区域以透明度 alpha 与原图合成
    final_frame = cv2.addWeighted(frame, 1 - alpha, highlighted_frame, alpha, 0)

    # 显示结果
    plt.figure(figsize=(10, 6))
    plt.imshow(cv2.cvtColor(final_frame, cv2.COLOR_BGR2RGB))
    plt.title("Highlighted Mask Area")
    plt.axis('off')
    plt.show()


def visualize_mask_blackwhite(frame_idx, masks):
    """
    可视化掩码
    :param frame_idx: 当前帧的索引
    :param masks: 掩码张量 (shape: [1, 1, H, W])
    """
    # 将掩码转换为 numpy 数组
    mask_np = masks.squeeze().cpu().numpy()  # 去除多余的维度并转换为 numpy 数组

    # 可视化掩码
    plt.figure(figsize=(10, 6))
    plt.imshow(mask_np, cmap='gray')
    plt.title(f"Frame {frame_idx} Mask")
    plt.axis('off')
    plt.show()


def visualize_mask(frame_idx, masks):
    """
    可视化掩码并应用颜色映射
    :param frame_idx: 当前帧的索引
    :param masks: 掩码张量 (shape: [1, 1, H, W])
    """
    # 将掩码转换为 numpy 数组
    mask_np = masks.squeeze().cpu().numpy()  # 去除多余的维度并转换为 numpy 数组

    # 创建一个空白背景
    output_image = np.zeros_like(mask_np)

    # 设置紫色区域 (根据掩码值为 1 的区域)
    output_image[mask_np > 0.5] = 255  # 设置高亮区域

    # 可视化
    plt.figure(figsize=(10, 6))
    plt.imshow(output_image, cmap='Purples')  # 使用紫色渐变
    plt.title(f"Frame {frame_idx} Mask")
    plt.axis('off')
    plt.show()



# 2. 保存掩码到文件
# 将掩码保存为图像文件。
def save_mask_to_file_blackwhite(frame_idx, masks, output_dir="output_masks12"):
    """
    将掩码保存为图像文件
    :param frame_idx: 当前帧的索引
    :param masks: 掩码张量 (shape: [1, 1, H, W])
    :param output_dir: 保存掩码的目录
    """
    # 创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)

    # 将掩码转换为 numpy 数组
    mask_np = masks.squeeze().cpu().numpy()  # 去除多余的维度并转换为 numpy 数组


    # 将掩码保存为图像文件
    mask_filename = os.path.join(output_dir, f"frame_{frame_idx}_mask_balckwhite.png")
    cv2.imwrite(mask_filename, mask_np * 255)  # 将掩码值从 [0, 1] 转换为 [0, 255]
    print(f"Saved mask for frame {frame_idx} to {mask_filename}")


def save_mask_to_file_visualization(frame_idx, masks, output_dir="output_masks12"):
    """
    将掩码保存为图像文件并加色
    :param frame_idx: 当前帧的索引
    :param masks: 掩码张量 (shape: [1, 1, H, W])
    :param output_dir: 保存掩码的目录
    """
    # 创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)

    # 将掩码转换为 numpy 数组
    mask_np = masks.squeeze().cpu().numpy()  # 去除多余的维度并转换为 numpy 数组

    # 创建一个空白背景
    output_image = np.zeros((mask_np.shape[0], mask_np.shape[1], 3), dtype=np.uint8)

    # 设置紫色区域 (根据掩码值为 1 的区域)
    output_image[mask_np > 0.5] = [128, 0, 128]  # 紫色

    # 将掩码保存为图像文件
    mask_filename = os.path.join(output_dir, f"frame_{frame_idx}_mask.png")
    cv2.imwrite(mask_filename, output_image)  # 保存为彩色掩码
    print(f"Saved mask for frame {frame_idx} to {mask_filename}")


def highlight_masked_area(frame, masks, threshold=0.5, color=(128, 0, 128), alpha=0.5):
    """
    将掩码区域高亮显示
    :param frame: 原始图像
    :param masks: 掩码张量 (shape: [1, 1, H, W])
    :param threshold: 掩码值的阈值，用于确定哪些区域需要高亮
    :param color: 高亮颜色 (BGR)
    :param alpha: 掩码区域与原图的透明度混合比例
    """
    # 将掩码转换为 numpy 数组
    mask_np = masks.squeeze().cpu().numpy()  # 去除多余的维度并转换为 numpy 数组

    # 创建一个与原图大小相同的背景图像
    highlighted_frame = frame.copy()

    # 找到掩码区域，并将其覆盖为指定颜色
    highlighted_frame[mask_np > threshold] = np.array(color)  # 高亮区域

    # 混合图像，将掩码区域以透明度 alpha 与原图合成
    final_frame = cv2.addWeighted(frame, 1 - alpha, highlighted_frame, alpha, 0)

    # 显示结果
    plt.figure(figsize=(10, 6))
    plt.imshow(cv2.cvtColor(final_frame, cv2.COLOR_BGR2RGB))
    plt.title("Highlighted Mask Area")
    plt.axis('off')
    plt.show()



def save_highlighted_mask(frame, masks, output_dir="output_masks12", frame_idx=0, threshold=0.5, color=(128, 0, 128),
                          alpha=0.5):
    """
    保存高亮掩码区域的分割结果
    :param frame: 原始图像
    :param masks: 掩码张量 (shape: [1, 1, H, W])
    :param output_dir: 保存结果的目录
    :param frame_idx: 当前帧的索引
    :param threshold: 掩码值的阈值
    :param color: 高亮颜色
    :param alpha: 掩码区域与原图的透明度
    """
    # 创建输出目录（如果不存在）
    os.makedirs(output_dir, exist_ok=True)

    # 将掩码转换为 numpy 数组
    mask_np = masks.squeeze().cpu().numpy()  # 去除多余的维度并转换为 numpy 数组

    # 创建一个与原图大小相同的背景图像
    highlighted_frame = frame.copy()

    # 找到掩码区域，并将其覆盖为指定颜色
    highlighted_frame[mask_np > threshold] = np.array(color)

    # 混合图像，将掩码区域与原图合成
    final_frame = cv2.addWeighted(frame, 1 - alpha, highlighted_frame, alpha, 0)

    # 保存图像
    mask_filename = os.path.join(output_dir, f"frame_{frame_idx}_highlighted_mask.png")
    cv2.imwrite(mask_filename, final_frame)
    print(f"Saved highlighted mask for frame {frame_idx} to {mask_filename}")


# 3. 进行后续处理
# 例如，计算掩码的面积或与其他掩码进行比较。

def process_mask(masks):
    """
    对掩码进行后续处理
    :param masks: 掩码张量 (shape: [1, 1, H, W])
    """
    # 将掩码转换为 numpy 数组
    mask_np = masks.squeeze().cpu().numpy()  # 去除多余的维度并转换为 numpy 数组

    # 计算掩码的面积（非零像素的数量）
    mask_area = mask_np.sum()
    print(f"Mask area: {mask_area} pixels")

    # 其他处理逻辑（例如与其他掩码进行比较）

# 在推理模式下进行计算
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    # 初始化推理状态
    state = predictor.init_state("/home/suyixuan/AI/Pose_Estimation/sam2/data/robot_mustard.mp4")

    # 假设 state 是之前初始化的推理状态
    frame_idx = 0  # 第一帧的索引
    obj_id = 1  # 对象 ID
    # points = torch.tensor([[480, 270], [400, 250]])  # 添加的点
    #
    # labels = torch.tensor([1, 1])  # 点的标签

    # box = np.array([140, 0, 550, 430])  # 示例边界框
    bounding_box = (100, 260, 175, 345)  # 示例边界框


    # 调用方法，添加点和边界框
    frame_idx, object_ids, masks = predictor.add_new_points_or_box(
        state,
        frame_idx,
        obj_id,
        clear_old_points=True,
        normalize_coords=True,
        box= bounding_box
    )

    # 传播提示以获取视频中的掩码
    for frame_idx, object_ids, masks in predictor.propagate_in_video(state, start_frame_idx=0, max_frame_num_to_track=5):
        # 在这里处理每一帧的掩码
        # 例如，可以将掩码可视化或保存到文件
        print(f"Frame Index: {frame_idx}, Object IDs: {object_ids}, Masks Shape: {masks.shape}")

        # 1. 可视化掩码
        visualize_mask_blackwhite(frame_idx, masks)
        visualize_mask(frame_idx, masks)
        highlight_masked_area(frame_idx, masks)

        # 2. 保存掩码到文件
        save_mask_to_file_blackwhite(frame_idx, masks)

        save_highlighted_mask(frame_idx, masks)


        # 3. 进行后续处理
        process_mask(masks)
        # 这里可以添加代码来处理掩码，例如：
        # 1. 可视化掩码
        # 2. 保存掩码到文件
        # 3. 进行后续处理
