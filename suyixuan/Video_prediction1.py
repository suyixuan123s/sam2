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


# 2. 保存掩码到文件
# 将掩码保存为图像文件。
def save_mask_to_file_blackwhite(frame_idx, masks,
                                 output_dir="/home/suyixuan/AI/Pose_Estimation/sam2/data/out_data/10ml_metal_rack/output_masks_10ml_metalrack"):
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
    # state = predictor.init_state("/home/suyixuan/AI/Pose_Estimation/sam2/data/vidoe_data/10ml_metal_rack.mp4")
    state = predictor.init_state("/home/suyixuan/AI/Pose_Estimation/sam2/data/vidoe_data/10ml_metal_rack.mp4")

    # state = predictor.init_state("/home/suyixuan/AI/Pose_Estimation/sam2/data/demo1.mp4")
    # 假设 state 是之前初始化的推理状态
    frame_idx = 0  # 第一帧的索引
    obj_id = 1  # 对象 ID
    # points = torch.tensor([[480, 270], [400, 250]])  # 添加的点
    #
    # labels = torch.tensor([1, 1])  # 点的标签

    # bounding_box = np.array([140, 0, 550, 430])  # 示例边界框
    # bounding_box = np.array([530, 255, 770, 450])  # 示例边界框
    # bounding_box = (760, 580, 913, 440)  # 替换为你的边界框坐标
    # bounding_box = (912, 431, 1044, 228)  # 替换为你的边界框坐标
    # bounding_box = (743, 595, 909, 443)

    # bounding_box = (100, 260, 175, 345)  # 示例边界框

    # 定义点的提示，格式为 (x, y) 10ml_rack1
    # points = [(753, 458), (810, 455), (843, 536), (891, 585), (900, 452), (791, 446), (839, 592)]  # 替换为您的点坐标
    # labels = [1, 1, 1, 1, 0, 0, 0]  # 替换为与点对应的标签，假设这两个点属于同一类别

    # 定义点的提示，格式为 (x, y) 10ml_metal_rack
    points = [(933, 422), (963, 291), (1043, 427), (900, 284), (1009, 232), (900, 340), (1048, 259),
              (1061, 428)]  # 替换为您的点坐标
    labels = [1, 1, 1, 1, 1, 0, 0, 0]  # 替换为与点对应的标签，假设这两个点属于同一类别

    # 添加初始提示
    frame_idx, object_ids, masks = predictor.add_new_points_or_box(
        state,
        frame_idx,
        obj_id,
        points=points,  # 添加点的提示
        labels=labels,  # 添加点的标签
        clear_old_points=True,
        normalize_coords=True,
        # box=bounding_box
    )

    # # 调用方法，添加点和边界框
    # frame_idx, object_ids, masks = predictor.add_new_points_or_box(
    #     state,
    #     frame_idx,
    #     obj_id,
    #     clear_old_points=True,
    #     normalize_coords=True,
    #     box=bounding_box
    # )

    # 传播提示以获取视频中的掩码
    for frame_idx, object_ids, masks in predictor.propagate_in_video(state, start_frame_idx=0):
        # 在这里处理每一帧的掩码
        # 例如，可以将掩码可视化或保存到文件
        print(f"Frame Index: {frame_idx}, Object IDs: {object_ids}, Masks Shape: {masks.shape}")

        # 1. 可视化掩码
        visualize_mask_blackwhite(frame_idx, masks)

        # 2. 保存掩码到文件
        save_mask_to_file_blackwhite(frame_idx, masks)

        # 3. 进行后续处理
        process_mask(masks)
