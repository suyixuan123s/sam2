import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import os

from sam2.build_sam import build_sam2_video_predictor


def visualize_mask_with_original(frame, masks, alpha=0.5, color=[0, 0, 255]):
    """
    在原图上可视化掩码，只有掩码区域显示为半透明颜色
    """
    if frame is None or masks is None:
        print("Error: Invalid frame or mask")
        return None

    try:
        # 将掩码转换为numpy数组并确保尺寸匹配
        mask_np = masks.squeeze().cpu().numpy()
        h, w = frame.shape[:2]
        mask_resized = cv2.resize(mask_np, (w, h), interpolation=cv2.INTER_NEAREST)

        # 创建结果图像（初始为原图的复制）
        # result = frame.copy()

        # 只在掩码为1的区域应用颜色混合
        mask_area = (mask_resized > 0.5)  # 使用阈值来确定掩码区域
        overlay = frame.copy()
        overlay[mask_area] = color

        # 使用cv2.addWeighted进行混合
        result = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        # 显示结果
        plt.figure(figsize=(10, 6))
        plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.show()

        return result

    except Exception as e:
        print(f"Error in visualization: {e}")
        return None


# 初始化模型
checkpoint = "../checkpoints/sam2.1_hiera_large.pt"
model_cfg = "../sam2/configs/sam2.1/sam2.1_hiera_l.yaml"
predictor = build_sam2_video_predictor(model_cfg, checkpoint)

# 视频路径和输出目录
video_path = "/home/suyixuan/AI/Pose_Estimation/sam2/data/vidoe_data/10ml_metal_rack.mp4"
highlight_dir = "/home/suyixuan/AI/Pose_Estimation/sam2/data/out_data/10ml_metal_rack/highlighted_frames_10ml_metal_rack"
os.makedirs(highlight_dir, exist_ok=True)

# 初始化视频捕获对象
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise ValueError("无法打开视频文件")

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    # 初始化推理状态
    state = predictor.init_state(video_path)

    # 添加初始交互（示例使用边界框）
    frame_idx = 0
    obj_id = 1

    # bounding_box = (530, 255, 770, 450)  # 格式：(x_min, y_min, x_max, y_max)
    # bounding_box = (740, 592, 913, 440)  # 替换为你的边界框坐标
    # bounding_box = (753, 588, 892, 453)  # 替换为你的边界框坐标
    # bounding_box = (900, 426, 1065, 215)  # 替换为你的边界框坐标
    # bounding_box = (140, 0, 550, 430)  # 格式：(x_min, y_min, x_max, y_max)
    # bounding_box = (754, 588, 890, 450)  # 替换为你的边界框坐标

    # # 定义点的提示，格式为 (x, y) 10ml_rack1
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

    # 进行视频传播
    for frame_idx, object_ids, masks in predictor.propagate_in_video(state, start_frame_idx=0):
        # 读取对应帧的原始图像
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            print(f"无法读取帧 {frame_idx}")
            continue

        # 可视化并保存结果
        result = visualize_mask_with_original(frame, masks, alpha=0.5, color=[0, 0, 255])

        if result is not None:
            # 保存结果
            output_path = os.path.join(highlight_dir, f"frame_{frame_idx:04d}_highlight.jpg")
            cv2.imwrite(output_path, result)
            print(f"已保存高亮帧：{output_path}")

# 释放资源
cap.release()
