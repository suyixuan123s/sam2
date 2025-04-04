"""
# Time： 2025/2/21 19:53
# Author： Yixuan Su
# File： demo3_highlighted_frames.py
# IDE： PyCharm
# Motto：ABC(Never give up!)
# Description：
"""
import torch
from sam2.build_sam import build_sam2_video_predictor
import cv2  # 导入 OpenCV 库，用于处理视频帧

checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
predictor = build_sam2_video_predictor(model_cfg, checkpoint)

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    video_path = "/home/suyixuan/AI/Pose_Estimation/sam2/notebooks/videos/bedroom.mp4"
    state = predictor.init_state(video_path)

    # 假设 state 是之前初始化的推理状态
    frame_idx = 0
    obj_id = 1
    points = torch.tensor([[100, 150], [200, 250]])  # 添加的点
    labels = torch.tensor([1, 1])  # 点的标签
    box = torch.tensor([90, 140, 210, 260])  # 边界框坐标

    # 调用方法，在第一帧添加提示
    frame_idx, object_ids, masks = predictor.add_new_points_or_box(
        state,
        frame_idx,
        obj_id,
        points=points,
        labels=labels,
        clear_old_points=True,
        normalize_coords=True,
        box=box
    )

    # propagate the prompts to get masklets throughout the video
    # 在视频的其余帧中传播提示，以获得分割掩码
    video = cv2.VideoCapture(video_path)  # 打开视频文件
    success, frame = video.read()  # 读取第一帧
    frame_count = 0  # 帧计数器

    for frame_idx, object_ids, masks in predictor.propagate_in_video(state, start_frame_idx=1):  # 从第二帧开始传播
        success, frame = video.read()  # 读取下一帧
        if not success:
            break  # 如果读取失败，则退出循环
        frame_count += 1

        # 在这里，你可以使用 masks 对当前帧进行处理
        # 例如，将掩码叠加到原始帧上，或者将掩码保存到文件中
        # 示例：将掩码叠加到原始帧上
        alpha = 0.5  # 透明度
        color = [0, 255, 0]  # 掩码颜色 (绿色)
        for mask in masks:
            mask = mask.cpu().numpy().astype(bool)  # 将掩码转换为 NumPy 数组
            frame[mask] = frame[mask] * (1 - alpha) + alpha * color  # 叠加掩码

        # 显示结果
        cv2.imshow("Segmented Frame", frame)
        cv2.waitKey(1)  # 等待 1 毫秒，以便显示图像

    video.release()  # 释放视频文件
    cv2.destroyAllWindows()  # 关闭所有窗口
