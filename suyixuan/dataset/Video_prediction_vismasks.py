import cv2
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from sam2.build_sam import build_sam2_video_predictor

# 加载模型的检查点和配置文件
checkpoint = "../checkpoints/sam2.1_hiera_large.pt"
model_cfg = "../sam2/configs/sam2.1/sam2.1_hiera_l.yaml"
predictor = build_sam2_video_predictor(model_cfg, checkpoint)

# 读取视频文件
cap = cv2.VideoCapture('video.mp4')

# 检查视频是否成功打开
if not cap.isOpened():
    print("Error: Unable to open video.")
    exit()

# 获取视频的总帧数
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# 确保 frame_idx 在视频帧的范围内
frame_idx = 0  # 假设我们需要获取第 0 帧
if frame_idx >= total_frames:
    print("Error: Frame index out of range.")
    exit()

# 设置视频读取的位置为 frame_idx
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

# 读取指定帧
ret, frame = cap.read()
if not ret:
    print(f"Error: Unable to read frame {frame_idx}.")
    exit()

# 将预测模型的推理状态初始化
state = predictor.init_state("/home/suyixuan/AI/Pose_Estimation/sam2/data/robot_mustard.mp4")

# 进行推理，获取掩码
bounding_box = (100, 260, 175, 345)  # 示例边界框

frame_idx, object_ids, masks = predictor.add_new_points_or_box(
    state,
    frame_idx,
    obj_id=1,
    clear_old_points=True,
    normalize_coords=True,
    box=bounding_box
)


# 函数：高亮掩码区域
def highlight_masked_area(frame, masks, threshold=0.5, color=(128, 0, 128), alpha=0.5):
    mask_np = masks.squeeze().cpu().numpy()
    highlighted_frame = frame.copy()
    highlighted_frame[mask_np > threshold] = np.array(color)
    final_frame = cv2.addWeighted(frame, 1 - alpha, highlighted_frame, alpha, 0)

    plt.figure(figsize=(10, 6))
    plt.imshow(cv2.cvtColor(final_frame, cv2.COLOR_BGR2RGB))
    plt.title("Highlighted Mask Area")
    plt.axis('off')
    plt.show()


# 函数：可视化掩码
def visualize_mask_blackwhite(frame_idx, masks):
    mask_np = masks.squeeze().cpu().numpy()
    plt.figure(figsize=(10, 6))
    plt.imshow(mask_np, cmap='gray')
    plt.title(f"Frame {frame_idx} Mask")
    plt.axis('off')
    plt.show()


def visualize_mask(frame_idx, masks):
    mask_np = masks.squeeze().cpu().numpy()
    output_image = np.zeros_like(mask_np)
    output_image[mask_np > 0.5] = 255
    plt.figure(figsize=(10, 6))
    plt.imshow(output_image, cmap='Purples')
    plt.title(f"Frame {frame_idx} Mask")
    plt.axis('off')
    plt.show()


# 函数：保存掩码到文件
def save_mask_to_file_blackwhite(frame_idx, masks, output_dir="output_masks12"):
    os.makedirs(output_dir, exist_ok=True)
    mask_np = masks.squeeze().cpu().numpy()
    mask_filename = os.path.join(output_dir, f"frame_{frame_idx}_mask_blackwhite.png")
    cv2.imwrite(mask_filename, mask_np * 255)  # 转换为 0-255 范围
    print(f"Saved mask for frame {frame_idx} to {mask_filename}")


def save_highlighted_mask(frame, masks, output_dir="output_masks12", frame_idx=0, threshold=0.5, color=(128, 0, 128),
                          alpha=0.5):
    os.makedirs(output_dir, exist_ok=True)
    mask_np = masks.squeeze().cpu().numpy()
    highlighted_frame = frame.copy()
    highlighted_frame[mask_np > threshold] = np.array(color)
    final_frame = cv2.addWeighted(frame, 1 - alpha, highlighted_frame, alpha, 0)

    mask_filename = os.path.join(output_dir, f"frame_{frame_idx}_highlighted_mask.png")
    cv2.imwrite(mask_filename, final_frame)
    print(f"Saved highlighted mask for frame {frame_idx} to {mask_filename}")


# 处理每一帧的掩码
for frame_idx, object_ids, masks in predictor.propagate_in_video(state, start_frame_idx=0, max_frame_num_to_track=5):
    # 从视频中读取指定帧
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret:
        print(f"Error: Unable to read frame {frame_idx}.")
        continue

    # 进行后续处理
    print(f"Processing Frame {frame_idx}")

    # 1. 可视化掩码
    visualize_mask_blackwhite(frame_idx, masks)
    visualize_mask(frame_idx, masks)
    highlight_masked_area(frame, masks)

    # 2. 保存掩码到文件
    save_mask_to_file_blackwhite(frame_idx, masks)
    save_highlighted_mask(frame, masks)

    # 3. 进行后续处理：例如计算掩码的面积
    mask_area = masks.squeeze().cpu().numpy().sum()
    print(f"Mask area for frame {frame_idx}: {mask_area} pixels")

# 关闭视频文件
cap.release()
