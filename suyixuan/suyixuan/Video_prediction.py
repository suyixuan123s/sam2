import numpy as np
import torch

from sam2.build_sam import build_sam2_video_predictor

# 加载模型
checkpoint = "../checkpoints/sam2.1_hiera_large.pt"
model_cfg = "../sam2/configs/sam2.1/sam2.1_hiera_l.yaml"
predictor = build_sam2_video_predictor(model_cfg, checkpoint)

# 初始化状态
video_path = "/data/vidoe_data/realsense_video_20250327-143551.mp4"
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    state = predictor.init_state(video_path)

    # 准备输入提示
    # point_coords = np.array([[100, 150], [200, 250]])  # 示例点坐标
    # point_labels = np.array([1, 0])  # 第一个点是前景，第二个点是背景
    box = np.array([753, 588, 892, 453])  # 示例边界框
    # bounding_box = (753, 588, 892, 453)  # 替换为你的边界框坐标

    # 添加新提示并获取输出
    # frame_idx, object_ids, masks = predictor.add_new_points_or_box(state, point_coords=point_coords, point_labels=point_labels, box=box)
    frame_idx, object_ids, masks = predictor.add_new_points_or_box(state, frame_idx=0, obj_id=5, box=box)
    # 在视频中传播提示
    for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
        # 处理每一帧的掩码
        print(f"Frame: {frame_idx}, Object IDs: {object_ids}, Masks: {masks}")
