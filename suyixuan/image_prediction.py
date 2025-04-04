import numpy as np
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

checkpoint = "../checkpoints/sam2.1_hiera_large.pt"
model_cfg = "../sam2//configs/sam2.1/sam2.1_hiera_l.yaml"
predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

# 输入图片
# video_path = "/home/suyixuan/AI/Pose_Estimation/sam2/output_frames1/frame_000000.png"

with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    predictor.set_image("/home/suyixuan/AI/Pose_Estimation/sam2/output_frames1/frame_000000.png")

    # 准备输入提示
    # point_coords = np.array([[100, 150], [200, 250]])  # 示例点坐标
    # point_labels = np.array([1, 0])  # 第一个点是前景，第二个点是背景
    # box = np.array([50, 100, 300, 400])  # 示例边界框

    box = np.array([140, 0, 550, 430])  # 示例边界框
    masks, _, _ = predictor.predict(box=box)
    print(masks.shape)  # (1, 430, 550)




