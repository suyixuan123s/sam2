import cv2
import numpy as np
import os

from demo import state
from sam2.benchmark import predictor


# 1. 可视化掩码
def visualize_mask(frame, mask, alpha=0.5):
    """
    将掩码叠加到原始帧上并可视化。
    :param frame: 原始帧 (H, W, 3)
    :param mask: 掩码 (H, W)
    :param alpha: 掩码透明度
    """
    # 将掩码转换为彩色图像
    mask_color = np.zeros_like(frame)
    mask_color[mask > 0] = [0, 255, 0]  # 绿色表示掩码区域

    # 将掩码叠加到原始帧上
    result = cv2.addWeighted(frame, 1, mask_color, alpha, 0)
    return result

# 2. 保存掩码到文件
def save_mask_to_file(mask, frame_idx, output_dir="output_masks"):
    """
    将掩码保存为图像文件。
    :param mask: 掩码 (H, W)
    :param frame_idx: 帧索引
    :param output_dir: 输出目录
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 将掩码保存为PNG文件
    mask_filename = os.path.join(output_dir, f"mask_frame_{frame_idx:04d}.png")
    cv2.imwrite(mask_filename, mask * 255)  # 将掩码值从0-1转换为0-255
    print(f"Saved mask for frame {frame_idx} to {mask_filename}")

# 3. 进行后续处理
def post_process_mask(mask):
    """
    对掩码进行后续处理，例如去除噪声、填充空洞等。
    :param mask: 掩码 (H, W)
    :return: 处理后的掩码
    """
    # 去除小噪声区域
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # 填充掩码中的空洞
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask

# 在 propagate_in_video 循环中处理每一帧的掩码
for frame_idx, object_ids, masks in predictor.propagate_in_video(state, start_frame_idx=0, max_frame_num_to_track=5):
    # 获取当前帧
    frame = state["frames"][frame_idx]  # 假设 state["frames"] 存储了所有帧

    # 处理每个对象的掩码
    for obj_id, mask in zip(object_ids, masks):
        # 将掩码转换为 numpy 数组
        mask_np = mask.squeeze().cpu().numpy()  # (H, W)

        # 1. 可视化掩码
        visualized_frame = visualize_mask(frame, mask_np)
        cv2.imshow(f"Frame {frame_idx} - Object {obj_id}", visualized_frame)
        cv2.waitKey(1)  # 显示1毫秒

        # 2. 保存掩码到文件
        save_mask_to_file(mask_np, frame_idx)

        # 3. 进行后续处理
        processed_mask = post_process_mask(mask_np)
        # 可以进一步处理或保存处理后的掩码

# 关闭所有 OpenCV 窗口
cv2.destroyAllWindows()
