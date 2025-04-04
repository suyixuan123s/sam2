import cv2
import os

def extract_frames(video_path, output_dir, frame_interval=1):
    """
    提取视频的每一帧图像，并保存为 PNG 格式。

    Args:
        video_path (str): 视频文件的路径。
        output_dir (str): 保存帧图像的目录。
        frame_interval (int): 提取帧的间隔。 默认为 1，表示提取每一帧。
                              如果设置为 2，则提取每隔一帧的图像，以此类推。
    """
    try:
        # 创建输出目录（如果不存在）
        os.makedirs(output_dir, exist_ok=True)

        # 打开视频文件
        cap = cv2.VideoCapture(video_path)

        # 检查视频是否成功打开
        if not cap.isOpened():
            print(f"无法打开视频文件: {video_path}")
            return

        # 获取视频的帧率
        fps = cap.get(cv2.CAP_PROP_FPS)

        # 获取视频的总帧数
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"视频帧率: {fps}")
        print(f"视频总帧数: {total_frames}")

        # 循环提取每一帧
        frame_count = 0
        success = True
        while success:
            # 读取下一帧
            success, image = cap.read()

            if success and frame_count % frame_interval == 0:
                # 构建输出文件名
                output_filename = os.path.join(output_dir, f"frame_{frame_count:06d}.png")

                # 保存帧图像为 PNG 格式
                cv2.imwrite(output_filename, image)
                print(f"已保存帧: {output_filename}")

            frame_count += 1

        # 释放视频对象
        cap.release()
        print("帧提取完成！")

    except Exception as e:
        print(f"发生错误: {e}")

# 示例用法
if __name__ == '__main__':
    video_file = "/home/suyixuan/AI/Pose_Estimation/sam2/data/vidoe_data/10ml_metal_rack.mp4"  # 替换为你的视频文件路径
    output_directory = "/home/suyixuan/AI/Pose_Estimation/sam2/data/out_data/10ml_metal_rack/output_frames_10ml_metalrack"  # 替换为你想要保存帧图像的目录
    frame_extraction_interval = 1  # 提取每一帧

    extract_frames(video_file, output_directory, frame_extraction_interval)
