import cv2

def get_video_frame_size(video_path):
    """
    获取视频每帧图像的大小。

    Args:
        video_path (str): 视频文件的路径。

    Returns:
        tuple: 包含视频宽度和高度的元组 (width, height)。
               如果无法打开视频文件，则返回 None。
    """
    try:
        # 打开视频文件
        cap = cv2.VideoCapture(video_path)

        # 检查视频是否成功打开
        if not cap.isOpened():
            print(f"无法打开视频文件: {video_path}")
            return None

        # 获取视频的宽度和高度
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 释放视频对象
        cap.release()

        return (width, height)

    except Exception as e:
        print(f"发生错误: {e}")
        return None

# 示例用法
if __name__ == '__main__':
    video_file = "/data/vidoe_data/realsense_video_20250325-184010 (1).mp4"  # 替换为你的视频文件路径
    frame_size = get_video_frame_size(video_file)

    if frame_size:
        width, height = frame_size
        print(f"视频帧大小: 宽度 = {width}, 高度 = {height}")
    else:
        print("无法获取视频帧大小。")
