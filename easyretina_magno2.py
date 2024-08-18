import cv2
import numpy as np
import os
'''
只提取动态部分，静态部分消失，但没有考虑on off
'''

class VideoProcessor:
    def __init__(self, temporal_coefficient, threshold=30, warmup_frames=10, noise_removal_size=5):
        self.temporal_coefficient = temporal_coefficient
        self.threshold = threshold
        self.noise_removal_size = noise_removal_size
        self.previous_input = None
        self.amacrine_cells_temp_output = None
        self.warmup_frames = warmup_frames  # 时域滤波前几帧乱码
        self.current_frame_index = 0  # 时域滤波前几帧

    def apply_temporal_filter(self, current_frame):
        if self.previous_input is None:
            self.previous_input = np.zeros_like(current_frame)
            self.amacrine_cells_temp_output = np.zeros_like(current_frame)

        # 计算当前帧和前一帧的差异
        # amacrine_cells_temp = (current_frame - self.previous_input).astype(np.int16)
        amacrine_cells_temp = current_frame.astype(np.int16) - self.previous_input.astype(np.int16)
        self.previous_input = current_frame

        # 只保留差异大于阈值的部分
        self.amacrine_cells_temp_output = np.where(np.abs(amacrine_cells_temp) > self.threshold,amacrine_cells_temp, 0).astype(np.uint8)
        return self.amacrine_cells_temp_output

    def remove_noise(self, image):
        # 使用形态学开运算去除小噪声
        kernel = np.ones((self.noise_removal_size, self.noise_removal_size), np.uint8)
        image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        return image

    def process_magno_channel(self, frame):
        # 应用时间域滤波器
        dynamic_part = self.apply_temporal_filter(frame)

        # 去掉负值部分，只保留运动的部分
        dynamic_part = np.maximum(dynamic_part, 0)

        # 去除噪声
        # dynamic_part = self.remove_noise(dynamic_part)

        return dynamic_part


def process_video(video_path):
    video_capture = cv2.VideoCapture(video_path)
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    print(f"fps/Frame Rate: {fps}")
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Total number of video frames: {total_frames}")

    if not video_capture.isOpened():
        print("Unable to open video")
        return

    temporal_coefficient = 0.9
    threshold = 50
    noise_removal_size = 3
    processor = VideoProcessor(temporal_coefficient, threshold, noise_removal_size=noise_removal_size)

    frame_counter = 0
    output_dir = "E:/dissertation/simulation/video/magnoframe"
    os.makedirs(output_dir, exist_ok=True)

    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            print("No frame available")
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        dynamic_part = processor.process_magno_channel(frame_gray)

        # 每隔5帧保存一张图片
        if frame_counter % 10 == 0:
            output_path = os.path.join(output_dir, f"frame_{frame_counter}.png")
            cv2.imwrite(output_path, dynamic_part)
            print(f"Saved {output_path}")

        cv2.imshow('Original Video', frame)
        cv2.imshow('Dynamic Parts', dynamic_part)

        if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
            break

        frame_counter += 1

    video_capture.release()
    cv2.destroyAllWindows()


video_path = 'E:/dissertation/simulation/video/moving_ball.mp4'
process_video(video_path)
