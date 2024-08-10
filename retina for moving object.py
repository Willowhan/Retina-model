import numpy as np
import cv2
import time

class VideoProcessor:
    def __init__(self, temporal_coefficient,warmup_frames=10):
        self.temporal_coefficient = temporal_coefficient
        self.previous_input_on = None
        self.amacrine_cells_temp_output_on = None
        self.previous_input_off = None
        self.amacrine_cells_temp_output_off = None
        self.warmup_frames = warmup_frames # 时域滤波前几帧乱码
        self.current_frame_index = 0 # 时域滤波前几帧

    # def apply_temporal_filter(self, current_frame, previous_input, amacrine_cells_temp_output):
    #     amacrine_cells_temp = self.temporal_coefficient * (amacrine_cells_temp_output + current_frame - previous_input)
    #     amacrine_cells_temp_output = np.maximum(amacrine_cells_temp, 0)
    #     previous_input = current_frame
    #     return amacrine_cells_temp_output, previous_input
    def apply_temporal_filter(self, current_frame, previous_input, amacrine_cells_temp_output):
        current_coefficient = self.temporal_coefficient * min(1, self.current_frame_index / self.warmup_frames)
        amacrine_cells_temp = current_coefficient * (amacrine_cells_temp_output + current_frame - previous_input)
        amacrine_cells_temp_output = np.maximum(amacrine_cells_temp, 0)
        previous_input = current_frame
        return amacrine_cells_temp_output, previous_input

    def process_magno_channel(self, BiplorOn_enhanced, BiplorOff_enhanced, frame):
        # 初始化 previous_input 和 amacrine_cells_temp_output
        if self.previous_input_on is None:
            self.previous_input_on = np.zeros_like(BiplorOn_enhanced)
            self.amacrine_cells_temp_output_on = np.zeros_like(BiplorOn_enhanced)
        if self.previous_input_off is None:
            self.previous_input_off = np.zeros_like(BiplorOff_enhanced)
            self.amacrine_cells_temp_output_off = np.zeros_like(BiplorOff_enhanced)

        # 应用时域滤波器
        self.amacrine_cells_temp_output_on, self.previous_input_on = self.apply_temporal_filter(
            BiplorOn_enhanced, self.previous_input_on, self.amacrine_cells_temp_output_on
        )
        self.amacrine_cells_temp_output_off, self.previous_input_off = self.apply_temporal_filter(
            BiplorOff_enhanced, self.previous_input_off, self.amacrine_cells_temp_output_off
        )

        # 空间滤波器
        kernel = np.array([[0, 1, 0],
                           [1, 2, 1],
                           [0, 1, 0]], dtype=np.float32)
        # amacrineOn_filtered = cv2.filter2D(self.amacrine_cells_temp_output_on, -1, kernel)
        # amacrineOff_filtered = cv2.filter2D(self.amacrine_cells_temp_output_off, -1, kernel)
        # 空间滤波，使用指定的边界处理策略
        amacrineOn_filtered = cv2.filter2D(self.amacrine_cells_temp_output_on, -1, kernel, borderType=cv2.BORDER_REFLECT)
        amacrineOff_filtered = cv2.filter2D(self.amacrine_cells_temp_output_off, -1, kernel, borderType=cv2.BORDER_REFLECT)

        # 伽玛校正增强
        # gamma = 0.5
        # frame_gamma_corrected = np.clip((frame / 255.0) ** gamma * 255.0, 0, 255).astype(np.uint8)
        # amacrineOn_enhanced = np.clip(amacrineOn_filtered + 0.2 * frame_gamma_corrected, 0, 255).astype(np.uint8)
        # amacrineOff_enhanced = np.clip(amacrineOff_filtered + 0.2 * frame_gamma_corrected, 0, 255).astype(np.uint8)
        gamma = 1.2
        amacrineOn_enhanced = np.clip(((amacrineOn_filtered / 255.0) ** gamma) * 255 + 0.3 * frame, 0, 255).astype(
            np.uint8)
        amacrineOff_enhanced = np.clip(((amacrineOff_filtered / 255.0) ** gamma) * 255 + 0.3 * frame, 0, 255).astype(
            np.uint8)

        # 将增强后的 ON 和 OFF 通道叠加以生成 Magno 通道输出
        magno_output = amacrineOn_enhanced + amacrineOff_enhanced

        return magno_output

def diffuse(image, iterations):
    for _ in range(iterations):
        image = (np.roll(image, 1, axis=0) + np.roll(image, -1, axis=0) +
                 np.roll(image, 1, axis=1) + np.roll(image, -1, axis=1) + image) / 5
    return image

def adjust_pixel_range(image):
    adjusted_image = (image / 2 + 127).astype(np.uint8)
    return adjusted_image

def process_frame(frame):
    adjusted_image = adjust_pixel_range(frame)
    kernel = np.array([[0, 1, 0],
                       [1, 2, 1],
                       [0, 1, 0]], dtype=np.float32)
    B = cv2.filter2D(adjusted_image.astype(np.float32), -1, kernel)
    B1 = B / 4
    C = np.clip(frame.astype(np.float32) - 10, 0, 255)
    A = B / 6 + C
    A = np.where(A > 255, 230, A)
    A = np.clip(A, 0, 255)
    diffuse_iterations = 5
    diffuse_image = diffuse(A, diffuse_iterations)
    E = np.maximum(A - diffuse_image, 0)
    F = np.maximum(diffuse_image - A, 0)
    E_normalized, F_normalized = adjust_edges(E, F, adjusted_image, detail_weight=0.1)
    F_normalized = F_normalized + 10
    BiplorOn_enhanced = np.clip(E_normalized + adjusted_image * 0.1, 127, 255).astype(np.uint8)
    BiplorOff_enhanced = np.clip(F_normalized + adjusted_image * 0.2, 127, 255).astype(np.uint8)
    parvo = BiplorOn_enhanced - BiplorOff_enhanced
    parvo_brightened = np.clip(parvo + 128, 0, 255)
    return parvo_brightened, BiplorOn_enhanced, BiplorOff_enhanced

def adjust_edges(E, F, enhanced_image, detail_weight=0.1):
    E_normalized = normalize_image_to_127_255(E)
    E_normalized = np.clip(E_normalized + enhanced_image * detail_weight, 127, 255).astype(np.uint8)
    F_normalized = normalize_image_to_127_255(F)
    F_normalized = 255 - np.clip(F_normalized + enhanced_image * detail_weight, 0, 255).astype(np.uint8)
    return E_normalized, F_normalized

def normalize_image_to_127_255(image):
    image_min = image.min()
    image_max = image.max()
    image_normalized = 128 * (image - image_min) / (image_max - image_min) + 127
    return image_normalized.astype(np.uint8)

def process_video(video_path):
    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        print("Unable to open video")
        return

    temporal_coefficient = 0.9
    processor = VideoProcessor(temporal_coefficient)

    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            print("No frame available")
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        parvo_brightened, BiplorOn_enhanced, BiplorOff_enhanced = process_frame(frame_gray)
        magno_output = processor.process_magno_channel(BiplorOn_enhanced, BiplorOff_enhanced, frame_gray)

        cv2.imshow('Original Video', frame)
        cv2.imshow('Parvo output', parvo_brightened)
        cv2.imshow('Magno output', magno_output)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

video_path = 'E:/dissertation/simulation/video/moving_cat_gray.mp4'
process_video(video_path)
