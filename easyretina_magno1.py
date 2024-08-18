import numpy as np
import cv2
"""
视频初始时由于时域滤波中previous_input 和 amacrine_cells_temp_output 这两个初始状态的不稳定性，导致滤波器输出极端值
故初始几帧内采用渐进调整的方式逐步引入时域滤波,以消除初始时的黑色伪影
"""
'''
静态部分保留，动态部分标记为白色,但是产生的噪声未能消除
噪声产生的原因：
使用绝对值的副作用
在 np.abs(magnoXonPixelResult) 处使用了绝对值操作，这可能导致正负噪声值都被放大并保留在最终输出中。虽然这在某些情况下有助于保留边缘信息，但它也会保留和放大噪声。

量化误差和数据类型转换
在计算过程中，从 np.int16 转换为 np.uint8 可能会导致一些量化误差和数据丢失，这在某些情况下可能会引入额外的噪声。
'''
class VideoProcessor:
    def __init__(self, temporal_coefficient, warmup_frames=100, threshold=20, noise_removal_size=3):
        self.temporal_coefficient = temporal_coefficient
        self.previous_input_on = None
        self.amacrine_cells_temp_output_on = None
        self.previous_input_off = None
        self.amacrine_cells_temp_output_off = None
        self.warmup_frames = warmup_frames
        self.current_frame_index = 0
        self.threshold = threshold
        self.noise_removal_size = noise_removal_size  # 用于去噪的核大小

    def apply_temporal_filter(self, current_frame, previous_input, amacrine_cells_temp_output):
        # 计算高通时域滤波器的输出
        magnoXonPixelResult = (current_frame - previous_input).astype(np.int16)
        # 更新前一时刻输入
        previous_input = current_frame.astype(np.uint8)
        # 确保输出的每个像素值非负
        amacrine_cells_temp_output = np.where(np.abs(magnoXonPixelResult) > self.threshold, np.abs(magnoXonPixelResult), 20).astype(np.uint8)

        return amacrine_cells_temp_output, previous_input

    def process_magno_channel(self, BiplorOn_enhanced, BiplorOff_enhanced, frame):
        # 初始化 previous_input 和 amacrine_cells_temp_output
        if self.previous_input_on is None:
            self.previous_input_on = np.zeros_like(BiplorOn_enhanced)
            self.amacrine_cells_temp_output_on = np.zeros_like(BiplorOn_enhanced)
        if self.previous_input_off is None:
            self.previous_input_off = np.zeros_like(BiplorOff_enhanced)
            self.amacrine_cells_temp_output_off = np.zeros_like(BiplorOff_enhanced)

        # 应用时域滤波
        self.amacrine_cells_temp_output_on, self.previous_input_on = self.apply_temporal_filter(
            BiplorOn_enhanced, self.previous_input_on, self.amacrine_cells_temp_output_on
        )
        self.amacrine_cells_temp_output_off, self.previous_input_off = self.apply_temporal_filter(
            BiplorOff_enhanced, self.previous_input_off, self.amacrine_cells_temp_output_off
        )

        # 空间滤波
        kernel = np.array([[0, 1, 0],
                           [1, 2, 1],
                           [0, 1, 0]], dtype=np.float32)
        amacrineOn_filtered = cv2.filter2D(self.amacrine_cells_temp_output_on, -1, kernel)
        amacrineOff_filtered = cv2.filter2D(self.amacrine_cells_temp_output_off, -1, kernel)

        # 伽玛校正增强
        gamma = 0.8
        frame_gamma_corrected = np.clip((frame / 255.0) ** gamma * 255.0, 0, 255).astype(np.uint8)
        amacrineOn_enhanced = np.clip(amacrineOn_filtered + 0.2 * frame_gamma_corrected, 0, 255).astype(np.uint8)
        amacrineOff_enhanced = np.clip(amacrineOff_filtered + 0.2 * frame_gamma_corrected, 0, 255).astype(np.uint8)

        # 将增强后的 ON 和 OFF 通道叠加以生成 Magno 通道输出
        magno_output = amacrineOn_enhanced + amacrineOff_enhanced

        # 去除噪点
        magno_output = self.apply_median_blur(magno_output)

        return magno_output, amacrineOn_filtered, amacrineOff_enhanced

    # 增加噪点去除的力度（增加阈值或者多次开运算）
    def remove_noise(self, frame):
        kernel = np.ones((self.noise_removal_size, self.noise_removal_size), np.uint8)
        frame = cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel)
        frame = cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel)  # 多次开运算
        return frame
    # def remove_noise(self, frame):
    #     kernel = np.ones((5, 5), np.uint8)  # 将结构元素大小增加到 7x7 或更大
    #     frame = cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel, iterations=2)  # 增加迭代次数到 3 次
    #     # frame = cv2.morphologyEx(frame, cv2.MORPH_CLOSE, kernel)  # 结合闭运算
    #     # frame = cv2.bilateralFilter(frame, d=9, sigmaColor=75, sigmaSpace=75)  # 双边滤波
    #     return frame
    def apply_median_blur(self, frame):
        return cv2.medianBlur(frame, 5)  # 中值滤波，核大小为 5

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
    return parvo,parvo_brightened, BiplorOn_enhanced, BiplorOff_enhanced

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
    processor = VideoProcessor(temporal_coefficient, threshold=threshold, noise_removal_size=noise_removal_size)

    frame_count = 0
    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            print("No frame available")
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        parvo, parvo_brightened, BiplorOn_enhanced, BiplorOff_enhanced = process_frame(frame_gray)
        magno_output, amacrineOn_filtered, amacrineOff_enhanced = processor.process_magno_channel(BiplorOn_enhanced, BiplorOff_enhanced, frame_gray)

        if frame_count % 5 == 0:
            cv2.imwrite(f'output_frame_{frame_count}.png', magno_output)

        cv2.imshow('Original Video', frame)
        cv2.imshow('Magno output', magno_output)

        frame_count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

video_path = 'E:/dissertation/simulation/video/moving_ball.mp4'
process_video(video_path)
