import numpy as np
import cv2
import matplotlib.pyplot as plt

def get_image(filepath):
    image = cv2.imread(filepath)
    # 调整图像尺寸为 256x256
    #image = cv2.resize(image, (256, 256))
    return image

#
# def diffuse(image, iterations):
#     for _ in range(iterations):
#         image = (np.roll(image, 1, axis=0) + np.roll(image, -1, axis=0) +
#                  np.roll(image, 1, axis=1) + np.roll(image, -1, axis=1) + image) / 5
#     return image

def diffuse(image,iterations, kappa, gamma, option=1):
    """
     参数：
      iterations：循环叠加滤波后图像
      kappa (float): 导电系数，控制对边缘的敏感度
      gamma (float): 控制扩散的速度
                     较大的会加快扩散速度，但也可能引起不稳定。稳定性最大值为0.25
     原理：
      计算4个方向的梯度值：区分边缘还是平坦区域
                       移位图像与原始图像的差值
      计算扩散系数：此系数与梯度和导电系数(kappa)有关；与梯度呈现负相关
                 计算方程体现扩散滤波的自适应性：平坦区域区域扩散强度大，边缘区域小，从而保留细节
      更新输出：计算梯度与扩散系数的加权和

    """
    image = image.astype(np.float32)
    diffused_image = image.copy()  # 结果初始为原始图像
    for _ in range(iterations):
        # 计算梯度
        deltaN = np.roll(diffused_image, -1, axis=0) - diffused_image  # 北方向梯度
        deltaS = np.roll(diffused_image, 1, axis=0) - diffused_image  # 南方向梯度
        deltaE = np.roll(diffused_image, -1, axis=1) - diffused_image  # 东方向梯度
        deltaW = np.roll(diffused_image, 1, axis=1) - diffused_image  # 西方向梯度

        # 计算梯度的模（大小）
        gradient_magnitude = np.sqrt(deltaN ** 2 + deltaS ** 2 + deltaE ** 2 + deltaW ** 2)

        # 计算扩散系数
        if option == 1:
            cN = np.exp(-(deltaN/kappa)**2)
            cS = np.exp(-(deltaS/kappa)**2)
            cE = np.exp(-(deltaE/kappa)**2)
            cW = np.exp(-(deltaW/kappa)**2)
        elif option == 2:
            cN = 1.0 / (1.0 + (deltaN/kappa)**2)
            cS = 1.0 / (1.0 + (deltaS/kappa)**2)
            cE = 1.0 / (1.0 + (deltaE/kappa)**2)
            cW = 1.0 / (1.0 + (deltaW/kappa)**2)

        # 计算更新图像
        diffused_image += gamma * (
            cN * deltaN + cS * deltaS + cE * deltaE + cW * deltaW
        )
    return diffused_image


def gaussian_blur(image, kernel_size, sigma):
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

def process_image(filepath):
    image = get_image(filepath)
    adjusted_image = adjust_pixel_range(image)

    kernel = np.array([[0, 1, 0],
                       [1, 2, 1],
                       [0, 1, 0]], dtype=np.float32)

    B = cv2.filter2D(adjusted_image.astype(np.float32), -1, kernel)
    B1 = B / 4


    C = np.clip(image.astype(np.float32) - 10, 0, 255)

    A = B / 6 + C # +C 保留一些细节

    # 将 A 的值限制在 0 到 255 之间，并将超过 255 的部分设为 128
    A = np.where(A > 255, 230, A)
    A = np.clip(A, 0, 255)


    diffuse_iterations = 5
    diffuse_image = diffuse(A, diffuse_iterations)

    E = np.maximum(A - diffuse_image, 0)  # 保留正值部分
    F = np.maximum(diffuse_image - A, 0)  # 保留负值部分

    return image, A, B, B1, diffuse_image, E, F

def adjust_pixel_range(image):
    """
    (image / 2 + 127):
      将图像像素值缩小一半,范围将变为 [0, 127.5]
      将所有像素值增加 127,范围将变为 [127,254.5 ]
    astype(np.uint8):
       将浮点型的图像数据转换为无符号 8 位整数
    """
    adjusted_image = (image / 2 + 127).astype(np.uint8) # 许多 OpenCV 函数要求输入图像为 uint8 类型
    return adjusted_image

def enhance_contrast(image):
    channels = cv2.split(image) # 将输入图像 image 分解成其独立的颜色通道
    eq_channels = [cv2.equalizeHist(channel) for channel in channels] # 每个通道分别进行直方图均衡化处理
    return cv2.merge(eq_channels) #均衡化后的通道重新合并成一个图像

def apply_clahe(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    channels = cv2.split(image)
    clahe_channels = [clahe.apply(channel) for channel in channels]
    return cv2.merge(clahe_channels)

def normalize_image_to_127_255(image):
    image_min = image.min()
    image_max = image.max()
    image_normalized = 128 * (image - image_min) / (image_max - image_min) + 127
    return image_normalized.astype(np.uint8)

def adjust_edges(E, F, original_image, detail_weight=0.1):
    E_normalized = normalize_image_to_127_255(E)
    E_normalized = np.clip(E_normalized + original_image * detail_weight, 127, 255).astype(np.uint8)

    F_normalized = normalize_image_to_127_255(F)
    F_normalized = 255 - np.clip(F_normalized + original_image * detail_weight, 0, 255).astype(np.uint8)

    return E_normalized, F_normalized

filepath = 'E:/dissertation/simulation/image/gray1.jpg'

image = get_image(filepath)   # cv2.imread
adjusted_image = adjust_pixel_range(image)   
enhanced_image = enhance_contrast(adjusted_image)
# clahe_image = apply_clahe(adjusted_image)

image, A, B, B1, diffuse_image, E, F = process_image(filepath)

# E_normalized, F_normalized = adjust_edges(E, F, clahe_image,detail_weight=0.6)
E_normalized, F_normalized = adjust_edges(E, F, enhanced_image,detail_weight=0.1)
F_normalized = F_normalized+10

BiplorOn_enhanced = np.clip(E_normalized+enhanced_image*0.1, 127, 255).astype(np.uint8)
BiplorOff_enhanced = np.clip(F_normalized+enhanced_image*0.2, 127, 255).astype(np.uint8)

parvo = BiplorOn_enhanced - BiplorOff_enhanced

# 增加 Parvo 图像的亮度
parvo_brightened = np.clip(parvo + 128, 0, 255)  # 你可以调整增加的亮度值，例如 50

print("Original Image - min:", image.min(), "max:", image.max(), "median:", np.median(image))
print("Adjust Image - min:", adjusted_image.min(), "max:", adjusted_image.max(), "median:", np.median(adjusted_image))
print("A Image - min:", A.min(), "max:", A.max(), "median:", np.median(A))
print("B Image - min:", B.min(), "max:", B.max(), "median:", np.median(B))
print("B1 Image - min:", B1.min(), "max:", B1.max(), "median:", np.median(B1))
print("diffuse Image - min:", diffuse_image.min(), "max:", diffuse_image.max(), "median:", np.median(diffuse_image))
print("E_normalized Image - min:", E_normalized.min(), "max:", E_normalized.max(), "median:", np.median(E_normalized))
print("F_normalized Image - min:", F_normalized.min(), "max:", F_normalized.max(), "median:", np.median(F_normalized))
print("BiplorOn Enhanced Image - min:", BiplorOn_enhanced.min(), "max:", BiplorOn_enhanced.max(), "median:", np.median(BiplorOn_enhanced))
print("BiplorOff Enhanced Image - min:", BiplorOff_enhanced.min(), "max:", BiplorOff_enhanced.max(), "median:", np.median(BiplorOff_enhanced))
print("Parvo Image - min:", parvo.min(), "max:", parvo.max(), "median:", np.median(parvo))
print("Parvo Brightened Image - min:", parvo_brightened.min(), "max:", parvo_brightened.max(), "median:", np.median(parvo_brightened))

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title('Input Image')

axes[0, 1].imshow(cv2.cvtColor(A.clip(0, 255).astype(np.uint8), cv2.COLOR_BGR2RGB), vmin=0, vmax=255)
axes[0, 1].set_title('Photoreceptors Output')


axes[0, 2].imshow(cv2.cvtColor(diffuse_image.clip(0, 255).astype(np.uint8), cv2.COLOR_BGR2RGB), vmin=0, vmax=255)
axes[0, 2].set_title('Horizontal Cell Output')

axes[1, 0].imshow(E_normalized, cmap='gray', vmin=127, vmax=255)
axes[1, 0].set_title('Bipolar ON')

axes[1, 1].imshow(F_normalized, cmap='gray', vmin=127, vmax=255)
axes[1, 1].set_title('Bipolar OFF')

axes[1, 2].imshow(parvo_brightened, cmap='gray', vmin=0, vmax=255)
axes[1, 2].set_title('Parvo Output Brightened')

for ax in axes.flatten():
    ax.axis('off')

plt.tight_layout()
plt.show()
