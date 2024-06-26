import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_spectrum(ax, image, title, zlim):
    # 将图像转换为灰度图像
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # 对灰度图像进行傅里叶变换
    dft = cv.dft(np.float32(gray), flags=cv.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    # 计算频谱
    magnitude_spectrum = 20 * np.log(cv.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]) + 1)  # 加1避免log(0)

    # 绘制3D频谱图
    X = np.arange(magnitude_spectrum.shape[1])
    Y = np.arange(magnitude_spectrum.shape[0])
    X, Y = np.meshgrid(X, Y)

    ax.plot_surface(X, Y, magnitude_spectrum, cmap='jet')
    ax.set_title(title)
    ax.set_xlim(500, 0)  # 根据图二的x轴范围
    ax.set_ylim(0, 500)  # 根据图二的y轴范围
    ax.set_zlim(50, zlim)  # 设置z轴范围
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

# 图片路径
image_path = "E:\\dissertation\\simulation\\Lena.bmp"
#Lena.bmp noiseman.png

# 读取图像
input_image = cv.imread(image_path)

# 检查图像是否成功加载
if input_image is not None:
    # 输出图像尺寸
    height, width, channels = input_image.shape
    print(f"Image size (height, width, number of channels): ({height}, {width}, {channels})")

    # 输出图像数据类型
    print(f"Image data type: {input_image.dtype}")

    # 输出图像像素值范围
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(cv.cvtColor(input_image, cv.COLOR_BGR2GRAY))
    print(f"Pixel value range of grayscale image: {min_val} - {max_val}")

    # 输出每个通道的平均值和标准差
    means, std_devs = cv.meanStdDev(input_image)
    print(f"The average of each channel: B={means[0][0]:.2f}, G={means[1][0]:.2f}, R={means[2][0]:.2f}")
    print(
        f"Standard deviation for each channel: B={std_devs[0][0]:.2f}, G={std_devs[1][0]:.2f}, R={std_devs[2][0]:.2f}")
    # 创建视网膜实例
    retina = cv.bioinspired_Retina.create((input_image.shape[1], input_image.shape[0]))
    retina.write('retinaParams.xml')
    # 从 xml 文件加载视网膜参数：这里我们加载刚刚写入文件的默认参数
    retina.setup('retinaParams.xml')

    # 在输入图像上运行视网膜
    retina.run(input_image)

    # 抓取视网膜输出
    retina_out_parvo = retina.getParvo()
    retina_out_magno = retina.getMagno()

    # 创建绘图窗口和子图
    fig = plt.figure(figsize=(14, 6))

    # 绘制输入图像的频谱
    ax1 = fig.add_subplot(121, projection='3d')
    plot_spectrum(ax1, input_image, 'Input spectrum',200)
    #plot_spectrum(ax1, input_image, 'Input spectrum', 200)

    # 绘制视网膜输出的频谱
    ax2 = fig.add_subplot(122, projection='3d')
    #plot_spectrum(ax2, retina_out_parvo, 'Retina OPL spectrum output', 50)
    plot_spectrum(ax2, retina_out_parvo, 'Retina retina parvo output',200)
    plt.show()

else:
    print("Unable to load image")
