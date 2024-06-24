import cv2 as cv

# 图片路径
image_path = "E:\\dissertation\\simulation\\noiseman.png"

# 读取图像
inputImage = cv.imread(image_path)

# 检查图像是否成功加载
if inputImage is not None:
    # 输出图像尺寸
    height, width, channels = inputImage.shape
    print(f"Image size (height, width, number of channels): ({height}, {width}, {channels})")

    # 输出图像数据类型
    print(f"Image data type: {inputImage.dtype}")

    # 输出图像像素值范围
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(cv.cvtColor(inputImage, cv.COLOR_BGR2GRAY))
    print(f"Pixel value range of grayscale image: {min_val} - {max_val}")

    # 输出每个通道的平均值和标准差
    means, std_devs = cv.meanStdDev(inputImage)
    print(f"The average of each channel: B={means[0][0]:.2f}, G={means[1][0]:.2f}, R={means[2][0]:.2f}")
    print(f"Standard deviation for each channel: B={std_devs[0][0]:.2f}, G={std_devs[1][0]:.2f}, R={std_devs[2][0]:.2f}")

    # 创建视网膜实例
    retina = cv.bioinspired_Retina.create((inputImage.shape[1], inputImage.shape[0]))
    # 将默认参数保存到XML文件
    retina.write('retinaParams.xml')
    print("Default parameters have been saved to retinaParams.xml")

    # 手动修改retinaParams.xml文件...

    # 从修改后的XML文件加载参数
    retina.setup('retinaParams_modified.xml')
    print("已从 retinaParams_modified.xml Loading parameter test")

    # 在输入图像上运行视网膜
    retina.run(inputImage)

    # 抓取视网膜输出
    retinaOut_parvo = retina.getParvo()
    retinaOut_magno = retina.getMagno()

    # 显示原始图像
    cv.imshow('input image', inputImage)
    # 显示视网膜输出
    cv.imshow('retina parvo out', retinaOut_parvo)
    cv.imshow('retina magno out', retinaOut_magno)

    cv.waitKey(0)
    cv.destroyAllWindows()
else:
    print("无法加载图像")
