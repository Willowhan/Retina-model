import numpy as np
import math
import cv2
import matplotlib.pyplot as plt


class BasicRetinaFilter:
    def __init__(self, NBrows, NBcolumns, parametersListSize, useProgressiveFilter=False):
        """
        BasicRetinaFilter 构造函数初始化滤波器，并根据输入参数设置滤波器的大小和其他参数。

        输入参数:
        NBrows (int): 图像的行数
        NBcolumns (int): 图像的列数
        parametersListSize (int): 参数列表的大小
        useProgressiveFilter (bool): 是否使用渐进滤波器
        """
        '''成员变量初始化'''
        # 初始化为一个大小为 NBrows x NBcolumns x 3的三维数组，用于存储滤波器的输出
        self._filterOutput = np.zeros((NBrows, NBcolumns, 3), dtype=np.float32)
        # 初始化为一个大小为 NBrows * NBcolumns * 3 的一维数组，用于存储局部缓冲区的数据。
        self._localBuffer = np.zeros(NBrows * NBcolumns * 3, dtype=np.float32)
        # 初始化为一个大小为 3 * parametersListSize 的一维数组，用于存储滤波器的系数表。
        self._filteringCoeficientsTable = np.zeros(3 * parametersListSize, dtype=np.float32)
        self._progressiveSpatialConstant = None
        self._progressiveGain = None
        # 分别初始化为图像行数和列数的一半。
        self._halfNBrows = NBrows // 2
        self._halfNBcolumns = NBcolumns // 2

        if useProgressiveFilter:
            self._progressiveSpatialConstant = np.zeros_like(self._filterOutput, dtype=np.float32)
            self._progressiveGain = np.zeros_like(self._filterOutput, dtype=np.float32)

        self._V0 = 0.9  # 亮度压缩参数
        self._maxInputValue = 255  # 最大输入值
        '''调用 clearAllBuffers 方法，清除所有缓冲区的数据，初始化为 0'''
        self.clearAllBuffers()

    def clearAllBuffers(self):
        """
        清空所有缓冲区
        """
        self._filterOutput.fill(0)
        self._localBuffer.fill(0)
        if self._progressiveSpatialConstant is not None:
            self._progressiveSpatialConstant.fill(0)
        if self._progressiveGain is not None:
            self._progressiveGain.fill(0)

    def resize(self, NBrows, NBcolumns):
        """
        调整滤波器大小并重新分配缓冲区

        参数:
        NBrows (int): 新的行数
        NBcolumns (int): 新的列数
        """
        self._filterOutput = np.zeros((NBrows, NBcolumns, 3), dtype=np.float32)
        self._localBuffer = np.zeros(NBrows * NBcolumns * 3, dtype=np.float32)
        self._halfNBrows = NBrows // 2
        self._halfNBcolumns = NBcolumns // 2

        if self._progressiveSpatialConstant is not None:
            self._progressiveSpatialConstant = np.zeros_like(self._filterOutput, dtype=np.float32)
            self._progressiveGain = np.zeros_like(self._filterOutput, dtype=np.float32)

        self.clearAllBuffers()


    def setLPfilterParameters(self, beta, tau, desired_k, filterIndex):
        """
         设置低通滤波器参数

         参数:
         beta (float): beta值
         tau (float): tau值
         desired_k (float): 期望的k值
         filterIndex (int): 滤波器索引
         """
        beta_tau = beta + tau
        k = max(desired_k, 0.001)  # 确保k值大于0以避免除零错误
        #alpha = 1 - math.exp(-2 * math.pi * k / self._halfNBrows)  # 根据需要调整 self._halfNBrows
        alpha = k * k
        mu = 0.8
        tableOffset = filterIndex * 3

        temp = (1.0 + beta_tau) / (2.0 * mu * alpha)
        a = 1.0 + temp - math.sqrt((1.0 + temp) * (1.0 + temp) - 1.0)
        # Calculate and store filter coefficients:
        self._filteringCoeficientsTable[tableOffset] = a
        self._filteringCoeficientsTable[tableOffset + 1] = (1.0 - a) ** 4 / (1.0 + beta_tau)
        self._filteringCoeficientsTable[tableOffset + 2] = tau

    def _spatiotemporalLPfilter(self, inputFrame, outputFrame, filterIndex):
        coefTableOffset = filterIndex * 3
        self._a = self._filteringCoeficientsTable[coefTableOffset]
        self._gain = self._filteringCoeficientsTable[coefTableOffset + 1]
        self._tau = self._filteringCoeficientsTable[coefTableOffset + 2]

        # Modification 1: Save the result of each filter operation in the intermediate_results list
        intermediate_results = [inputFrame.copy()]
        # Process each color channel
        for channel in range(3):
            self._horizontalCausalFilter_addInput(inputFrame[:, :, channel], outputFrame[:, :, channel], 0,
                                                  self._filterOutput.shape[0])
        intermediate_results.append(outputFrame.copy())
        # Both input and output are OutputFrames
        for channel in range(3):
            self._horizontalAnticausalFilter(outputFrame[:, :, channel], 0, self._filterOutput.shape[0])
        intermediate_results.append(outputFrame.copy())

        for channel in range(3):
            self._verticalCausalFilter(outputFrame[:, :, channel], 0, self._filterOutput.shape[1])
        intermediate_results.append(outputFrame.copy())

        for channel in range(3):
            self._verticalAnticausalFilter_multGain(outputFrame[:, :, channel], 0, self._filterOutput.shape[1])
        intermediate_results.append(outputFrame.copy())

        return intermediate_results

    def _horizontalCausalFilter_addInput(self, inputFrame, outputFrame, IDrowStart, IDrowEnd):
        # Specifies the range of rows to process行数
        for IDrow in range(IDrowStart, IDrowEnd):
            # 获取当前行的所有列
            inputPTR = inputFrame[IDrow * self._filterOutput.shape[1]: (IDrow + 1) * self._filterOutput.shape[1]]
            outputPTR = outputFrame[IDrow * self._filterOutput.shape[1]: (IDrow + 1) * self._filterOutput.shape[1]]
            result = 0
            # len(outputPTR) ：当前列数
            for index in range(len(outputPTR)):
                "y[n]=x[n]+τ⋅y[n]+a⋅y[n−1]"
                result = inputPTR[index] + self._tau * outputPTR[index] + self._a * result
                outputPTR[index] = result

    def _horizontalAnticausalFilter(self, outputFrame, IDrowStart, IDrowEnd):
        for IDrow in range(IDrowStart, IDrowEnd):
            outputPTR = outputFrame[(IDrowEnd - IDrow - 1) * self._filterOutput.shape[1]: (IDrowEnd - IDrow) * self._filterOutput.shape[1]]
            result = 0
            for index in range(len(outputPTR)):
                "y[n]=y[−(n+1)]+a⋅y[−(n+1)]"
                result = outputPTR[-(index + 1)] + self._a * result
                outputPTR[-(index + 1)] = result


    def _verticalCausalFilter(self, outputFrame, IDcolumnStart, IDcolumnEnd):
        for IDcolumn in range(IDcolumnStart, IDcolumnEnd):
            result = 0
            for IDrow in range(self._filterOutput.shape[0]):
                "y[i]=x[i]+a⋅y[i−1]"
                outputFrame[IDrow, IDcolumn] = outputFrame[IDrow, IDcolumn] + self._a * result
                result = outputFrame[IDrow, IDcolumn]

    def _verticalAnticausalFilter_multGain(self, outputFrame, IDcolumnStart, IDcolumnEnd):
        outputOffset = outputFrame[-self._filterOutput.shape[1]:]
        for IDcolumn in range(IDcolumnStart, IDcolumnEnd):
            result = 0
            outputPTR = outputOffset[:, IDcolumn]
            for index in range(self._filterOutput.shape[0]):
                "y[i]=gain⋅(x[i]+a⋅y[i+1])"
                result = outputPTR[index] + self._a * result
                outputPTR[index] = self._gain * result

    def calculateLocalLuminance(self, inputFrame):
        """Use LP filter to calculate the local brightness, which assumes a Gaussian filter"""
        localLuminance = np.zeros_like(
            inputFrame)  # “np.zeros_like”： same shape and type as the inputFrame, but with all elements set to zero
        # The local brightness is calculated using a low-pass filter, which assumes a Gaussian filter
        kernel_size = 15  # Resize the kernel
        for channel in range(3):
            localLuminance[:, :, channel] = cv2.GaussianBlur(inputFrame[:, :, channel], (kernel_size, kernel_size), 0)
        return localLuminance

    def _localLuminanceAdaptation(self, inputFrame, localLuminance, outputFrame, V0=None, updateLuminanceMean=False):
        """
        inputFrame：输入帧，是一个包含像素值的数组。
        localLuminance：局部亮度缓冲区，是一个包含局部亮度值的数组。
        outputFrame：输出帧，是一个将存储结果的数组。
        updateLuminanceMean：布尔值，指示是否更新平均亮度。v0=NONE,
        """

        if updateLuminanceMean:
            meanLuminance = np.mean(inputFrame)
            self.updateCompressionParameter(meanLuminance)

        if V0 is None:
            V0 = self._V0
        Vmax = self._maxInputValue  # Assume that maxInputValue is Vmax
        epsilon = 1e-10  # # Avoid dividing by a small value of zero

        # cal:  R0(p) = V0 * L(p) + Vmax * (1 - V0)
        R0 = V0 * localLuminance + Vmax * (1 - V0)

        # cal： C(p) = (R(p) / (R(p) + R0(p))) * Vmax
        for i in range(len(inputFrame)):
            R = inputFrame[i]
            outputFrame[i] = (R / (R + R0[i] + epsilon)) * Vmax

# test
if __name__ == "__main__":

    image = cv2.imread('E:/dissertation/simulation/image/Cat.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Create a BasicRetinaFilter instance
    NBrows, NBcolumns, _ = image.shape
    print(f"设置的 NBrows: {NBrows}, NBcolumns: {NBcolumns}")

    filter_instance = BasicRetinaFilter(NBrows, NBcolumns, 1)

    # Set the low-pass filter parameters
    beta = -0.44
    tau = 0.001
    desired_k = 0.1
    filterIndex = 0
    filter_instance.setLPfilterParameters(beta, tau, desired_k, filterIndex)

    # Prepare the input and output frames
    inputFrame = image.astype(np.float32)
    outputFrame = np.zeros_like(inputFrame, dtype=np.float32)
    #
    # # 运行 spatiotemporalLPfilter
    # filter_instance._spatiotemporalLPfilter(inputFrame, outputFrame, filterIndex)
    #
    # # 显示输入和输出图像
    # fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    # axes[0].imshow(inputFrame.astype(np.uint8))
    # axes[0].set_title('Input Image')
    # axes[0].axis('off')
    # axes[1].imshow(outputFrame.astype(np.uint8))
    # axes[1].set_title('Filtered Image')
    # axes[1].axis('off')
    # plt.show()
    intermediate_results = filter_instance._spatiotemporalLPfilter(inputFrame, outputFrame, filterIndex)

    plt.figure(figsize=(15, 10))

    plt.subplot(2, 3, 1)
    plt.imshow(inputFrame.astype(np.uint8))
    plt.title('Input Image')
    plt.axis('off')

    filter_steps = ['Horizontal Causal Filter', 'Horizontal Anti-Causal Filter', 'Vertical Causal Filter', 'Vertical Anti-Causal Filter']

    # for i, result in enumerate([intermediate_results[0], intermediate_results[1], intermediate_results[2], intermediate_results[3]]):
    for i, result in enumerate(intermediate_results[1:]):
        plt.subplot(2, 3, i + 2)
        plt.imshow(result.astype(np.uint8))
        plt.title(filter_steps[i])
        plt.axis('off')

    plt.tight_layout()
    plt.show()