import numpy as np
from basic_retina_filter import BasicRetinaFilter
import cv2
import matplotlib.pyplot as plt

class ParvoRetinaFilter(BasicRetinaFilter):
    def __init__(self, NBrows, NBcolumns):
        super().__init__(NBrows, NBcolumns, 3)
        self._NBrows = NBrows
        self._NBcolumns = NBcolumns
        self._initialize_buffers(NBrows, NBcolumns)
        self.clearAllBuffers()
        # self._ON_OFF_diff = None
        self._V0 = 0.9
        self._maxInputValue = 255.0

    def _initialize_buffers(self, NBrows, NBcolumns):
        self._photoreceptorsLPOutput = np.zeros((NBrows, NBcolumns, 3), dtype=np.float32)
        self._photoreceptorsAdaptation = np.zeros((NBrows, NBcolumns, 3), dtype=np.float32)
        self._horizontalCellsOutput = np.zeros((NBrows, NBcolumns, 3), dtype=np.float32)
        self._parvocellularOutputON = np.zeros((NBrows, NBcolumns, 3), dtype=np.float32)
        self._parvocellularOutputOFF = np.zeros((NBrows, NBcolumns, 3), dtype=np.float32)
        self._bipolarCellsOutputON = np.zeros((NBrows, NBcolumns, 3), dtype=np.float32)
        self._bipolarCellsOutputOFF = np.zeros((NBrows, NBcolumns, 3), dtype=np.float32)
        self._localAdaptationOFF = np.zeros((NBrows, NBcolumns, 3), dtype=np.float32)
        #self._localAdaptationON = self._localBuffer.reshape((NBrows, NBcolumns, 3))
        self._ON_OFF_diff = np.zeros((NBrows, NBcolumns, 3), dtype=np.float32)
        self._parvocellularOutputON = np.zeros((NBrows, NBcolumns, 3), dtype=np.float32)
        self._parvocellularOutputOFF = np.zeros((NBrows, NBcolumns, 3), dtype=np.float32)
        self._parvocellularOutputONminusOFF = np.zeros((NBrows, NBcolumns, 3), dtype=np.float32)
    #
    def clearAllBuffers(self):
        super().clearAllBuffers()
        if hasattr(self, '_photoreceptorsLPOutput'):
            self._photoreceptorsLPOutput.fill(0)
        if hasattr(self, '_photoreceptorsAdaptation '):
            self._photoreceptorsAdaptation.fill(0)
        if hasattr(self, '_horizontalCellsOutput'):
            self._horizontalCellsOutput.fill(0)
        if hasattr(self, '_parvocellularOutputON'):
            self._parvocellularOutputON.fill(0)
        if hasattr(self, '_parvocellularOutputOFF'):
            self._parvocellularOutputOFF.fill(0)
        if hasattr(self, '_bipolarCellsOutputON'):
            self._bipolarCellsOutputON.fill(0)
        if hasattr(self, '_bipolarCellsOutputOFF'):
            self._bipolarCellsOutputOFF.fill(0)
        if hasattr(self, '_localAdaptationOFF'):
            self._localAdaptationOFF.fill(0)
        if hasattr(self, '_parvocellularOutputON'):
            self._parvocellularOutputON.fill(0)
        if hasattr(self, '_parvocellularOutputOFF'):
            self._parvocellularOutputOFF.fill(0)
        if hasattr(self, '_parvocellularOutputONminusOFF'):
            self._parvocellularOutputONminusOFF.fill(0)
    # def clearAllBuffers(self):
    #     super().clearAllBuffers()
    #     self._photoreceptorsLPOutput.fill(0)
    #     self._photoreceptorsAdaptation.fill(0)
    #     self._horizontalCellsOutput.fill(0)
    #     self._parvocellularOutputON.fill(0)
    #     self._parvocellularOutputOFF.fill(0)
    #     self._bipolarCellsOutputON.fill(0)
    #     self._bipolarCellsOutputOFF.fill(0)
    #     self._localAdaptationOFF.fill(0)
    #     self._ON_OFF_diff.fill(0)

    def resize(self, NBrows, NBcolumns):
        self._NBrows = NBrows
        self._NBcolumns = NBcolumns
        super().resize(NBrows, NBcolumns)
        self._initialize_buffers(NBrows, NBcolumns)
        self.clearAllBuffers()

    def setOPLandParvoFiltersParameters(self, beta1, tau1, k1, beta2, tau2, k2):
        self.setLPfilterParameters(beta1, tau1, k1, 0)
        self.setLPfilterParameters(beta2, tau2, k2, 1)
        self.setLPfilterParameters(0, tau1, k1, 2)

    def runFilter(self, inputFrame, useParvoOutput=True):
        # self._spatiotemporalLPfilter(inputFrame, self._photoreceptorsOutput, 0)
        localLuminance0 = self.calculateLocalLuminance(inputFrame)
        self._localLuminanceAdaptation(inputFrame, localLuminance0, self._photoreceptorsAdaptation)
        self._spatiotemporalLPfilter(self._photoreceptorsAdaptation, self._photoreceptorsLPOutput, 0)
        self._spatiotemporalLPfilter(self._photoreceptorsLPOutput, self._horizontalCellsOutput, 1)
        self._OPL_OnOffWaysComputing()
        bipolarCellsOutputON, bipolarCellsOutputOFF, _ON_OFF_diff = self._OPL_OnOffWaysComputing()
        localLuminance1 = self.calculateLocalLuminance(_ON_OFF_diff)
        self._localLuminanceAdaptation(self._ON_OFF_diff, localLuminance1, self._parvocellularOutputONminusOFF,V0=-6)

    def _OPL_OnOffWaysComputing(self):
        # reshape: Transform 3 dimensions into 2 dimensions for pixel-by-pixel processing
        photoreceptorsLPOutput_PTR = self._photoreceptorsLPOutput.reshape(-1, 3)
        horizontalCellsOutput_PTR = self._horizontalCellsOutput.reshape(-1, 3)
        bipolarCellsON_PTR = self._bipolarCellsOutputON.reshape(-1, 3)
        bipolarCellsOFF_PTR = self._bipolarCellsOutputOFF.reshape(-1, 3)

        self._ON_OFF_diff = np.zeros_like(photoreceptorsLPOutput_PTR)
        # 创建像素点的索引0~行数-1
        for IDpixel in range(photoreceptorsLPOutput_PTR.shape[0]):
            pixelDifference = photoreceptorsLPOutput_PTR[IDpixel] - horizontalCellsOutput_PTR[IDpixel]
            isPositive = pixelDifference > 0 # 判断 pixelDifference 是否为正。如果为正，则 isPositive=1

            # print(f"Pixel {IDpixel} difference: {pixelDifference}") #76799个pixel
            bipolarCellsON_PTR[IDpixel] = isPositive * pixelDifference # pixelDifference为正 在on输出
            bipolarCellsOFF_PTR[IDpixel] = (~isPositive) * pixelDifference  # pixelDifference为负 在off输出

            self._ON_OFF_diff[IDpixel] = bipolarCellsON_PTR[IDpixel] - bipolarCellsOFF_PTR[IDpixel]

        self._bipolarCellsOutputON = bipolarCellsON_PTR.reshape(self._NBrows, self._NBcolumns, 3)
        self._bipolarCellsOutputOFF = bipolarCellsOFF_PTR.reshape(self._NBrows, self._NBcolumns, 3)
        self._ON_OFF_diff = self._ON_OFF_diff.reshape(self._NBrows, self._NBcolumns, 3)
        return self._bipolarCellsOutputON, self._bipolarCellsOutputOFF,self._ON_OFF_diff

    def get_ON_OFF_difference(self):
        if self._ON_OFF_diff is not None:
            return self._ON_OFF_diff
        else:
            return None

    def normalize_image(self, img):
        img_min = img.min()
        img_max = img.max()
        img_normalized = (img - img_min) / (img_max - img_min) * 255
        return img_normalized.astype(np.uint8)

    # 测试代码
if __name__ == "__main__":
    image = cv2.imread('E:/dissertation/simulation/image/xs.png') #image is in BGR format and the data type is uint8
    # The shape of image is (height, width, channels)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    NBrows, NBcolumns, channels = image.shape
    print(f"Image size (height, width, number of channels): ({NBrows}, {NBcolumns}, {channels})")

    parvo_filter = ParvoRetinaFilter(NBrows, NBcolumns)

    beta1, tau1, k1 = 0.1, 0.001, 1  # -0.44, 0.001, 0.1
    beta2, tau2, k2 = -0.44, 0.001, 0.1
    parvo_filter.setOPLandParvoFiltersParameters(beta1, tau1, k1, beta2, tau2, k2)

    inputFrame = image.astype(np.float32) # “astype(np.float32)” Converts the data type to a 32-bit floating point number
    # “inputFrame” is a NumPy array containing image data of shape (height, width, 3), data type float32, and color channel order RGB
    enhanced_ON_OFF_diff = parvo_filter.runFilter(inputFrame)
    print("inputFrame dimensions：",inputFrame.shape)
    #print(f"input frame:{inputFrame}")

    on_off_difference_normalized = parvo_filter.normalize_image(parvo_filter._ON_OFF_diff)
    parvocellularOutputONminusOFF_normalized = parvo_filter.normalize_image(parvo_filter._parvocellularOutputONminusOFF)

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    axes[0].imshow(inputFrame.astype(np.uint8))
    axes[0].set_title('Input Image')
    axes[0].axis('off')

    axes[1].imshow(parvo_filter._photoreceptorsAdaptation.astype(np.uint8))
    axes[1].set_title('Photoreceptors Adaption')
    axes[1].axis('off')

    axes[2].imshow(parvo_filter._photoreceptorsLPOutput.astype(np.uint8))
    axes[2].set_title('Photoreceptors Output')
    axes[2].axis('off')

    axes[3].imshow(parvo_filter._horizontalCellsOutput.astype(np.uint8))
    axes[3].set_title('Horizontal Output')
    axes[3].axis('off')


    axes[4].imshow(on_off_difference_normalized)
    axes[4].set_title('Bipolar ON-OFF Difference')
    axes[4].axis('off')


    axes[5].imshow(parvo_filter._bipolarCellsOutputON.astype(np.uint8))
    axes[5].set_title('Bipolar Cells ON Output')
    axes[5].axis('off')

    axes[6].imshow(parvo_filter._bipolarCellsOutputOFF.astype(np.uint8))
    axes[6].set_title('Bipolar Cells OFF Output')
    axes[6].axis('off')


    axes[7].imshow(parvocellularOutputONminusOFF_normalized)
    axes[7].set_title('Parvo Channel Output')
    axes[7].axis('off')

    plt.show()
