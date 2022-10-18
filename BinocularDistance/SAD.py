import cv2
from pylab import *
import time


def SAD(Img_L, Img_R, winsize, DSR):  # 输入左右图像，窗口尺寸，搜索范围
    width, height = Img_L.shape
    kernel_L = np.zeros((winsize, winsize), dtype='uint8')
    kernel_R = np.zeros((winsize, winsize), dtype='uint8')
    disparity = np.zeros((width, height), dtype='uint8')
    for i in range(width-winsize):
        for j in range(height-winsize):
            kernel_L = Img_L[i:i+winsize, j:j+winsize]
            v = [0]*DSR
            for k in range(DSR):
                x = i-k
                if x >= 0:
                    kernel_R = Img_R[x:x+winsize, j:j+winsize]
                for m in range(winsize):
                    for n in range(winsize):
                        v[k] = v[k]+abs(kernel_R[m, n]-kernel_L[m, n])
            disparity[i, j] = min(v)
    return disparity


start = time.process_time()  # 获取代码运行时间
img_L = cv2.imread('./MODEL/left1.bmp', 0)
img_R = cv2.imread('./MODEL/right1.bmp', 0)
sad = SAD(img_L, img_R, 3, 30)
cv2.imshow('Origion_L', img_L)
cv2.imshow('Origion_R', img_R)
cv2.imshow('After', sad)
cv2.waitKey()
cv2.destroyAllWindows()
end = time.process_time()
print('Running time:', end-start)  # 显示运行时间
