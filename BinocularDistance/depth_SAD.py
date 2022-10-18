from tokenize import triple_quoted
import cv2 as cv
from matplotlib.image import imread
import numpy as np
import camera_configs
import time
import matplotlib as plt


maxDisparity = 25  # 最大视差
window_size = 5


lraw = imread("./paperimg/img_l1.bmp")
rraw = imread("./paperimg/img_r1.bmp")

img1_rectified = cv.remap(lraw, camera_configs.left_map1, camera_configs.left_map2, cv.INTER_LINEAR,
                          cv.BORDER_CONSTANT)
img2_rectified = cv.remap(rraw, camera_configs.right_map1, camera_configs.right_map2, cv.INTER_LINEAR,
                          cv.BORDER_CONSTANT)

'''这一部分是转换彩色图像为灰度图像，并且转为double格式'''
# ------------------------------
limg = cv.cvtColor(lraw, cv.COLOR_BGR2GRAY)
rimg = cv.cvtColor(rraw, cv.COLOR_BGR2GRAY)
limg = np.asanyarray(limg, dtype=np.double)
rimg = np.asanyarray(rimg, dtype=np.double)
img_size = np.shape(limg)[0:2]

# -------------------------------
'''这一部分是加速后的SAD算法，具体做法是先计算右图按照视差由0到maxDisparity减去左图所得的矩阵'''
# ------------------------------
tic1 = time.time()
imgDiff = np.zeros((img_size[0], img_size[1], maxDisparity))
e = np.zeros(img_size)
for i in range(0, maxDisparity):
    # 视差为多少，那么生成的图像就会少多少像素列,e负责计算视差为i时，两张图整体的差距
    e = np.abs(rimg[:, 0:(img_size[1]-i)] - limg[:, i:img_size[1]])
    e2 = np.zeros(img_size)  # 计算窗口内的和
    for x in range((window_size), (img_size[0]-window_size)):
        for y in range((window_size), (img_size[1]-window_size)):
            # 其实相当于用111 111 111的卷积核去卷积，如果用tensorflow会不会更快一些，其实就是先做差再加和以及先加和再做差的关系
            e2[x, y] = np.sum(
                e[(x-window_size):(x+window_size), (y-window_size):(y+window_size)])
        imgDiff[:, :, i] = e2
dispMap = np.zeros(img_size)
# -------------------------------
'''这一部分整找到使灰度差最小的视差，并绘图'''
# ------------------------------
for x in range(0, img_size[0]):
    for y in range(0, img_size[1]):
        val = np.sort(imgDiff[x, y, :])
        if np.abs(val[0]-val[1]) > 10:
            val_id = np.argsort(imgDiff[x, y, :])
            # 其实dispmap计算的是视差的大小，如果视差大，那么就相当于两张图片中同样物品的位移大，就是距离近
            dispMap[x, y] = val_id[0]/maxDisparity*255
print('用时:', time.time()-tic1)
i = 1

while True:
    disp = dispMap

    cv.imshow("img1", disp)
    c = cv.waitKey(10) & 0xff
    if c == ord('q'):
        break
    elif c == ord('s'):
        cv.imwrite('SAD%d.bmp' % i, disp)
        print("Save %d image" % i)
cv.destroyAllWindows()
