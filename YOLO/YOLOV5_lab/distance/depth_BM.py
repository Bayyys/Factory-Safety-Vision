from cv2 import cv2
import numpy as np
from distance import camera_configs
from distance.BM import BM

def depth_BM(img, frame1, frame2, imageWidth, imageHeight):
    ####### 深度图测量开始 ##   #####
    # 立体匹配这里使用BM算法，
    # 根据标定数据对图片进行重构消除图片的畸变
    img1_rectified = cv2.remap(frame1, camera_configs.left_map1, camera_configs.left_map2, cv2.INTER_LINEAR,
                               cv2.BORDER_CONSTANT)
    img2_rectified = cv2.remap(frame2, camera_configs.right_map1, camera_configs.right_map2, cv2.INTER_LINEAR,
                               cv2.BORDER_CONSTANT)

    # 将图片置为灰度图，为StereoBM作准备，BM算法只能计算单通道的图片，即灰度图
    # 单通道就是黑白的，一个像素只有一个值如[123]
    # opencv默认的是BGR(注意不是RGB)——[B,G,R]
    imgL = cv2.cvtColor(img1_rectified, cv2.COLOR_BGR2GRAY)
    imgR = cv2.cvtColor(img2_rectified, cv2.COLOR_BGR2GRAY)

    out = np.hstack((img1_rectified, img2_rectified))
    for i in range(0, out.shape[0], 30):
        cv2.line(out, (0, i), (out.shape[1], i), (0, 255, 0), 1)

    disparity = BM(imgL, imgR)

    # 按照深度矩阵生产深度图
    disp = cv2.normalize(disparity, disparity, alpha=0,
                         beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # 将深度图扩展至三维空间中，其z方向的值则为当前的距离
    threeD = cv2.reprojectImageTo3D(
        disparity.astype(np.float32) / 16., camera_configs.Q)
    # 将深度图转为伪色图，show
    fakeColorDepth = cv2.applyColorMap(disp, cv2.COLORMAP_JET)
    
    return out, disp, fakeColorDepth, threeD