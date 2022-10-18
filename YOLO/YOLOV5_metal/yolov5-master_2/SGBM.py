
from cv2 import cv2


def SGBM(imgL, imgR):
    # SGBM 参数
    SADWindowSize = 5

    sgbm = cv2.StereoSGBM_create(minDisparity=0,
                                 numDisparities=100,
                                 blockSize=5,
                                 P1=8 * 3 * SADWindowSize ** 2,
                                 P2=32 * 3 * SADWindowSize ** 2,
                                 disp12MaxDiff=1,
                                 uniquenessRatio=15,
                                 speckleWindowSize=100,
                                 speckleRange=32
                                 )

    # 对深度进行计算，获取深度矩阵
    disparity = sgbm.compute(imgL, imgR)
    return disparity
