from cv2 import cv2
import numpy as np
import camera_configs
from SGBM import SGBM


def depth_SGBM(img, frame1, frame2, imageWidth, imageHeight):
    ####### 深度图测量开始 ##   #####
    # 立体匹配这里使用SGBM算法，
    # 根据标定数据对图片进行重构消除图片的畸变
    img1_rectified = cv2.remap(frame1, camera_configs.left_map1, camera_configs.left_map2, cv2.INTER_LINEAR,
                                   cv2.BORDER_CONSTANT)
    img2_rectified = cv2.remap(frame2, camera_configs.right_map1, camera_configs.right_map2, cv2.INTER_LINEAR,
                                   cv2.BORDER_CONSTANT)
    imgL = cv2.cvtColor(img1_rectified, cv2.COLOR_BGR2GRAY)
    imgR = cv2.cvtColor(img2_rectified, cv2.COLOR_BGR2GRAY)
    out_line = img.copy()
        
    for i in range(0, out_line.shape[0], 30):
            cv2.line(out_line, (0, i), (out_line.shape[1], i), (0, 255, 0), 1)
    
    disparity = SGBM(imgL, imgR)
    # 按照深度矩阵生产深度图
    disp = cv2.normalize(disparity, disparity, alpha=0,
                             beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # 将深度图扩展至三维空间中，其z方向的值则为当前的距离
    threeD = cv2.reprojectImageTo3D(
            disparity.astype(np.float32) / 16., camera_configs.Q)
    # 将深度图转为伪色图，show
    fakeColorDepth = cv2.applyColorMap(disp, cv2.COLORMAP_JET)
        
    ####### 任务1：测距结束 #######
    
    return out_line, disp, fakeColorDepth, threeD