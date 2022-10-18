from cv2 import cv2
from distance import camera_configs

def BM(imgL, imgR):
    # BM 参数
    numberOfDisparities = ((640 // 8) + 15) & -16  # 640对应是分辨率的宽

    stereo = cv2.StereoBM_create(numDisparities=16, blockSize=9)  # 立体匹配
    stereo.setROI1(camera_configs.validPixROI1)
    stereo.setROI2(camera_configs.validPixROI2)
    stereo.setPreFilterCap(31)
    stereo.setBlockSize(9)
    stereo.setMinDisparity(0)
    stereo.setNumDisparities(numberOfDisparities)
    stereo.setTextureThreshold(10)
    stereo.setUniquenessRatio(15)
    stereo.setSpeckleWindowSize(100)
    stereo.setSpeckleRange(32)
    stereo.setDisp12MaxDiff(1)
    # ———————————————————————— #
    # 对深度进行计算，获取深度矩阵

    disparity = stereo.compute(imgL, imgR)
    
    return disparity
