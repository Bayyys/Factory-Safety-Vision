import cv2
import numpy as np

left_camera_matrix = np.array([[574.3820, -0.3295, 340.9976],
                               [0, 575.5628, 263.0289],
                               [0, 0, 1]])
left_distortion = np.array([0.0787, -0.0860, 0.00063, -0.00094, 0.00000])

right_camera_matrix = np.array([[573.5875, -0.1309, 343.1904],
                                [0, 574.5145, 274.4080],
                                [0, 0, 1]])
right_distortion = np.array([0.0660, -0.0669, 0.0020, 0.0019, 0.00000])


# rec旋转向量
rec = np.array([0.0025, 0.0057, 0.0017])

R = cv2.Rodrigues(rec)[0]

# print(R)

T = np.array([-60.7867, 0.0302, -0.0531])  # 平移关系向量

size = (640, 480)  # 图像尺寸

# 进行立体更正
R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(left_camera_matrix, left_distortion,
                                                                  right_camera_matrix, right_distortion, size, R,
                                                                  T)
# 计算更正map
left_map1, left_map2 = cv2.initUndistortRectifyMap(
    left_camera_matrix, left_distortion, R1, P1, size, cv2.CV_16SC2)
right_map1, right_map2 = cv2.initUndistortRectifyMap(
    right_camera_matrix, right_distortion, R2, P2, size, cv2.CV_16SC2)
