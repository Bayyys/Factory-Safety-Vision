# 可以运行后根据效果展示
# 如果效果极差——重新标定
# 如果效果较差——修改显示深度的窗口和调节参数的bar
# 一般情况下建议重新标定

from cv2 import cv2
import numpy as np
import camera_configs  # 摄像头的标定数据

imageWidth = 640    # 分辨率宽度
imageHeight = 360   # 分辨率高度
imageSize = (imageWidth, imageHeight)

# cam1 = cv2.VideoCapture(1)  # 摄像头的ID不同设备上可能不同
# cam2 = cv2.VideoCapture(1)  # 摄像头的ID不同设备上可能不同
# cam1 = cv2.VideoCapture(1 + cv2.CAP_DSHOW)  # 摄像头的ID不同设备上可能不同
# cam1.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # 设置双目的宽度
# cam1.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # 设置双目的高度

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, imageWidth * 2)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, imageHeight)

# 创建用于显示深度的窗口和调节参数的bar
cv2.namedWindow("depth")
# cv2.moveWindow("depth", 0, 0)
cv2.moveWindow("depth", 600, 0)


# 创建用于显示深度的窗口和调节参数的bar
# 默认状态已经调好，如需要重新调整则打开此项
# cv2.namedWindow("depth")
cv2.namedWindow("config", cv2.WINDOW_NORMAL)
cv2.resizeWindow("config", 1000, 400)

cv2.createTrackbar("model", "config", 1, 1, lambda x: None)
cv2.createTrackbar("num", "config", 1, 60, lambda x: None)
cv2.createTrackbar("PreFilterCap", "config", 31, 65,
                   lambda x: None)  # 注意调节的时候这个值必须是奇数
cv2.createTrackbar("BlockSize", "config", 9, 255, lambda x: None)
cv2.createTrackbar("MinDisparity", "config", 0, 255, lambda x: None)
cv2.createTrackbar("NumDisparities", "config", 5, 10, lambda x: None)
cv2.createTrackbar("TextureThreshold", "config", 10, 255, lambda x: None)
cv2.createTrackbar("UniquenessRatio", "config", 15, 50, lambda x: None)
cv2.createTrackbar("SpeckleWindowSize", "config", 100, 500, lambda x: None)
cv2.createTrackbar("SpeckleRange", "config", 32, 255, lambda x: None)
cv2.createTrackbar("MaxDiff", "config", 1, 100, lambda x: None)


# 添加点击事件，打印当前点的距离
def callbackFunc(e, x, y, f, p):
    if e == cv2.EVENT_LBUTTONDOWN:
        print(threeD[y][x])
        dis = abs(threeD[y][x][2])
        if dis < 6000:
            print("当前距离:  %.3f cm" % (dis / 10))
        else:
            print("当前距离过大或请点击色块的位置")


cv2.setMouseCallback("depth", callbackFunc, None)

# # 初始化计算FPS需要用到参数
# # 注意不要用opencv自带fps的函数，那个函数得到的是摄像头最大的FPS
# frame_rate_calc = 1
# freq = cv2.getTickFrequency()
# font = cv2.FONT_HERSHEY_SIMPLEX

imageCount = 1

j = 1


while True:
    # t1 = cv2.getTickCount()
    ret1, frame = cap.read()

    # 这里将左右两个摄像头的图像进行一下分割
    frame1 = frame[:, 0:imageWidth, :]
    frame2 = frame[:, imageWidth:imageWidth * 2, :]

    if not ret1:
        print("camera is not connected!")
        break

    ####### 深度图测量开始 ##   #####
    # 立体匹配这里使用BM算法，

    # 根据标定数据对图片进行重构消除图片的畸变
    img1_rectified = cv2.remap(frame1, camera_configs.left_map1, camera_configs.left_map2, cv2.INTER_LINEAR,
                               cv2.BORDER_CONSTANT)
    img2_rectified = cv2.remap(frame2, camera_configs.right_map1, camera_configs.right_map2, cv2.INTER_LINEAR,
                               cv2.BORDER_CONSTANT)

    # 如有remap()的图是反的 需要对角翻转
    # img1_rectified = cv2.flip(img1_rectified, -1)
    # img2_rectified = cv2.flip(img2_rectified, -1)

    # 将图片置为灰度图，为StereoBM作准备，BM算法只能计算单通道的图片，即灰度图
    # 单通道就是黑白的，一个像素只有一个值如[123]，opencv默认的是BGR(注意不是RGB), 如[123,4,134]分别代表这个像素点的蓝绿红的值
    imgL = cv2.cvtColor(img1_rectified, cv2.COLOR_BGR2GRAY)
    imgR = cv2.cvtColor(img2_rectified, cv2.COLOR_BGR2GRAY)

    out = np.hstack((img1_rectified, img2_rectified))
    for i in range(0, out.shape[0], 30):
        cv2.line(out, (0, i), (out.shape[1], i), (0, 255, 0), 1)

    flag_model = cv2.getTrackbarPos("model", "config")
    if flag_model == 0:
        # 自定义设置参数
        # 通过bar来获取到当前的参数
        # BM算法对参数非常敏感，耐心调整！！！
        # 前两个参数影响大 后面的参数也需要精细的调整

        num = cv2.getTrackbarPos("num", "config")
        PreFilterCap = cv2.getTrackbarPos("PreFilterCap", "config")
        BlockSize = cv2.getTrackbarPos("BlockSize", "config")
        MinDisparity = cv2.getTrackbarPos("MinDisparity", "config")
        NumDisparities = cv2.getTrackbarPos("NumDisparities", "config")
        TextureThreshold = cv2.getTrackbarPos("TextureThreshold", "config")
        UniquenessRatio = cv2.getTrackbarPos("UniquenessRatio", "config")
        SpeckleWindowSize = cv2.getTrackbarPos("SpeckleWindowSize", "config")
        SpeckleRange = cv2.getTrackbarPos("SpeckleRange", "config")
        MaxDiff = cv2.getTrackbarPos("MaxDiff", "config")
        if PreFilterCap % 2 != 0:
            PreFilterCap -= 1
        if NumDisparities == 0:
            NumDisparities += 1
        NumDisparities *= 16
        if BlockSize % 2 == 0:
            BlockSize += 1
        if BlockSize < 5:
            BlockSize = 5

        stereo = cv2.StereoBM_create(
            numDisparities=16 * num, blockSize=9)  # 立体匹配
        stereo.setROI1(camera_configs.validPixROI1)
        stereo.setROI2(camera_configs.validPixROI2)
        stereo.setPreFilterCap(PreFilterCap)
        stereo.setBlockSize(BlockSize)
        stereo.setMinDisparity(MinDisparity)
        stereo.setNumDisparities(NumDisparities)
        stereo.setTextureThreshold(TextureThreshold)
        stereo.setUniquenessRatio(UniquenessRatio)
        stereo.setSpeckleWindowSize(SpeckleWindowSize)
        stereo.setSpeckleRange(SpeckleRange)
        stereo.setDisp12MaxDiff(MaxDiff)

        # ———————————————————————— #
        # 对深度进行计算，获取深度矩阵

    elif flag_model == 1:
        # 默认默认参数
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

    # 按照深度矩阵生产深度图
    disp = cv2.normalize(disparity, disparity, alpha=0,
                         beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # 将深度图扩展至三维空间中，其z方向的值则为当前的距离
    threeD = cv2.reprojectImageTo3D(
        disparity.astype(np.float32) / 16., camera_configs.Q)
    # 将深度图转为伪色图，show
    fakeColorDepth = cv2.applyColorMap(disp, cv2.COLORMAP_JET)

    # cv2.putText(frame1, "FPS: {0:.2f}".format(
    # frame_rate_calc), (30, 50), font, 1, (255, 255, 0), 2, cv2.LINE_AA)

    # 按下S可以保存图片
    interrupt = cv2.waitKey(10)
    if interrupt & 0xFF == 27:  # 按下ESC退出程序
        break
    elif interrupt == ord('s'):
        cv2.imwrite('./lab/fake/fakecolordepth%d.bmp' % j, fakeColorDepth)
        cv2.imwrite('./lab/img/img_o_all%d.bmp' % j, frame)
        cv2.imwrite('./lab/imgl_r/img_l%d.bmp' % j, frame1)
        cv2.imwrite('./lab/imgl_r/img_r%d.bmp' % j, frame2)
        cv2.imwrite('./lab/depth/depth%d.bmp' % j, disp)
        print("Save %d image" % j)
        j += 1

    ####### 任务1：测距结束 #######

    # 需要对深度图进行滤波将下面几行开启即可 开启后FPS会降低
    img_medianBlur = cv2.medianBlur(disp, 25)
    img_medianBlur_fakeColorDepth = cv2.applyColorMap(
        img_medianBlur, cv2.COLORMAP_JET)
    img_GaussianBlur = cv2.GaussianBlur(disp, (7, 7), 0)
    img_Blur = cv2.blur(disp, (5, 5))
    # cv2.imshow("img_GaussianBlur", img_GaussianBlur)  # 右边原始输出
    # cv2.imshow("img_medianBlur_fakeColorDepth",
    #            img_medianBlur_fakeColorDepth)  # 右边原始输出
    # cv2.imshow("img_Blur", img_Blur)  # 右边原始输出
    # cv2.imshow("img_medianBlur", img_medianBlur)  # 右边原始输出
    # fakeColorDepth = cv2.applyColorMap(img_medianBlur, cv2.COLORMAP_JET)
    # fakeColorDepth = cv2.applyColorMap(img_Blur, cv2.COLORMAP_JET)
    # fakeColorDepth = cv2.applyColorMap(img_GaussianBlur, cv2.COLORMAP_JET)

    # 显示
    cv2.imshow("img_all", np.vstack((frame, out)))  # 原始输出，用于检测左右
    cv2.imshow("depth", disp)  # 输出深度图及调整的bar
    cv2.imshow("fakeColor", fakeColorDepth)  # 输出深度图的伪色图，这个图没有用只是好看

    # t2 = cv2.getTickCount()
    # time1 = (t2 - t1) / freq
    # frame_rate_calc = 1 / time1

cap.release()
cv2.destroyAllWindows()
