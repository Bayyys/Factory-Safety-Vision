# 可以运行后根据效果展示
# 如果效果极差——重新标定
# 如果效果较差——修改显示深度的窗口和调节参数的bar
# 一般情况下建议重新标定

from cv2 import cv2
import numpy as np
import camera_configs  # 摄像头的标定数据
from SGBM import SGBM

imageWidth = 640    # 分辨率宽度
imageHeight = 360   # 分辨率高度
imageSize = (imageWidth, imageHeight)

# 创建用于显示深度的窗口和调节参数的bar
cv2.namedWindow("depth")
# cv2.moveWindow("depth", 0, 0)
cv2.moveWindow("depth", 600, 0)


# 添加点击事件，打印当前点的距离
def callbackFunc(e, x, y, f, p):
    if e == cv2.EVENT_LBUTTONDOWN:
        print(threeD[y][x])
        if abs(threeD[y][x][2]) < 3000:
            print("当前距离:"+str(abs(threeD[y][x][2])))
        else:
            print("当前距离过大或请点击色块的位置")


cv2.setMouseCallback("depth", callbackFunc, None)

j = 16

while True:
    # 这里将左右两个摄像头的图像进行一下分割
    frame1 = cv2.imread('./MODEL/paperimg/img_l11.bmp')
    frame2 = cv2.imread('./MODEL/paperimg/img_r11.bmp')

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

    disparity = SGBM(imgL, imgR)

   # 按照深度矩阵生产深度图
    disp = cv2.normalize(disparity, disparity, alpha=0,
                         beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # 将深度图扩展至三维空间中，其z方向的值则为当前的距离
    threeD = cv2.reprojectImageTo3D(
        disparity.astype(np.float32) / 16., camera_configs.Q)
    # 将深度图转为伪色图，show
    fakeColorDepth = cv2.applyColorMap(disp, cv2.COLORMAP_JET)

    # 按下S可以保存图片
    interrupt = cv2.waitKey(10)
    if interrupt & 0xFF == 27:  # 按下ESC退出程序
        break

    ####### 任务1：测距结束 #######

    # 需要对深度图进行滤波将下面几行开启即可 开启后FPS会降低
    # img_medianBlur = cv2.medianBlur(disp, 25)
    # img_medianBlur_fakeColorDepth = cv2.applyColorMap(
    #     img_medianBlur, cv2.COLORMAP_JET)
    # img_GaussianBlur = cv2.GaussianBlur(disp, (7, 7), 0)
    # img_Blur = cv2.blur(disp, (5, 5))
    # cv2.imshow("img_GaussianBlur", img_GaussianBlur)  # 右边原始输出
    # cv2.imshow("img_medianBlur_fakeColorDepth",
    #            img_medianBlur_fakeColorDepth)  # 右边原始输出
    # cv2.imshow("img_Blur", img_Blur)  # 右边原始输出
    # cv2.imshow("img_medianBlur", img_medianBlur)  # 右边原始输出
    # fakeColorDepth = cv2.applyColorMap(img_medianBlur, cv2.COLORMAP_JET)
    # fakeColorDepth = cv2.applyColorMap(img_Blur, cv2.COLORMAP_JET)
    # fakeColorDepth = cv2.applyColorMap(img_GaussianBlur, cv2.COLORMAP_JET)
    blur = cv2.medianBlur(disp, 1)
    cv2.imshow("blur", blur)
    # 显示
    cv2.imshow("img_all", np.vstack(
        (np.hstack((frame1, frame2)), out)))  # 原始输出，用于检测左右
    gray = cv2.cvtColor(np.hstack((frame1, frame2)),cv2.COLOR_RGB2GRAY)
    ret, gray = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    cv2.imshow("111",gray)
    cv2.imshow("depth", disp)  # 输出深度图及调整的bar
    cv2.imshow("fakeColor", fakeColorDepth)  # 输出深度图的伪色图，这个图没有用只是好看

cv2.destroyAllWindows()
