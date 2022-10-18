from bdb import Breakpoint
import cv2

imageWidth = 640
imageHeight = 360

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, imageWidth * 2)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, imageHeight)
i = 1

while True:
    # 从摄像头读取图片
    success, img = cap.read()

    if not success:
        break

    if success:
        # 获取左右摄像头的图像
        rgbImageL = img[:, 0:imageWidth, :]
        rgbImageR = img[:, imageWidth:imageWidth * 2, :]
        cv2.imshow('Left', rgbImageL)
        cv2.imshow('Right', rgbImageR)
        cv2.imshow("img", img)
        # 按“回车”保存图片
        c = cv2.waitKey(10) & 0xff
        if c == 13:
            cv2.imwrite('./calib/left/Left%d.bmp' % i, rgbImageL)
            cv2.imwrite('./calib/right/Right%d.bmp' % i, rgbImageR)
            # cv2.imwrite('./labimg/all%d.bmp' % i, img)
            print("Save %d image" % i)
            i += 1
        elif c == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
