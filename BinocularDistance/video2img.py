import cv2
import os,sys

cap = cv2.VideoCapture(r"video/output_stairs2.avi")
isOpened = cap.isOpened
print(isOpened)

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(fps,width,height)

file_path = os.getcwd()    # 当前文件目录
img_dir = file_path + '/video/img'  # 图片存储文件夹命名

if not os.path.exists(img_dir):
    os.makedirs(img_dir)

print(os.getcwd())

i = 0
while(isOpened):
    i = i+1
    (flag,frame) = cap.read()   #读取每一帧，flag表示是否读取成功，frame为图片内容。
    fileName = "image" +str(i) +".png"
    print(fileName)
    if flag == True:
        cv2.imwrite(fileName,frame,[cv2.IMWRITE_JPEG_QUALITY,100])

print(os.getcwd())
print("end!")
