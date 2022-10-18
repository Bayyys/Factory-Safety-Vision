import cv2
cap = cv2.VideoCapture(1)

# *mp4v 就是解包操作，等同于'm','p','4','v'
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# 尝试avi格式
fourcc = cv2.VideoWriter_fourcc(*'XVID')

# (1280, 720)表示摄像头分辨率，这个大小搞错了也不行
# 主要是这个分辨率
# vw = cv2.VideoWriter('output.mp4', fourcc, 30, (640, 480))
vw = cv2.VideoWriter('./video/output_stairs5.avi', fourcc, 30, (640, 240))

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    # 写每一帧数据
    vw.write(frame)
    cv2.imshow('frame', frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
vw.release()
cv2.destroyAllWindows()
