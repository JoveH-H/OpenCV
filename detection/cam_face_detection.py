import cv2 as cv

xml_path = '../xml/haarcascade_frontalface_default.xml'  # 人脸检测器（默认）xml文件的路径
face_cascade = cv.CascadeClassifier(xml_path)  # 加载级联分类器

cam = cv.VideoCapture(0)  # 从摄像头中取得视频

cv.namedWindow('face_detection')  # 创建一个窗口

while 1:
    # 读取帧摄像头
    ret, frame = cam.read()
    frame = cv.resize(frame, (240, 180))
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,  # 需要被检测的图像，一般为灰度图像加快检测速度
                                          scaleFactor=1.3,  # 在前后两次相继的扫描中，搜索窗口的缩放比例
                                          minNeighbors=5,  # 构成检测目标的相邻矩形的最小个数
                                          minSize=(5, 5),  # 需要被检测的最小尺寸
                                          flags=cv.CASCADE_SCALE_IMAGE)  # 级联扩展图像
    for (x, y, w, h) in faces:  # 在人脸范围画矩形框
        cv.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
    print("发现{0}个人脸!".format(len(faces)))
    cv.imshow('face_detection', frame)
    # 键盘按 Q 退出
    if (cv.waitKey(50) & 0xFF) == ord('q'):
        break

# 释放资源
cam.release()
cv.destroyAllWindows()
