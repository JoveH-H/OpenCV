import cv2 as cv

xml_path = '../xml/haarcascade_frontalface_default.xml'  # 人脸检测器（默认）xml文件的路径
recognizer_path = '../xml/userFace.yml'  # LBPH人脸识别模型 yml文件的路径

user_name_list = ["Unkown", 'H-H', 'Andrew Ng', 'Jack Ma']  # 用户名单

face_cascade = cv.CascadeClassifier(xml_path)  # 加载级联分类器

recognizer = cv.face.LBPHFaceRecognizer_create()  # 创建LBPH模型
recognizer.read(recognizer_path)  # 加载LBPH人脸识别模型
font = cv.FONT_HERSHEY_SIMPLEX  # 设置正常大小无衬线字体

cam = cv.VideoCapture(0)  # 从摄像头中取得视频

cv.namedWindow('face_recognition')  # 创建一个窗口

while 1:
    # 读取帧摄像头
    ret, frame = cam.read()
    frame = cv.resize(frame, (320, 240))
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,  # 需要被检测的图像，一般为灰度图像加快检测速度
                                          scaleFactor=1.3,  # 在前后两次相继的扫描中，搜索窗口的缩放比例
                                          minNeighbors=5,  # 构成检测目标的相邻矩形的最小个数
                                          minSize=(5, 5),  # 需要被检测的最小尺寸
                                          flags=cv.CASCADE_SCALE_IMAGE)  # 级联扩展图像
    for (x, y, w, h) in faces:
        cv.rectangle(frame, (x, y), (x + w, y + w), (255, 255, 0), 2)  # 在人脸范围画矩形框
        f = cv.resize(gray[y: y + h, x:x + w], (64, 64))
        frame_id, conf = recognizer.predict(f)  # 人脸预测
        print("{0} {1}".format(frame_id, conf))
        if conf < 130:  # 置信度
            cv.putText(frame, user_name_list[frame_id], (x, y - 10), font, 0.8, (0, 255, 0), 2)  # 显示用户名
        else:  # 未能识别的人脸
            cv.putText(frame, user_name_list[0], (x, y - 5), font, 0.5, (255, 255, 0), 1)  # 显示未能识别
    cv.imshow('face_recognition', frame)
    # 键盘按 Q 退出
    if (cv.waitKey(10) & 0xFF) == ord('q'):
        break

# 释放资源
cam.release()
cv.destroyAllWindows()
