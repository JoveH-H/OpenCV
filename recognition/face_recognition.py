import cv2 as cv

img_path = '../testPhoto/four_people.jpg'  # 检测图片的路径
face_xml_path = '../xml/haarcascade_frontalface_default.xml'  # 人脸检测器（默认）xml文件的路径
eye_xml_path = '../xml/haarcascade_eye_tree_eyeglasses.xml'  # 眼睛和眼镜检测器 xml文件的路径
recognizer_path = '../xml/userFace.yml'  # LBPH人脸识别模型 yml文件的路径

user_name_list = ["Unkown", 'H-H', 'Andrew Ng', 'Jack Ma']  # 用户名单

recognizer = cv.face.LBPHFaceRecognizer_create()  # 创建LBPH模型
recognizer.read(recognizer_path)  # 加载LBPH人脸识别模型
font = cv.FONT_HERSHEY_SIMPLEX  # 设置正常大小无衬线字体

img = cv.imread(img_path)  # 读取图像
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # 转换为灰度图

face_cascade = cv.CascadeClassifier(face_xml_path)  # 加载人脸分类器
eye_cascade = cv.CascadeClassifier(eye_xml_path)  # 加载眼睛和眼镜分类器

faces = face_cascade.detectMultiScale(gray, 1.3, 3)  # 人脸检测

for (x, y, w, h) in faces:
    cv.rectangle(img, (x, y), (x + w, y + w), (255, 255, 0), 2)  # 在人脸范围画矩形框
    f = cv.resize(gray[y: y + h, x:x + w], (64, 64))
    roi_gray = cv.resize(gray[y: y + h, x:x + w], (128, 128))  # 提取出人脸所在区域转换为128*128像素灰度图用于检测眼睛
    roi_color = img[y:y + h, x:x + w]  # 提取出人脸所在区域用于绘制矩形框显示眼睛的位置
    eyes = eye_cascade.detectMultiScale(roi_gray)  # 眼睛检测
    for (ex, ey, ew, eh) in eyes:  # 在眼睛范围画矩形框
        cv.rectangle(roi_color, (ex * w // 128, ey * h // 128),
                     (ex * w // 128 + ew * w // 128, ey * h // 128 + eh * h // 128), (255, 255, 0), 1)

    img_id, conf = recognizer.predict(f)  # 人脸预测
    if conf < 125:  # 置信度
        cv.putText(img, user_name_list[img_id], (x, y - 10), font, 0.8, (0, 255, 0), 2)  # 显示用户名
    else:  # 未能识别的人脸
        cv.putText(img, user_name_list[0], (x, y - 5), font, 0.5, (255, 255, 0), 1)  # 显示未能识别

cv.imshow('face recognition', img)  # 显示图片
cv.waitKey(0)
cv.destroyAllWindows()
