import cv2 as cv

FaceID = 2  # 设置用户ID
user_name = "Andrew_Ng"  # 设置用户名
face_num = 15  # 15张用户单人图片
img_size = 64  # 调整后正方形图片的单边像素

xml_path = '../xml/haarcascade_frontalface_default.xml'  # 人脸检测器（默认）xml文件的路径
img_path = '../testPhoto/userFace/' + str(user_name)  # 检测图片文件夹的路径
date_path = '../testPhoto/userFace/dateSet'  # 存储获取的灰度图片文件夹路径

face_cascade = cv.CascadeClassifier(xml_path)  # 加载人脸分类器

for sampleNum in range(1, face_num + 1):  # 获取用户单人照片的人脸，处理后保存
    img = cv.imread(img_path + "/" + user_name + "(" + str(sampleNum) + ").jpg")  # 加载用户的单人图片
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # 转换为灰度图
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)  # 人脸检测
    for (x, y, w, h) in faces:
        f = cv.resize(gray[y: y + h, x:x + w], (img_size, img_size))  # 调整人脸灰度图的尺寸
        cv.imwrite(date_path + "/user." + str(FaceID) + '.' + str(sampleNum) + ".jpg", f)  # 保存人脸灰度图
