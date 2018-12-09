import cv2 as cv

img_path = '../testPhoto/Andrew_Ng.jpg'  # 检测图片的路径
face_xml_path = '../xml/haarcascade_frontalface_default.xml'  # 人脸检测器（默认）xml文件的路径
eye_xml_path = '../xml/haarcascade_eye_tree_eyeglasses.xml'  # 眼睛和眼镜检测器 xml文件的路径

img = cv.imread(img_path)  # 读取图像，支持 bmp、jpg、png、tiff 等常用格式
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # 转换为灰度图

face_cascade = cv.CascadeClassifier(face_xml_path)  # 加载人脸分类器
eye_cascade = cv.CascadeClassifier(eye_xml_path)  # 加载眼睛和眼镜分类器

faces = face_cascade.detectMultiScale(gray, 1.3, 5)  # 人脸检测
for (x, y, w, h) in faces:
    cv.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)  # 在人脸范围画矩形框
    roi_gray = gray[y:y + h, x:x + w]  # 提取出人脸所在区域用于转换为灰度图检测眼睛和眼镜
    roi_color = img[y:y + h, x:x + w]  # 提取出人脸所在区域用于绘制矩形框显示检测的位置
    eyes = eye_cascade.detectMultiScale(roi_gray)  # 眼睛和眼镜检测
    for (ex, ey, ew, eh) in eyes:  # 在眼睛和眼镜范围画矩形框
        cv.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
print("发现{0}个人脸，{0}双眼睛!".format(len(faces), len(eyes)))
cv.imshow('face_and_eye_detection', img)# 显示图片
cv.waitKey(0)
cv.destroyAllWindows()
