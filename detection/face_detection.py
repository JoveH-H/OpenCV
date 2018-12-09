import cv2 as cv
from datetime import datetime

img_path = '../testPhoto/Andrew_Ng.jpg'  # 检测图片的路径
xml_path = '../xml/haarcascade_frontalface_default.xml'  # 人脸检测器（默认）xml文件的路径

a = datetime.now()  # 记录开始时间

img = cv.imread(img_path)  # 读取图像，支持 bmp、jpg、png、tiff 等常用格式
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # 转换为灰度图

# 人脸检测
face_cascade = cv.CascadeClassifier(xml_path)  # 加载级联分类器

faces = face_cascade.detectMultiScale(gray,  # 需要被检测的图像，一般为灰度图像加快检测速度
                                      scaleFactor=1.3,  # 在前后两次相继的扫描中，搜索窗口的缩放比例
                                      minNeighbors=5,  # 构成检测目标的相邻矩形的最小个数
                                      minSize=(5, 5),  # 需要被检测的最小尺寸
                                      maxSize=(200, 200),  # 需要被检测的最大尺寸
                                      flags=cv.CASCADE_SCALE_IMAGE)  # 级联扩展图像
print("发现{0}个人脸!".format(len(faces)))

for (x, y, w, h) in faces:  # 在人脸范围画矩形框
    cv.rectangle(img, (x, y), (x + w, y + w), (255, 255, 0), 2)

b = datetime.now()  # 记录结束时间
print(b - a)  # 计算运算时间
cv.imshow("face_detection", img)  # 显示图片

cv.waitKey(0)
cv.destroyAllWindows()