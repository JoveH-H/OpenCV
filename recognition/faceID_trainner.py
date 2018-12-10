import cv2 as cv
import os
import numpy as np
from PIL import Image

date_path = '../testPhoto/userFace/dateSet'  # 用户灰度图片文件夹路径
xml_path = '../xml/haarcascade_frontalface_default.xml'  # 人脸检测器（默认）xml文件的路径
yml_path = '../xml/userFace.yml'  # 储存模型 yml文件的路径

face_cascade = cv.CascadeClassifier(xml_path)  # 加载人脸分类器
recognizer = cv.face.LBPHFaceRecognizer_create()  # 创建LBPH模型


def get_images_and_labels(path):
    image_paths = [os.path.join(path, f) for f in os.listdir(path)]  # 获取图片地址
    face_samples = []
    ids = []
    for image_path in image_paths:  # 遍历图片路径，导入图片和ID
        image = Image.open(image_path).convert('L')  # 图片转换成“L”模式，像素值为[0,255]之间的某个数值
        image_np = np.array(image, 'uint8')  # 创建图片数组
        if os.path.split(image_path)[-1].split(".")[-1] != 'jpg':  # 非jpg图片跳过
            continue
        image_id = int(os.path.split(image_path)[-1].split(".")[1])  # 获取ID号
        face = face_cascade.detectMultiScale(image_np)  # 检测人脸
        for (x, y, w, h) in face:
            face_samples.append(image_np[y:y + h, x:x + w])  # 添加人脸数据
            ids.append(image_id)  # 添加ID数据
    return face_samples, ids  # 返回人脸和ID数据列表


faces, Ids = get_images_and_labels(date_path)  # 提取人脸和对应ID数据
recognizer.train(faces, np.array(Ids))  # 训练模型
# recognizer.update(faces, np.array(Ids))  # 升级模型
recognizer.save(yml_path)  # 保存模型
