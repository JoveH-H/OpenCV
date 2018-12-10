from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import numpy as np

FaceID = 2  # 设置用户ID
face_num = 15  # 15张用户单人图片
new_face_num = 49  # 每张用户图片新生成图片的张数，即有50张图片来自同一张照片,需大于0

date_path = '../testPhoto/userFace/dateSet'  # 用户灰度图片文件夹路径

datagen = ImageDataGenerator(  # 图片生成器
    rotation_range=45,  # 旋转范围, 随机旋转(0-180)度
    width_shift_range=0.2,  # 随机沿着水平，以图像的宽小部分百分比为变化范围进行平移;
    height_shift_range=0.2,  # 随机沿着垂直，以图像的长小部分百分比为变化范围进行平移;
    rescale=1 / 255,  # 对图像按照指定的尺度因子, 进行放大或缩小, 设置值在0 - 1之间，通常为1 / 255;
    shear_range=0.2,  # 水平或垂直投影变换
    zoom_range=0.2,  # 按比例随机缩放图像尺寸
    horizontal_flip=True,  # 水平翻转图像
    fill_mode="nearest"  # 出现在旋转或平移之后，取最近的像素填充
)

for sampleNum in range(1, face_num + 1):  # 获取用户单人照片的人脸，生成新图片后保存
    img = load_img(date_path + "/user." + str(FaceID) + "." + str(sampleNum) + ".jpg")  # 加载用户单人图片
    X = img_to_array(img)  # 图片转化成数组
    X = np.expand_dims(X, 0)  # 拓展维度
    i = 0
    for batch in datagen.flow(X, batch_size=1, save_to_dir=date_path,  # 设置保存参数
                              save_prefix="user." + str(FaceID) + ".", save_format="jpg"):
        i += 1
        if i == new_face_num:  # 每张用户图片新生成 new_face_num 张图片
            break
# 为了后期的使用，还需对生成的图片进行重命名 user."FaceID". ("编号").jpg
