import cv2
import pyzbar.pyzbar as pyzbar


class point_polygon:
    def __init__(self, polygon):
        self.x1, self.y1 = polygon[0]
        self.x2, self.y2 = polygon[1]
        self.x3, self.y3 = polygon[2]
        self.x4, self.y4 = polygon[3]


def decodeDisplay(image):
    # 转为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    barcodes = pyzbar.decode(gray, symbols=[pyzbar.ZBarSymbol.QRCODE])
    for barcode in barcodes:
        # 提取条形码的边界框的位置
        # 画出图像中二维码范围的正方形边界框和中心点
        (x, y, w, h) = barcode.rect
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(image, (int(x + w / 2), int(y + h / 2)), 7, (255, 255, 0), -1)

        # 绘画二维码的边缘框
        cv2.line(image, barcode.polygon[0], barcode.polygon[1], (0, 255, 0), 2)
        cv2.line(image, barcode.polygon[1], barcode.polygon[2], (0, 255, 0), 2)
        cv2.line(image, barcode.polygon[2], barcode.polygon[3], (0, 255, 0), 2)
        cv2.line(image, barcode.polygon[3], barcode.polygon[0], (0, 255, 0), 2)

        # 二维码数据为字节对象
        barcodeData = barcode.data.decode("utf-8")

        # 向终端打印二维码信息
        print("[INFO] 中心位置：({:3.1f},{:3.1f}) 内容：{}".format(x + w / 2, y + h / 2, barcodeData))
    return image


def detect():
    # 读取当前帧
    frame = cv2.imread("./QR.png")

    im = decodeDisplay(frame)
    cv2.imshow("camera", im)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    detect()
