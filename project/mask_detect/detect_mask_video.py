# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import pymysql
import winsound
from threading import Thread
import os
import time

def task():
    winsound.Beep(600, 100)

prototxtPath = r"static/model/deploy.prototxt"
weightsPath = r"static/model/res10_300x300_ssd_iter_140000.caffemodel"

#加载人脸检测的模型 参数分别为 模型 和 模型的参数 来自于网络
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# 加载口罩模型
maskNet = load_model("static/model/mask_detector.model")

# 连接数据库 这里连接的是mysql数据库 因为是个人电脑 所以没设置密码 连的是主机 其中存放的信息只有一张user表 对应账号密码邮箱 用于登陆处理
# conn = pymysql.connect(host='localhost',port=3306,user='root',passwd='',db='maskdetector')
# cur = conn.cursor()
# login实现登录  ** login代码修改到了views.py中
# def login():

# frame为图片（视频流中的一帧） faceNet与maskNet分别为人脸识别模型和口罩识别模型
def detect_and_predict_mask(frame, faceNet, maskNet):

    # 窗口的长和宽
    (h, w) = frame.shape[:2]

    #windowPara = (h,w)

    #对图像进行预处理 frame是捕获到的图片 1.0是缩放比例（选择不缩放） 裁剪大小为（244，244）这么大 最后一个元组降低光照的影响    !!!opencv
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224), (104.0, 177.0, 123.0))

    # print(blob)
    # print('****')
    faceNet.setInput(blob)
    detections = faceNet.forward()

    #print(detections.shape)
    #print(detections)

    # 初始化三个列表 分别存放处理后的face ; face坐标 ;对于face的mask检验预测值
    faces = []
    locs = []
    preds = []

    # 以下为人脸识别的代码块，来自于网络
    for i in range(0, detections.shape[2]):

        # 他可能框出来很多是人脸的地方 算置信度 如果大于50%就框出来
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:

            # box类似于容器 找出face框的坐标
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # 进一步确认 出现的face框同样出现在镜头中
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # 裁剪出来face图
            face = frame[startY:endY, startX:endX]
            # 脸变成黑白图
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            # 缩放脸大小 224*224 网路的输入
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            # 数据处理完放入数组中 在口罩识别备用
            faces.append(face)
            locs.append((startX, startY, endX, endY))

    # faces中存有值 即识别出人脸
    if len(faces) > 0:
        # 进行口罩检测
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)

    # 返回元组 (坐标列表,预测值列表) 坐标列表中每个元素为人脸识别框对角线顶点坐标（共四个坐标） 预测值列表中每个元素为二元组（戴口罩概率,不戴口罩概率）
    return (locs, preds)


if __name__ == '__main__':

    #
    print("[INFO] starting video stream...")
    # VideoCapture参数为0代表笔记本摄像头 为1代表其他摄像头 其他数字类推 代码获取视频流
    webcam = cv2.VideoCapture(0)
    if not webcam.isOpened():
        raise Exception("There is something wrong with the camera!")

    # 设置窗口分辨率 1920 x 1080
    webcam.set(3,1920)
    webcam.set(4,1080)


    # 循环读取视频流每一帧
    while True:
        # 获取摄像头 ret为true or false frame为视频流当前帧
        ret, frame = webcam.read()

        # 检测是否佩戴口罩 先检测人脸，再识别口罩 三个帧率分别为 摄像头传来的实时帧 人脸识别模型 口罩识别模型 （其中人脸识别模型来自于网络 口罩识别模型为自己训练而得）
        (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)


        # 找出每一个人脸的位置并裁剪出框以及显示口罩识别的结果
        for (box, pred) in zip(locs, preds):

            # 对应人脸识别后裁剪出的人脸坐标框的对角线顶点
            (startX, startY, endX, endY) = box

            #这里mask和withoutMask是两个预测的概率 加和为1

            (mask, withoutMask) = pred



            if mask > withoutMask:
                label = 'with mask'  # 想做成中文标签 但是cv2对于中文并不支持
                color = (0, 255, 0)  # green
            else:
                label = "without mask"
                color = (0, 0, 255)  # red
                # 利用线程处理 否则因为执行Beep时间过长 视觉效果上会丢帧
                t = Thread(target=task)
                t.start()
                # winsound.Beep(600,300)

            # 展示标签
            cv2.putText(frame, label, (startX+100, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

        # imgName = "static/photo/frame.jpg"
        # cv2.imwrite(imgName,frame)

        cv2.imshow("Mask Detect", frame)
        key = cv2.waitKey(1) & 0xFF
        # 监控键盘输入 只要输入q 无论大小写直接退出窗口
        if key == ord("q") or key == ord("Q"):
            break
    #关闭视频流
    webcam.release()
    cv2.destroyAllWindows()
