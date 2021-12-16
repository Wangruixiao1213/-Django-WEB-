# 模拟体温生成 实则随机数生成
import random
import time


def temperView(LTX,LTY,RBX,RBY):
    # 对摄像头在 (LTX,LTY) 到 (RBX,RBY)这两个对角线顶点的框中进行体温检测 由于未有温度测量模块 故选择随机数生成
    
    temperMea = random.uniform(36.0,37.0)
    temperChe = str(temperMea)
    temperChe = temperChe[:4]  #保留三位有效数字
    # time.sleep(0.5)
    return temperChe