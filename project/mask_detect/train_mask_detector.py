
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os

INIT_LR = 1e-4#学习率设置
EPOCHS = 10#定型周期数量
# batch size
BS = 32#单次训练所抓取的数据亮量


#得到用于训练的数据集
DIRECTORY = r"D:\code\dataset"
CATEGORIES = ["with_mask", "without_mask"]

print("[INFO] loading images...")

data = []
labels = []

#拼接数据集中图片的路径，并将图片数据规范储存
for category in CATEGORIES:
    path = os.path.join(DIRECTORY, category)
    for img in os.listdir(path):
    	img_path = os.path.join(path, img)#将数据集中图片的路径拼接完整
    	image = load_img(img_path, target_size=(224, 224))#指定图片大小为224*224
    	image = img_to_array(image)
    	image = preprocess_input(image)

#将新图片添加到数组
    	data.append(image)
    	labels.append(category)

# perform one-hot encoding on the labels
#使用热独立编码
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

#将结果存为新数组
data = np.array(data, dtype="float32")
labels = np.array(labels)

#使用train_test_split函数分割数据集
#训练集和测试集八二分，
(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.20, stratify=labels, random_state=42)

# 使用keras.preprocessing.image模块中的ImageDataGenerator
# 构建图片生成器产生作为神经网络输入的训练图像
aug = ImageDataGenerator(
	rotation_range=20,#旋转范围
	zoom_range=0.15,#缩放范围
	width_shift_range=0.2,#水平平移范围
	height_shift_range=0.2,#垂直平移范围
	shear_range=0.15,#透视变换的范围
	horizontal_flip=True,#水平反转
	fill_mode="nearest")#填充模式

# 载入MobileNetV2模型作为基础模型
# left off
baseModel = MobileNetV2(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(224, 224, 3)))

# 构建基础模型的顶部模型
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)#使用7*7的池化
headModel = Flatten(name="flatten")(headModel)#进行flatten操作图片数据为一维
#将池化结果放入128输出的全连接层，使用relu函数为激活函数
headModel = Dense(128, activation="relu")(headModel)
#由于数据样本偏少，使用dropout函数避免过拟合，
headModel = Dropout(0.5)(headModel)
#二输出的全连接层，使用softmax函数进行归一化处理得到概率
headModel = Dense(2, activation="softmax")(headModel)

#将顶部模型置于基础模型上
model = Model(inputs=baseModel.input, outputs=headModel)

#遍历基础模型每层并使其权值无法改变来
for layer in baseModel.layers:
	layer.trainable = False

print("[INFO] compiling model...")

#使用Adam算法代替传统的随机梯度下降算法作为优化算法
#损失函数（二元分类交叉熵损失函数）和评价函数直接调用keras自带函数
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# 训练网络
print("[INFO] training head...")
H = model.fit(
	aug.flow(trainX, trainY, batch_size=BS),#训练元组
	steps_per_epoch=len(trainX) // BS,#样本采集数
	validation_data=(testX, testY),#评估元组
	validation_steps=len(testX) // BS,#验证集上的step总数
	epochs=EPOCHS)#迭代数20

# 进行预测与优化
print("[INFO] evaluating network...")
#使用testX作为测试数据
predIdxs = model.predict(testX, batch_size=BS)

# 找出计算结果中最大预测概率对应标签
predIdxs = np.argmax(predIdxs, axis=1)

# 显示较好的结果
#testY.argmax(axis=1)为真实值
print(classification_report(testY.argmax(axis=1), predIdxs,
	target_names=lb.classes_))

# 保存模型为h5格式
print("[INFO] saving mask detector model...")
model.save("mask_detector.model", save_format="h5")

# 绘制准确旅曲线
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")