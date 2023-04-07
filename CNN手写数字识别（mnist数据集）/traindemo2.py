# -*- coding: utf-8 -*-

import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation
from tensorflow.keras.optimizers import RMSprop
import os
import tensorflow as tf

from PIL import Image
import numpy as np



#---------------------------载入数据及预处理---------------------------
# 下载MNIST数据 
# X shape(60000, 28*28) y shape(10000, )
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# 数据预处理
X_train = X_train.reshape(X_train.shape[0], -1) / 255  # normalize
X_test = X_test.reshape(X_test.shape[0], -1) / 255     # normalize

# 将类向量转化为类矩阵  数字 5 转换为 0 0 0 0 0 1 0 0 0 0 矩阵
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)

#---------------------------创建神经网络层---------------------------
# Another way to build your neural net
model = Sequential([
        Dense(32, input_dim=784),  # 输入值784(28*28) => 输出值32
        Activation('relu'),        # 激励函数 转换成非线性数据
        Dense(10),                 # 输出为10个单位的结果
        Activation('softmax')      # 激励函数 调用softmax进行分类
        ])

# Another way to define your optimizer
rmsprop = RMSprop(learning_rate=0.001, rho=0.9, epsilon=1e-08, decay=0.0) #学习率lr

# We add metrics to get more results you want to see
# 激活神经网络
model.compile(
        optimizer = rmsprop,                 # 加速神经网络
        loss = 'categorical_crossentropy',   # 损失函数
        metrics = ['accuracy'],               # 计算误差或准确率
        )

#------------------------------训练及预测  保存------------------------------

# 保存训练结果
check_path = os.path.abspath(os.path.dirname( __file__)) +  '\\cnn\\ckpt2\\cp-{epoch:04d}.ckpt'
save_model_cb = tf.keras.callbacks.ModelCheckpoint(check_path, save_weights_only=True, verbose=1)

print("Training")
model.fit(X_train, y_train, epochs=1000, batch_size=50000, callbacks=[save_model_cb])    # 训练次数及每批训练大小
print("Testing")
loss, accuracy = model.evaluate(X_test, y_test)

print("loss:", loss)
print("accuracy:", accuracy)





latest = tf.train.latest_checkpoint('.\\cnn\\ckpt2')
# # 恢复网络权重
model.load_weights(latest)




image_path = ".\\test_images\\3.png"

img = Image.open(image_path)
img.show()
img = img.convert('L').resize((28, 28))
img.show()
img_array = np.array(img)
# 将像素值转换为0-1之间的浮点数
img_array = img_array.astype('float32') / 255.0
img_array_result = np.reshape(img_array, (1, 784))


prediction  = model.predict(img_array_result)
prediction_result = np.argmax(prediction)

# 因为x只传入了一张图片，取y[0]即可
# np.argmax()取得最大值的下标，即代表的数字
print(image_path)
print(prediction_result)