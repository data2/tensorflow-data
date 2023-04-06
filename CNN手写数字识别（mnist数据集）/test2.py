import tensorflow as tf
from PIL import Image
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation
from tensorflow.keras.optimizers import RMSprop

'''
python 3.7 3.9
tensorflow 2.0.0b0
pillow(PIL) 4.3.0
'''



#-------------------------test ---------------


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

#---------------------------------------


latest = tf.train.latest_checkpoint('.\\cnn\\ckpt2')
# 恢复网络权重
model.load_weights(latest)

image_path = ".\\test_images\\2.png"

img = Image.open(image_path)
img = img.convert('L').resize((28, 28))
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