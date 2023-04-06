import os
import tensorflow as tf
from tensorflow.keras import datasets, layers, models

from keras.models import Sequential
from keras.layers import Dense, Activation
from tensorflow.keras.optimizers import RMSprop
'''
python 3.7、3.9
tensorflow 2.0.0b0
'''

class CNN(object):
    def __init__(self):
        # 模型定义的前半部分主要使用Keras.layers 提供的Conv2D（卷积）与MaxPooling2D（池化）函数。
        # CNN的输入是维度为（image_height, image_width, color_channels）的张量，
        # mnist数据集是黑白的，因此只有一个color_channels 颜色通道；一般的彩色图片有3个（R, G, B），
        # 也有4个通道的（R, G, B, A），A代表透明度；
        # 对于mnist数据集，输入的张量维度为(28, 28, 1)，通过参数input_shapa 传给网络的第一层
        # CNN模型处理:

        # model = models.Sequential()
        # # 第1层卷积，卷积核大小为3*3，32个，28*28为待训练图片的大小
        # model.add(layers.Conv2D(
        #     32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
        # model.add(layers.MaxPooling2D((2, 2)))
        # # 第2层卷积，卷积核大小为3*3，64个
        # model.add(layers.Conv2D(64, (3, 3), activation='relu'))  # 使用神经网络中激活函数ReLu
        # model.add(layers.MaxPooling2D((2, 2)))
        # # 第3层卷积，卷积核大小为3*3，64个
        # model.add(layers.Conv2D(64, (3, 3), activation='relu'))

        # model.add(layers.Flatten())
        # model.add(layers.Dense(64, activation='relu'))
        # model.add(layers.Dense(10, activation='softmax'))
        # # Flatten层用来将输入“压平”，即把多维的输入一维化，常用在从卷积层到全连接层的过渡。Flatten不影响batch的大小
        # # dense ：全连接层相当于添加一个层
        # # softmax用于多分类过程中，它将多个神经元的输出，映射到（0,1）区间内，可以看成概率来理解，从而来进行多分类！
        # model.summary()  # 输出模型各层的参数状况


        # 另外一种方式
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


        self.model = model


# mnist数据集预处理
class DataSource(object):
    def __init__(self):
        # mnist数据集存储的位置，如果不存在将自动下载
        data_path = os.path.abspath(os.path.dirname(
            __file__)) + '\\data_set_tf2\\mnist.npz'
        (train_images, train_labels), (test_images,test_labels) = datasets.mnist.load_data(path=data_path)
        # (train_images, train_labels), (test_images,test_labels) = datasets.mnist.load_data()
        # 6万张训练图片，1万张测试图片

        train_images = train_images.reshape((60000, 28, 28, 1))
        test_images = test_images.reshape((10000, 28, 28, 1))
        # 像素值映射到 0 - 1 之间
        train_images, test_images = train_images / 255.0, test_images / 255.0

        self.train_images, self.train_labels = train_images, train_labels
        self.test_images, self.test_labels = test_images, test_labels



# 开始训练并保存训练结果
class Train:
    def __init__(self):
        self.cnn = CNN()
        self.data = DataSource()

    def train(self):
        check_path = os.path.abspath(os.path.dirname( __file__)) +  '\\cnn\\ckpt1\\cp-{epoch:04d}.ckpt'
        print(check_path)
        # period 每隔5epoch保存一次
        save_model_cb = tf.keras.callbacks.ModelCheckpoint(check_path, save_weights_only=True, verbose=1)

        # self.cnn.model.compile(optimizer='adam',
        #                        loss='sparse_categorical_crossentropy',
        #                        metrics=['accuracy'])
        self.cnn.model.fit(self.data.train_images, self.data.train_labels,
                           epochs=2,  batch_size=32, callbacks=[save_model_cb])

        test_loss, test_acc = self.cnn.model.evaluate(self.data.test_images, self.data.test_labels)
        print(111)
        print("准确率: %.4f，共测试了%d张图片 " % (test_acc, len(self.data.test_labels)))


if __name__ == "__main__":
    app = Train()
    app.train