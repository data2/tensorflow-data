
import numpy as np
import paddle as paddle
import paddle.fluid as fluid
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import os
from multiprocessing import cpu_count

########################################################
# 生成车牌字符图像列表
data_path = './data'
character_folders = os.listdir(data_path)

label = 0
if(os.path.exists('./train_data.list')):
    os.remove('./train_data.list')
if(os.path.exists('./test_data.list')):
    os.remove('./test_data.list')
for character_folder in character_folders:
    with open('./train_data.list', 'a') as f_train:
        with open('./test_data.list', 'a') as f_test:
            if character_folder == '.DS_Store' or character_folder == '.ipynb_checkpoints' or character_folder == 'data18414':
                continue
            print(character_folder + " " + str(label))
            character_imgs = os.listdir(os.path.join(data_path, character_folder))
            for i in range(len(character_imgs)):
                if i%10 == 0:
                    f_test.write(os.path.join(os.path.join(data_path, character_folder), character_imgs[i]) + "\t" + str(label) + '\n')
                else:
                    f_train.write(os.path.join(os.path.join(data_path, character_folder), character_imgs[i]) + "\t" + str(label) + '\n')
    label = label + 1
print('图像列表已生成')

########################################################
# 用上一步生成的图像列表定义车牌字符训练集和测试集的reader
def train_mapper(sample):
    img, label = sample
    img = paddle.dataset.image.load_image(file=img, is_color=False)
    img = img.flatten().astype('float32') / 255.0
    return img, label


def train_r(train_list_path):
    def reader():
        with open(train_list_path, 'r') as f:
            lines = f.readlines()
            del lines[len(lines) - 1]
            for line in lines:
                img, label = line.split('\t')
                yield img, int(label)

    return paddle.reader.xmap_readers(train_mapper, reader, cpu_count(), 1024)


def test_mapper(sample):
    img, label = sample
    img = paddle.dataset.image.load_image(file=img, is_color=False)
    img = img.flatten().astype('float32') / 255.0
    return img, label


def test_r(test_list_path):
    def reader():
        with open(test_list_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                img, label = line.split('\t')
                yield img, int(label)

    return paddle.reader.xmap_readers(test_mapper, reader, cpu_count(), 1024)


########################################################

# 用于训练的数据提供器
train_reader = paddle.batch(reader=paddle.reader.shuffle(reader=train_r('./train_data.list'), buf_size=3000), batch_size=128)
# 用于测试的数据提供器
test_reader = paddle.batch(reader=test_r('./test_data.list'), batch_size=128)


########################################################

# 搭建leNet5卷积神经网络，用于识别车牌的每一个字符
def convolutional_neural_network(input):
    # 卷积层，卷积核大小为3*3，步长是1，一共有32个卷积核
    conv_1 = fluid.layers.conv2d(input=input, num_filters=50, filter_size=5, stride=1)
    # 池化层，池化核大小为2*2，步长是1，最大池化
    pool_1 = fluid.layers.pool2d(input=conv_1, pool_size=2, pool_stride=1, pool_type='max')
    # 第二个卷积层，卷积核大小为3*3，步长1，一共有64个卷积核
    conv_2 = fluid.layers.conv2d(input=pool_1, num_filters=32, filter_size=3, stride=1)
    # 第二个池化层，池化核大小是2*2，步长1，最大池化
    pool_2 = fluid.layers.pool2d(input=conv_2, pool_size=2, pool_stride=1, pool_type='max')

    # 以softmax为激活函数的全连接输出层，大小为label的大小
    # softmax一般用于多分类问题最后一层输出层的激活函数，作用是对输出归一化，这种情况下一般损失函数使用交叉熵

    fc = fluid.layers.fc(input=pool_2, size=65, act='softmax')
    return fc

########################################################
# 定义占位输入层和标签层
# 图像是20*20的灰度图，所以输入的形状是[1,20,20]（灰度图是1通道，彩图3通道），理论上应该还有一个维度是Batch，PaddlePaddle帮我们默认设置，可以不设置Batch
image = fluid.layers.data(name='image', shape=[1, 20, 20], dtype='float32')
label = fluid.layers.data(name='label', shape=[1], dtype='int64')

# 获取前向传播网络结果
result = convolutional_neural_network(image)

# 定义损失函数和准确率函数
cost = fluid.layers.cross_entropy(input=result, label=label)  # 交叉熵
avg_cost = fluid.layers.mean(cost)  # 整个Batch的平均值
accuracy = fluid.layers.accuracy(input=result, label=label)

# 在定义优化之前，克隆主程序，获得一个预测程序
test_program = fluid.default_main_program().clone(for_test=True)

# 定义优化器，使用Adam优化器
optimizer = fluid.optimizer.AdamOptimizer(learning_rate=0.001)

# 指定优化目标函数
opts = optimizer.minimize(avg_cost)


########################################################
# 创建执行器
use_cuda = False
place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()

exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())

########################################################
feeder = fluid.DataFeeder( feed_list=[image, label],place=place)

########################################################

# 开始训练，20个pass
for pass_id in range(20):
    for batch_id, data in enumerate(train_reader()):
        train_cost, train_acc = exe.run(program=fluid.default_main_program(),
                                        feed=feeder.feed(data),
                                        fetch_list=[avg_cost, accuracy])
        if batch_id % 100 == 0:
            print('\nPass：%d, Batch：%d, Cost：%f, Accuracy：%f' % (pass_id, batch_id, train_cost[0], train_acc[0]))
        else:
            print('.', end="")

    # 每一个pass进行一次测试
    test_costs = []
    test_accs = []
    for batch_id, data in enumerate(test_reader()):
        test_cost, test_acc = exe.run(program=test_program,
                                      feed=feeder.feed(data),
                                      fetch_list=[avg_cost, accuracy])
        test_costs.append(test_cost[0])
        test_accs.append(test_acc[0])
    test_cost = sum(test_costs) / len(test_costs)
    test_acc = sum(test_accs) / len(test_accs)
    print('\nTest：%d, Cost：%f, Accuracy：%f' % (pass_id, test_cost, test_acc))

    fluid.io.save_inference_model(dirname='./model', feeded_var_names=['image'], target_vars=[result], executor=exe)

########################################################

# 对车牌图片进行处理，分割出车牌中的每一个字符并保存
license_plate = cv2.imread('./车牌.png')
gray_plate = cv2.cvtColor(license_plate, cv2.COLOR_RGB2GRAY)
ret, binary_plate = cv2.threshold(gray_plate, 175, 255, cv2.THRESH_BINARY)
result = []
for col in range(binary_plate.shape[1]):
    result.append(0)
    for row in range(binary_plate.shape[0]):
        result[col] = result[col] + binary_plate[row][col]/255
character_dict = {}
num = 0
i = 0
while i < len(result):
    if result[i] == 0:
        i += 1
    else:
        index = i + 1
        while result[index] != 0:
            index += 1
        character_dict[num] = [i, index-1]
        num += 1
        i = index

characters = []
for i in range(8):
    if i==2:
        continue
    padding = (170 - (character_dict[i][1] - character_dict[i][0])) / 2
    ndarray = np.pad(binary_plate[:,character_dict[i][0]:character_dict[i][1]], ((0,0), (int(padding), int(padding))), 'constant', constant_values=(0,0))
    ndarray = cv2.resize(ndarray, (20,20))
    cv2.imwrite('./' + str(i) + '.png', ndarray)
    characters.append(ndarray)


########################################################
[infer_program, feeded_var_names, target_vars] = fluid.io.load_inference_model(dirname='./model', executor=exe)

########################################################
# 对字符图片进行预处理
def load_image(path):
    img = paddle.dataset.image.load_image(file=path, is_color=False)
    img = img.astype('float32')
    img = img[np.newaxis, ] / 255.0
    return img
########################################################
# 执行预测
labels = []
for i in range(8):
    if i==2:
        continue
    infer_imgs = []
    infer_imgs.append(load_image('./' + str(i) + '.png'))
    infer_imgs = np.array(infer_imgs)
    result = exe.run(program=infer_program,
                 feed={feeded_var_names[0]:infer_imgs},
                 fetch_list=target_vars)
    labels.append(np.argsort(result)[0][0][-1])
########################################################
# 将原车牌图片和预测后对结果打印出来
# match = {0:'Z', 1:'云', 2:'桂', 3:'G', 4:'E', 5:'2', 6:'甘', 7:'5', 8:'3', 9:'6', 10:'C', 11:'F', 12:'川', 13:'京', 14:'沪', 15:'R', 16:'新',
#     17:'0', 18:'X', 19:'闽', 20:'4', 21:'J', 22:'湘', 23:'苏', 24:'陕', 25:'藏', 26:'冀', 27:'皖', 28:'青', 29:'K', 30:'渝', 31:'A', 32:'N', 33:'W',
#     34:'P', 35:'7', 36:'吉', 37:'1', 38:'V', 39:'浙', 40:'D', 41:'豫', 42:'宁', 43:'蒙', 44:'L', 45:'Q', 46:'鲁', 47:'津', 48:'晋', 49:'S', 50:'M',
#     51:'8', 52:'B', 53:'9', 54:'赣', 55:'琼', 56:'黑', 57:'Y', 58:'贵', 59:'辽', 60:'鄂', 61:'T', 62:'H', 63:'粤', 64:'U'
# }
# display(Image.open('./车牌.png'))
# print('\n车牌识别结果为：',end='')
# for i in range(len(labels)):
#     print(match[labels[i]], end='')
