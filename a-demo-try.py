
import tensorflow as tf

# 张量（Tensor）
# TensorFlow 内部的计算都是基于张量的，因此我们有必要先对张量有个认识。张量是在我们熟悉的标量、向量之上定义的，详细的定义比较复杂，我们可以先简单的将它理解为一个多维数组：

# 创建一个整型常量，即 0 阶 Tensor
t0 = tf.constant(3, dtype=tf.int32)

# 创建一个浮点数的一维数组，即 1 阶 Tensor
t1 = tf.constant([3., 4.1, 5.2], dtype=tf.float32)

# 创建一个字符串的2x2数组，即 2 阶 Tensor
t2 = tf.constant([['Apple', 'Orange'], ['Potato', 'Tomato']], dtype=tf.string)

# 创建一个 2x3x1 数组，即 3 阶张量，数据类型默认为整型
t3 = tf.constant([[[5], [6], [7]], [[4], [3], [2]]])


print(t0)
print(t1)
print(t2)