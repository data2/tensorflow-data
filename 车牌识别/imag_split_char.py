import numpy as np
import cv2

# 尝试读取车牌图片
license_plate = cv2.imread('C:/Users/admin/Desktop/plate.png')
# 打印图片数据，如果为None则表示图片未成功读取
print(license_plate)

# 将图片从BGR颜色空间转换为灰度空间
# 注意：cv2默认读取图片为BGR格式，因此使用COLOR_BGR2GRAY
gray_plate = cv2.cvtColor(license_plate, cv2.COLOR_BGR2GRAY)

# 使用固定阈值进行二值化处理
ret, binary_plate = cv2.threshold(gray_plate, 175, 255, cv2.THRESH_BINARY)

# 初始化一个列表，用于存储每列的像素和（这个方法对于分割字符并不高效）
result = []
for col in range(binary_plate.shape[1]):
    result.append(0)
    for row in range(binary_plate.shape[0]):
        # 累加每列的像素值（但此处/255似乎没有意义，因为binary_plate已经是二值化的）
        result[col] = result[col] + binary_plate[row][col] / 255

    # 初始化一个字典，用于存储每个字符的边界索引
character_dict = {}
num = 0
i = 0
# 遍历result列表，根据非零值来划分字符边界
while i < len(result):
    if result[i] == 0:
        i += 1
    else:
        index = i + 1
        # 寻找字符的结束边界
        while index < len(result) and result[index] != 0:
            index += 1
            # 将字符的起始和结束索引存储到字典中
        character_dict[num] = [i, index - 1]
        num += 1
        i = index

    # 初始化一个列表，用于存储分割后的字符图像
characters = []
# 遍历所有字符（这里假设车牌有8个字符，但跳过第二个位置，可能考虑到分隔符）
for i in range(8):
    if i == 2:
        continue
        # 计算字符区域的宽度，并计算需要添加的padding量以保持字符宽度一致
    padding = (170 - (character_dict[i][1] - character_dict[i][0])) / 2
    # 使用np.pad在字符周围添加padding
    ndarray = np.pad(binary_plate[:, character_dict[i][0]:character_dict[i][1]],
                     ((0, 0), (int(padding), int(padding))),
                     'constant', constant_values=(0, 0))
    # 调整字符图像大小为20x20
    ndarray = cv2.resize(ndarray, (20, 20))
    # 保存字符图像
    cv2.imwrite('./' + str(i) + '.png', ndarray)
    # 将字符图像添加到列表中
    characters.append(ndarray)

# 注意：此代码存在多个问题和潜在的改进点
# 1. 使用列像素和来划分字符可能不准确，特别是对于大小不一或紧密排列的字符。
# 2. 硬编码车牌字符数量和跳过特定位置（如第二个位置）可能不适用于所有情况。
# 3. 添加padding的数值（如170）和最终尺寸（20x20）可能是根据特定情况选择的，可能需要调整以适应不同大小的车牌。