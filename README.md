# pytorch

# paddlepaddle

# tensorflow

# setup environment

查看tensorflow与python的版本，然后确定各自安装版本。eg:python-3.9,tensorflow-2.6.0

## setup python 

https://www.python.org/downloads/windows/

python安装过程中，选择加入环境变量 & 关闭window路径限长（disable path length limit）两个选项

## setup tensorflow

pip install tensorflow-cpu==2.6.0 -i https://pypi.tuna.tsinghua.edu.cn/simple

## 其他可能的注意事项

### 错误1-protobuf

![image](https://user-images.githubusercontent.com/13504729/230704379-dc7c4ee9-4dc8-4402-bbda-30b60867a37a.png)

pip install protobuf==3.20

### 错误2-numpy版本

![image](https://user-images.githubusercontent.com/13504729/230704389-a7fa9c2f-e86e-4be6-9be0-bddd375669ce.png)

错误：pip 的依赖项解析器当前未考虑所有已安装的包。此行为是以下依赖项冲突的根源。tensorflow-cpu 2.6.0 需要 numpy~=1.19.2，但你有 numpy 1.24.2 不兼容。

pip install numpy==1.19.2

### 错误3-numpy依赖visual c++ 14.0

![image](https://user-images.githubusercontent.com/13504729/230704733-f3bb9423-77dc-4b11-ab16-b7f94b42d530.png)

此时需要安装visual c++ 14.0，,可以去微软下载安装  [https://learn.microsoft.com/en-US/cpp/windows/latest-supported-vc-redist?view=msvc-170](https://visualstudio.microsoft.com/zh-hans/downloads/)

安装完后再 pip install numpy==1.19.2

## 需要注意tensorflow与keras版本对应

![image](https://user-images.githubusercontent.com/13504729/230291129-e606ec43-9fec-4091-94b8-95e7a2819f00.png)

pip install keras==2.6

## 测试tensorflow

![image](https://user-images.githubusercontent.com/13504729/230268366-bbd3c479-f90d-47e0-9e14-0830d9dcb107.png)




