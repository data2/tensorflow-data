#谷歌tensorflow物体检测使用

## github 

https://github.com/tensorflow/models

手动下载zip 

![image](https://user-images.githubusercontent.com/13504729/231032885-dd3402fd-5be5-4c97-b4d1-e3bb27ec7c2d.png)

或者

git clone git@github.com:tensorflow/models.git

## 安装pycocotools

进入到object_detection目录下执行谷歌官方tf2的官方demo

报错如下：

![image](https://user-images.githubusercontent.com/13504729/231033058-6a9eb50f-9491-4524-9dae-3c7a8e562354.png)

执行pip install pycocotools

## 安装protoc 

再次执行

![image](https://user-images.githubusercontent.com/13504729/231033260-1ac0d85e-99b8-427e-849f-5a79243b9ff3.png)

```
ImportError: cannot import name 'string_int_label_map_pb2'
```

cannot import name 'string_int_label_map_pb2'

安装protoc

https://github.com/protocolbuffers/protobuf/releases

加入path变量

编译 Protobuf 依赖包，执行以下命令

```
%%bash
cd models/research
protoc object_detection/protos/*.proto --python_out=.

```
## 安装model/research

python setup.py build

python setup.py install


## 测试

