#谷歌tensorflow物体检测使用



## 安装protoc

执行谷歌官方tf2的官方demo,报错如下：

```
ImportError: cannot import name 'string_int_label_map_pb2'
```

cannot import name 'string_int_label_map_pb2'

安装protoc

https://github.com/protocolbuffers/protobuf/releases

加入path变量

%%bash
cd models/research
protoc object_detection/protos/*.proto --python_out=.



