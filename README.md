
# tensorflow

Check the versions of tensorflow and python, and then determine the installed versions of each. eg:python-3.9,tensorflow-2.6.0

## setup python 

https://www.python.org/downloads/windows/

During the python installation process, select the two options of adding environment variables & turning off window path length limit (disable path length limit)

## setup tensorflow

pip install tensorflow-cpu==2.6.0 -i https://pypi.tuna.tsinghua.edu.cn/simple

## Other possible considerations

### err1-protobuf

![image](https://user-images.githubusercontent.com/13504729/230704379-dc7c4ee9-4dc8-4402-bbda-30b60867a37a.png)

pip install protobuf==3.20

### err2-numpy version

![image](https://user-images.githubusercontent.com/13504729/230704389-a7fa9c2f-e86e-4be6-9be0-bddd375669ce.png)

Error: pip's dependency resolver currently does not consider all installed packages. This behavior is the source of the following dependency conflicts. tensorflow-cpu 2.6.0 requires numpy~=1.19.2, but you have numpy 1.24.2 which is incompatible.
pip install numpy==1.19.2

### err3-Numpy depends on visual c++ 14.0

![image](https://user-images.githubusercontent.com/13504729/230704733-f3bb9423-77dc-4b11-ab16-b7f94b42d530.png)

At this time, visual c++ 14.0 needs to be installed. You can download and install it from Microsoft.  [https://learn.microsoft.com/en-US/cpp/windows/latest-supported-vc-redist?view=msvc-170](https://visualstudio.microsoft.com/zh-hans/downloads/)

after setup, pip install numpy==1.19.2

## Need to pay attention to the correspondence between tensorflow and keras versions

![image](https://user-images.githubusercontent.com/13504729/230291129-e606ec43-9fec-4091-94b8-95e7a2819f00.png)

pip install keras==2.6

## test tensorflow

![image](https://user-images.githubusercontent.com/13504729/230268366-bbd3c479-f90d-47e0-9e14-0830d9dcb107.png)


# pytorch

[pytorch setup](https://github.com/data2/tensorflow-pytorch-paddlepaddle/blob/main/pytorch.md)

# paddlepaddle

[paddlepaddle setup](https://github.com/data2/tensorflow-pytorch-paddlepaddle/blob/main/%E9%A3%9E%E6%B5%86/README.md)

 Administrator execution cmd

cpu： pip install paddlepaddle



