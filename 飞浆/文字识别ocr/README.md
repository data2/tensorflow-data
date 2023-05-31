## 安装
pip install paddleocr --user

## python>>>>>>>>>>>

```
from paddleocr import PaddleOCR, draw_ocr

ocr = PaddleOCR(use_angle_cls=True, lang="ch")  # need to run only once to download and load model into memory
img_path = 'C:\\Users\\data\\Desktop\\test.png' # 注意上传一张图片，并修改正确的图片地址
result = ocr.ocr(img_path, cls=True)
for line in result:
    print(line)

```
