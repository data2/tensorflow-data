git clone --depth 1  https://gitee.com/paddlepaddle/PaddleSeg

cd ./PaddleSeg/Matting
pip install -r requirements.txt


# 下载预训练模型
wget https://paddleseg.bj.bcebos.com/matting/models/ppmatting-hrnet_w18-human_512.pdparams
# 下载图片
wget https://user-images.githubusercontent.com/30919197/200645066-6898ec5a-f1c5-4bf7-aa41-473a29977a86.jpeg
# 在GPU上预测一张图片
export CUDA_VISIBLE_DEVICES=0
python tools/predict.py \
    --config configs/ppmatting/ppmatting-hrnet_w18-human_512.yml \
    --model_path ppmatting-hrnet_w18-human_512.pdparams \
    --image_path 200645066-6898ec5a-f1c5-4bf7-aa41-473a29977a86.jpeg \
    --save_dir ./output/results \
    --fg_estimate True
    

python tools/predict.py  --config configs/ppmatting/ppmatting-hrnet_w18-human_512.yml   --model_path ppmatting-hrnet_w18-human_512.pdparams  --image_path 200645066-6898ec5a-f1c5-4bf7-aa41-473a29977a86.jpeg  --save_dir ./output/results  --fg_estimate True
