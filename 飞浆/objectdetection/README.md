# download

git clone https://github.com/PaddlePaddle/PaddleDetection.git

cd PaddleDetection

mkdir -p demo_input demo_output

pip install -r requirements.txt

# train

## 单卡训练

!python tools/train.py -c configs/ppyoloe/ppyoloe_plus_crn_l_80e_coco.yml --eval --amp

## 多卡训练

!python -m paddle.distributed.launch --gpus 0,1,2,3,4,5,6,7 tools/train.py -c configs/ppyoloe/ppyoloe_plus_crn_l_80e_coco.yml --eval --amp

如果需要边训练边评估，请添加--eval.

PP-YOLOE支持混合精度训练，请添加--amp.

# predict

## export model

use_gpu 

python tools/export_model.py -c configs/ppyoloe/ppyoloe_plus_crn_l_80e_coco.yml -o weights=https://paddledet.bj.bcebos.com/models/ppyoloe_plus_crn_l_80e_coco.pdparams trt=True

use_cpu

python tools/export_model.py -c configs/ppyoloe/ppyoloe_plus_crn_l_80e_coco.yml -o weights=https://paddledet.bj.bcebos.com/models/ppyoloe_plus_crn_l_80e_coco.pdparams trt=True use_gpu=False

## predict 

python deploy/python/infer.py --model_dir=output_inference/ppyoloe_plus_crn_l_80e_coco --image_file=demo_input/000000014439.jpg --device=gpu --run_mode=trt_fp16 --output_dir=demo_output 
