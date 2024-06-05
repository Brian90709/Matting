# Install PaddlePaddle
```
pip3 install paddlepaddle-gpu
```
# Install PaddleSeg
```
git clone https://github.com/Brian90709/Matting.git
pip3 install "paddleseg>=2.5"
pip3 install -r requirements.txt
```

# Cuda Environment
```
module load cuda/10.2
module load cudnn/cuda101_7.6
```
# Train
## PP-Matting
```
CUDA_VISIBLE_DEVICES=1 python3 tools/train.py --config configs/ppmatting/ppmatting-hrnet_w18-human_512_NTU1000.yml --do_eval --use_vdl --save_interval 500 --num_workers 5 --save_dir output/PP-Matting_test
```
## MODNet
```
CUDA_VISIBLE_DEVICES=1 python3 tools/train.py --config configs/modnet/modnet-mobilenetv2.yml --do_eval --use_vdl --save_interval 500 --num_workers 5 --save_dir output/MODNet_test
```


# Inference
## PP-Matting
```
CUDA_VISIBLE_DEVICES=1 python3 tools/predict.py --config configs/ppmatting/ppmatting-hrnet_w18-human_512_NTU1000.yml --model_path output/PP-Matting_convgf_after/best_model/model.pdparams --image_path /project/g/r10922161/2023/matting_data/NTU_1000/val/fg/ --save_dir ./output/results/PP-Matting_convgf_after
```
## MODNet
```
CUDA_VISIBLE_DEVICES=1 python3 tools/predict.py --config configs/modnet/modnet-mobilenetv2.yml --model_path output/MODNet/best_model/model.pdparams --image_path /project/g/r10922161/2023/matting_data/NTU_1000/val/fg/ --save_dir ./output/results/MODNet
```

# Evaluation
```
python3 evaluate2.py
```