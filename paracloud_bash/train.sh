#!/bin/bash
module load anaconda  compilers/cuda/12.1   cudnn/8.8.1.3_cuda12.x  compilers/gcc/12.2.0 llvm/triton-llvm_14.0.6
export LD_PRELOAD=/home/bingxing2/apps/compilers/gcc/12.2.0/lib64/libstdc++.so
export LD_PRELOAD=$LD_PRELOAD:/home/bingxing2/home/scx8ana/.conda/envs/videollava/lib/python3.10/site-packages/scikit_learn.libs/libgomp-d22c30c5.so.1.0.0

source activate videollava
export PYTHONUNBUFFERED=1 
export HF_ENDPOINT=https://hf-mirror.com


JSON_FOLDER="llava_all_image_video/pt_json"
IMAGE_FOLDER="llava_all_image_video/llava_image"
VIDEO_FOLDER="llava_all_image_video/valley"

# cd /path/to/Video-LLaVA
HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 deepspeed videollava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path ./checkpoints/vicuna-7b-v1.5 \
    --version v1 \
    --data_path ${JSON_FOLDER}/llava_image_.json ${JSON_FOLDER}/valley_.json \
    --image_folder ${IMAGE_FOLDER} \
    --image_tower LanguageBind/LanguageBind_Image \
    --video_folder ${VIDEO_FOLDER} \
    --video_tower LanguageBind/LanguageBind_Video_merge \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ./checkpoints/videollava-7b-pretrain \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048  --tokenizer_model_max_length 3072 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --cache_dir "./cache_dir"
