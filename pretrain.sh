unset LD_LIBRARY_PATH

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/root/anaconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/root/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/root/anaconda3/etc/profile.d/conda.sh"
    else
        export PATH="/root/anaconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

conda activate mam

torchrun --standalone --nproc_per_node=4 --master_port 1221 main_pretrain.py \
    --batch_size 128 \
    --accum_iter 1 \
    --input_size 224 \
    --model star_base_patch16_224 \
    --norm_pix_loss \
    --epochs 200 \
    --warmup_epochs 40 \
    --blr 1.5e-4 --weight_decay 0.05 \
    --data_path /lpai/dataset/imagenet-1k/0-1-0/ --output_dir ./test
    # --data_path $LPAI_INPUT_DATASET_0 --output_dir $LPAI_OUTPUT_DATA_0/lhp/mam/eso_longinput_4/


python -m torch.distributed.launch --nproc_per_node=8 --use_env main.py \
  --model mambar_base_patch16_224 \
  --batch 64 --lr 1e-5 --weight-decay 0.1 --unscale-lr \
  --data-path /PATH/TO/IMAGENET \
  --finetune ./output/mambar_base_patch16_224/mid/checkpoint.pth \
  --output_dir ./output/mambar_base_patch16_224/ft \
  --reprob 0.0 --smoothing 0.1 --no-repeated-aug \
  --aa rand-m9-mstd0.5-inc1 --eval-crop-ratio 1.0 \
  --epochs 20 --input-size 224 --drop-path 0.4

# cd ./Finetuning

# torchrun --standalone --nproc_per_node 8 main_finetune.py --batch_size 128 \
#     --accum_iter 1 \
#     --model arm_base_pz16 --finetune $LPAI_OUTPUT_DATA_0/lhp/mam/eso_longinput_4/checkpoint-199.pth \
#     --epochs 100 --global_pool True \
#     --blr 5e-4 --layer_decay 0.65 --ema_decay 0.99992 \
#     --weight_decay 0.05 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
#     --dist_eval --data_path $LPAI_INPUT_DATASET_0 --output_dir $LPAI_OUTPUT_DATA_0/lhp/mam/eso_longinput_4_ft/