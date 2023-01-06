export CUDA_VISIBLE_DEVICES=2
dataset=cifar10

python3 main_pretrain.py \
    --dataset cifar10 \
    --backbone vit_small \
    --train_data_path ./datasets \
    --val_data_path ./datasets \
    --max_epochs 1000 \
    --warmup_epochs 10 \
    --devices 0 \
    --accelerator gpu \
    --optimizer adamw \
    --scheduler warmup_cosine \
    --lr 5.0e-4 \
    --classifier_lr 5.0e-4 \
    --weight_decay 0.1 \
    --batch_size 128 \
    --num_workers 4 \
    --brightness 0.4 \
    --contrast 0.4 \
    --saturation 0.4 \
    --hue 0.1 \
    --gaussian_prob 1.0 0.1 \
    --solarization_prob 0.0 0.2 \
    --min_scale 0.2 \
    --crop_size 32 \
    --num_crops_per_aug 1 1 \
    --name mocov3-${dataset}-baseline \
    --entity jmeng15 \
    --project iclr2023_sparse_ssl \
    --wandb \
    --save_checkpoint \
    --method mocov3 \
    --proj_hidden_dim 4096 \
    --pred_hidden_dim 4096 \
    --temperature 0.2 \
    --base_tau_momentum 0.99 \
    --final_tau_momentum 1.0
