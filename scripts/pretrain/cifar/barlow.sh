alpha=0.99
width=0.167

export CUDA_VISIBLE_DEVICES=1
python3 main_pretrain.py \
    --dataset cifar10 \
    --backbone resnet20_6x \
    --train_data_path ./datasets \
    --val_data_path ./datasets \
    --max_epochs 1000 \
    --devices 0 \
    --accelerator gpu \
    --precision 16 \
    --num_workers 4 \
    --optimizer lars \
    --grad_clip_lars \
    --eta_lars 0.02 \
    --exclude_bias_n_norm_lars \
    --scheduler warmup_cosine \
    --lr 0.3 \
    --weight_decay 1e-4 \
    --batch_size 256 \
    --brightness 0.4 \
    --contrast 0.4 \
    --saturation 0.2 \
    --hue 0.1 \
    --gaussian_prob 0.0 0.0 \
    --solarization_prob 0.0 0.2 \
    --crop_size 32 \
    --num_crops_per_aug 1 1 \
    --wandb \
    --name barlow-1000ep-cifar10-resnet20-6x-1x-cl-symm-distill-bt-alpha${alpha}-loss-iter196 \
    --entity jmeng15 \
    --project lightssl-hparam-search \
    --save_checkpoint \
    --method barlow_twins \
    --proj_hidden_dim 2048 \
    --proj_output_dim 2048 \
    --scale_loss 0.1 \
    --alpha ${alpha} \
    --width ${width} \
