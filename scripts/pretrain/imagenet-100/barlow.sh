export CUDA_VISIBLE_DEVICES=0,1,2,3

python3 main_pretrain.py \
    --dataset imagenet100 \
    --backbone mobilenetv1_2x \
    --train_data_path /home/jmeng15/data/imagenet-100/train \
    --val_data_path /home/jmeng15/data/imagenet-100/val \
    --max_epochs 400 \
    --devices 0,1,2,3 \
    --accelerator gpu \
    --strategy ddp \
    --sync_batchnorm \
    --num_workers 4 \
    --precision 16 \
    --optimizer lars \
    --grad_clip_lars \
    --eta_lars 0.02 \
    --exclude_bias_n_norm_lars \
    --scheduler warmup_cosine \
    --lr 0.3 \
    --weight_decay 1e-4 \
    --batch_size 64 \
    --data_format dali \
    --brightness 0.4 \
    --contrast 0.4 \
    --saturation 0.2 \
    --hue 0.1 \
    --gaussian_prob 1.0 0.1 \
    --solarization_prob 0.0 0.2 \
    --num_crops_per_aug 1 1 \
    --name barlow-400ep-imagenet100-mobilenetv1-2x-1x-alpha0.95-b64-interval4000 \
    --wandb \
    --entity jmeng15 \
    --project light-ssl-arxiv \
    --save_checkpoint \
    --scale_loss 0.1 \
    --method barlow_twins \
    --proj_hidden_dim 2048 \
    --proj_output_dim 2048
