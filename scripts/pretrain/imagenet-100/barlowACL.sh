xdistll=True
distype="copy"
llamb=1e-4
alpha=0.95
width=1.0

export CUDA_VISIBLE_DEVICES=0,1,3

python3 main_pretrain.py \
    --dataset imagenet100 \
    --backbone mobilenetv1 \
    --train_data_path /home/jmeng15/data/imagenet-100/train \
    --val_data_path /home/jmeng15/data/imagenet-100/val \
    --max_epochs 400 \
    --devices 0,1,2 \
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
    --weight_decay 5e-5 \
    --batch_size 128 \
    --data_format dali \
    --brightness 0.4 \
    --contrast 0.4 \
    --saturation 0.2 \
    --hue 0.1 \
    --gaussian_prob 1.0 0.1 \
    --solarization_prob 0.0 0.2 \
    --num_crops_per_aug 1 1 \
    --name barlow-400ep-imagenet100-mobilenetv1-1x-1x-alpha0.95-xdistill${xdistll}-${distype}-lamb${llamb}-alpha${alpha} \
    --wandb \
    --entity jmeng15 \
    --project light-ssl-acl \
    --save_checkpoint \
    --scale_loss 0.1 \
    --method barlow_acl \
    --proj_hidden_dim 2048 \
    --proj_output_dim 2048 \
    --xdistill ${xdistll} \
    --distype ${distype} \
    --alpha ${alpha} \
    --width ${width} \
