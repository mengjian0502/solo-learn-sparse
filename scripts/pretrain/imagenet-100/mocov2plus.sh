export CUDA_VISIBLE_DEVICES=0,1,2,3

python3 main_pretrain.py \
    --dataset imagenet100 \
    --backbone mobilenetv2_2x \
    --train_data_path /home/jmeng15/data/imagenet-100/train \
    --val_data_path /home/jmeng15/data/imagenet-100/val \
    --max_epochs 400 \
    --devices 0,1,2,3 \
    --accelerator gpu \
    --strategy ddp \
    --sync_batchnorm \
    --precision 16 \
    --optimizer sgd \
    --scheduler warmup_cosine \
    --lr 0.2 \
    --classifier_lr 0.2 \
    --weight_decay 1e-4 \
    --batch_size 64 \
    --num_workers 4 \
    --data_format dali \
    --brightness 0.4 \
    --contrast 0.4 \
    --saturation 0.4 \
    --hue 0.1 \
    --num_crops_per_aug 2 \
    --name mocov2plus-400ep-mobilenetv2_1.25x \
    --entity jmeng15 \
    --project light-ssl \
    --save_checkpoint \
    --wandb \
    --method mocov2plus \
    --proj_hidden_dim 2048 \
    --queue_size 65536 \
    --temperature 0.2 \
    --base_tau_momentum 0.99 \
    --final_tau_momentum 0.999 \
    --momentum_classifier
