export CUDA_VISIBLE_DEVICES=2
dataset=cifar10

python3 main_linear.py \
    --dataset cifar10 \
    --backbone vit_base \
    --train_data_path ./datasets \
    --val_data_path ./datasets \
    --max_epochs 100 \
    --devices 0 \
    --accelerator gpu \
    --precision 16 \
    --optimizer sgd \
    --scheduler step \
    --lr 0.3 \
    --patch_size 8 \
    --lr_decay_steps 60 80 \
    --weight_decay 0 \
    --batch_size 64 \
    --num_workers 4 \
    --pretrained_feature_extractor /home/jmeng15/solo-learn-sparse/trained_models/dino/2a49a0ws/dino-cifar10-vit-baseline-2a49a0ws-ep=999.ckpt \
    --name dino-${dataset}-vit-baseline \
    --entity jmeng15 \
    --project iclr2023_sparse_ssl \
    --wandb \
    --save_checkpoint \
    --auto_resume
