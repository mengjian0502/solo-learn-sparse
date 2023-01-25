export CUDA_VISIBLE_DEVICES=0

python3 main_linear.py \
    --dataset cifar10 \
    --backbone resnet20_4x \
    --train_data_path ./datasets \
    --val_data_path ./datasets \
    --max_epochs 100 \
    --devices 0 \
    --accelerator gpu \
    --precision 16 \
    --optimizer sgd \
    --scheduler step \
    --lr 0.1 \
    --lr_decay_steps 60 80 \
    --weight_decay 0 \
    --batch_size 128 \
    --num_workers 10 \
    --pretrained_feature_extractor /home2/jmeng15/solo-learn-sparse/trained_models/barlow_twins/2hry7emc/barlow-1000ep-cifar10-resnet20-4x-2hry7emc-ep=999.ckpt \
    --name barlow-1000ep-cifar10-resnet20-4x-linear \
    --entity jmeng15 \
    --project light-ssl \
    --wandb \
    --save_checkpoint \
    --auto_resume
