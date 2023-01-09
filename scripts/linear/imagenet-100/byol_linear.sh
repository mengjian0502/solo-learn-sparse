export CUDA_VISIBLE_DEVICES=1

python3 main_linear.py \
    --dataset imagenet100 \
    --backbone mobilenetv1_2x \
    --train_data_path /home2/jmeng15/data/imagenet-100/train \
    --val_data_path /home2/jmeng15/data/imagenet-100/val \
    --max_epochs 100 \
    --devices 0 \
    --accelerator gpu \
    --precision 16 \
    --optimizer sgd \
    --scheduler step \
    --lr 0.3 \
    --lr_decay_steps 60 80 \
    --weight_decay 0 \
    --batch_size 256 \
    --num_workers 4 \
    --data_format dali \
    --name byol-imagenet100-linear-eval \
    --pretrained_feature_extractor /home2/jmeng15/solo-learn-sparse/trained_models/byol/1nbqka9j/byol-400ep-imagenet100-mobilenetv1-2x-1nbqka9j-ep=399.ckpt \
    --entity jmeng15 \
    --project light-ssl \
    --wandb \
    --save_checkpoint \
    --auto_resume
