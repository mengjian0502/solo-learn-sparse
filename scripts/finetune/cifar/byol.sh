export CUDA_VISIBLE_DEVICES=2

python3 main_linear.py \
    --dataset cifar10 \
    --backbone sresnet50 \
    --train_data_path ./datasets \
    --val_data_path ./datasets \
    --max_epochs 100 \
    --devices 0 \
    --accelerator gpu \
    --strategy ddp \
    --sync_batchnorm \
    --precision 16 \
    --optimizer sgd \
    --scheduler step \
    --lr 0.1 \
    --lr_decay_steps 60 80 \
    --weight_decay 0 \
    --batch_size 256 \
    --num_workers 10 \
    --data_format dali \
    --pretrained_feature_extractor /home/jmeng15/solo-learn-sparse/trained_models/byol/146nx8a3/byol-resnet50-imagenet-200ep-i1.0-f1.0-d0.0-m0.99-momentumTrue-baseline-146nx8a3-ep=199.ckpt \
    --name byol-resnet50-imagenet-linear-eval-i0.7-f0.2-d0.3-m0.99 \
    --entity jmeng15 \
    --project iclr2023_sparse_ssl \
    --wandb \
    --save_checkpoint \
    --auto_resume \
    --finetune
