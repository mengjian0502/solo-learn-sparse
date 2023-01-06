export CUDA_VISIBLE_DEVICES=0,1

# prune
prune=True
crm=False
momentum_mask=True
sparse_enck=True
prune_rate=0.0
init_density=1.0
final_density=1.0
density_gap=0.0
momentum=0.99
update_frequency=4000


python3 main_pretrain.py \
    --dataset imagenet \
    --backbone sresnet50 \
    --train_data_path /home/zwang586/imagenet/train \
    --val_data_path /home/jmeng15/data/val/ \
    --max_epochs 200 \
    --devices 0 1 \
    --accelerator gpu \
    --strategy ddp \
    --sync_batchnorm \
    --precision 16 \
    --optimizer lars \
    --eta_lars 0.001 \
    --exclude_bias_n_norm_lars \
    --scheduler warmup_cosine \
    --lr 0.45 \
    --accumulate_grad_batches 16 \
    --classifier_lr 0.2 \
    --weight_decay 1e-6 \
    --batch_size 128 \
    --num_workers 4 \
    --data_format dali \
    --brightness 0.4 \
    --contrast 0.4 \
    --saturation 0.2 \
    --hue 0.1 \
    --gaussian_prob 1.0 0.1 \
    --solarization_prob 0.0 0.2 \
    --num_crops_per_aug 1 1 \
    --name byol-resnet50-imagenet-200ep-i${init_density}-f${final_density}-d${density_gap}-m${momentum}-momentum${momentum_mask}-baseline \
    --wandb \
    --entity jmeng15 \
    --project iclr2023_sparse_ssl \
    --save_checkpoint \
    --auto_resume \
    --method byol \
    --proj_output_dim 256 \
    --proj_hidden_dim 4096 \
    --pred_hidden_dim 4096 \
    --base_tau_momentum 0.99 \
    --final_tau_momentum 1.0 \
    --momentum_classifier \
    --prune ${prune} \
    --crm ${crm} \
    --momentum_mask ${momentum_mask} \
    --ema_momentum ${momentum} \
    --sparse_enck ${sparse_enck} \
    --prune-rate ${prune_rate} \
    --init-density ${init_density} \
    --final-density ${final_density} \
    --init-prune-epoch 60 \
    --final-prune-epoch 138 \
    --density_gap ${density_gap} \
    --update-frequency ${update_frequency} \
    --slist ${final_density};
