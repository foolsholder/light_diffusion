#rm -r wandb
#export WANDB_MODE=offline
HYDRA_FULL_ERROR=1 python3 utils/train.py \
    project=cross_attention exp_name=art-t5-bert-t5t \
    +lightning_wrapper=slava_contextual_t5t \
    +lightning_wrapper/optim_partial=slava_adam \
    +lightning_wrapper/sched_partial=linear_warmup \
    ++lightning_wrapper.optim_partial.lr=2e-4 \
    +lightning_wrapper/sde_cfg=cosine_sd \
    ++lightning_wrapper.sched_partial.warmup_start_lr=1e-6 \
    every_n_train_steps=100000 \
    +datamodule=contextual_art \
    datamodule.train_dataloader_cfg.batch_size=512 \
    datamodule.train_dataloader_cfg.num_workers=8 \
    pretrained_path=data/newest-slav/ema_weights.pth \
    max_steps=1000000
#python3 utils/validate.py experiments/smth 0
