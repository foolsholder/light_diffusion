#rm -r wandb
#export WANDB_MODE=offline
#python3 utils/train.py exp_name=first_voc2 +lightning_wrapper=first_voc2 +datamodule=first_sst2
#python3 utils/validate.py experiments/smth 0
#python3 utils/train.py exp_name=sst2_freeze_cls
HYDRA_FULL_ERROR=1 python3 utils/train_by_epoch.py \
    project=cross_attention exp_name=ns_cola \
    ++monitor=mat-corr/valid@1 \
    +lightning_wrapper=contextual_cls \
    +lightning_wrapper/sde_cfg=cosine_sd \
    +lightning_wrapper/optim_partial=adam \
    ++lightning_wrapper.optim_partial.lr=2e-4 \
    ++lightning_wrapper.sched_partial.warmup_start_lr=1e-6 \
    +datamodule=contextual_cola \
    ++datamodule.train_dataloader_cfg.batch_size=512 \
    ++datamodule.train_dataloader_cfg.num_workers=2 \
    ++pretrained_path=data/newest-slav/ema_weights.pth \
    max_epochs=500 \
    precision=16
