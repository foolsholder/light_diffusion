#rm -r wandb
#export WANDB_MODE=offline
#python3 utils/train.py exp_name=first_voc2 +lightning_wrapper=first_voc2 +datamodule=first_sst2
#python3 utils/validate.py experiments/smth 0
#python3 utils/train.py exp_name=sst2_freeze_cls
HYDRA_FULL_ERROR=1 python3 utils/train.py \
    project=cross_attention exp_name=sc_adamw_cosine_sd \
    +lightning_wrapper=slava_contextual \
    +lightning_wrapper/sde_cfg=cosine_sd \
    +lightning_wrapper/optim_partial=slava_adam \
    ++lightning_wrapper.optim_partial.lr=2e-4 \
    ++lightning_wrapper.sched_partial.warmup_start_lr=1e-6 \
    ++every_n_train_steps=100000 \
    +datamodule=wiki \
    ++datamodule.train_dataloader_cfg.batch_size=256 \
    ++datamodule.train_dataloader_cfg.num_workers=30 \
    ++max_steps=800000
