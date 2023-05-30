#rm -r wandb
#export WANDB_MODE=offline
HYDRA_FULL_ERROR=1 python3 utils/train.py \
    project=cross_attention exp_name=second_zero_voc2 \
    +lightning_wrapper=new_zero_voc2 \
    +lightning_wrapper/optim_partial=slava_adam \
    ++lightning_wrapper.optim_partial.lr=2e-4 \
    +lightning_wrapper/sde_cfg=cosine_sd \
    ++lightning_wrapper.sched_partial.warmup_start_lr=1e-6 \
    every_n_train_steps=10000 \
    +datamodule=zero_sst2 \
    datamodule.train_dataloader_cfg.batch_size=256 \
    datamodule.train_dataloader_cfg.num_workers=16 \
    max_steps=100000
#python3 utils/validate.py experiments/smth 0
