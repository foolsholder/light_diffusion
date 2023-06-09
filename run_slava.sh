#rm -r wandb
#export WANDB_MODE=offline
HYDRA_FULL_ERROR=1 python3 utils/train.py \
    project=cross_attention exp_name=25k_second_zero_roberta_pretrain_sc_cos_adamw \
    +lightning_wrapper=slava_contextual \
    ++lightning_wrapper.roberta_pretrain=True \
    +lightning_wrapper/optim_partial=slava_adam \
    +lightning_wrapper/sched_partial=linear_warmup_second_zero \
    ++lightning_wrapper.optim_partial.lr=2e-4 \
    +lightning_wrapper/sde_cfg=cosine_sd \
    ++lightning_wrapper.sched_partial.warmup_start_lr=1e-6 \
    every_n_train_steps=100000 \
    +datamodule=wiki \
    datamodule.train_dataloader_cfg.batch_size=128 \
    datamodule.train_dataloader_cfg.num_workers=12 \
    max_steps=800000 precision=32
#python3 utils/validate.py experiments/smth 0
