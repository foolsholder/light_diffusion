#rm -r wandb
#export WANDB_MODE=offline
HYDRA_FULL_ERROR=1 python3 utils/train.py \
    project=cross_attention exp_name=wiki-planner-nam-noisy-067-bs512-t2 \
    +lightning_wrapper=planner_contextual \
    +lightning_wrapper/optim_partial=slava_adam \
    +lightning_wrapper/sched_partial=linear_warmup \
    ++lightning_wrapper.optim_partial.lr=2e-4 \
    +lightning_wrapper/sde_cfg=cosine_sd \
    ++lightning_wrapper.sched_partial.warmup_start_lr=1e-6 \
    every_n_train_steps=50000 \
    +datamodule=wiki \
    datamodule.train_dataloader_cfg.batch_size=512 \
    datamodule.train_dataloader_cfg.num_workers=16 \
    max_steps=500000
#python3 utils/validate.py experiments/smth 0
