#rm -r wandb
#export WANDB_MODE=offline
#python3 utils/train.py exp_name=first_voc2 +lightning_wrapper=first_voc2 +datamodule=first_sst2
#python3 utils/validate.py experiments/smth 0
#python3 utils/train.py exp_name=sst2_freeze_cls

HYDRA_FULL_ERROR=1 python3 utils/train_by_epoch.py \
    project=cross_attention exp_name=mrpc_pretrained_t5t \
    ++monitor=f1/valid@1 \
    +lightning_wrapper=contextual_cls_trainable_t5 \
    +lightning_wrapper/sde_cfg=cosine_sd \
    +lightning_wrapper/optim_partial=adam \
    ++lightning_wrapper.optim_partial.lr=2e-4 \
    ++lightning_wrapper.sched_partial.warmup_start_lr=1e-6 \
    +datamodule=contextual_mrpc \
    ++datamodule.train_dataloader_cfg.batch_size=512 \
    ++datamodule.train_dataloader_cfg.num_workers=16 \
    ++pretrained_path=experiments/sc_adamw_cosine_sd/step_200000.ckpt \
    precision=bf16-mixed max_epochs=500
