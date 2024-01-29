#rm -r wandb
#export WANDB_MODE=offline
HYDRA_FULL_ERROR=1 python3 utils/train.py \
    project=cross_attention exp_name=commongen-baseline-lr5e-5-fixdataset-wopret-fixencs \
    +lightning_wrapper=slava_contextual \
    +lightning_wrapper/optim_partial=slava_adam \
    ++lightning_wrapper.optim_partial.lr=5e-5 \
    +lightning_wrapper/sde_cfg=cosine_sd \
    every_n_train_steps=5000 \
    +datamodule=contextual_common_gen \
    datamodule.train_dataloader_cfg.batch_size=256 \
    datamodule.train_dataloader_cfg.num_workers=8 \
    max_steps=120000
#python3 utils/validate.py experiments/smth 0
