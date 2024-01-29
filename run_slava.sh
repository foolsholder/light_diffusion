#rm -r wandb
#export WANDB_MODE=offline
HYDRA_FULL_ERROR=1 python3 utils/train.py \
    project=cross_attention exp_name=xsum-baseline-lr2e-4-fixdataset \
    +lightning_wrapper=slava_contextual \
    +lightning_wrapper/optim_partial=slava_adam \
    ++lightning_wrapper.optim_partial.lr=2e-4 \
    +lightning_wrapper/sde_cfg=cosine_sd \
    every_n_train_steps=10000 \
    +datamodule=contextual_xsum \
    datamodule.train_dataloader_cfg.batch_size=256 \
    datamodule.train_dataloader_cfg.num_workers=8 \
    max_steps=120000 \
    pretrained_path=experiments/wiki-pretrain-nam-noisy-067-bs512-t2/step_500000.ckpt
#python3 utils/validate.py experiments/smth 0
