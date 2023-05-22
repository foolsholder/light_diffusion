#rm -r wandb
#export WANDB_MODE=offline
HYDRA_FULL_ERROR=1 python3 utils/train.py \
    project=cross_attention exp_name=cross_attention_t5_bert_se_800k_lr2e-4 \
    every_n_train_steps=100000 \
    ++lightning_wrapper.optim_partial.lr=2e-4 \
    datamodule.train_dataloader_cfg.batch_size=256 \
    datamodule.train_dataloader_cfg.num_workers=32 \
    max_steps=800000
#python3 utils/validate.py experiments/smth 0
