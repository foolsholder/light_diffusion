#rm -r wandb
#export WANDB_MODE=offline
HYDRA_FULL_ERROR=1 python3 utils/train.py \
    project=cross_attention exp_name=cross_attention_t5_bert_se \
    every_n_train_steps=10000 \
    datamodule.train_dataloader_cfg.batch_size=256 \
    datamodule.train_dataloader_cfg.num_workers=32
#python3 utils/validate.py experiments/smth 0
