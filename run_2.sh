#rm -r wandb
#export WANDB_MODE=offline
python3 utils/train.py exp_name=zero_voc2 #+lightning_wrapper=first_voc2 +datamodule=first_sst2
#python3 utils/validate.py experiments/smth 0
