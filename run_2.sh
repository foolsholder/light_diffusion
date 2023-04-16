#rm -r wandb
#export WANDB_MODE=offline
HYDRA_FULL_ERROR=1 python3 utils/check_cfg.py
#python3 utils/validate.py experiments/smth 0
