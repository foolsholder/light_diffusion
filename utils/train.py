import hydra
import os
import os.path as osp

import torch

from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate

from typing import Dict, List, Optional, Union, Tuple

from lightning import seed_everything, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies.ddp import DDPStrategy

from diffusion import Config
from diffusion.callbacks import EMACallback

from sys import exit


@hydra.main(version_base=None, config_path='../configs', config_name='launch')
def main(cfg: Config):
    seed_everything(cfg.seed, workers=True)

    if cfg.exp_name == 'None':
        raise "Specify experiment name - exp_name"

    yaml_cfg = OmegaConf.to_yaml(cfg)
    to_wandb_cfg = OmegaConf.to_container(cfg)
    print(yaml_cfg)
    print(osp.abspath('.'))

    wrapped_model = instantiate(cfg.lightning_wrapper, _recursive_=False)
    #wrapped_model.score_estimator = torch.compile(wrapped_model.score_estimator, mode='default')
    #wrapped_model = torch.compile(wrapped_model, mode='max-autotune')

    exp_name = cfg.exp_name

    exp_folder = osp.join(os.environ['EXP_PATH'], exp_name)
    if not osp.exists(exp_folder):
        os.makedirs(exp_folder)

    with open(osp.join(exp_folder, 'config.yaml'), 'w') as fout:
        print(yaml_cfg, file=fout)
    #exit(0)

    if torch.cuda.device_count() > 1:
        strategy = DDPStrategy(
            find_unused_parameters=False,
        )
    else:
        strategy = 'auto'

    trainer = Trainer(
        max_steps=cfg.max_steps,
        logger=WandbLogger(
            project=cfg.project,
            name=exp_name,
            config=to_wandb_cfg
        ),
        log_every_n_steps=50,
        callbacks=[
            ModelCheckpoint(
                dirpath=exp_folder,
                filename='{epoch:02d}',
                #every_n_train_steps=cfg.every_n_train_steps,
                save_top_k=-1,
                auto_insert_metric_name=True,
                save_weights_only=False
            ),
            LearningRateMonitor(logging_interval='step'),
            EMACallback(0.9999)
        ],
        enable_checkpointing=True,
        gradient_clip_algorithm="norm",
        gradient_clip_val=cfg.grad_clip_norm,
        precision='32',
        accelerator='auto',
        strategy=strategy
    )

    trainer.fit(
        wrapped_model,
        datamodule=instantiate(cfg.datamodule, _recursive_=False)
    )


if __name__ == '__main__':
    os.environ['EXP_PATH'] = osp.abspath('experiments/')
    main()
