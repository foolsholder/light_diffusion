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
import diffusion
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

    wrapped_model: torch.nn.Module = instantiate(cfg.lightning_wrapper, _recursive_=False)
    #wrapped_model.score_estimator = torch.compile(wrapped_model.score_estimator, mode='default')
    #wrapped_model = torch.compile(wrapped_model, mode='max-autotune')

    exp_name = cfg.exp_name

    exp_folder = osp.join(os.environ['EXP_PATH'], exp_name)
    if not osp.exists(exp_folder):
        os.makedirs(exp_folder)

    with open(osp.join(exp_folder, 'config.yaml'), 'w') as fout:
        print(yaml_cfg, file=fout)
    #exit(0)
    if cfg.pretrained_path:
        dct = torch.load(
            osp.join(
                os.environ['BASE_PATH'],
                cfg.pretrained_path
            ),
            map_location='cpu'
        )['state_dict']
        st_dct = wrapped_model.state_dict()
        for k, v in dct.items():
            if k in st_dct:
                assert(v.shape == st_dct[k].shape)
        missing_keys = []
        for k, v in st_dct.items():
            if not k in dct:
                missing_keys += [k]
        print("Missing keys:", ", ".join(missing_keys))
        wrapped_model.load_state_dict(
            dct, strict=False
        )

    if torch.cuda.device_count() > 1:
        strategy = DDPStrategy(
            find_unused_parameters=False,
        )
    else:
        strategy = 'auto'

    trainer = Trainer(
        max_epochs=cfg.max_epochs,
        logger=WandbLogger(
            project=cfg.project,
            name=exp_name,
            config=to_wandb_cfg
        ),
        log_every_n_steps=50,
        callbacks=[
            ModelCheckpoint(
                dirpath=exp_folder,
                filename='epoch_{epoch:d}',
                every_n_epochs=cfg.every_n_epochs,
                save_top_k=-1,
                auto_insert_metric_name=False,
                save_weights_only=False
            ),
            LearningRateMonitor(logging_interval='step'),
            EMACallback(0.9999)
        ],
        enable_checkpointing=True,
        gradient_clip_algorithm="norm",
        gradient_clip_val=cfg.grad_clip_norm,
        precision=cfg.precision,
        accelerator='auto',
        strategy=strategy
    )
    datamodule: diffusion.SimpleDataModule = instantiate(cfg.datamodule, _recursive_=False)
    datamodule.setup()
    if len(datamodule.valid_dataset) > 0:
        trainer.fit(
            wrapped_model,
            datamodule=datamodule
        )
    else:
        trainer.fit(
            wrapped_model,
            train_dataloaders=datamodule.train_dataloader()
        )

if __name__ == '__main__':
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ['BASE_PATH'] = './'
    os.environ['EXP_PATH'] = osp.abspath('experiments/')
    main()
