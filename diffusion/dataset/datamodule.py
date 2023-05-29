from hydra.utils import instantiate

import lightning as L

from torch.utils.data import DataLoader, Dataset

from typing import List
from diffusion.configs import DatasetCfg, DataLoaderCfg


class SimpleDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_dataset_cfg: DatasetCfg,
        valid_dataset_cfg: DatasetCfg,
        train_dataloader_cfg: DataLoaderCfg,
        valid_dataloader_cfg: DataLoaderCfg,
    ):
        super().__init__()
        self.train_dataset_cfg = train_dataset_cfg
        self.valid_dataset_cfg = valid_dataset_cfg

        self.train_dataloader_cfg = train_dataloader_cfg
        self.valid_dataloader_cfg = valid_dataloader_cfg

    def setup(self, stage: str = "smth") -> None:
        self.train_dataset: Dataset = instantiate(self.train_dataset_cfg)
        self.valid_dataset: Dataset = instantiate(self.valid_dataset_cfg, train=False)

    def train_dataloader(self):
        return instantiate(self.train_dataloader_cfg, dataset=self.train_dataset)

    def val_dataloader(self) -> List[DataLoader]:
        if len(self.valid_dataset) == 0:
            return []
        return [instantiate(self.valid_dataloader_cfg, dataset=self.valid_dataset)]

    def test_dataloader(self):
        return self.val_dataloader()
