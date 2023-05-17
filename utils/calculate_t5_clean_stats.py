import torch
import os
import os.path as osp

import diffusion

from lightning import seed_everything
from tqdm.auto import tqdm
from torch import nn, FloatTensor
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast, T5EncoderModel, T5Config
from torchsummary import summary
from transformers.models.t5.modeling_t5 import BaseModelOutput
from typing import Dict, List


def filter_by_prefix(
    dict_obj: Dict[str, FloatTensor],
    prefix: str = 'clean_'
) -> Dict[str, FloatTensor]:
    result = dict()
    for k, v in dict_obj.items():
        if k.startswith(prefix):
            result[k[len(prefix):]] = v
    return result

from diffusion.utils import dict_to_device

if __name__ == '__main__':
    seed_everything(42)
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda:0'

    model = T5EncoderModel.from_pretrained('t5-base').eval().to(device)
    for param in model.parameters():
        param.requires_grad = False

    dataset = diffusion.dataset.wiki_dataset.WikiDataset(train=True, max_length=96)
    dict_to_model = dict()
    prefix = 'clean_'
    print(len(dataset))
    dataloader = DataLoader(dataset, batch_size=512, num_workers=30, shuffle=True, drop_last=True)

    target_folder = osp.join('data', 't5-base-stats')
    if not osp.exists(target_folder):
        os.makedirs(target_folder)

    config: T5Config = T5Config.from_pretrained('t5-base')

    layer = nn.BatchNorm1d(config.d_model).train().to(device)
    old_mean, old_var = None, None
    eps = 1e-7

    layer.weight.requires_grad = False
    layer.bias.requires_grad = False

    bar = tqdm(dataloader, desc='Calculating t5 stats: ')

    total_count = 0

    for batch in bar:
        batch = filter_by_prefix(batch, prefix=prefix)
        batch = dict_to_device(batch, device)
        attention_mask = batch['attention_mask'].bool()

        outs: BaseModelOutput = model.forward(**batch)
        last_hidden_state = outs.last_hidden_state
        # [BS; SENT_LEN; H]
        last_hidden_state = last_hidden_state.view(-1, last_hidden_state.shape[-1])
        mask = attention_mask.view(-1)
        embs = last_hidden_state[mask]
        # [SEQ; H]
        n = len(mask)
        layer.momentum = n / float(n + total_count)
        layer(embs[None].permute(0, 2, 1))
        total_count += n

        if old_mean is not None:
            diff = torch.max(torch.abs(old_mean - layer.running_mean.data))
            vdiff = torch.max(torch.abs(old_var - layer.running_var.data))
            bar.set_description(f'mean_diff: {diff:.7f}, var_diff: {vdiff:.7f}')
            if diff <= eps and vdiff <= eps:
                break

        old_mean = torch.clone(layer.running_mean.data)
        old_var = torch.clone(layer.running_var.data)
        torch.save(layer.running_mean, osp.join(target_folder, 'mean.pth'))
        torch.save(torch.sqrt(layer.running_var), osp.join(target_folder, 'std.pth'))

    print(layer.running_mean.shape, layer.running_var.shape)

