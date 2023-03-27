import torch

from torch import nn


class EncNormalizer(nn.Module):
    def __init__(self, enc_mean_path: str, enc_std_path: str):
        super().__init__()
        self.enc_mean = nn.Parameter(
            torch.load(enc_mean_path, map_location='cpu')[None, None, :],
            requires_grad=False
        )
        self.enc_std = nn.Parameter(
            torch.load(enc_std_path, map_location='cpu')[None, None, :],
            requires_grad=False
        )

    def forward(self, *args, **kwargs):
        return nn.Identity()(*args, **kwargs)

    def normalize(self, encoding):
        #print(encoding.shape, self.enc_mean.shape, flush=True)
        #print(torch.sqrt(torch.sum(self.enc_mean**2)), flush=True)
        return (encoding - self.enc_mean) / self.enc_std

    def denormalize(self, pred_x_0):
        return pred_x_0 * self.enc_std + self.enc_mean
