from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.time_embedding import SinusoidalPositionEmbedding


def zero_module(module):
    for p in module.parameters():
        p.detach().zero_()
    return module


class Upsample(nn.Module):
    """
    An upsampling layer implementing convolution
    """

    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.op = nn.Conv2d(self.channels, self.channels, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.op(x)


class Downsample(nn.Module):
    """
    A downsampling layer implementing convolution
    """

    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.op = nn.Conv2d(self.channels, self.channels, 3, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)


class ResidualBlock(nn.Module):
    def __init__(
        self,
        model_channels: int,
        emb_channels: int,
        num_groups: int,
        out_channels: int,
        dropout_rate: float = 0.0,
    ):
        super().__init__()
        self.model_channels = model_channels
        self.emb_channels = emb_channels
        self.out_channels = out_channels
        self.dropout_rate = dropout_rate

        self.in_layers = nn.Sequential(
            nn.GroupNorm(num_groups, self.model_channels),
            nn.SiLU(),
            nn.Conv2d(self.model_channels, self.out_channels, 3, padding=1),
        )

        self.emb_layers = nn.Sequential(
            nn.SiLU(), nn.Linear(emb_channels, self.out_channels)
        )

        self.out_layers = nn.Sequential(
            nn.GroupNorm(num_groups, self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout_rate),
            zero_module(nn.Conv2d(self.out_channels, self.out_channels, 3, padding=1)),
        )

        if self.out_channels == self.model_channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Conv2d(self.model_channels, self.out_channels, 1)

    def forward(self, x, time_emb):
        h = self.in_layers(x)
        emb_out = self.emb_layers(time_emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        h = h + emb_out
        h = self.out_layers(h)
        return self.skip_connection(x) + h


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        model_channels: int = 128,
        out_channels: int = 128,
        num_residual_blocks: int = 2,
        num_normalization_groups: int = 32,
        dropout_rate: float = 0.0,
        channel_mult=(1, 2, 4, 8),
    ):
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_residual_blocks = num_residual_blocks
        self.num_normalization_groups = num_normalization_groups
        self.dropout_rate = dropout_rate
        self.channel_mult = channel_mult

        self.time_embed_dim = model_channels * 4
        self.time_embed = SinusoidalPositionEmbedding(
            model_channels, self.time_embed_dim
        )

        self.input_blocks = nn.ModuleList(
            [nn.Conv2d(in_channels, model_channels, 3, padding=1)]
        )

        input_block_chans = [model_channels]
        ch = model_channels
        for level, mult in enumerate(channel_mult):
            for _ in range(num_residual_blocks):
                res_block = ResidualBlock(
                    model_channels=ch,
                    emb_channels=self.time_embed_dim,
                    num_groups=self.num_normalization_groups,
                    out_channels=mult * model_channels,
                    dropout_rate=self.dropout_rate,
                )
                ch = mult * model_channels
                self.input_blocks.append(res_block)
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                self.input_blocks.append(Downsample(ch))
                input_block_chans.append(ch)

        self.middle_block = nn.ModuleList(
            [
                ResidualBlock(
                    model_channels=ch,
                    emb_channels=self.time_embed_dim,
                    num_groups=self.num_normalization_groups,
                    out_channels=ch,
                    dropout_rate=self.dropout_rate,
                ),
                ResidualBlock(
                    model_channels=ch,
                    emb_channels=self.time_embed_dim,
                    num_groups=self.num_normalization_groups,
                    out_channels=ch,
                    dropout_rate=self.dropout_rate,
                ),
            ]
        )
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_residual_blocks + 1):
                res_block = ResidualBlock(
                    model_channels=ch + input_block_chans.pop(),
                    emb_channels=self.time_embed_dim,
                    num_groups=32,
                    out_channels=mult * model_channels,
                    dropout_rate=self.dropout_rate,
                )
                ch = model_channels * mult
                self.output_blocks.append(res_block)
                if level and i == num_residual_blocks:
                    self.output_blocks.append(Upsample(ch))

        self.out = nn.Sequential(
            nn.GroupNorm(self.num_normalization_groups, ch),
            nn.SiLU(),
            zero_module(nn.Conv2d(model_channels, out_channels, 3, padding=1)),
        )

    @property
    def inner_dtype(self):
        return next(self.input_blocks.parameters()).dtype

    def forward(self, x, timesteps):
        hs = []
        emb = self.time_embed(timesteps)

        h = x.type(self.inner_dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            cat_in = torch.cat([h, hs.pop()], dim=1)
            h = module(cat_in, emb)
        h = h.type(x.dtype)
        return self.out(h)
