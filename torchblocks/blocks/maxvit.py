from functools import partial
import torch
from torch import nn
from torchvision.models.maxvit import MaxVitBlock
from torchvision.models.maxvit import MaxVitLayer


def maxvit(channel, height, width):
    block = MaxVitLayer(
        in_channels=channel,
        out_channels=channel,
        squeeze_ratio=0.5,
        expansion_ratio=4,
        stride=1,
        # conv + transformer parameters
        norm_layer=partial(nn.BatchNorm2d, eps=1e-3, momentum=0.99),
        activation_layer=nn.GELU,
        # transformer parameters
        head_dim=16,
        mlp_ratio=4,
        mlp_dropout=0.0,
        attention_dropout=0.0,
        p_stochastic_dropout=0.5,
        partition_size=4,
        grid_size=[height, width],
    )
    return block


if __name__ == "__main__":
    feat = torch.rand([2, 32, 128, 128])
    block = MaxVitBlock(
        in_channels=32,
        out_channels=32,
        squeeze_ratio=0.5,
        expansion_ratio=4,
        # conv + transformer parameters
        norm_layer=partial(nn.BatchNorm2d, eps=1e-3, momentum=0.99),
        activation_layer=nn.GELU,
        # transformer parameters
        head_dim=32,
        mlp_ratio=4,
        mlp_dropout=0.0,
        attention_dropout=0.0,
        # partitioning parameters
        partition_size=4,
        input_grid_size=[128, 128],
        # number of layers
        n_layers=2,
        p_stochastic=[0.5, 0.5],
    )
    out = block(feat)
    print(out.shape)
