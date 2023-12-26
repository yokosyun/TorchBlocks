# https://github.com/pytorch/vision/blob/main/torchvision/models/regnet.py


import torch
from torch import nn
from torchvision.models.regnet import ResBottleneckBlock, BlockParams


def x_block(channel, height, width):
    block = ResBottleneckBlock(
        width_in=channel,
        width_out=channel,
        stride=1,
        norm_layer=nn.BatchNorm2d,
        activation_layer=nn.ReLU,
        group_width=channel,
        bottleneck_multiplier=1.0,
        se_ratio=0.25,
    )
    return block


def x_block_gw16(channel, height, width):
    group_width = 16 if channel > 16 else channel
    block = ResBottleneckBlock(
        width_in=channel,
        width_out=channel,
        stride=1,
        norm_layer=nn.BatchNorm2d,
        activation_layer=nn.ReLU,
        group_width=group_width,
        bottleneck_multiplier=1.0,
        se_ratio=0.25,
    )
    return block


def x_block_gw32(channel, height, width):
    group_width = 32 if channel > 32 else channel
    block = ResBottleneckBlock(
        width_in=channel,
        width_out=channel,
        stride=1,
        norm_layer=nn.BatchNorm2d,
        activation_layer=nn.ReLU,
        group_width=group_width,
        bottleneck_multiplier=1.0,
        se_ratio=0.25,
    )
    return block


def x_block_gw64(channel, height, width):
    group_width = 64 if channel > 64 else channel
    block = ResBottleneckBlock(
        width_in=channel,
        width_out=channel,
        stride=1,
        norm_layer=nn.BatchNorm2d,
        activation_layer=nn.ReLU,
        group_width=group_width,
        bottleneck_multiplier=1.0,
        se_ratio=0.25,
    )
    return block


def x_block_gw128(channel, height, width):
    group_width = 128 if channel > 128 else channel
    block = ResBottleneckBlock(
        width_in=channel,
        width_out=channel,
        stride=1,
        norm_layer=nn.BatchNorm2d,
        activation_layer=nn.ReLU,
        group_width=group_width,
        bottleneck_multiplier=1.0,
        se_ratio=0.25,
    )
    return block


if __name__ == "__main__":
    feat = torch.rand([2, 32, 128, 128])

    params = BlockParams.from_init_params(
        depth=23, w_0=320, w_a=69.86, w_m=2.0, group_width=168
    )

    block = ResBottleneckBlock(
        width_in=32,
        width_out=32,
        stride=1,
        norm_layer=nn.BatchNorm2d,
        activation_layer=nn.ReLU,
        group_width=32,
        bottleneck_multiplier=1.0,
        se_ratio=0.25,
    )

    out = block(feat)
    print(out.shape)
