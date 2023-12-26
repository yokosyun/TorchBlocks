# https://github.com/pytorch/vision/blob/main/torchvision/models/efficientnet.py#L346
# https://arxiv.org/pdf/2203.06717v2.pdf
from functools import partial
import torch
from torch import nn
from torchvision.models.efficientnet import (
    MBConvConfig,
    FusedMBConvConfig,
    MBConv,
    FusedMBConv,
)
from typing import Callable, List, Optional
from torch import Tensor
from torchvision.ops.misc import Conv2dNormActivation, SqueezeExcitation
from torchvision.ops import StochasticDepth
from torchvision.models._utils import _make_divisible


def adjust_channels(
    channels: int, width_mult: float, min_value: Optional[int] = None
) -> int:
    return _make_divisible(channels * width_mult, 8, min_value)


class LargeMBConv(nn.Module):
    def __init__(
        self,
        input_channels,
        out_channels,
        stochastic_depth_prob: float,
        norm_layer: Callable[..., nn.Module],
        group_width=1,
        kernel_size=3,
        stride=1,
        expand_ratio=4,
        se_layer: Callable[..., nn.Module] = SqueezeExcitation,
    ) -> None:
        super().__init__()

        if not (1 <= stride <= 2):
            raise ValueError("illegal stride value")

        self.use_res_connect = stride == 1 and input_channels == out_channels

        layers: List[nn.Module] = []
        activation_layer = nn.SiLU

        # expand
        expanded_channels = adjust_channels(input_channels, expand_ratio)
        if expanded_channels != input_channels:
            layers.append(
                Conv2dNormActivation(
                    input_channels,
                    expanded_channels,
                    kernel_size=1,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                )
            )

        # depthwise
        layers.append(
            Conv2dNormActivation(
                expanded_channels,
                expanded_channels,
                kernel_size=kernel_size,
                stride=stride,
                groups=expanded_channels // group_width,
                norm_layer=norm_layer,
                activation_layer=activation_layer,
            )
        )

        # squeeze and excitation
        squeeze_channels = max(1, input_channels // 4)
        layers.append(
            se_layer(
                expanded_channels,
                squeeze_channels,
                activation=partial(nn.SiLU, inplace=True),
            )
        )

        # project
        layers.append(
            Conv2dNormActivation(
                expanded_channels,
                out_channels,
                kernel_size=1,
                norm_layer=norm_layer,
                activation_layer=None,
            )
        )

        self.block = nn.Sequential(*layers)
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")
        self.out_channels = out_channels

    def forward(self, input: Tensor) -> Tensor:
        result = self.block(input)
        if self.use_res_connect:
            result = self.stochastic_depth(result)
            result += input
        return result


class GroupFusedMBConv(nn.Module):
    def __init__(
        self,
        input_channels,
        out_channels,
        stochastic_depth_prob: float,
        norm_layer: Callable[..., nn.Module],
        groups=1,
        kernel_size=3,
        stride=1,
        expand_ratio=4,
    ) -> None:
        super().__init__()

        if not (1 <= stride <= 2):
            raise ValueError("illegal stride value")

        self.use_res_connect = stride == 1 and input_channels == out_channels

        layers: List[nn.Module] = []
        activation_layer = nn.SiLU

        expanded_channels = adjust_channels(input_channels, expand_ratio)

        # groups = input_channels // group_width if input_channels > group_width else 1
        if expanded_channels != input_channels:
            # fused expand
            layers.append(
                Conv2dNormActivation(
                    input_channels,
                    expanded_channels,
                    kernel_size=kernel_size,
                    groups=groups,
                    stride=stride,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                )
            )

            # project
            layers.append(
                Conv2dNormActivation(
                    expanded_channels,
                    out_channels,
                    kernel_size=1,
                    norm_layer=norm_layer,
                    activation_layer=None,
                )
            )
        else:
            layers.append(
                Conv2dNormActivation(
                    input_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    groups=groups,
                    stride=stride,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                )
            )

        self.block = nn.Sequential(*layers)
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")
        self.out_channels = out_channels

    def forward(self, input: Tensor) -> Tensor:
        result = self.block(input)
        if self.use_res_connect:
            result = self.stochastic_depth(result)
            result += input
        return result


def mbconv(channel, height, width):
    cfg = MBConvConfig(
        expand_ratio=4,
        kernel=3,
        stride=1,
        input_channels=channel,
        out_channels=channel,
        num_layers=0,
    )
    block = MBConv(cfg, stochastic_depth_prob=0.2, norm_layer=nn.BatchNorm2d)
    return block


def mbconv_k3_e4(channel, height, width):
    block = LargeMBConv(
        input_channels=channel,
        out_channels=channel,
        group_width=1,
        kernel_size=3,
        stride=1,
        expand_ratio=4,
        stochastic_depth_prob=0.2,
        norm_layer=nn.BatchNorm2d,
    )
    return block


def mbconv_k5_e4(channel, height, width):
    block = LargeMBConv(
        input_channels=channel,
        out_channels=channel,
        group_width=1,
        kernel_size=5,
        stride=1,
        expand_ratio=4,
        stochastic_depth_prob=0.2,
        norm_layer=nn.BatchNorm2d,
    )
    return block


def mbconv_k7_e4(channel, height, width):
    block = LargeMBConv(
        input_channels=channel,
        out_channels=channel,
        group_width=1,
        kernel_size=7,
        stride=1,
        expand_ratio=4,
        stochastic_depth_prob=0.2,
        norm_layer=nn.BatchNorm2d,
    )
    return block


def mbconv_k13_e4(channel, height, width):
    block = LargeMBConv(
        input_channels=channel,
        out_channels=channel,
        group_width=1,
        kernel_size=13,
        stride=1,
        expand_ratio=4,
        stochastic_depth_prob=0.2,
        norm_layer=nn.BatchNorm2d,
    )
    return block


def mbconv_k23_e4(channel, height, width):
    block = LargeMBConv(
        input_channels=channel,
        out_channels=channel,
        group_width=1,
        kernel_size=23,
        stride=1,
        expand_ratio=4,
        stochastic_depth_prob=0.2,
        norm_layer=nn.BatchNorm2d,
    )
    return block


def mbconv_k33_e4(channel, height, width):
    block = LargeMBConv(
        input_channels=channel,
        out_channels=channel,
        group_width=1,
        kernel_size=33,
        stride=1,
        expand_ratio=4,
        stochastic_depth_prob=0.2,
        norm_layer=nn.BatchNorm2d,
    )
    return block


def mbconv_k3_e2(channel, height, width):
    block = LargeMBConv(
        input_channels=channel,
        out_channels=channel,
        group_width=1,
        kernel_size=3,
        stride=1,
        expand_ratio=2,
        stochastic_depth_prob=0.2,
        norm_layer=nn.BatchNorm2d,
    )
    return block


def mbconv_k5_e2(channel, height, width):
    block = LargeMBConv(
        input_channels=channel,
        out_channels=channel,
        group_width=1,
        kernel_size=5,
        stride=1,
        expand_ratio=2,
        stochastic_depth_prob=0.2,
        norm_layer=nn.BatchNorm2d,
    )
    return block


def mbconv_k7_e2(channel, height, width):
    block = LargeMBConv(
        input_channels=channel,
        out_channels=channel,
        group_width=1,
        kernel_size=7,
        stride=1,
        expand_ratio=2,
        stochastic_depth_prob=0.2,
        norm_layer=nn.BatchNorm2d,
    )
    return block


def mbconv_k13_e2(channel, height, width):
    block = LargeMBConv(
        input_channels=channel,
        out_channels=channel,
        group_width=1,
        kernel_size=13,
        stride=1,
        expand_ratio=2,
        stochastic_depth_prob=0.2,
        norm_layer=nn.BatchNorm2d,
    )
    return block


def mbconv_k23_e2(channel, height, width):
    block = LargeMBConv(
        input_channels=channel,
        out_channels=channel,
        group_width=1,
        kernel_size=23,
        stride=1,
        expand_ratio=2,
        stochastic_depth_prob=0.2,
        norm_layer=nn.BatchNorm2d,
    )
    return block


def mbconv_k33_e2(channel, height, width):
    block = LargeMBConv(
        input_channels=channel,
        out_channels=channel,
        group_width=1,
        kernel_size=33,
        stride=1,
        expand_ratio=2,
        stochastic_depth_prob=0.2,
        norm_layer=nn.BatchNorm2d,
    )
    return block


def fused_mbconv_g1_e1(channel, height, width):
    block = GroupFusedMBConv(
        input_channels=channel,
        out_channels=channel,
        groups=1,
        kernel_size=3,
        stride=1,
        expand_ratio=1,
        stochastic_depth_prob=0.2,
        norm_layer=nn.BatchNorm2d,
    )
    return block


def fused_mbconv_g1_e2(channel, height, width):
    block = GroupFusedMBConv(
        input_channels=channel,
        out_channels=channel,
        groups=1,
        kernel_size=3,
        stride=1,
        expand_ratio=2,
        stochastic_depth_prob=0.2,
        norm_layer=nn.BatchNorm2d,
    )
    return block


def fused_mbconv_g1_e3(channel, height, width):
    block = GroupFusedMBConv(
        input_channels=channel,
        out_channels=channel,
        groups=1,
        kernel_size=3,
        stride=1,
        expand_ratio=3,
        stochastic_depth_prob=0.2,
        norm_layer=nn.BatchNorm2d,
    )
    return block


def fused_mbconv_g1_e4(channel, height, width):
    block = GroupFusedMBConv(
        input_channels=channel,
        out_channels=channel,
        groups=1,
        kernel_size=3,
        stride=1,
        expand_ratio=4,
        stochastic_depth_prob=0.2,
        norm_layer=nn.BatchNorm2d,
    )
    return block


def fused_mbconv_g2_e1(channel, height, width):
    block = GroupFusedMBConv(
        input_channels=channel,
        out_channels=channel,
        groups=2,
        kernel_size=3,
        stride=1,
        expand_ratio=1,
        stochastic_depth_prob=0.2,
        norm_layer=nn.BatchNorm2d,
    )
    return block


def fused_mbconv_g2_e2(channel, height, width):
    block = GroupFusedMBConv(
        input_channels=channel,
        out_channels=channel,
        groups=2,
        kernel_size=3,
        stride=1,
        expand_ratio=2,
        stochastic_depth_prob=0.2,
        norm_layer=nn.BatchNorm2d,
    )
    return block


def fused_mbconv_g2_e4(channel, height, width):
    block = GroupFusedMBConv(
        input_channels=channel,
        out_channels=channel,
        groups=2,
        kernel_size=3,
        stride=1,
        expand_ratio=4,
        stochastic_depth_prob=0.2,
        norm_layer=nn.BatchNorm2d,
    )
    return block


def fused_mbconv_g4_e1(channel, height, width):
    block = GroupFusedMBConv(
        input_channels=channel,
        out_channels=channel,
        groups=4,
        kernel_size=3,
        stride=1,
        expand_ratio=1,
        stochastic_depth_prob=0.2,
        norm_layer=nn.BatchNorm2d,
    )
    return block


def fused_mbconv_g4_e2(channel, height, width):
    block = GroupFusedMBConv(
        input_channels=channel,
        out_channels=channel,
        groups=4,
        kernel_size=3,
        stride=1,
        expand_ratio=2,
        stochastic_depth_prob=0.2,
        norm_layer=nn.BatchNorm2d,
    )
    return block


def fused_mbconv_g4_e4(channel, height, width):
    block = GroupFusedMBConv(
        input_channels=channel,
        out_channels=channel,
        groups=4,
        kernel_size=3,
        stride=1,
        expand_ratio=4,
        stochastic_depth_prob=0.2,
        norm_layer=nn.BatchNorm2d,
    )
    return block


def fused_mbconv_g8_e1(channel, height, width):
    block = GroupFusedMBConv(
        input_channels=channel,
        out_channels=channel,
        groups=8,
        kernel_size=3,
        stride=1,
        expand_ratio=1,
        stochastic_depth_prob=0.2,
        norm_layer=nn.BatchNorm2d,
    )
    return block


def fused_mbconv_g8_e2(channel, height, width):
    block = GroupFusedMBConv(
        input_channels=channel,
        out_channels=channel,
        groups=8,
        kernel_size=3,
        stride=1,
        expand_ratio=2,
        stochastic_depth_prob=0.2,
        norm_layer=nn.BatchNorm2d,
    )
    return block


def fused_mbconv_g8_e4(channel, height, width):
    block = GroupFusedMBConv(
        input_channels=channel,
        out_channels=channel,
        groups=8,
        kernel_size=3,
        stride=1,
        expand_ratio=4,
        stochastic_depth_prob=0.2,
        norm_layer=nn.BatchNorm2d,
    )
    return block


def fused_mbconv(channel, height, width):
    cfg = FusedMBConvConfig(
        expand_ratio=4,
        kernel=3,
        stride=1,
        input_channels=channel,
        out_channels=channel,
        num_layers=0,
    )
    block = FusedMBConv(cfg, stochastic_depth_prob=0.2, norm_layer=nn.BatchNorm2d)
    return block


if __name__ == "__main__":
    feat = torch.rand([2, 32, 128, 128])

    cfg = MBConvConfig(
        expand_ratio=4,
        kernel=3,
        stride=1,
        input_channels=32,
        out_channels=32,
        num_layers=0,
    )
    block = MBConv(cfg, stochastic_depth_prob=0.2, norm_layer=nn.BatchNorm2d)
    out = block(feat)
    print(out.shape)

    cfg = FusedMBConvConfig(
        expand_ratio=4,
        kernel=3,
        stride=1,
        input_channels=32,
        out_channels=32,
        num_layers=0,
    )
    block = FusedMBConv(cfg, stochastic_depth_prob=0.2, norm_layer=nn.BatchNorm2d)
    out = block(feat)
    print(out.shape)
