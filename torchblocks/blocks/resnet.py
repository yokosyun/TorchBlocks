# https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
import torch
from torchvision.models.resnet import BasicBlock, Bottleneck
from typing import Optional, Callable
from torch import nn
from torch import Tensor


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        kernel_size: int = 3,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        padding = (kernel_size - 1) // 2
        self.conv1 = nn.Conv2d(
            inplanes,
            planes,
            kernel_size=kernel_size,
            stride=stride,
            groups=groups,
            padding=padding,
            bias=False,
        )
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            inplanes,
            planes,
            kernel_size=kernel_size,
            stride=stride,
            groups=1,
            padding=padding,
            bias=False,
        )
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


def basic_k3_g1(channel, height, width):
    block = BasicBlock(inplanes=channel, planes=channel, kernel_size=3, groups=1)
    return block


def basic_k3_g2(channel, height, width):
    block = BasicBlock(inplanes=channel, planes=channel, kernel_size=3, groups=2)
    return block


def basic_k3_g4(channel, height, width):
    block = BasicBlock(inplanes=channel, planes=channel, kernel_size=3, groups=4)
    return block


def basic_k3_g8(channel, height, width):
    block = BasicBlock(inplanes=channel, planes=channel, kernel_size=3, groups=8)
    return block


def basic_k5(channel, height, width):
    block = BasicBlock(inplanes=channel, planes=channel, kernel_size=5)
    return block


def basic_k7(channel, height, width):
    block = BasicBlock(inplanes=channel, planes=channel, kernel_size=7)
    return block


def basic_k9(channel, height, width):
    block = BasicBlock(inplanes=channel, planes=channel, kernel_size=9)
    return block


def basic_k11(channel, height, width):
    block = BasicBlock(inplanes=channel, planes=channel, kernel_size=11)
    return block


def basic_k13(channel, height, width):
    block = BasicBlock(inplanes=channel, planes=channel, kernel_size=13)
    return block


def bottleneck(channel, height, width):
    block = Bottleneck(inplanes=channel, planes=channel // 4)
    return block


if __name__ == "__main__":
    feat = torch.rand([2, 32, 128, 128])
    block = BasicBlock(inplanes=32, planes=32)
    out = block(feat)
    print(out.shape)

    block = Bottleneck(inplanes=32, planes=32 // 4)
    out = block(feat)
    print(out.shape)
