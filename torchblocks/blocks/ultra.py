# https://github.com/ultralytics/ultralytics/blob/main/ultralytics/nn/modules/block.py
from torch import nn
from ultralytics.nn.modules import (
    C2f,
    C2,
    HGBlock,
    HGStem,
    C3,
    GhostBottleneck,
    C3Ghost,
    RepC3,
)


def c2f(channel, height, width):
    return C2f(c1=channel, c2=channel, n=1, shortcut=False, g=1, e=0.5)


def hgb_block(channel, height, width):
    return HGBlock(
        c1=channel,
        cm=channel // 2,
        c2=channel * 2,
        k=3,
        n=6,
        lightconv=False,
        shortcut=False,
        act=nn.ReLU(),
    )


# def hgs_stem(channel, height, width):
#     return HGStem(c1=channel, cm=channel, c2=channel)


def c2(channel, height, width):
    return C2(c1=channel, c2=channel, n=1, shortcut=True, g=1, e=0.5)


def c3(channel, height, width):
    return C3(c1=channel, c2=channel, n=1, shortcut=True, g=1, e=0.5)


def ghost_bottleneck(channel, height, width):
    return GhostBottleneck(c1=channel, c2=channel, k=3, s=1)


def c3ghost(channel, height, width):
    return C3Ghost(c1=channel, c2=channel, n=1, shortcut=True, g=1, e=0.5)


def rep_c3(channel, height, width):
    return RepC3(c1=channel, c2=channel, n=3, e=1.0)
