from torch import nn


def conv_k3_s2(in_channels, out_channels):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=2,
        padding=1,
    )


def conv_k5_s2(in_channels, out_channels):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=5,
        stride=2,
        padding=2,
    )


def conv_k5_s4(in_channels, out_channels):
    return nn.Conv2d(
        in_channels,
        out_channels * 2,
        kernel_size=5,
        stride=4,
        padding=2,
    )


def conv_k6_s2(in_channels, out_channels):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=6,
        stride=2,
        padding=2,
    )


def conv_k7_s2(in_channels, out_channels):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=7,
        stride=2,
        padding=3,
    )


def conv_k7_s4(in_channels, out_channels):
    return nn.Conv2d(
        in_channels,
        out_channels * 2,
        kernel_size=7,
        stride=4,
        padding=3,
    )


def conv_k9_s2(in_channels, out_channels):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=9,
        stride=2,
        padding=4,
    )


def conv_k9_s4(in_channels, out_channels):
    return nn.Conv2d(
        in_channels,
        out_channels * 2,
        kernel_size=9,
        stride=4,
        padding=4,
    )


def conv_k9_s8(in_channels, out_channels):
    return nn.Conv2d(
        in_channels,
        out_channels * 4,
        kernel_size=9,
        stride=8,
        padding=4,
    )


def conv_k11_s2(in_channels, out_channels):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=11,
        stride=2,
        padding=5,
    )


def conv_k11_s4(in_channels, out_channels):
    return nn.Conv2d(
        in_channels,
        out_channels * 2,
        kernel_size=11,
        stride=4,
        padding=5,
    )


def conv_k11_s8(in_channels, out_channels):
    return nn.Conv2d(
        in_channels,
        out_channels * 4,
        kernel_size=11,
        stride=8,
        padding=5,
    )


def conv_k13_s2(in_channels, out_channels):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=13,
        stride=2,
        padding=5,
    )


def conv_k13_s4(in_channels, out_channels):
    return nn.Conv2d(
        in_channels,
        out_channels * 2,
        kernel_size=13,
        stride=4,
        padding=5,
    )


def conv_k13_s8(in_channels, out_channels):
    return nn.Conv2d(
        in_channels,
        out_channels * 4,
        kernel_size=13,
        stride=8,
        padding=5,
    )


def conv_k15_s4(in_channels, out_channels):
    return nn.Conv2d(
        in_channels,
        out_channels * 2,
        kernel_size=15,
        stride=4,
        padding=6,
    )


def conv_k17_s4(in_channels, out_channels):
    return nn.Conv2d(
        in_channels,
        out_channels * 2,
        kernel_size=17,
        stride=4,
        padding=7,
    )


def conv_k19_s4(in_channels, out_channels):
    return nn.Conv2d(
        in_channels,
        out_channels * 2,
        kernel_size=19,
        stride=4,
        padding=8,
    )


def conv_k21_s4(in_channels, out_channels):
    return nn.Conv2d(
        in_channels,
        out_channels * 2,
        kernel_size=21,
        stride=4,
        padding=9,
    )
