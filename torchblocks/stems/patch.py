from torch import nn


def patch_k4_s2(in_channels, out_channels):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=4,
        stride=2,
        padding=1,
    )


def patch_k4_s4(in_channels, out_channels):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=4,
        stride=4,
    )


def patch_k8_s2(in_channels, out_channels):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=8,
        stride=2,
        padding=3,
    )


def patch_k8_s4(in_channels, out_channels):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=8,
        stride=4,
        padding=3,
    )


def patch_k8_s8(in_channels, out_channels):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=8,
        stride=8,
    )


def patch_k16_s2(in_channels, out_channels):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=16,
        stride=2,
    )


def patch_k16_s4(in_channels, out_channels):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=16,
        stride=4,
    )


def patch_k16_s8(in_channels, out_channels):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=16,
        stride=8,
    )


def patch_k16_s16(in_channels, out_channels):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=16,
        stride=16,
    )


def patch_k32(in_channels, out_channels):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=32,
        stride=32,
    )
