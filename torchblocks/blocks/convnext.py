import torch
from torchvision.models.convnext import CNBlock


def cnb(channel, height, width):
    block = CNBlock(
        dim=channel,
        layer_scale=1e-6,
        stochastic_depth_prob=0.5,
    )
    return block


if __name__ == "__main__":
    feat = torch.rand([2, 32, 128, 128])
    block = CNBlock(
        dim=32,
        layer_scale=1e-6,
        stochastic_depth_prob=0.5,
    )
    out = block(feat)
    print(out.shape)
