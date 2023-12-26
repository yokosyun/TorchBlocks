import torch
from torchvision.models.densenet import _DenseBlock


def denseblock(channel, height, width):
    block = _DenseBlock(
        num_layers=4,
        num_input_features=channel,
        bn_size=4,
        growth_rate=32,
        drop_rate=0,
        memory_efficient=False,
    )
    return block


if __name__ == "__main__":
    feat = torch.rand([2, 32, 128, 128])
    block = _DenseBlock(
        num_layers=6,
        num_input_features=32,
        bn_size=4,
        growth_rate=32,
        drop_rate=0,
        memory_efficient=False,
    )

    out = block(feat)
    print(out.shape)
