import torch
from torchvision.models.shufflenetv2 import InvertedResidual


def shuffle_block(channel, height, width):
    block = InvertedResidual(inp=channel, oup=channel, stride=1)
    return block


if __name__ == "__main__":
    feat = torch.rand([2, 32, 128, 128])
    block = InvertedResidual(inp=32, oup=32, stride=1)
    out = block(feat)
    print(out.shape)
