from torch import nn


def transformer_encoder_layer_1(channel, heigh, width):
    block = nn.TransformerEncoderLayer(d_model=channel, nhead=1, batch_first=True)
    return block


def transformer_encoder_layer_2(channel, heigh, width):
    block = nn.TransformerEncoderLayer(d_model=channel, nhead=2, batch_first=True)
    return block


def transformer_encoder_layer_4(channel, heigh, width):
    block = nn.TransformerEncoderLayer(d_model=channel, nhead=4, batch_first=True)
    return block


def transformer_encoder_layer_8(channel, heigh, width):
    block = nn.TransformerEncoderLayer(d_model=channel, nhead=8, batch_first=True)
    return block


def transformer_encoder_layer_16(channel, heigh, width):
    block = nn.TransformerEncoderLayer(d_model=channel, nhead=16, batch_first=True)
    return block


def transformer_encoder_layer_32(channel, heigh, width):
    block = nn.TransformerEncoderLayer(d_model=channel, nhead=32, batch_first=True)
    return block
