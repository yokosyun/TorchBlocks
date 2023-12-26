from torch import nn
from torchvision.models.swin_transformer import (
    SwinTransformerBlock,
    ShiftedWindowAttention,
    SwinTransformerBlockV2,
    ShiftedWindowAttentionV2,
)
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from mmengine.model import BaseModule
from mmcv.cnn import ConvModule
from torch import Tensor
import torch


class ELANBlock(BaseModule):
    """Efficient layer aggregation networks for YOLOv7.

    Args:
        in_channels (int): The input channels of this Module.
        out_channels (int): The out channels of this Module.
        middle_ratio (float): The scaling ratio of the middle layer
            based on the in_channels.
        block_ratio (float): The scaling ratio of the block layer
            based on the in_channels.
        num_blocks (int): The number of blocks in the main branch.
            Defaults to 2.
        num_convs_in_block (int): The number of convs pre block.
            Defaults to 1.
        conv_cfg (dict): Config dict for convolution layer. Defaults to None.
            which means using conv2d. Defaults to None.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to dict(type='BN', momentum=0.03, eps=0.001).
        act_cfg (dict): Config dict for activation layer.
            Defaults to dict(type='SiLU', inplace=True).
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        middle_ratio: float,
        block_ratio: float,
        num_blocks: int = 2,
        num_convs_in_block: int = 1,
        conv_cfg: OptConfigType = None,
        norm_cfg: ConfigType = dict(type="BN", momentum=0.03, eps=0.001),
        act_cfg: ConfigType = dict(type="SiLU", inplace=True),
        init_cfg: OptMultiConfig = None,
    ):
        super().__init__(init_cfg=init_cfg)
        assert num_blocks >= 1
        assert num_convs_in_block >= 1

        middle_channels = int(in_channels * middle_ratio)
        block_channels = int(in_channels * block_ratio)
        final_conv_in_channels = int(num_blocks * block_channels) + 2 * middle_channels
        self.final_conv_in_channels = final_conv_in_channels

        self.main_conv = ConvModule(
            in_channels,
            middle_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )

        self.short_conv = ConvModule(
            in_channels,
            middle_channels,
            1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
        )

        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            if num_convs_in_block == 1:
                internal_block = ConvModule(
                    middle_channels,
                    block_channels,
                    3,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                )
            else:
                internal_block = []
                for _ in range(num_convs_in_block):
                    internal_block.append(
                        ConvModule(
                            middle_channels,
                            block_channels,
                            3,
                            padding=1,
                            conv_cfg=conv_cfg,
                            norm_cfg=norm_cfg,
                            act_cfg=act_cfg,
                        )
                    )
                    middle_channels = block_channels
                internal_block = nn.Sequential(*internal_block)

            middle_channels = block_channels
            self.blocks.append(internal_block)

        # self.final_conv = ConvModule(
        #     final_conv_in_channels,
        #     out_channels,
        #     1,
        #     conv_cfg=conv_cfg,
        #     norm_cfg=norm_cfg,
        #     act_cfg=act_cfg,
        # )

    def forward(self, x: Tensor) -> Tensor:
        """Forward process
        Args:
            x (Tensor): The input tensor.
        """
        x_short = self.short_conv(x)
        x_main = self.main_conv(x)
        block_outs = []
        x_block = x_main
        for block in self.blocks:
            x_block = block(x_block)
            block_outs.append(x_block)
        x_final = torch.cat((*block_outs[::-1], x_main, x_short), dim=1)
        # return self.final_conv(x_final)
        return x_final


class ELANSwin(nn.Module):
    def __init__(
        self,
        in_channels: int,
        window_size: int,
        heads: int,
    ):
        super().__init__()

        self.elan = ELANBlock(
            in_channels=in_channels,
            out_channels=in_channels,
            middle_ratio=0.5,
            block_ratio=0.5,
            num_blocks=2,
            num_convs_in_block=2,
        )

        window_size = [window_size, window_size]
        self.swin = SwinTransformerBlockV2(
            dim=self.elan.final_conv_in_channels,
            num_heads=heads,
            window_size=window_size,
            # shift_size=[0 if i_layer % 2 == 0 else w // 2 for w in window_size],
            # shift_size=[window_size[0] // 2, window_size[1] // 2],
            shift_size=[0, 0],
            mlp_ratio=1.0,
            dropout=0.0,
            attention_dropout=0.0,
            stochastic_depth_prob=0.0,
            norm_layer=nn.LayerNorm,
            # attn_layer=ShiftedWindowAttention,
            attn_layer=ShiftedWindowAttentionV2,
        )

    def forward(self, x):
        e = self.elan(x)
        e = e.permute(0, 2, 3, 1)
        s = self.swin(e)
        s = s.permute(0, 3, 1, 2)
        return s


def elanswin_w4_h4(channel, height, width):
    block = ELANSwin(channel, window_size=4, heads=4)
    return block


def elanswin_w4_h8(channel, height, width):
    block = ELANSwin(channel, window_size=4, heads=8)
    return block


def elanswin_w8_h4(channel, height, width):
    block = ELANSwin(channel, window_size=8, heads=4)
    return block


def elanswin_w8_h8(channel, height, width):
    block = ELANSwin(channel, window_size=8, heads=8)
    return block
