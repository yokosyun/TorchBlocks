from torch import nn
from torchvision.models.swin_transformer import (
    SwinTransformerBlock,
    ShiftedWindowAttention,
    SwinTransformerBlockV2,
    ShiftedWindowAttentionV2,
)


def swintransformerblockv2_w4_h4(channel, height, width):
    W = 4
    heads = 4
    window_size = [W, W]
    block = SwinTransformerBlockV2(
        dim=channel,
        num_heads=heads,
        window_size=window_size,
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
    return block


def swintransformerblockv2_w4_h8(channel, height, width):
    W = 4
    heads = 8
    window_size = [W, W]
    block = SwinTransformerBlockV2(
        dim=channel,
        num_heads=heads,
        window_size=window_size,
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
    return block


def swintransformerblockv2_w4_h16(channel, height, width):
    W = 4
    heads = 16
    window_size = [W, W]
    block = SwinTransformerBlockV2(
        dim=channel,
        num_heads=heads,
        window_size=window_size,
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
    return block


def swintransformerblockv2_w4_h32(channel, height, width):
    W = 4
    heads = 32
    window_size = [W, W]
    block = SwinTransformerBlockV2(
        dim=channel,
        num_heads=heads,
        window_size=window_size,
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
    return block


def swintransformerblockv2_w8_h4(channel, height, width):
    W = 8
    heads = 4
    window_size = [W, W]
    block = SwinTransformerBlockV2(
        dim=channel,
        num_heads=heads,
        window_size=window_size,
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
    return block


def swintransformerblockv2_w8_h8(channel, height, width):
    W = 8
    heads = 8
    window_size = [W, W]
    block = SwinTransformerBlockV2(
        dim=channel,
        num_heads=heads,
        window_size=window_size,
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
    return block


def swintransformerblockv2_w8_h16(channel, height, width):
    W = 8
    heads = 16
    window_size = [W, W]
    block = SwinTransformerBlockV2(
        dim=channel,
        num_heads=heads,
        window_size=window_size,
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
    return block


def swintransformerblockv2_w8_h32(channel, height, width):
    W = 8
    heads = 32
    window_size = [W, W]
    block = SwinTransformerBlockV2(
        dim=channel,
        num_heads=heads,
        window_size=window_size,
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
    return block


def swintransformerblockv2_w16_h4(channel, height, width):
    W = 16
    heads = 4
    window_size = [W, W]
    block = SwinTransformerBlockV2(
        dim=channel,
        num_heads=heads,
        window_size=window_size,
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
    return block


def swintransformerblockv2_w16_h8(channel, height, width):
    W = 16
    heads = 8
    window_size = [W, W]
    block = SwinTransformerBlockV2(
        dim=channel,
        num_heads=heads,
        window_size=window_size,
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
    return block


def swintransformerblockv2_w16_h16(channel, height, width):
    W = 16
    heads = 16
    window_size = [W, W]
    block = SwinTransformerBlockV2(
        dim=channel,
        num_heads=heads,
        window_size=window_size,
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
    return block


def swintransformerblockv2_w16_h32(channel, height, width):
    W = 16
    heads = 32
    window_size = [W, W]
    block = SwinTransformerBlockV2(
        dim=channel,
        num_heads=heads,
        window_size=window_size,
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
    return block


def swintransformerblockv2_w32_h4(channel, height, width):
    W = 32
    heads = 4
    window_size = [W, W]
    block = SwinTransformerBlockV2(
        dim=channel,
        num_heads=heads,
        window_size=window_size,
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
    return block


def swintransformerblockv2_w32_h8(channel, height, width):
    W = 32
    heads = 8
    window_size = [W, W]
    block = SwinTransformerBlockV2(
        dim=channel,
        num_heads=heads,
        window_size=window_size,
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
    return block


def swintransformerblockv2_w32_h16(channel, height, width):
    W = 32
    heads = 16
    window_size = [W, W]
    block = SwinTransformerBlockV2(
        dim=channel,
        num_heads=heads,
        window_size=window_size,
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
    return block


def swintransformerblockv2_w32_h32(channel, height, width):
    W = 32
    heads = 32
    window_size = [W, W]
    block = SwinTransformerBlockV2(
        dim=channel,
        num_heads=heads,
        window_size=window_size,
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
    return block


def swintransformerblockv2_w64_h8(channel, height, width):
    W = 64
    heads = 8
    window_size = [W, W]
    block = SwinTransformerBlockV2(
        dim=channel,
        num_heads=heads,
        window_size=window_size,
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
    return block


if __name__ == "__main__":
    if False:
        import torch
        from torchblocks.utils.profile import get_latency

        torch.backends.cudnn.deterministic = False
        # torch.backends.cudnn.benchmark = True

        scale_factor = 1
        channel = 32 * scale_factor
        height = 216 // scale_factor
        width = 216 // scale_factor
        batch_size = 1
        device = "cuda"
        dtype = torch.float32

        block = swintransformerblockv2(channel, height, width)
        block.eval()

        feat = torch.rand([batch_size, height, width, channel])
        feat = feat.to(dtype=dtype, device=device)
        block = block.to(dtype=dtype, device=device)
        latency = get_latency(block, feat)
        print("median time=", latency)

        window_size = [8, 8]
        block = SwinTransformerBlockV2(
            dim=channel,
            num_heads=16,
            window_size=window_size,
            shift_size=[0, 0],
            mlp_ratio=4.0,
            dropout=0.0,
            attention_dropout=0.0,
            stochastic_depth_prob=0.0,
            norm_layer=nn.LayerNorm,
            attn_layer=ShiftedWindowAttention,
        )

        feat = torch.rand([batch_size, height, width, channel])
        feat = feat.to(dtype=dtype, device=device)
        block = block.to(dtype=dtype, device=device)

        ouput = block(feat)

    if False:
        import torch
        from torchvision.models.swin_transformer import shifted_window_attention

        batch_size = 2
        channels = 96
        height = 128
        width = 128
        window_size = [7, 7]
        num_heads = 3
        shift_size = [0, 0]

        input = torch.rand([batch_size, height, width, channels])
        qkv_weight = torch.rand([3 * channels, channels])
        proj_weight = torch.rand([channels, channels])
        relative_position_bias = torch.rand(
            [
                1,
                num_heads,
                window_size[0] * window_size[1],
                window_size[0] * window_size[1],
            ]
        )

        old_attn = shifted_window_attention(
            input=input,
            qkv_weight=qkv_weight,
            proj_weight=proj_weight,
            relative_position_bias=relative_position_bias,
            window_size=window_size,
            num_heads=num_heads,
            shift_size=shift_size,
            attention_dropout=0.0,
            dropout=0.0,
            qkv_bias=None,
            proj_bias=None,
            logit_scale=None,
            training=True,
            use_efficient_attention=False,
        )

        new_attn = shifted_window_attention(
            input=input,
            qkv_weight=qkv_weight,
            proj_weight=proj_weight,
            relative_position_bias=relative_position_bias,
            window_size=window_size,
            num_heads=num_heads,
            shift_size=shift_size,
            attention_dropout=0.0,
            dropout=0.0,
            qkv_bias=None,
            proj_bias=None,
            logit_scale=None,
            training=True,
            use_efficient_attention=True,
        )

        diff_attn = torch.abs(old_attn - new_attn)
        print(torch.sum(diff_attn))

    if True:
        import torch
        import torchvision

        device = "cuda"
        dtype = torch.float16

        batch_size = 1
        height = 512 * 4
        width = 512 * 4
        img = torch.rand([batch_size, 3, height, width])

        model = torchvision.models.swin_t()
        img = img.to(dtype=dtype, device=device)
        model = model.to(dtype=dtype, device=device)
        ouput = model(img)

        model = torchvision.models.swin_v2_t()
        model = model.to(dtype=dtype, device=device)
        ouput = model(img)
