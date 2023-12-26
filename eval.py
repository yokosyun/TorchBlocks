import sys

sys.path.append("fvcore")
from fvcore.nn import FlopCountAnalysis
from fvcore.nn import ActivationCountAnalysis
import numpy as np
import os
import statistics
from collections import defaultdict
from typing import Dict, Any
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import Tensor
from torchblocks.blocks.resnet import (
    basic_k3_g1,
    basic_k3_g2,
    basic_k3_g4,
    basic_k3_g8,
    basic_k5,
    basic_k7,
    bottleneck,
)
from torchblocks.blocks.efficientnet import (
    mbconv_k3_e2,
    mbconv_k5_e2,
    mbconv_k7_e2,
    mbconv_k13_e2,
    mbconv_k23_e2,
    mbconv_k33_e2,
    mbconv_k3_e4,
    mbconv_k5_e4,
    mbconv_k7_e4,
    mbconv_k13_e4,
    mbconv_k23_e4,
    mbconv_k33_e4,
    fused_mbconv_g1_e1,
    fused_mbconv_g1_e2,
    fused_mbconv_g1_e3,
    fused_mbconv_g1_e4,
    fused_mbconv_g2_e1,
    fused_mbconv_g2_e2,
    fused_mbconv_g2_e4,
    fused_mbconv_g4_e1,
    fused_mbconv_g4_e2,
    fused_mbconv_g4_e4,
    fused_mbconv_g8_e1,
    fused_mbconv_g8_e2,
    fused_mbconv_g8_e4,
)
from torchblocks.blocks.regnet import (
    x_block,
    x_block_gw16,
    x_block_gw32,
    x_block_gw64,
    x_block_gw128,
)

from torchblocks.blocks.transformer import (
    transformer_encoder_layer_1,
    transformer_encoder_layer_2,
    transformer_encoder_layer_4,
    transformer_encoder_layer_8,
    transformer_encoder_layer_16,
    transformer_encoder_layer_32,
)
from torchblocks.blocks.yolo import (
    elan_e100,
    elan_e875,
    elan_e75,
    elan_e625,
    elan_e50,
    elan_e375,
    elan_e25,
    eelan,
    cspsspfbottleneck,
)

from torchblocks.blocks.ultra import (
    c2f,
    hgb_block,
    c3,
    ghost_bottleneck,
    c3ghost,
    rep_c3,
)
from torchblocks.blocks.elan_swin import (
    elanswin_w4_h4,
    elanswin_w4_h8,
    elanswin_w8_h4,
    elanswin_w8_h8,
)

from torchblocks.blocks.maxvit import maxvit
from torchblocks.blocks.swin import (
    swintransformerblockv2_w4_h4,
    swintransformerblockv2_w4_h8,
    swintransformerblockv2_w4_h16,
    swintransformerblockv2_w4_h32,
    swintransformerblockv2_w8_h4,
    swintransformerblockv2_w8_h8,
    swintransformerblockv2_w8_h16,
    swintransformerblockv2_w8_h32,
    swintransformerblockv2_w16_h4,
    swintransformerblockv2_w16_h8,
    swintransformerblockv2_w16_h16,
    swintransformerblockv2_w16_h32,
    swintransformerblockv2_w32_h4,
    swintransformerblockv2_w32_h8,
    swintransformerblockv2_w32_h16,
    swintransformerblockv2_w32_h32,
    swintransformerblockv2_w64_h8,
)
from torchblocks.blocks.convnext import cnb
import torch._dynamo

torch._dynamo.config.suppress_errors = True


def get_latency(
    model: nn.Module,
    input: Tensor,
    iterations: int = 20,
) -> float:
    latencies = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        output = model(input)
        end.record()
        torch.cuda.synchronize()
        latencies.append(start.elapsed_time(end))
    return statistics.median(latencies)  # / 1.0e3


def plot_models(
    models_info: Dict[str, Any],
    key_x: str,
    key_y: str,
    x_scale="linear",
    y_scale="linear",
    save_dir: str = "results",
) -> None:
    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots()

    for model_name, model_info in models_info.items():
        plt.plot(
            model_info[key_x],
            model_info[key_y],
            marker="o",
            label=model_name,
        )
    plt.ylabel(key_y)
    plt.xlabel(key_x)
    plt.xscale(x_scale)
    plt.yscale(y_scale)
    ax.legend()
    fig.savefig(save_dir + "/" + key_y + "_" + key_x + ".jpg")


if __name__ == "__main__":
    batch_size = 1
    in_c = 32 // 2
    in_h = 1024
    in_w = 1024
    num_stages = 7

    # torch.backends.cudnn.deterministic = True

    # stages = [
    #     [80, 192, 672],
    #     [160, 96, 336],
    #     [320, 48, 168],
    #     [640, 24, 84],
    #     [960, 12, 42],
    # ]
    # stages = [
    #     [64, 640, 640],
    #     [128, 320, 320],
    #     [256, 160, 160],
    #     [512, 80, 80],
    #     [1024, 40, 40],
    # ]
    stages = [
        [16, 512, 512],
        [64, 256, 256],
        [256, 128, 128],
        [512, 64, 64],
        [1024, 32, 32],
        [2048, 16, 16],
        # [64, 512, 512],
        # [128, 256, 256],
        # [256, 128, 128],
        # [512, 64, 64],
        # [1024, 32, 32],
        # [2048, 16, 16],
    ]
    device = "cuda"
    dtype = torch.float16
    memory_format = torch.channels_last
    # memory_format = torch.contiguous_format

    results = defaultdict(lambda: defaultdict(list))
    stages_res = defaultdict(lambda: defaultdict(list))

    feat = torch.rand([batch_size, in_c, in_h, in_w])
    block = basic_k3_g1(in_c, in_h, in_w)

    block.eval()
    feat = feat.to(dtype=dtype, device=device)
    block = block.to(dtype=dtype, device=device)
    get_latency(block, feat)

    # for stage in range(num_stages):
    #     growth = 2**stage
    #     channel = in_c * growth
    #     height = in_h // growth
    #     width = in_w // growth
    # feat = torch.rand([batch_size, channel, height, width])
    for stage, (channel, height, width) in enumerate(stages):
        feat = torch.rand([batch_size, channel, height, width])
        stages_res[stage]["shape"] = feat.shape

        for build_block in [
            # mbconv_k3_e2,
            # mbconv_k5_e2,
            # mbconv_k7_e2,
            # mbconv_k13_e2,
            # mbconv_k23_e2,
            # mbconv_k33_e2,
            # mbconv_k3_e4,
            # mbconv_k5_e4,
            # mbconv_k7_e4,
            # mbconv_k13_e4,
            # mbconv_k23_e4,
            # mbconv_k33_e4,
            # fused_mbconv_g1_e1,
            # fused_mbconv_g1_e2,
            # fused_mbconv_g1_e3,
            fused_mbconv_g1_e4,
            # fused_mbconv_g2_e1,
            # fused_mbconv_g2_e2,
            # fused_mbconv_g2_e4,
            # fused_mbconv_g4_e1,
            # fused_mbconv_g4_e2,
            # fused_mbconv_g4_e4,
            # fused_mbconv_g8_e1,
            # fused_mbconv_g8_e2,
            # fused_mbconv_g8_e4,
            # x_block,
            # x_block_gw16,
            # x_block_gw32,
            # x_block_gw64,
            # x_block_gw128,
            # cnb,
            basic_k3_g1,
            # basic_k3_g2,
            # basic_k3_g4,
            # basic_k3_g8,
            # basic_k5,
            # basic_k7,
            # bottleneck,
            # elan_e100,
            # elan_e875,
            # elan_e75,
            # elan_e625,
            # elan_e50,
            # elan_e375,
            # eelan,
            elan_e50,
            # elanswin_w4_h4,
            # elanswin_w4_h8,
            # elanswin_w8_h4,
            # elanswin_w8_h8,
            # cspsspfbottleneck,
            # c2f,
            # hgb_block,
            # c3,
            # ghost_bottleneck,
            # c3ghost,
            # rep_c3,
            # maxvit,
            # swintransformerblockv2_w4_h4,
            # swintransformerblockv2_w4_h8,
            # swintransformerblockv2_w4_h16,
            # swintransformerblockv2_w4_h32,
            # swintransformerblockv2_w8_h4,
            swintransformerblockv2_w8_h8,
            # swintransformerblockv2_w8_h16,
            # swintransformerblockv2_w8_h32,
            # swintransformerblockv2_w16_h4,
            # swintransformerblockv2_w16_h8,
            # swintransformerblockv2_w16_h16,
            # swintransformerblockv2_w16_h32,
            # swintransformerblockv2_w32_h4,
            # swintransformerblockv2_w32_h8,
            # swintransformerblockv2_w32_h16,
            # swintransformerblockv2_w32_h32,
            # swintransformerblockv2_w64_h8,
            # transformer_encoder_layer_1,
            # transformer_encoder_layer_2,
            # transformer_encoder_layer_4,
            # transformer_encoder_layer_8,
            # transformer_encoder_layer_16,
            transformer_encoder_layer_32,
        ]:
            block_name = build_block.__name__
            if block_name.startswith("transformer_encoder_layer"):
                # if height * width > 128 * 128:
                if height * width > 64 * 64:
                    continue
                feat = torch.rand([batch_size, height * width, channel])
            elif block_name.startswith(
                "swintransformerblockv2_w32"
            ) or block_name.startswith("swintransformerblockv2_w16"):
                if height * width > 128 * 128:
                    continue
                feat = torch.rand([batch_size, height, width, channel])
            elif block_name.startswith("swintransformerblockv2"):
                # if height * width > 128 * 128:
                if height * width > 256 * 256:
                    continue
                feat = torch.rand([batch_size, height, width, channel])
            elif block_name.startswith("elanswin"):
                if height * width > 256 * 256:
                    continue
            elif block_name.startswith("maxvit"):
                if height * width > 256 * 256 or channel < 16:
                    continue
            elif block_name.startswith("mbconv"):
                if channel < 8:
                    continue
            elif block_name.startswith("x_block_gw16"):
                if channel < 16:
                    continue
            elif block_name.startswith("x_block_gw32"):
                if channel < 32:
                    continue
            elif block_name.startswith("x_block_gw64"):
                if channel < 64:
                    continue
            elif block_name.startswith("x_block_gw128"):
                if channel < 128:
                    continue

            block = build_block(channel, height, width)

            block.eval()
            feat = feat.to(dtype=dtype, device=device)
            block = block.to(dtype=dtype, device=device)

            print(feat.shape)

            if (
                feat.dim() >= 4
                and block_name.startswith("swintransformerblockv2") is False
                and block_name.startswith("elanswin") is False
            ):
                feat = feat.to(memory_format=memory_format)
                block = block.to(memory_format=memory_format)
                block = torch.compile(block, disable=True)

            latency = get_latency(block, feat)

            params = sum(x.numel() for x in block.parameters())
            flops_counter = FlopCountAnalysis(block, feat)
            macs = flops_counter.total()

            activation_counter = ActivationCountAnalysis(block, feat)
            activations = activation_counter.total()

            memory_acc = activations + params
            tops = macs / latency
            tpps = params / latency / (batch_size * height * width)

            results[block_name]["latency"].append(latency)
            results[block_name]["macs"].append(macs)
            results[block_name]["params"].append(params)
            results[block_name]["activations"].append(activations)
            results[block_name]["memory_acc"].append(memory_acc)
            results[block_name]["tops"].append(tops)
            results[block_name]["tpps"].append(tpps)
            results[block_name]["stage"].append(stage)

            stages_res[stage]["block_name"].append(block_name)
            stages_res[stage]["tops"].append(tops)
            stages_res[stage]["tpps"].append(tpps)
            stages_res[stage]["latency"].append(latency)
            stages_res[stage]["macs"].append(macs)
            stages_res[stage]["params"].append(params)

    plot_models(results, key_x="params", key_y="tops", x_scale="log", y_scale="linear")
    plot_models(
        results, key_x="stage", key_y="tops", x_scale="linear", y_scale="linear"
    )
    plot_models(
        results, key_x="stage", key_y="params", x_scale="linear", y_scale="linear"
    )
    plot_models(
        results, key_x="stage", key_y="latency", x_scale="linear", y_scale="linear"
    )
    plot_models(
        results, key_x="stage", key_y="macs", x_scale="linear", y_scale="linear"
    )
    plot_models(
        results, key_x="stage", key_y="activations", x_scale="linear", y_scale="linear"
    )

    plot_models(
        results, key_x="stage", key_y="memory_acc", x_scale="linear", y_scale="linear"
    )

    top_k = 8

    print("---tops----")

    for stage_idx, stage_res in stages_res.items():
        (
            stage_res["tops"],
            stage_res["block_name"],
            stage_res["macs"],
            stage_res["latency"],
            stage_res["params"],
        ) = zip(
            *sorted(
                zip(
                    stage_res["tops"],
                    stage_res["block_name"],
                    stage_res["macs"],
                    stage_res["latency"],
                    stage_res["params"],
                ),
                reverse=True,
            )
        )

        for key in stage_res.keys():
            stage_res[key] = stage_res[key][:top_k]
        tops_score = np.array(stage_res["tops"]) / max(stage_res["tops"])
        tpps_score = np.array(stage_res["tpps"]) / max(stage_res["tpps"])

        alpha = 1

        stage_res["score"] = alpha * tops_score + (1 - alpha) * tpps_score

    print("---score----")
    for stage_idx, stage_res in stages_res.items():
        (
            stage_res["score"],
            stage_res["latency"],
            stage_res["block_name"],
            stage_res["macs"],
            stage_res["tops"],
            stage_res["params"],
        ) = zip(
            *sorted(
                zip(
                    stage_res["score"],
                    stage_res["latency"],
                    stage_res["block_name"],
                    stage_res["macs"],
                    stage_res["tops"],
                    stage_res["params"],
                ),
                reverse=True,
            )
        )

        block_names = stage_res["block_name"]
        feat_shape = stage_res["shape"]
        score = stage_res["score"]

        print(f"{stage_idx}: {feat_shape} => {block_names} {score}")
