import sys

sys.path.append("fvcore")
from fvcore.nn import FlopCountAnalysis
import numpy as np
import os
import statistics
from collections import defaultdict
from typing import Dict, Any
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import Tensor
from torchblocks.stems.patch import (
    patch_k4_s2,
    patch_k4_s4,
    patch_k8_s2,
    patch_k8_s4,
    patch_k8_s8,
    patch_k16_s2,
    patch_k16_s4,
    patch_k16_s8,
    patch_k16_s16,
)

from torchblocks.stems.conv import (
    conv_k3_s2,
    conv_k5_s2,
    conv_k5_s4,
    conv_k7_s2,
    conv_k7_s4,
    conv_k9_s2,
    conv_k9_s4,
    conv_k9_s8,
    conv_k11_s2,
    conv_k11_s4,
    conv_k11_s8,
    conv_k13_s2,
    conv_k13_s4,
    conv_k13_s8,
    conv_k15_s4,
    conv_k17_s4,
    conv_k19_s4,
    conv_k21_s4,
    conv_k6_s2,
)

from eval import get_latency
import torch._dynamo

torch._dynamo.config.suppress_errors = True


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
    in_c = 3
    out_c = 32
    in_h = 1024
    in_w = 1024

    device = "cuda"
    dtype = torch.float16
    memory_format = torch.channels_last

    results = defaultdict(lambda: defaultdict(list))
    stages_res = defaultdict(lambda: defaultdict(list))

    feat = torch.rand([batch_size, in_c, in_h, in_w])
    block = patch_k8_s8(in_c, out_c)

    block.eval()
    feat = feat.to(dtype=dtype, device=device)
    block = block.to(dtype=dtype, device=device)
    get_latency(block, feat)

    for build_block in [
        # patch_k4_s2,
        patch_k4_s4,
        # patch_k8_s2,
        # patch_k8_s4,
        # patch_k8_s8,
        # patch_k16_s2,
        # patch_k16_s4,
        # patch_k16_s8,
        # patch_k16_s16,
        # conv_k3_s2,
        # conv_k5_s2,
        conv_k5_s4,
        conv_k7_s2,
        conv_k7_s4,
        # conv_k9_s2,
        conv_k9_s4,
        # conv_k9_s8,
        # conv_k11_s2,
        conv_k11_s4,
        # conv_k11_s8,
        # conv_k13_s2,
        conv_k13_s4,
        # conv_k13_s8,
        conv_k15_s4,
        # conv_k17_s4,
        # conv_k19_s4,
        # conv_k21_s4,
        conv_k6_s2,
    ]:
        block_name = build_block.__name__

        block = build_block(in_c, out_c)

        block.eval()
        feat = feat.to(dtype=dtype, device=device)
        block = block.to(dtype=dtype, device=device)

        if feat.dim() >= 4:
            feat = feat.to(memory_format=memory_format)
            block = block.to(memory_format=memory_format)

        block = torch.compile(block, disable=False)

        latency = get_latency(block, feat)

        params = sum(x.numel() for x in block.parameters())
        flops_counter = FlopCountAnalysis(block, feat)
        macs = flops_counter.total()

        tops = macs / latency
        # tpps = params / latency / (batch_size * in_h * in_w)

        results[block_name]["latency"] = latency * 1.0e3
        results[block_name]["macs"] = macs
        results[block_name]["params"] = params
        results[block_name]["tops"] = tops
        results[block_name]["params_macs"] = params / macs
        # results[block_name]["tpps"] = tpps

    # print(results["conv_k11_s8"]["macs"])
    # print(results["conv_k11_s4"]["macs"])
    # print(results["conv_k11_s2"]["macs"])

    plot_models(
        results, key_x="params", key_y="tops", x_scale="linear", y_scale="linear"
    )
    plot_models(
        results, key_x="params", key_y="latency", x_scale="linear", y_scale="linear"
    )
    plot_models(
        results, key_x="params", key_y="macs", x_scale="linear", y_scale="linear"
    )
