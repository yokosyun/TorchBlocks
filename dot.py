import torch
from torch.nn import functional as F
import statistics
import math


# Optionally use the context manager to ensure one of the fused kernels is run
# Efficient implementation equivalent to the following:
def scaled_dot_product_attention(
    query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None
) -> torch.Tensor:
    # Efficient implementation equivalent to the following:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_mask.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask
    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    # print(attn_weight.shape)
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value


max_img = 256
min_c = 64
num_stages = 5

for idx in range(5):
    scale = 2**idx
    print(scale, max_img // scale, min_c * scale)

    img_size = max_img // scale
    W = 16
    C = min_c * scale
    batch = 1
    if False:
        C_PER_HEAD = 128
        heads = C // C_PER_HEAD
    else:
        heads = 8
        C_PER_HEAD = C // heads

    WW = W**2
    nW = (img_size // W) ** 2

    q = torch.rand(batch, heads, nW, WW, C_PER_HEAD, dtype=torch.float16, device="cuda")
    k = torch.rand(batch, heads, nW, WW, C_PER_HEAD, dtype=torch.float16, device="cuda")
    v = torch.rand(batch, heads, nW, WW, C_PER_HEAD, dtype=torch.float16, device="cuda")

    print(q.shape)
    # q = q.reshape(-1, q.size(-2), q.size(-1))
    # k = k.reshape(-1, k.size(-2), k.size(-1))
    # v = v.reshape(-1, v.size(-2), v.size(-1))
    # q = q.view(-1, q.size(-2), q.size(-1))
    # k = k.view(-1, k.size(-2), k.size(-1))
    # v = v.view(-1, v.size(-2), v.size(-1))
    print(q.shape)

    iterations = 30
    latencies = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        # with torch.backends.cuda.sdp_kernel(
        #     enable_flash=True, enable_math=True, enable_mem_efficient=True
        # ):
        out = scaled_dot_product_attention(q, k, v)
        # print(out.shape)
        end.record()
        torch.cuda.synchronize()
        latencies.append(start.elapsed_time(end))
    latency = statistics.median(latencies)  # / 1.0e3
    print(latency, "local")

    iterations = 30
    latencies = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        m = F.scaled_dot_product_attention
        # m = torch.compile(m)
        m(q, k, v)
        end.record()
        torch.cuda.synchronize()
        latencies.append(start.elapsed_time(end))
    latency = statistics.median(latencies)  # / 1.0e3
    print(latency, "F")

    # iterations = 30
    # latencies = []
    # for _ in range(iterations):
    #     start = torch.cuda.Event(enable_timing=True)
    #     end = torch.cuda.Event(enable_timing=True)
    #     start.record()
    #     with torch.backends.cuda.sdp_kernel(
    #         enable_flash=False,
    #         enable_math=False,
    #         enable_mem_efficient=True,
    #     ):
    #         F.scaled_dot_product_attention(q, k, v)
    #     end.record()
    #     torch.cuda.synchronize()
    #     latencies.append(start.elapsed_time(end))
    # latency = statistics.median(latencies)  # / 1.0e3
    # print(latency, False, False, True)

    iterations = 30
    latencies = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.backends.cuda.sdp_kernel(
            enable_flash=True, enable_math=True, enable_mem_efficient=True
        ):
            F.scaled_dot_product_attention(q, k, v)
        end.record()
        torch.cuda.synchronize()
        latencies.append(start.elapsed_time(end))
    latency = statistics.median(latencies)  # / 1.0e3
    print(latency, True, True, True)

    iterations = 30
    latencies = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.backends.cuda.sdp_kernel(
            enable_flash=False, enable_math=True, enable_mem_efficient=True
        ):
            F.scaled_dot_product_attention(q, k, v)
        end.record()
        torch.cuda.synchronize()
        latencies.append(start.elapsed_time(end))
    latency = statistics.median(latencies)  # / 1.0e3
    print(latency, False, True, True)

    iterations = 30
    latencies = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.backends.cuda.sdp_kernel(
            enable_flash=False, enable_math=True, enable_mem_efficient=False
        ):
            F.scaled_dot_product_attention(q, k, v)
        end.record()
        torch.cuda.synchronize()
        latencies.append(start.elapsed_time(end))
    latency = statistics.median(latencies)  # / 1.0e3
    print(latency, False, True, False)

    iterations = 30
    latencies = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.backends.cuda.sdp_kernel(
            enable_flash=True, enable_math=True, enable_mem_efficient=False
        ):
            F.scaled_dot_product_attention(q, k, v)
        end.record()
        torch.cuda.synchronize()
        latencies.append(start.elapsed_time(end))
    latency = statistics.median(latencies)  # / 1.0e3
    print(latency, True, True, False)

    iterations = 30
    latencies = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        # with torch.backends.cuda.sdp_kernel(
        #     enable_flash=True, enable_math=True, enable_mem_efficient=True
        # ):
        scaled_dot_product_attention(q, k, v)
        end.record()
        torch.cuda.synchronize()
        latencies.append(start.elapsed_time(end))
    latency = statistics.median(latencies)  # / 1.0e3
    print(latency, "local")
