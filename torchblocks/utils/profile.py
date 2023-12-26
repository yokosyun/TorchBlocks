from torch import nn, Tensor
import torch
import statistics


def get_latency(
    model: nn.Module,
    input: Tensor,
    use_efficent_attention,
    iterations: int = 20,
) -> float:
    latencies = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        output = model(input, use_efficent_attention)
        end.record()
        torch.cuda.synchronize()
        latencies.append(start.elapsed_time(end))
    return statistics.median(latencies)  # / 1.0e3
