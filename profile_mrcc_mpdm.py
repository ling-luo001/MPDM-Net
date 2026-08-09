"""Profile baseline-disabled and MRCC-enabled MPDM-Net generators."""

import argparse
import copy
import gc
import statistics
import time

import torch
import yaml

from models.generator import MambaSEUNet


PARAMETER_MIN = 2_250_000
PARAMETER_MAX = 2_450_000
COMPUTE_RATIO_MAX = 1.70


def parameter_count(model):
    return sum(parameter.numel() for parameter in model.parameters())


def profile_step(model, noisy_magnitude, noisy_phase, warmup, iterations):
    model.train()
    timings = []
    peak_memory = 0
    for iteration in range(warmup + iterations):
        model.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.perf_counter()
        magnitude, phase, complex_spectrum = model(noisy_magnitude, noisy_phase)
        loss = magnitude.square().mean() + phase.square().mean() + complex_spectrum.square().mean()
        loss.backward()
        torch.cuda.synchronize()
        elapsed_ms = 1000.0 * (time.perf_counter() - start)
        if iteration >= warmup:
            timings.append(elapsed_ms)
            peak_memory = max(peak_memory, torch.cuda.max_memory_allocated())
    return statistics.median(timings), peak_memory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="recipes/Mamba-SEUNet/MRCC-MPDM-v1-mini.yaml")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--frames", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=3)
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as handle:
        enabled_cfg = yaml.safe_load(handle)
    disabled_cfg = copy.deepcopy(enabled_cfg)
    disabled_cfg["model_cfg"]["mrcc_enabled"] = False

    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda") if cuda_available else None
    noisy_magnitude = None
    noisy_phase = None
    if cuda_available:
        torch.manual_seed(1234)
        noisy_magnitude = torch.rand(args.batch_size, 256, args.frames, device=device) + 0.05
        noisy_phase = (
            torch.rand(args.batch_size, 256, args.frames, device=device) * (2.0 * torch.pi)
            - torch.pi
        )

    baseline = MambaSEUNet(disabled_cfg)
    baseline_parameters = parameter_count(baseline)
    baseline_ms = None
    baseline_memory = None
    if cuda_available:
        baseline = baseline.to(device)
        baseline_ms, baseline_memory = profile_step(
            baseline, noisy_magnitude, noisy_phase, args.warmup, args.iterations
        )
    del baseline
    gc.collect()
    if cuda_available:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    enabled = MambaSEUNet(enabled_cfg)
    enabled_parameters = parameter_count(enabled)
    print(f"baseline_disabled_parameters={baseline_parameters}")
    print(f"mrcc_enabled_parameters={enabled_parameters}")
    print(f"mrcc_added_parameters={enabled_parameters - baseline_parameters}")
    if not PARAMETER_MIN <= enabled_parameters <= PARAMETER_MAX:
        raise SystemExit(
            f"MRCC generator parameter gate failed: {enabled_parameters} not in "
            f"[{PARAMETER_MIN}, {PARAMETER_MAX}]"
        )

    if not cuda_available:
        print("CUDA unavailable: runtime, memory, and compute-ratio profiling skipped")
        return

    enabled = enabled.to(device)
    enabled_ms, enabled_memory = profile_step(
        enabled, noisy_magnitude, noisy_phase, args.warmup, args.iterations
    )
    compute_ratio = enabled_ms / baseline_ms
    print(f"baseline_forward_backward_median_ms={baseline_ms:.3f}")
    print(f"mrcc_forward_backward_median_ms={enabled_ms:.3f}")
    print(f"baseline_peak_allocated_mib={baseline_memory / 2**20:.2f}")
    print(f"mrcc_peak_allocated_mib={enabled_memory / 2**20:.2f}")
    print(f"measured_compute_ratio={compute_ratio:.4f}")
    if compute_ratio > COMPUTE_RATIO_MAX:
        raise SystemExit(
            f"MRCC compute-ratio gate failed: {compute_ratio:.4f} > {COMPUTE_RATIO_MAX:.2f}"
        )


if __name__ == "__main__":
    main()
