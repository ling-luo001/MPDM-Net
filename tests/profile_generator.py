"""Profile one MPDM-Net generator forward/backward pass on CUDA."""

import argparse
import json
import time

import torch
import yaml

from models.generator import MambaSEUNet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="recipes/Mamba-SEUNet/Mamba-SEUNet.yaml"
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--time-frames", type=int, default=256)
    parser.add_argument("--frequency-bins", type=int, default=256)
    parser.add_argument("--forward-only", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this profiling script")

    with open(args.config, "r", encoding="utf-8") as config_file:
        cfg = yaml.safe_load(config_file)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = MambaSEUNet(cfg).cuda()
    model.eval() if args.forward_only else model.train()

    shape = (args.batch_size, args.frequency_bins, args.time_frames)
    noisy_magnitude = torch.rand(shape, device="cuda")
    noisy_phase = (torch.rand(shape, device="cuda") * 2.0 - 1.0) * torch.pi

    torch.cuda.synchronize()
    start_time = time.time()
    if args.forward_only:
        with torch.no_grad():
            outputs = model(noisy_magnitude, noisy_phase)
    else:
        outputs = model(noisy_magnitude, noisy_phase)
        loss = sum(output.float().square().mean() for output in outputs)
        loss.backward()
    torch.cuda.synchronize()

    result = {
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "input_shape": shape,
        "output_shapes": [tuple(output.shape) for output in outputs],
        "outputs_finite": [bool(torch.isfinite(output).all()) for output in outputs],
        "elapsed_seconds": round(time.time() - start_time, 3),
        "peak_allocated_gib": round(
            torch.cuda.max_memory_allocated() / 1024 ** 3, 3
        ),
        "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 1024 ** 3, 3),
        "wavelet_scales": model.wavelet_diagnostics(),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
