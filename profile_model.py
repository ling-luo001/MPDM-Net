import argparse
import time

import torch
import yaml

from models.generator import MambaSEUNet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', default='recipes/Mamba-SEUNet/Mamba-SEUNet.yaml'
    )
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--frequency-bins', type=int, default=256)
    parser.add_argument('--frames', type=int, default=256)
    parser.add_argument('--forward-only', action='store_true')
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for model profiling')
    with open(args.config, encoding='utf-8') as config_file:
        cfg = yaml.safe_load(config_file)

    device = torch.device('cuda')
    model = MambaSEUNet(cfg).to(device).train()
    magnitude = torch.rand(
        args.batch_size,
        args.frequency_bins,
        args.frames,
        device=device,
    )
    phase = torch.rand_like(magnitude) * (2.0 * torch.pi) - torch.pi

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    output = model(magnitude, phase)
    if not args.forward_only:
        loss = (
            output[0].mean()
            + model.latest_aux['temporal_noise_log_ratio'].square().mean()
            + model.latest_aux['spectral_noise_log_ratio'].square().mean()
        )
        loss.backward()
    torch.cuda.synchronize()

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f'parameters={parameter_count}')
    print(f'output_shape={tuple(output[2].shape)}')
    print(f'forward_only={args.forward_only}')
    print(f'elapsed_seconds={time.time() - start:.3f}')
    print(
        'peak_allocated_gib='
        f'{torch.cuda.max_memory_allocated() / (1024 ** 3):.3f}'
    )
    print(
        'peak_reserved_gib='
        f'{torch.cuda.max_memory_reserved() / (1024 ** 3):.3f}'
    )


if __name__ == '__main__':
    main()
