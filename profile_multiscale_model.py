import argparse

import torch
import yaml

from models.generator import MambaSEUNet


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        default='recipes/Mamba-SEUNet/Mamba-SEUNet-mini-3090.yaml',
    )
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--frequency-bins', type=int, default=256)
    parser.add_argument('--frames', type=int, default=256)
    args = parser.parse_args()

    with open(args.config, encoding='utf-8') as config_file:
        cfg = yaml.safe_load(config_file)

    torch.cuda.reset_peak_memory_stats()
    model = MambaSEUNet(cfg).cuda().train()
    magnitude = torch.rand(
        args.batch_size,
        args.frequency_bins,
        args.frames,
        device='cuda',
        requires_grad=True,
    )
    phase = (
        torch.rand_like(magnitude, requires_grad=False) * 2.0 - 1.0
    ) * torch.pi

    outputs = model(magnitude, phase)
    loss = sum(output.float().square().mean() for output in outputs)
    loss.backward()

    print(f'parameters={sum(parameter.numel() for parameter in model.parameters())}')
    print(f'output_shapes={[tuple(output.shape) for output in outputs]}')
    print(f'loss={loss.item():.6f}')
    print(f'peak_allocated_gib={torch.cuda.max_memory_allocated() / 1024 ** 3:.3f}')
    print(
        'local_channel_scale='
        f'{model.latest_aux["local_channel_scales"].abs().mean().item():.6f}'
    )


if __name__ == '__main__':
    main()
