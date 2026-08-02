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
    parser.add_argument('--max-stabilization-parameters', type=int, default=128)
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

    stabilization_parameters = sum(
        refiner.dense_residual_scales.numel() + refiner.branch_logits.numel()
        for refiner in model.local_channel_refiners.values()
    )
    if stabilization_parameters > args.max_stabilization_parameters:
        raise RuntimeError(
            'TF-LCA stabilization parameter budget exceeded: '
            f'{stabilization_parameters} > {args.max_stabilization_parameters}'
        )
    aux = model.latest_aux
    expected_shapes = {
        'local_channel_scales': (12,),
        'local_channel_dense_scales': (12, 3),
        'local_channel_branch_weights': (12, 3),
        'local_channel_channel_gain': (12,),
        'local_channel_update_ratio': (12,),
    }
    for key, expected_shape in expected_shapes.items():
        if aux[key].shape != expected_shape:
            raise RuntimeError(
                f'{key} has shape {tuple(aux[key].shape)}, expected {expected_shape}.'
            )
        if not torch.isfinite(aux[key]).all():
            raise RuntimeError(f'{key} contains NaN/Inf.')

    print(f'parameters={sum(parameter.numel() for parameter in model.parameters())}')
    print(f'stabilization_parameters={stabilization_parameters}')
    print(f'output_shapes={[tuple(output.shape) for output in outputs]}')
    print(f'loss={loss.item():.6f}')
    print(f'peak_allocated_gib={torch.cuda.max_memory_allocated() / 1024 ** 3:.3f}')
    print(
        'local_channel_scale_suppression/restoration='
        f'{aux["local_channel_suppression_scale_mean"].item():.6f}/'
        f'{aux["local_channel_restoration_scale_mean"].item():.6f}'
    )
    print(
        'local_channel_update_ratio_suppression/restoration='
        f'{aux["local_channel_suppression_update_ratio_mean"].item():.6f}/'
        f'{aux["local_channel_restoration_update_ratio_mean"].item():.6f}'
    )
    print(
        'local_channel_dense_scale='
        f'{aux["local_channel_dense_scales"].abs().mean().item():.6f}'
    )
    print(
        'local_channel_branch_weights='
        f'{aux["local_channel_branch_weights"].mean(dim=0).tolist()}'
    )
    print(
        'local_channel_channel_gain='
        f'{aux["local_channel_channel_gain"].mean().item():.6f}'
    )


if __name__ == '__main__':
    main()
