import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_config(path):
    with path.open(encoding='utf-8') as config_file:
        return yaml.safe_load(config_file)


def run_step(model, magnitude, phase):
    model.zero_grad(set_to_none=True)
    outputs = model(magnitude, phase)
    loss = sum(output.float().square().mean() for output in outputs)
    loss.backward()
    return outputs, loss


def profile_model(model, magnitude, phase, warmup, iterations):
    model = model.cuda().train()
    for _ in range(warmup):
        outputs, loss = run_step(model, magnitude, phase)
        del outputs, loss
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    last_shapes = None
    last_loss_tensor = None
    for iteration in range(iterations):
        outputs, loss = run_step(model, magnitude, phase)
        if iteration == iterations - 1:
            last_shapes = [list(output.shape) for output in outputs]
            last_loss_tensor = loss.detach()
        del outputs, loss
    end.record()
    torch.cuda.synchronize()

    result = {
        'forward_backward_ms': float(start.elapsed_time(end) / iterations),
        'max_allocated_gib': float(torch.cuda.max_memory_allocated() / 1024 ** 3),
        'max_reserved_gib': float(torch.cuda.max_memory_reserved() / 1024 ** 3),
        'output_shapes': last_shapes,
        'loss': float(last_loss_tensor.item()),
    }
    model.zero_grad(set_to_none=True)
    model.cpu()
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--baseline-config',
        type=Path,
        default=ROOT / 'recipes/Mamba-SEUNet/Mamba-SEUNet-UILS-baseline-mini.yaml',
    )
    parser.add_argument(
        '--candidate-config',
        type=Path,
        default=ROOT / 'recipes/Mamba-SEUNet/Mamba-SEUNet-UILS-candidate-mini.yaml',
    )
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--frequency-bins', type=int, default=256)
    parser.add_argument('--frames', type=int, default=256)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--iterations', type=int, default=5)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print(json.dumps({'status': 'error', 'reason': 'cuda_unavailable'}))
        return 2
    if args.warmup < 1 or args.iterations < 1:
        print(json.dumps({'status': 'error', 'reason': 'invalid_iteration_count'}))
        return 2

    baseline_cfg = load_config(args.baseline_config)
    candidate_cfg = load_config(args.candidate_config)
    normalized = deepcopy(candidate_cfg)
    normalized['model_cfg']['uils_enabled'] = False
    if normalized != baseline_cfg:
        print(json.dumps({'status': 'error', 'reason': 'config_mismatch'}))
        return 2

    try:
        from models.generator import MambaSEUNet
    except Exception as error:
        print(json.dumps({
            'status': 'error',
            'reason': 'model_import_failed',
            'error_type': type(error).__name__,
            'message': str(error),
        }, sort_keys=True))
        return 2

    seed = int(baseline_cfg['env_setting']['seed'])
    torch.manual_seed(seed)
    baseline_model = MambaSEUNet(deepcopy(baseline_cfg)).cpu()
    torch.manual_seed(seed)
    candidate_model = MambaSEUNet(deepcopy(candidate_cfg)).cpu()
    baseline_parameters = sum(parameter.numel() for parameter in baseline_model.parameters())
    candidate_parameters = sum(parameter.numel() for parameter in candidate_model.parameters())

    input_generator = torch.Generator(device='cuda').manual_seed(seed + 1)
    magnitude = torch.rand(
        args.batch_size,
        args.frequency_bins,
        args.frames,
        generator=input_generator,
        device='cuda',
    )
    phase = (
        torch.rand(
            magnitude.shape,
            generator=input_generator,
            device='cuda',
        )
        * 2.0
        - 1.0
    ) * torch.pi

    baseline = profile_model(
        baseline_model, magnitude, phase, args.warmup, args.iterations
    )
    candidate = profile_model(
        candidate_model, magnitude, phase, args.warmup, args.iterations
    )
    ratios = {
        'time': candidate['forward_backward_ms'] / baseline['forward_backward_ms'],
        'allocated': candidate['max_allocated_gib'] / baseline['max_allocated_gib'],
        'reserved': candidate['max_reserved_gib'] / baseline['max_reserved_gib'],
    }
    checks = {
        'time_ratio_le_1_15': ratios['time'] <= 1.15,
        'allocated_ratio_le_1_12': ratios['allocated'] <= 1.12,
        'reserved_ratio_le_1_12': ratios['reserved'] <= 1.12,
        'candidate_allocated_le_11_5_gib': candidate['max_allocated_gib'] <= 11.5,
        'candidate_reserved_le_12_gib': candidate['max_reserved_gib'] <= 12.0,
        'added_parameters_le_15000': candidate_parameters - baseline_parameters <= 15000,
    }
    report = {
        'status': 'pass' if all(checks.values()) else 'fail',
        'device': torch.cuda.get_device_name(torch.cuda.current_device()),
        'settings': {
            'batch_size': args.batch_size,
            'frequency_bins': args.frequency_bins,
            'frames': args.frames,
            'warmup': args.warmup,
            'iterations': args.iterations,
            'same_process_same_input': True,
        },
        'parameters': {
            'baseline': baseline_parameters,
            'candidate': candidate_parameters,
            'added': candidate_parameters - baseline_parameters,
        },
        'baseline': baseline,
        'candidate': candidate,
        'ratios': ratios,
        'checks': checks,
    }
    print(json.dumps(report, sort_keys=True))
    return 0 if report['status'] == 'pass' else 1


if __name__ == '__main__':
    sys.exit(main())
