import json
import math
import sys
import types
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.uils import UILS, diagonal_kalman_rts


BASELINE_CONFIG = ROOT / 'recipes/Mamba-SEUNet/Mamba-SEUNet-UILS-baseline-mini.yaml'
CANDIDATE_CONFIG = ROOT / 'recipes/Mamba-SEUNet/Mamba-SEUNet-UILS-candidate-mini.yaml'


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def load_config(path):
    with path.open(encoding='utf-8') as config_file:
        return yaml.safe_load(config_file)


def reference_kalman_rts(y, a, q, r, initial_covariance=1.0, gain_limit=1.0):
    y_time = y.permute(2, 0, 3, 1).contiguous()
    a_time = a.permute(2, 0, 3, 1).contiguous()
    q_time = q.permute(2, 0, 3, 1).contiguous()
    r_time = r.permute(2, 0, 3, 1).contiguous()
    state = torch.zeros_like(y_time[0])
    covariance = torch.full_like(y_time[0], initial_covariance)
    predicted_state = []
    predicted_covariance = []
    filtered_state = []
    filtered_covariance = []
    for index in range(y_time.shape[0]):
        state_prior = a_time[index] * state
        covariance_prior = a_time[index].square() * covariance + q_time[index]
        gain = covariance_prior / (covariance_prior + r_time[index])
        state = state_prior + gain * (y_time[index] - state_prior)
        covariance = (
            (1.0 - gain).square() * covariance_prior
            + gain.square() * r_time[index]
        )
        predicted_state.append(state_prior)
        predicted_covariance.append(covariance_prior)
        filtered_state.append(state)
        filtered_covariance.append(covariance)

    smoothed_state = [None] * y_time.shape[0]
    smoothed_state[-1] = filtered_state[-1]
    for index in range(y_time.shape[0] - 2, -1, -1):
        gain = (
            filtered_covariance[index]
            * a_time[index + 1]
            / predicted_covariance[index + 1]
        ).clamp(-gain_limit, gain_limit)
        smoothed_state[index] = filtered_state[index] + gain * (
            smoothed_state[index + 1] - predicted_state[index + 1]
        )

    def restore(values):
        return torch.stack(values).permute(1, 3, 0, 2).contiguous()

    return restore(filtered_state), restore(smoothed_state)


def test_identity_and_vjp():
    torch.manual_seed(5)
    module = UILS().cpu()
    require(module.mag_gate.item() == 0.0, 'Magnitude gate is not exactly zero.')
    require(module.pha_gate.item() == 0.0, 'Phase gate is not exactly zero.')
    internal_weight = sum(
        parameter.detach().abs().sum().item()
        for name, parameter in module.named_parameters()
        if name not in {'mag_gate', 'pha_gate'}
    )
    require(internal_weight > 0.0, 'All internal projections are zero.')

    mag_source = torch.randn(2, 48, 9, 7)
    pha_source = torch.randn(2, 24, 9, 7)
    mag_weight = torch.randn_like(mag_source)
    pha_weight = torch.randn_like(pha_source)

    mag_off = mag_source.clone().requires_grad_(True)
    pha_off = pha_source.clone().requires_grad_(True)
    off_outputs = module(mag_off, pha_off, enabled=False)
    off_loss = (off_outputs[0] * mag_weight).sum() + (off_outputs[1] * pha_weight).sum()
    off_vjp = torch.autograd.grad(off_loss, (mag_off, pha_off))

    mag_on = mag_source.clone().requires_grad_(True)
    pha_on = pha_source.clone().requires_grad_(True)
    on_outputs = module(mag_on, pha_on)
    on_loss = (on_outputs[0] * mag_weight).sum() + (on_outputs[1] * pha_weight).sum()
    on_vjp = torch.autograd.grad(on_loss, (mag_on, pha_on))

    output_error = max(
        (off_outputs[0] - on_outputs[0]).abs().max().item(),
        (off_outputs[1] - on_outputs[1]).abs().max().item(),
    )
    difference = torch.cat([
        (off_vjp[0] - on_vjp[0]).reshape(-1),
        (off_vjp[1] - on_vjp[1]).reshape(-1),
    ])
    reference = torch.cat([off_vjp[0].reshape(-1), off_vjp[1].reshape(-1)])
    relative_vjp_error = difference.norm().item() / max(reference.norm().item(), 1e-12)
    require(output_error <= 1e-6, f'Zero-gate output error is {output_error}.')
    require(relative_vjp_error <= 1e-5, f'Input VJP error is {relative_vjp_error}.')
    return {
        'zero_gate_output_max_abs': output_error,
        'input_vjp_relative_error': relative_vjp_error,
    }


def test_reference_system():
    generator = torch.Generator().manual_seed(17)
    shape = (2, 3, 96, 2)
    a = torch.full(shape, 0.80, dtype=torch.float64)
    q = torch.full(shape, 0.04, dtype=torch.float64)
    r = torch.full(shape, 0.15, dtype=torch.float64)
    true_states = []
    state = torch.zeros(shape[0], shape[1], shape[3], dtype=torch.float64)
    for _ in range(shape[2]):
        state = 0.80 * state + math.sqrt(0.04) * torch.randn(
            state.shape, generator=generator, dtype=torch.float64
        )
        true_states.append(state)
    truth = torch.stack(true_states, dim=2)
    observations = truth + math.sqrt(0.15) * torch.randn(
        truth.shape, generator=generator, dtype=torch.float64
    )

    smoothed, diagnostics = diagonal_kalman_rts(
        observations, a, q, r, force_fp32=True
    )
    require(smoothed.dtype == torch.float32, 'Recursive path did not force FP32.')
    reference_filtered, reference_smoothed = reference_kalman_rts(
        observations, a, q, r
    )
    filter_error = (
        diagnostics['filtered'].double() - reference_filtered
    ).abs().max().item()
    smoother_error = (smoothed.double() - reference_smoothed).abs().max().item()
    filter_mse = (diagnostics['filtered'].double() - truth).square().mean().item()
    smoother_mse = (smoothed.double() - truth).square().mean().item()
    require(filter_error <= 1e-4, f'FP64 filter reference error is {filter_error}.')
    require(smoother_error <= 1e-4, f'FP64 smoother reference error is {smoother_error}.')
    require(smoother_mse <= filter_mse, 'Smoother MSE exceeds filter MSE.')
    return {
        'filter_reference_max_abs': filter_error,
        'smoother_reference_max_abs': smoother_error,
        'filter_mse': filter_mse,
        'smoother_mse': smoother_mse,
    }


def check_diagnostic_bounds(diagnostics):
    for name in ('a', 'Q', 'R', 'P', 'K', 'G'):
        require(torch.isfinite(diagnostics[name]).all(), f'{name} is not finite.')
    require(diagnostics['a'].abs().max().item() <= 0.98 + 1e-7, 'a exceeds 0.98.')
    for name in ('Q', 'R'):
        require(diagnostics[name].min().item() >= 1e-4 - 1e-8, f'{name} below floor.')
        require(diagnostics[name].max().item() <= 10.0 + 1e-7, f'{name} above cap.')
    require(diagnostics['P'].min().item() >= 0.0, 'P is negative.')
    require(diagnostics['K'].min().item() >= 0.0, 'K is negative.')
    require(diagnostics['K'].max().item() <= 1.0 + 1e-7, 'K exceeds one.')
    require(diagnostics['G'].abs().max().item() <= 1.0 + 1e-7, 'G is unbounded.')


def test_sequences_shapes_and_gradients():
    torch.manual_seed(23)
    module = UILS().cpu()
    sequence_cases = {
        'constant': torch.full((1, 48, 320, 3), 0.25),
        'pulse': torch.zeros(1, 48, 320, 3),
        'random_long': torch.randn(1, 48, 512, 3),
    }
    phase_cases = {
        'constant': torch.full((1, 24, 320, 3), -0.1),
        'pulse': torch.zeros(1, 24, 320, 3),
        'random_long': torch.randn(1, 24, 512, 3),
    }
    sequence_cases['pulse'][:, :, 160] = 1.0
    phase_cases['pulse'][:, :, 160] = -1.0
    for name in sequence_cases:
        mag_output, pha_output, diagnostics = module(
            sequence_cases[name], phase_cases[name], return_diagnostics=True
        )
        require(mag_output.shape == sequence_cases[name].shape, 'Magnitude shape changed.')
        require(pha_output.shape == phase_cases[name].shape, 'Phase shape changed.')
        require(torch.isfinite(mag_output).all(), f'{name} magnitude is not finite.')
        require(torch.isfinite(pha_output).all(), f'{name} phase is not finite.')
        check_diagnostic_bounds(diagnostics)

    module.zero_grad(set_to_none=True)
    mag_input = torch.randn(1, 48, 13, 5)
    pha_input = torch.randn(1, 24, 13, 5)
    mag_weight = torch.randn_like(mag_input)
    pha_weight = torch.randn_like(pha_input)
    mag_output, pha_output = module(mag_input, pha_input)
    ((mag_output * mag_weight).sum() + (pha_output * pha_weight).sum()).backward()
    for gate in (module.mag_gate, module.pha_gate):
        require(gate.grad is not None, 'Zero gate has no gradient.')
        require(torch.isfinite(gate.grad), 'Zero-gate gradient is not finite.')
        require(gate.grad.abs().item() > 0.0, 'Zero-gate gradient is zero.')
    for name, parameter in module.named_parameters():
        if name in {'mag_gate', 'pha_gate'} or parameter.grad is None:
            continue
        require(torch.isfinite(parameter.grad).all(), f'{name} gradient is not finite.')
        require(parameter.grad.abs().max().item() == 0.0, f'{name} should be zero-gated.')

    with torch.no_grad():
        module.mag_gate.fill_(1e-3)
        module.pha_gate.fill_(1e-3)
    module.zero_grad(set_to_none=True)
    mag_output, pha_output = module(mag_input, pha_input)
    (mag_output.square().mean() + pha_output.square().mean()).backward()
    nonzero_internal = []
    for name, parameter in module.named_parameters():
        if name in {'mag_gate', 'pha_gate'} or parameter.grad is None:
            continue
        require(torch.isfinite(parameter.grad).all(), f'{name} gradient is not finite.')
        if parameter.grad.abs().sum().item() > 0.0:
            nonzero_internal.append(name)
    require(nonzero_internal, 'No internal gradient at gate=1e-3.')
    return {
        'sequence_cases': sorted(sequence_cases),
        'nonzero_internal_gradients': nonzero_internal,
    }


def install_inspection_stubs():
    if not hasattr(nn, 'RMSNorm'):
        nn.RMSNorm = nn.LayerNorm

    class InspectionMamba(nn.Module):
        def __init__(self, d_model, *args, **kwargs):
            super().__init__()
            self.projection = nn.Linear(d_model, d_model, bias=False)

        def forward(self, inputs, *args, **kwargs):
            return self.projection(inputs)

    class InspectionBlock(nn.Module):
        def __init__(self, d_model, mixer_cls, norm_cls=nn.LayerNorm, **kwargs):
            super().__init__()
            self.mixer = mixer_cls(d_model)
            self.norm = norm_cls(d_model)

        def forward(self, inputs, residual=None, **kwargs):
            return self.norm(self.mixer(inputs)), residual

    def add_package(name):
        module = types.ModuleType(name)
        module.__path__ = []
        sys.modules[name] = module
        return module

    root = add_package('mamba_ssm')
    add_package('mamba_ssm.modules')
    add_package('mamba_ssm.models')
    add_package('mamba_ssm.ops')
    add_package('mamba_ssm.ops.triton')
    root.Mamba = InspectionMamba

    simple = types.ModuleType('mamba_ssm.modules.mamba_simple')
    simple.Mamba = InspectionMamba
    simple.Block = InspectionBlock
    sys.modules[simple.__name__] = simple

    mixer = types.ModuleType('mamba_ssm.models.mixer_seq_simple')
    mixer._init_weights = lambda module, **kwargs: None
    sys.modules[mixer.__name__] = mixer

    layernorm = types.ModuleType('mamba_ssm.ops.triton.layernorm')
    layernorm.RMSNorm = nn.LayerNorm
    sys.modules[layernorm.__name__] = layernorm

    selective = types.ModuleType('mamba_ssm.ops.selective_scan_interface')
    selective.selective_scan_fn = lambda *args, **kwargs: args[0]
    selective.selective_scan_ref = selective.selective_scan_fn
    selective.mamba_inner_fn = selective.selective_scan_fn
    sys.modules[selective.__name__] = selective

    extension = types.ModuleType('selective_scan_cuda')
    extension.fwd = lambda *args, **kwargs: None
    extension.bwd = lambda *args, **kwargs: None
    sys.modules[extension.__name__] = extension


def test_model_rng_and_configs():
    baseline_cfg = load_config(BASELINE_CONFIG)
    candidate_cfg = load_config(CANDIDATE_CONFIG)
    normalized = deepcopy(candidate_cfg)
    normalized['model_cfg']['uils_enabled'] = False
    require(normalized == baseline_cfg, 'Mini configs differ beyond uils_enabled.')
    require(baseline_cfg['env_setting']['seed'] == 1234, 'Baseline seed changed.')
    require(baseline_cfg['training_cfg']['batch_size'] == 2, 'Baseline batch changed.')
    require('mini_' in baseline_cfg['data_cfg']['train_clean_json'], 'Train data is not mini.')
    require('mini_' in baseline_cfg['data_cfg']['valid_clean_json'], 'Validation data is not mini.')

    install_inspection_stubs()
    from models.generator import MambaSEUNet

    torch.manual_seed(1234)
    baseline = MambaSEUNet(deepcopy(baseline_cfg)).cpu()
    baseline_rng = torch.random.get_rng_state().clone()
    baseline_state = baseline.state_dict()

    torch.manual_seed(1234)
    candidate = MambaSEUNet(deepcopy(candidate_cfg)).cpu()
    candidate_rng = torch.random.get_rng_state().clone()
    candidate_state = candidate.state_dict()

    require(torch.equal(baseline_rng, candidate_rng), 'Post-construction CPU RNG differs.')
    shared_names = [name for name in candidate_state if not name.startswith('uils.')]
    require(set(shared_names) == set(baseline_state), 'Shared parameter names differ.')
    mismatches = [
        name for name in shared_names
        if not torch.equal(candidate_state[name], baseline_state[name])
    ]
    require(not mismatches, f'Shared parameters differ: {mismatches[:3]}')
    require(candidate.uils is not None, 'Candidate did not construct UILS.')
    require(baseline.uils is None, 'Baseline constructed UILS.')
    return {
        'inspection_stub_used': True,
        'shared_tensor_count': len(shared_names),
        'post_rng_equal': True,
    }


def test_parameter_budget():
    module = UILS()
    parameters = sum(parameter.numel() for parameter in module.parameters())
    require(parameters <= 15000, f'UILS adds {parameters} parameters.')
    return {'uils_parameters': parameters, 'parameter_limit': 15000}


def main():
    torch.set_num_threads(1)
    results = {
        'identity_vjp': test_identity_and_vjp(),
        'reference_system': test_reference_system(),
        'sequence_gradient_interface': test_sequences_shapes_and_gradients(),
        'rng_and_configs': test_model_rng_and_configs(),
        'parameter_budget': test_parameter_budget(),
    }
    print(json.dumps({'status': 'pass', 'device': 'cpu', 'results': results}, sort_keys=True))


if __name__ == '__main__':
    main()
