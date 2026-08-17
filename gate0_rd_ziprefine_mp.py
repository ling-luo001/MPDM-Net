"""Local CPU Gate 0 for RD-ZipRefine-MP.

This script never trains and never reads a checkpoint.  When the local CUDA
selective-scan extension is unavailable it installs an explicitly labeled
structural forward stub. Repository module definitions and parameter tensors are
still constructed for a structural count, which can differ from the native CUDA
aggregate; the stub result is not CUDA/runtime validation.
"""

import copy
import pathlib
import sys
import types


ROOT = pathlib.Path(__file__).resolve().parent
BASE_RECIPE = ROOT / 'recipes' / 'Mamba-SEUNet' / 'Mamba-SEUNet.yaml'
CANDIDATE_RECIPE = ROOT / 'recipes' / 'RD-ZipRefine-MP' / 'RD-ZipRefine-MP.yaml'
STRUCTURAL_STUB = False


def _install_import_stubs():
    """Install dependency stubs only for unavailable local runtime packages."""
    global STRUCTURAL_STUB
    import importlib.util
    import torch
    import torch.nn as nn

    if importlib.util.find_spec('selective_scan_cuda') is None:
        sys.modules['selective_scan_cuda'] = types.ModuleType('selective_scan_cuda')
        STRUCTURAL_STUB = True

    if importlib.util.find_spec('triton') is None:
        layernorm = types.ModuleType('mamba_ssm.ops.triton.layernorm')

        class StructuralRMSNorm(nn.Module):
            def __init__(self, hidden_size, eps=1e-5, device=None, dtype=None):
                super().__init__()
                self.eps = eps
                self.weight = nn.Parameter(
                    torch.ones(hidden_size, device=device, dtype=dtype)
                )

            def forward(self, value):
                normalized = value * torch.rsqrt(
                    value.float().square().mean(dim=-1, keepdim=True) + self.eps
                ).to(value.dtype)
                return normalized * self.weight

        layernorm.RMSNorm = StructuralRMSNorm
        layernorm.layer_norm_fn = None
        layernorm.rms_norm_fn = None
        if not hasattr(nn, 'RMSNorm'):
            nn.RMSNorm = StructuralRMSNorm
        sys.modules['mamba_ssm.ops.triton.layernorm'] = layernorm
        STRUCTURAL_STUB = True

    if importlib.util.find_spec('transformers') is None:
        transformers = types.ModuleType('transformers')
        generation = types.ModuleType('transformers.generation')
        configuration_utils = types.ModuleType('transformers.configuration_utils')
        file_utils = types.ModuleType('transformers.file_utils')
        transformer_utils = types.ModuleType('transformers.utils')
        transformer_hub = types.ModuleType('transformers.utils.hub')
        configuration_utils.PretrainedConfig = type('PretrainedConfig', (), {})
        file_utils.ModelOutput = type('ModelOutput', (dict,), {})
        generation.GreedySearchDecoderOnlyOutput = type(
            'GreedySearchDecoderOnlyOutput', (), {}
        )
        generation.SampleDecoderOnlyOutput = type(
            'SampleDecoderOnlyOutput', (), {}
        )
        generation.TextStreamer = type('TextStreamer', (), {})
        transformer_utils.WEIGHTS_NAME = 'pytorch_model.bin'
        transformer_utils.CONFIG_NAME = 'config.json'
        transformer_hub.cached_file = lambda *args, **kwargs: None
        transformers.configuration_utils = configuration_utils
        transformers.file_utils = file_utils
        sys.modules['transformers'] = transformers
        sys.modules['transformers.generation'] = generation
        sys.modules['transformers.configuration_utils'] = configuration_utils
        sys.modules['transformers.file_utils'] = file_utils
        sys.modules['transformers.utils'] = transformer_utils
        sys.modules['transformers.utils.hub'] = transformer_hub

    # The installed torchvision build is not usable with this Windows PyTorch
    # environment. This replacement uses ordinary convolution only for
    # structural execution; aggregate counts remain labeled structural below.
    try:
        from torchvision.ops.deform_conv import DeformConv2d  # noqa: F401
    except Exception:
        torchvision = types.ModuleType('torchvision')
        ops = types.ModuleType('torchvision.ops')
        deform_conv = types.ModuleType('torchvision.ops.deform_conv')

        class StructuralDeformConv2d(nn.Conv2d):
            def forward(self, input_tensor, offset, mask=None):
                return super().forward(input_tensor)

        deform_conv.DeformConv2d = StructuralDeformConv2d
        torchvision.ops = ops
        ops.deform_conv = deform_conv
        sys.modules['torchvision'] = torchvision
        sys.modules['torchvision.ops'] = ops
        sys.modules['torchvision.ops.deform_conv'] = deform_conv
        STRUCTURAL_STUB = True

    try:
        from timm.layers import Mlp  # noqa: F401
    except Exception:
        timm = types.ModuleType('timm')
        layers = types.ModuleType('timm.layers')

        class StructuralMlp(nn.Module):
            def __init__(self, in_features, hidden_features=None, out_features=None, **kwargs):
                super().__init__()
                hidden_features = hidden_features or in_features
                out_features = out_features or in_features
                self.fc1 = nn.Linear(in_features, hidden_features)
                self.act = nn.GELU()
                self.fc2 = nn.Linear(hidden_features, out_features)

            def forward(self, value):
                return self.fc2(self.act(self.fc1(value)))

        layers.Mlp = StructuralMlp
        timm.layers = layers
        sys.modules['timm'] = timm
        sys.modules['timm.layers'] = layers
        STRUCTURAL_STUB = True


def _install_mamba_forward_stub():
    """Bypass selective scan while retaining repository parameter connectivity."""
    if not STRUCTURAL_STUB:
        return
    import torch
    from models.mamba_block import MambaBlock

    def structural_forward(module, value):
        parameter_signal = value.new_zeros(())
        for parameter in module.parameters():
            parameter_signal = parameter_signal + parameter.to(value.dtype).mean()
        factor = 1.0 + 1e-5 * torch.tanh(parameter_signal)
        modeled = value * factor
        return torch.cat((modeled, modeled), dim=-1)

    MambaBlock.forward = structural_forward


def _load_yaml(path):
    import yaml
    with path.open('r', encoding='utf-8') as stream:
        return yaml.safe_load(stream)


def _diff_paths(left, right, prefix=''):
    if isinstance(left, dict) and isinstance(right, dict):
        paths = set()
        for key in left.keys() | right.keys():
            path = f'{prefix}.{key}' if prefix else str(key)
            if key not in left or key not in right:
                paths.add(path)
            else:
                paths.update(_diff_paths(left[key], right[key], path))
        return paths
    return set() if left == right else {prefix}


def _assert_config_control():
    baseline = _load_yaml(BASE_RECIPE)
    candidate = _load_yaml(CANDIDATE_RECIPE)
    differences = _diff_paths(baseline, candidate)
    allowed = {
        'experiment_cfg',
        'env_setting.dist_cfg.dist_url',
        'model_cfg.zip_refine_mp_enabled',
        'model_cfg.zip_refine_mp_activation_checkpointing',
        'model_cfg.zip_refine_mp_channels',
        'model_cfg.zip_refine_mp_eps',
        'model_cfg.zip_refine_mp_delta_limit',
    }
    assert differences == allowed, (differences, allowed)
    assert candidate['experiment_cfg']['name'] == 'rd_ziprefine_mp_mini_v1'
    assert candidate['experiment_cfg']['log_name'] == 'rd_ziprefine_mp_mini_v1'
    assert candidate['model_cfg']['zip_refine_mp_activation_checkpointing'] is True
    assert candidate['model_cfg']['zip_refine_mp_delta_limit'] == 1.0

    forbidden = ('load_state_dict', 'torch.load', 'resume_from', 'pretrained', '.pth')
    for relative in (
        'models/zip_refine_mp.py',
        'models/generator.py',
        'recipes/RD-ZipRefine-MP/RD-ZipRefine-MP.yaml',
    ):
        source = (ROOT / relative).read_text(encoding='utf-8').lower()
        assert not any(token.lower() in source for token in forbidden), relative
    print('PASS config control and no weight-loading/reuse code')


def _actual_cfg():
    return _load_yaml(CANDIDATE_RECIPE)


def _assert_refiner_contract():
    import torch
    from models.mamba_block import FMambaBlock, TMambaBlock
    from models.zip_refine_mp import ZipRefineMP

    torch.manual_seed(101)
    refiner = ZipRefineMP(_actual_cfg())
    assert refiner.compression_ratios == (1, 2, 2, 1)
    assert refiner.activation_checkpointing is True
    assert refiner.delta_limit == 1.0
    for mag_stage, phase_stage in zip(refiner.mag_stages, refiner.phase_stages):
        assert isinstance(mag_stage.axis_blocks[0], FMambaBlock)
        assert isinstance(mag_stage.axis_blocks[1], TMambaBlock)
        assert isinstance(phase_stage.axis_blocks[0], TMambaBlock)
        assert isinstance(phase_stage.axis_blocks[1], FMambaBlock)
    for time_bins, freq_bins in ((5, 7), (6, 8)):
        noisy = torch.randn(1, 2, time_bins, freq_bins)
        base = torch.randn(1, 2, time_bins, freq_bins)
        maps = refiner.build_eight_map_input(noisy, base)
        assert maps.shape == (1, 8, time_bins, freq_bins)
        expected_maps = torch.cat((
            noisy,
            base,
            noisy - base,
            torch.log1p(torch.linalg.vector_norm(noisy, dim=1, keepdim=True)),
            torch.log1p(torch.linalg.vector_norm(base, dim=1, keepdim=True)),
        ), dim=1)
        assert torch.equal(maps, expected_maps)
        output, aux = refiner(noisy, base)
        assert output.shape == base.shape
        assert torch.isfinite(output).all()
        assert aux['delta_log_mag'].abs().max() <= 1.0
        assert aux['stage_scales'].shape == (2, 4)
        assert aux['interaction_scales'].shape == (4, 2)
    with torch.no_grad():
        refiner.rotation_head.weight.zero_()
        refiner.rotation_head.bias.zero_()
        refiner.delta_log_mag_head.weight.zero_()
        refiner.delta_log_mag_head.bias.fill_(-20.0)
        refiner.outer_mag_gate.fill_(3.0)
        refiner.outer_phase_gate.fill_(3.0)
    low_energy_base = torch.zeros(1, 2, 5, 7)
    low_energy_base[:, :1] = refiner.eps * 0.1
    low_energy_output, low_energy_aux = refiner(
        torch.randn_like(low_energy_base), low_energy_base
    )
    identity_rotation = torch.zeros_like(low_energy_aux['rotation'])
    identity_rotation[:, :1] = 1.0
    assert torch.equal(low_energy_aux['rotation'], identity_rotation)
    assert low_energy_aux['corrected_magnitude'].min() >= 0.0
    assert torch.allclose(
        torch.linalg.vector_norm(low_energy_output, dim=1, keepdim=True),
        low_energy_aux['corrected_magnitude'],
        atol=1e-8,
        rtol=1e-6,
    )
    print('PASS eight-map input, ratios [1,2,2,1], odd/even shape and finite')


def _assert_activation_checkpointing():
    import torch
    import models.zip_refine_mp as zip_refine_module
    from models.zip_refine_mp import ZipRefineMP

    enabled_cfg = _actual_cfg()
    disabled_cfg = copy.deepcopy(enabled_cfg)
    disabled_cfg['model_cfg']['zip_refine_mp_activation_checkpointing'] = False
    default_cfg = copy.deepcopy(enabled_cfg)
    default_cfg['model_cfg'].pop('zip_refine_mp_activation_checkpointing')

    torch.manual_seed(151)
    disabled = ZipRefineMP(disabled_cfg).train()
    enabled = ZipRefineMP(enabled_cfg).train()
    enabled.load_state_dict(disabled.state_dict())
    assert ZipRefineMP(default_cfg).activation_checkpointing is False

    with torch.no_grad():
        for refiner in (disabled, enabled):
            refiner.outer_mag_gate.fill_(1e-3)
            refiner.outer_phase_gate.fill_(1e-3)

    noisy_source = torch.randn(1, 2, 5, 7)
    base_source = torch.randn(1, 2, 5, 7)
    cotangent = torch.randn_like(base_source)
    checkpoint_calls = []
    real_checkpoint = zip_refine_module.checkpoint

    def tracked_checkpoint(function, *args, **kwargs):
        checkpoint_calls.append(kwargs.copy())
        return real_checkpoint(function, *args, **kwargs)

    zip_refine_module.checkpoint = tracked_checkpoint
    try:
        enabled_noisy = noisy_source.clone().requires_grad_(True)
        enabled_base = base_source.clone().requires_grad_(True)
        enabled_output, _ = enabled(enabled_noisy, enabled_base)
        enabled_loss = (enabled_output * cotangent).sum()
        enabled_loss.backward()
    finally:
        zip_refine_module.checkpoint = real_checkpoint

    assert len(checkpoint_calls) == 8, len(checkpoint_calls)
    assert all(call == {'use_reentrant': False} for call in checkpoint_calls)

    disabled_noisy = noisy_source.clone().requires_grad_(True)
    disabled_base = base_source.clone().requires_grad_(True)
    disabled_output, _ = disabled(disabled_noisy, disabled_base)
    disabled_loss = (disabled_output * cotangent).sum()
    disabled_loss.backward()

    assert torch.equal(enabled_output, disabled_output)
    assert torch.equal(enabled_loss, disabled_loss)
    for enabled_input, disabled_input in (
            (enabled_noisy, disabled_noisy), (enabled_base, disabled_base)):
        assert torch.allclose(enabled_input.grad, disabled_input.grad, atol=1e-6, rtol=1e-5)
    for (enabled_name, enabled_parameter), (disabled_name, disabled_parameter) in zip(
            enabled.named_parameters(), disabled.named_parameters()):
        assert enabled_name == disabled_name
        if enabled_parameter.grad is None or disabled_parameter.grad is None:
            assert enabled_parameter.grad is disabled_parameter.grad, enabled_name
        else:
            assert torch.allclose(
                enabled_parameter.grad, disabled_parameter.grad, atol=1e-6, rtol=1e-5
            ), enabled_name

    checkpoint_calls.clear()
    zip_refine_module.checkpoint = tracked_checkpoint
    try:
        enabled.eval()
        eval_output, _ = enabled(noisy_source, base_source)
    finally:
        zip_refine_module.checkpoint = real_checkpoint
    assert not checkpoint_calls
    disabled.eval()
    disabled_eval_output, _ = disabled(noisy_source, base_source)
    assert torch.equal(eval_output, disabled_eval_output)

    zip_refine_module.checkpoint = tracked_checkpoint
    try:
        enabled.train()
        with torch.no_grad():
            enabled(noisy_source, base_source)
    finally:
        zip_refine_module.checkpoint = real_checkpoint
    assert not checkpoint_calls
    print(
        'PASS activation checkpoint default-off, eight non-reentrant training-stage '
        'calls, eval/no-grad bypass, exact forward/loss, and matching backward gradients'
    )


def _assert_identity_and_gradients():
    import torch
    from models.zip_refine_mp import ZipRefineMP

    torch.manual_seed(202)
    refiner = ZipRefineMP(_actual_cfg())
    noisy = torch.randn(1, 2, 5, 7)
    base = (torch.randn(1, 2, 5, 7) + 0.25).requires_grad_(True)
    output, _ = refiner(noisy, base)
    assert torch.equal(output, base), (output - base).abs().max().item()

    cotangent = torch.randn_like(output)
    input_vjp = torch.autograd.grad(
        output, base, cotangent, retain_graph=True
    )[0]
    assert torch.equal(input_vjp, cotangent), (input_vjp - cotangent).abs().max().item()

    weighted_loss = (output * cotangent).sum()
    mag_gate_grad, phase_gate_grad = torch.autograd.grad(
        weighted_loss,
        (refiner.outer_mag_gate, refiner.outer_phase_gate),
        retain_graph=False,
    )
    for gradient in (mag_gate_grad, phase_gate_grad):
        assert torch.isfinite(gradient) and gradient.abs() > 0

    with torch.no_grad():
        refiner.outer_mag_gate.fill_(1e-3)
        refiner.outer_phase_gate.fill_(1e-3)
    refiner.zero_grad(set_to_none=True)
    micro_output, micro_aux = refiner(noisy, base.detach())
    micro_magnitude = torch.linalg.vector_norm(micro_output, dim=1, keepdim=True)
    assert torch.allclose(
        micro_magnitude, micro_aux['corrected_magnitude'], atol=1e-5, rtol=1e-5
    )
    assert torch.allclose(
        torch.linalg.vector_norm(micro_aux['rotation'], dim=1),
        torch.ones_like(micro_aux['rotation'][:, 0]),
        atol=1e-5,
        rtol=1e-5,
    )
    micro_loss = (micro_output * cotangent).sum()
    micro_loss.backward()
    internal_gradients = [
        parameter.grad
        for name, parameter in refiner.named_parameters()
        if not name.startswith('outer_') and parameter.grad is not None
    ]
    assert internal_gradients
    assert all(torch.isfinite(gradient).all() for gradient in internal_gradients)
    assert sum(float(gradient.abs().sum()) for gradient in internal_gradients) > 0.0
    print(
        'PASS exact zero-gate identity/input VJP, nonzero finite gate gradients, '
        'and micro-gate internal gradients'
    )


def _assert_rng_and_parameters():
    import torch
    from models.generator import MambaSEUNet

    candidate_cfg = _actual_cfg()
    baseline_cfg = copy.deepcopy(candidate_cfg)
    baseline_cfg.pop('experiment_cfg', None)
    baseline_cfg['model_cfg'].pop('zip_refine_mp_enabled', None)
    baseline_cfg['model_cfg'].pop('zip_refine_mp_activation_checkpointing', None)
    baseline_cfg['model_cfg'].pop('zip_refine_mp_channels', None)
    baseline_cfg['model_cfg'].pop('zip_refine_mp_eps', None)
    baseline_cfg['model_cfg'].pop('zip_refine_mp_delta_limit', None)

    torch.manual_seed(303)
    baseline = MambaSEUNet(baseline_cfg)
    baseline_rng = torch.random.get_rng_state().clone()
    torch.manual_seed(303)
    candidate = MambaSEUNet(candidate_cfg)
    candidate_rng = torch.random.get_rng_state().clone()

    baseline_state = baseline.state_dict()
    candidate_state = candidate.state_dict()
    assert baseline.zip_refiner is None
    assert candidate.zip_refiner is not None
    assert baseline_state.keys() <= candidate_state.keys()
    for key, value in baseline_state.items():
        assert torch.equal(value, candidate_state[key]), key
    assert torch.equal(baseline_rng, candidate_rng)

    baseline_parameters = sum(parameter.numel() for parameter in baseline.parameters())
    candidate_parameters = sum(parameter.numel() for parameter in candidate.parameters())
    added_parameters = candidate_parameters - baseline_parameters
    known_real_baseline = 1_963_626
    projected_real_candidate = known_real_baseline + added_parameters
    assert 8_000_000 <= candidate_parameters <= 10_000_000, candidate_parameters
    assert 8_000_000 <= projected_real_candidate <= 10_000_000, projected_real_candidate
    assert added_parameters == sum(
        parameter.numel() for parameter in candidate.zip_refiner.parameters()
    )
    print(
        f'PASS RNG/shared-state equality; structural parameters '
        f'baseline={baseline_parameters:,}, added={added_parameters:,}, '
        f'total={candidate_parameters:,}; known-real-baseline projection='
        f'{projected_real_candidate:,}'
    )
    return candidate


def _assert_generator_forward_backward(generator):
    import torch

    # Harmonic templates depend on n_fft at construction, so use fresh paired models.
    candidate_cfg = copy.deepcopy(generator.cfg)
    candidate_cfg['stft_cfg'].update({'n_fft': 30, 'win_size': 30})
    candidate_cfg['model_cfg']['pitch_candidates'] = 8
    baseline_cfg = copy.deepcopy(candidate_cfg)
    baseline_cfg.pop('experiment_cfg', None)
    for key in (
        'zip_refine_mp_enabled',
        'zip_refine_mp_activation_checkpointing',
        'zip_refine_mp_channels',
        'zip_refine_mp_eps',
        'zip_refine_mp_delta_limit',
    ):
        baseline_cfg['model_cfg'].pop(key, None)
    from models.generator import MambaSEUNet
    torch.manual_seed(404)
    baseline = MambaSEUNet(baseline_cfg)
    torch.manual_seed(404)
    candidate = MambaSEUNet(candidate_cfg)
    noisy_mag_source = torch.rand(1, 16, 8) + 0.1
    noisy_phase_source = torch.randn(1, 16, 8)
    cotangents = (
        torch.randn(1, 16, 8),
        torch.randn(1, 16, 8),
        torch.randn(1, 16, 8, 2),
    )

    def paired_forward_vjp(model):
        noisy_mag = noisy_mag_source.clone().requires_grad_(True)
        noisy_phase = noisy_phase_source.clone().requires_grad_(True)
        outputs = model(noisy_mag, noisy_phase)
        objective = sum(
            (output * cotangent).sum()
            for output, cotangent in zip(outputs, cotangents)
        )
        input_vjp = torch.autograd.grad(objective, (noisy_mag, noisy_phase))
        return outputs, input_vjp

    baseline_outputs, baseline_vjp = paired_forward_vjp(baseline)
    candidate_outputs, candidate_vjp = paired_forward_vjp(candidate)
    for baseline_value, candidate_value in zip(baseline_outputs, candidate_outputs):
        assert torch.equal(baseline_value, candidate_value)
    for baseline_value, candidate_value in zip(baseline_vjp, candidate_vjp):
        assert torch.equal(baseline_value, candidate_value)

    magnitude, phase, complex_spectrum = candidate_outputs
    assert magnitude.shape == phase.shape == (1, 16, 8)
    assert complex_spectrum.shape == (1, 16, 8, 2)
    assert all(torch.isfinite(value).all() for value in (magnitude, phase, complex_spectrum))
    assert all(torch.isfinite(value).all() for value in candidate_vjp)
    required_aux = {
        'base_complex', 'delta_log_mag', 'applied_delta_magnitude', 'rotation',
        'outer_mag_gate', 'outer_phase_gate', 'stage_scales', 'interaction_scales',
    }
    assert required_aux <= candidate.latest_aux.keys()
    print(
        'PASS paired small-generator exact output/input VJP, finite tensors, '
        'shapes, and audit aux'
    )


def main():
    _install_import_stubs()
    _install_mamba_forward_stub()
    _assert_config_control()
    _assert_refiner_contract()
    _assert_activation_checkpointing()
    _assert_identity_and_gradients()
    generator = _assert_rng_and_parameters()
    _assert_generator_forward_backward(generator)
    if STRUCTURAL_STUB:
        print(
            'LIMITATION structural CPU stub used: structural parameter tensors '
            'were counted, but aggregate count differs from the known real baseline; '
            'this is NOT selective_scan_cuda or real CUDA validation.'
        )
    else:
        print('INFO real selective-scan dependency path was available.')
    print('GATE 0 PASS')


if __name__ == '__main__':
    main()
