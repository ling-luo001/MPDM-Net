"""Local Gate 0 for RD-Asymmetric-Polar-ZipRefine.

This script does not train or read checkpoints. If native selective-scan or
other local runtime dependencies are unavailable, it installs explicitly
labeled structural forward stubs. A structural pass is not CUDA validation.
"""

import argparse
import copy
import importlib.util
import json
import pathlib
import sys
import types


ROOT = pathlib.Path(__file__).resolve().parent
RD_ZIP_RECIPE = ROOT / 'recipes' / 'RD-ZipRefine-MP' / 'RD-ZipRefine-MP.yaml'
CANDIDATE_RECIPE = (
    ROOT / 'recipes' / 'RD-Asymmetric-Polar-ZipRefine'
    / 'RD-Asymmetric-Polar-ZipRefine.yaml'
)
STRUCTURAL_STUB = False
TEST_DEVICE = None
EXPECTED_PARENT_PARAMETERS = 1_961_130
PARAMETER_CAP = 4_525_424
PARENT_DOUBLE_CAP = 2 * EXPECTED_PARENT_PARAMETERS


def _install_import_stubs():
    global STRUCTURAL_STUB
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
        STRUCTURAL_STUB = True

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
        from timm.models.layers import DropPath  # noqa: F401
    except Exception:
        timm = types.ModuleType('timm')
        layers = types.ModuleType('timm.layers')
        models = types.ModuleType('timm.models')
        model_layers = types.ModuleType('timm.models.layers')

        class StructuralMlp(nn.Module):
            def __init__(
                self, in_features, hidden_features=None, out_features=None,
                act_layer=nn.GELU, drop=0.0, **kwargs
            ):
                super().__init__()
                hidden_features = hidden_features or in_features
                out_features = out_features or in_features
                self.fc1 = nn.Linear(in_features, hidden_features)
                self.act = act_layer()
                self.fc2 = nn.Linear(hidden_features, out_features)

            def forward(self, value):
                return self.fc2(self.act(self.fc1(value)))

        class StructuralDropPath(nn.Identity):
            def __init__(self, drop_prob=0.0):
                super().__init__()
                self.drop_prob = drop_prob

        def trunc_normal_(tensor, std=0.02, **kwargs):
            return nn.init.normal_(tensor, std=std)

        layers.Mlp = StructuralMlp
        model_layers.DropPath = StructuralDropPath
        model_layers.to_2tuple = lambda value: (value, value)
        model_layers.trunc_normal_ = trunc_normal_
        timm.layers = layers
        timm.models = models
        models.layers = model_layers
        sys.modules['timm'] = timm
        sys.modules['timm.layers'] = layers
        sys.modules['timm.models'] = models
        sys.modules['timm.models.layers'] = model_layers
        STRUCTURAL_STUB = True


def _install_forward_stubs():
    if not STRUCTURAL_STUB:
        return
    import torch
    from models.cross import SS2D_cross_new
    from models.mamba_block import MambaBlock

    def parameter_signal(module, value):
        signal = value.new_zeros(())
        for parameter in module.parameters():
            signal = signal + parameter.to(value.dtype).mean()
        return 1.0 + 1e-5 * torch.tanh(signal)

    def structural_mamba_forward(module, value):
        modeled = value * parameter_signal(module, value)
        return torch.cat((modeled, modeled), dim=-1)

    def structural_cross_forward(module, value1, value2, **kwargs):
        factor = parameter_signal(module, value1)
        return value1 * factor, value2 * factor

    MambaBlock.forward = structural_mamba_forward
    SS2D_cross_new.forward = structural_cross_forward


def _configure_runtime(require_native_cuda):
    global STRUCTURAL_STUB, TEST_DEVICE
    import torch

    if not STRUCTURAL_STUB and not torch.cuda.is_available():
        if require_native_cuda:
            raise RuntimeError('Native Gate 0 requires an available CUDA device.')
        STRUCTURAL_STUB = True
    if require_native_cuda and STRUCTURAL_STUB:
        raise RuntimeError(
            'Native Gate 0 refused to run because structural stubs are required.'
        )
    TEST_DEVICE = torch.device('cpu' if STRUCTURAL_STUB else 'cuda')
    _install_forward_stubs()
    if TEST_DEVICE.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(TEST_DEVICE)


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


def _actual_cfg():
    return _load_yaml(CANDIDATE_RECIPE)


def _baseline_cfg():
    cfg = _actual_cfg()
    for key in tuple(cfg['model_cfg']):
        if key.startswith('asymmetric_polar_zip_refine_'):
            cfg['model_cfg'].pop(key)
    return cfg


def _assert_config_control():
    reference = _load_yaml(RD_ZIP_RECIPE)
    candidate = _actual_cfg()
    differences = _diff_paths(reference, candidate)
    allowed = {
        'experiment_cfg.name',
        'experiment_cfg.log_name',
        'env_setting.dist_cfg.dist_url',
        'model_cfg.zip_refine_mp_enabled',
        'model_cfg.zip_refine_mp_activation_checkpointing',
        'model_cfg.zip_refine_mp_channels',
        'model_cfg.zip_refine_mp_eps',
        'model_cfg.zip_refine_mp_delta_limit',
        'model_cfg.asymmetric_polar_zip_refine_enabled',
        'model_cfg.asymmetric_polar_zip_refine_activation_checkpointing',
        'model_cfg.asymmetric_polar_zip_refine_mag_channels',
        'model_cfg.asymmetric_polar_zip_refine_phase_channels',
        'model_cfg.asymmetric_polar_zip_refine_stage_common_channels',
        'model_cfg.asymmetric_polar_zip_refine_expand',
        'model_cfg.asymmetric_polar_zip_refine_eps',
        'model_cfg.asymmetric_polar_zip_refine_delta_limit',
        'model_cfg.asymmetric_polar_zip_refine_interaction_gate_bias',
        'model_cfg.asymmetric_polar_zip_refine_complex_residual_scale',
        'model_cfg.asymmetric_polar_zip_refine_complex_residual_gate_bias',
    }
    assert differences == allowed, (differences, allowed)
    assert candidate['data_cfg'] == reference['data_cfg']
    assert candidate['training_cfg'] == reference['training_cfg']
    assert candidate['stft_cfg'] == reference['stft_cfg']
    assert candidate['model_cfg']['expand'] == 4
    assert 'zip_refine_mp_enabled' not in candidate['model_cfg']
    forbidden = ('torch.load', 'load_state_dict', 'resume_from', 'pretrained', '.pth')
    for relative in (
        'models/asymmetric_polar_zip_refine.py',
        'models/generator.py',
        'recipes/RD-Asymmetric-Polar-ZipRefine/RD-Asymmetric-Polar-ZipRefine.yaml',
    ):
        source = (ROOT / relative).read_text(encoding='utf-8').lower()
        assert not any(token in source for token in forbidden), relative

    def load_manifest(relative):
        return json.loads((ROOT / relative).read_text(encoding='utf-8'))

    valid_clean = load_manifest(candidate['data_cfg']['valid_clean_json'])
    valid_noisy = load_manifest(candidate['data_cfg']['valid_noisy_json'])
    mini_valid_clean = load_manifest('data/mini_val_clean_list.json')
    mini_valid_noisy = load_manifest('data/mini_val_noisy_list.json')
    assert len(valid_clean) == len(valid_noisy) == 824
    assert len(mini_valid_clean) == len(mini_valid_noisy) == 165
    assert all(
        pathlib.PurePosixPath(clean).name == pathlib.PurePosixPath(noisy).name
        for clean, noisy in zip(valid_clean, valid_noisy)
    )
    assert all(
        pathlib.PurePosixPath(clean).name == pathlib.PurePosixPath(noisy).name
        for clean, noisy in zip(mini_valid_clean, mini_valid_noisy)
    )
    assert set(mini_valid_clean) <= set(valid_clean)
    assert set(mini_valid_noisy) <= set(valid_noisy)
    print(
        'PASS config changes restricted; full validation=824, mini validation=165 '
        'paired subset; training settings unchanged'
    )


def _assert_switches_and_contract():
    import torch
    import models.generator as generator_module
    from models.asymmetric_polar_zip_refine import (
        AsymmetricPolarZipRefine,
        _AlignedSS2DCross,
    )
    from models.generator import MambaSEUNet
    from models.mamba_block import FMambaBlock, TMambaBlock

    cfg = _actual_cfg()
    original_expand = cfg['model_cfg']['expand']
    refiner = AsymmetricPolarZipRefine(cfg).to(TEST_DEVICE)
    assert cfg['model_cfg']['expand'] == original_expand == 4
    assert refiner.core_cfg is not cfg
    assert refiner.core_cfg['model_cfg']['expand'] == 2
    assert refiner.compression_ratios == (1, 2, 2, 1)
    assert refiner.mag_channels == 80 and refiner.phase_channels == 40
    assert refiner.eps == 1e-6 and refiner.phase_eps == 1e-3
    assert len(refiner.paired_stages) == 4
    assert [stage.interaction_position for stage in refiner.paired_stages] == [
        'none', 'compressed_pre_up', 'compressed_pre_up', 'full_resolution'
    ]
    interactions = [
        stage.interaction for stage in refiner.paired_stages
        if stage.interaction is not None
    ]
    assert len(interactions) == 3
    assert [module.common_channels for module in interactions] == [64, 64, 40]
    assert len({id(module) for module in interactions}) == 3
    for interaction in interactions:
        assert isinstance(interaction.cross, _AlignedSS2DCross)
        assert torch.count_nonzero(interaction.mag_cross_gate.weight) == 0
        assert torch.count_nonzero(interaction.phase_cross_gate.weight) == 0
        assert torch.equal(
            interaction.mag_cross_gate.bias,
            torch.full_like(interaction.mag_cross_gate.bias, -2.0),
        )
        assert torch.equal(
            interaction.phase_cross_gate.bias,
            torch.full_like(interaction.phase_cross_gate.bias, -2.0),
        )
    for stage in refiner.paired_stages:
        assert isinstance(stage.mag_path.axis_blocks[0], FMambaBlock)
        assert isinstance(stage.mag_path.axis_blocks[1], TMambaBlock)
        assert isinstance(stage.phase_path.axis_blocks[0], TMambaBlock)
        assert isinstance(stage.phase_path.axis_blocks[1], FMambaBlock)
    forbidden_types = {'Cross_layer', 'VSSBlock_Cross_new'}
    assert not forbidden_types & {
        module.__class__.__name__ for module in refiner.modules()
    }

    disabled_cfg = _baseline_cfg()
    real_constructor = generator_module.AsymmetricPolarZipRefine

    class ForbiddenConstructor:
        def __init__(self, *args, **kwargs):
            raise AssertionError('disabled refiner was constructed')

    generator_module.AsymmetricPolarZipRefine = ForbiddenConstructor
    try:
        disabled = MambaSEUNet(disabled_cfg)
    finally:
        generator_module.AsymmetricPolarZipRefine = real_constructor
    assert disabled.asymmetric_polar_zip_refiner is None

    conflicting_cfg = copy.deepcopy(cfg)
    conflicting_cfg['model_cfg']['zip_refine_mp_enabled'] = True
    try:
        MambaSEUNet(conflicting_cfg)
    except ValueError as error:
        assert 'mutually exclusive' in str(error)
    else:
        raise AssertionError('mutually exclusive refiners were accepted')
    print('PASS disabled construction, mutual exclusion, widths/order/placement, expand isolation')
    return refiner


def _assert_directional_alignment():
    import torch
    from models.asymmetric_polar_zip_refine import (
        _build_aligned_scan_sequences,
        _merge_aligned_scan_outputs,
    )

    position_codes = torch.arange(1, 7, dtype=torch.float32).reshape(1, 1, 2, 3)
    directional = _build_aligned_scan_sequences(position_codes)
    aligned = _merge_aligned_scan_outputs(directional, height=2, width=3)
    assert torch.equal(aligned, 4.0 * position_codes)
    naive = directional.sum(dim=1).reshape_as(position_codes)
    assert not torch.equal(naive, aligned)
    print('PASS four-direction inverse flip/transpose alignment')


def _assert_bidirectional_exchange(refiner):
    import torch

    torch.manual_seed(151)
    interaction = refiner.paired_stages[1].interaction
    mag_features = torch.randn(
        1, 80, 3, 4, device=TEST_DEVICE, requires_grad=True
    )
    phase_features = torch.randn(
        1, 40, 3, 4, device=TEST_DEVICE, requires_grad=True
    )
    mag_output, phase_output = interaction(mag_features, phase_features)
    mag_cotangent = torch.randn_like(mag_output)
    phase_cotangent = torch.randn_like(phase_output)
    phase_to_mag = torch.autograd.grad(
        (mag_output * mag_cotangent).sum(),
        phase_features,
        retain_graph=True,
    )[0]
    mag_to_phase = torch.autograd.grad(
        (phase_output * phase_cotangent).sum(),
        mag_features,
    )[0]
    assert phase_to_mag.abs().sum() > 0
    assert mag_to_phase.abs().sum() > 0
    assert torch.isfinite(phase_to_mag).all()
    assert torch.isfinite(mag_to_phase).all()
    print('PASS explicit phase-to-mag and mag-to-phase feature transfer')


def _assert_shapes_identity_and_gradients(refiner):
    import torch

    torch.manual_seed(202)
    for time_bins, freq_bins in ((5, 7), (6, 8)):
        noisy = torch.randn(1, 2, time_bins, freq_bins, device=TEST_DEVICE)
        base = torch.randn(1, 2, time_bins, freq_bins, device=TEST_DEVICE)
        maps = refiner.build_eight_map_input(noisy, base)
        expected_maps = torch.cat((
            noisy,
            base,
            noisy - base,
            torch.log1p(torch.linalg.vector_norm(noisy, dim=1, keepdim=True)),
            torch.log1p(torch.linalg.vector_norm(base, dim=1, keepdim=True)),
        ), dim=1)
        assert torch.equal(maps, expected_maps)
        output, _ = refiner(noisy, base)
        assert output.shape == base.shape
        assert torch.equal(output, base)
        assert torch.isfinite(output).all()

    noisy = torch.randn(1, 2, 5, 7, device=TEST_DEVICE)
    base = (
        torch.randn(1, 2, 5, 7, device=TEST_DEVICE) + 0.25
    ).requires_grad_(True)
    output, aux = refiner(noisy, base)
    cotangent = torch.randn_like(output)
    input_vjp = torch.autograd.grad(
        output, base, cotangent, retain_graph=True
    )[0]
    assert torch.equal(input_vjp, cotangent)
    loss = (output * cotangent).sum()
    gradients = torch.autograd.grad(
        loss,
        (
            refiner.outer_mag_gate,
            refiner.outer_phase_gate,
            refiner.ri_residual_head[-1].weight,
            refiner.ri_residual_gate.weight,
        ),
    )
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert gradients[0].abs() > 0 and gradients[1].abs() > 0
    assert gradients[2].abs().sum() > 0
    assert torch.count_nonzero(gradients[3]) == 0
    assert torch.count_nonzero(refiner.ri_residual_head[-1].weight) == 0
    assert torch.count_nonzero(refiner.ri_residual_head[-1].bias) == 0
    assert torch.count_nonzero(refiner.ri_residual_gate.weight) == 0
    assert torch.equal(
        refiner.ri_residual_gate.bias,
        torch.full_like(refiner.ri_residual_gate.bias, -2.0),
    )
    assert torch.count_nonzero(aux['ri_residual']) == 0
    assert torch.count_nonzero(aux['ri_residual_applied']) == 0

    zero_noisy = torch.zeros(
        1, 2, 5, 7, device=TEST_DEVICE, requires_grad=True
    )
    zero_base = torch.zeros(
        1, 2, 5, 7, device=TEST_DEVICE, requires_grad=True
    )
    zero_output, zero_aux = refiner(zero_noisy, zero_base)
    zero_cotangent = torch.randn_like(zero_output)
    zero_loss = (zero_output * zero_cotangent).sum()
    zero_gradients = torch.autograd.grad(
        zero_loss, (zero_noisy, zero_base), allow_unused=False
    )
    assert torch.equal(zero_output, zero_base)
    assert all(torch.isfinite(gradient).all() for gradient in zero_gradients)
    assert torch.count_nonzero(zero_gradients[0]) == 0
    assert torch.equal(zero_gradients[1], zero_cotangent)
    assert torch.count_nonzero(zero_aux['ri_residual_energy_gate']) == 0

    with torch.no_grad():
        refiner.outer_mag_gate.fill_(1e-3)
        refiner.outer_phase_gate.fill_(1e-3)
    refiner.zero_grad(set_to_none=True)
    finite_output, finite_aux = refiner(noisy, base.detach())
    finite_loss = finite_output.square().mean()
    finite_loss.backward()
    assert torch.isfinite(finite_output).all()
    assert all(
        torch.isfinite(value).all() for value in finite_aux.values()
        if torch.is_tensor(value)
    )
    parameter_gradients = [
        parameter.grad for parameter in refiner.parameters()
        if parameter.grad is not None
    ]
    assert parameter_gradients
    assert all(torch.isfinite(gradient).all() for gradient in parameter_gradients)
    assert refiner.mag_stem[0].weight.grad.abs().sum() > 0
    assert refiner.phase_stem[0].weight.grad.abs().sum() > 0
    first_interaction = refiner.paired_stages[1].interaction
    assert first_interaction.mag_cross_gate.weight.grad.abs().sum() > 0
    assert first_interaction.phase_cross_gate.weight.grad.abs().sum() > 0

    with torch.no_grad():
        refiner.ri_residual_head[-1].weight.fill_(1e-3)
    refiner.zero_grad(set_to_none=True)
    _, opened_aux = refiner(noisy, base.detach())
    ri_cotangent = torch.randn_like(opened_aux['ri_residual_applied'])
    (opened_aux['ri_residual_applied'] * ri_cotangent).sum().backward()
    assert refiner.ri_residual_gate.weight.grad.abs().sum() > 0
    assert refiner.ri_residual_gate.bias.grad.abs().sum() > 0
    assert refiner.phase_stem[0].weight.grad.abs().sum() > 0
    print(
        'PASS eight maps, odd/even restore, identity/VJP, staged polar/A-B '
        'opening and finite gradients'
    )


def _assert_checkpoint_parity():
    import torch
    import models.asymmetric_polar_zip_refine as refiner_module
    from models.asymmetric_polar_zip_refine import AsymmetricPolarZipRefine

    enabled_cfg = _actual_cfg()
    disabled_cfg = copy.deepcopy(enabled_cfg)
    disabled_cfg['model_cfg'][
        'asymmetric_polar_zip_refine_activation_checkpointing'
    ] = False
    default_cfg = copy.deepcopy(enabled_cfg)
    default_cfg['model_cfg'].pop(
        'asymmetric_polar_zip_refine_activation_checkpointing'
    )
    torch.manual_seed(303)
    disabled = AsymmetricPolarZipRefine(disabled_cfg).to(TEST_DEVICE).train()
    enabled = AsymmetricPolarZipRefine(enabled_cfg).to(TEST_DEVICE).train()
    enabled.load_state_dict(disabled.state_dict())
    assert AsymmetricPolarZipRefine(default_cfg).activation_checkpointing is False
    with torch.no_grad():
        for module in (disabled, enabled):
            module.outer_mag_gate.fill_(1e-3)
            module.outer_phase_gate.fill_(1e-3)

    noisy_source = torch.randn(1, 2, 5, 7, device=TEST_DEVICE)
    base_source = torch.randn(1, 2, 5, 7, device=TEST_DEVICE)
    cotangent = torch.randn_like(base_source)
    checkpoint_calls = []
    real_checkpoint = refiner_module.checkpoint

    def tracked_checkpoint(function, *args, **kwargs):
        checkpoint_calls.append(kwargs.copy())
        return real_checkpoint(function, *args, **kwargs)

    refiner_module.checkpoint = tracked_checkpoint
    try:
        enabled_noisy = noisy_source.clone().requires_grad_(True)
        enabled_base = base_source.clone().requires_grad_(True)
        enabled_output, _ = enabled(enabled_noisy, enabled_base)
        (enabled_output * cotangent).sum().backward()
    finally:
        refiner_module.checkpoint = real_checkpoint
    assert len(checkpoint_calls) == 4
    assert all(call == {'use_reentrant': False} for call in checkpoint_calls)

    disabled_noisy = noisy_source.clone().requires_grad_(True)
    disabled_base = base_source.clone().requires_grad_(True)
    disabled_output, _ = disabled(disabled_noisy, disabled_base)
    (disabled_output * cotangent).sum().backward()
    assert torch.equal(enabled_output, disabled_output)
    assert torch.allclose(enabled_noisy.grad, disabled_noisy.grad, atol=1e-6, rtol=1e-5)
    assert torch.allclose(enabled_base.grad, disabled_base.grad, atol=1e-6, rtol=1e-5)
    for (enabled_name, enabled_parameter), (disabled_name, disabled_parameter) in zip(
        enabled.named_parameters(), disabled.named_parameters()
    ):
        assert enabled_name == disabled_name
        if enabled_parameter.grad is None or disabled_parameter.grad is None:
            assert enabled_parameter.grad is disabled_parameter.grad, enabled_name
        else:
            assert torch.allclose(
                enabled_parameter.grad,
                disabled_parameter.grad,
                atol=1e-6,
                rtol=1e-5,
            ), enabled_name

    checkpoint_calls.clear()
    refiner_module.checkpoint = tracked_checkpoint
    try:
        enabled.eval()
        enabled(noisy_source, base_source)
        enabled.train()
        with torch.no_grad():
            enabled(noisy_source, base_source)
    finally:
        refiner_module.checkpoint = real_checkpoint
    assert not checkpoint_calls
    print('PASS four whole-paired-stage checkpoints, tuple output, parity, eval/no-grad bypass')


def _assert_state_rng_parameters_and_generator():
    import torch
    from models.generator import MambaSEUNet

    baseline_cfg = _baseline_cfg()
    candidate_cfg = _actual_cfg()
    torch.manual_seed(404)
    baseline = MambaSEUNet(baseline_cfg)
    baseline_rng = torch.random.get_rng_state().clone()
    torch.manual_seed(404)
    candidate = MambaSEUNet(candidate_cfg)
    candidate_rng = torch.random.get_rng_state().clone()
    assert baseline.asymmetric_polar_zip_refiner is None
    assert candidate.asymmetric_polar_zip_refiner is not None
    baseline_state = baseline.state_dict()
    candidate_state = candidate.state_dict()
    assert baseline_state.keys() <= candidate_state.keys()
    for key, value in baseline_state.items():
        assert torch.equal(value, candidate_state[key]), key
    assert torch.equal(baseline_rng, candidate_rng)

    baseline_parameters = sum(parameter.numel() for parameter in baseline.parameters())
    assert baseline_parameters == EXPECTED_PARENT_PARAMETERS, baseline_parameters
    added_parameters = sum(
        parameter.numel()
        for parameter in candidate.asymmetric_polar_zip_refiner.parameters()
    )
    projected_total = baseline_parameters + added_parameters
    assert projected_total <= PARAMETER_CAP, projected_total
    assert projected_total <= PARENT_DOUBLE_CAP, projected_total
    actual_total = sum(parameter.numel() for parameter in candidate.parameters())
    assert actual_total == projected_total, (actual_total, projected_total)
    assert actual_total <= PARAMETER_CAP, actual_total
    assert actual_total <= PARENT_DOUBLE_CAP, actual_total
    parent_ratio = projected_total / baseline_parameters
    print(
        f'PASS state/RNG isolation; parent={baseline_parameters:,}, '
        f'refiner={added_parameters:,}, total={projected_total:,} '
        f'({parent_ratio:.3f}x parent), '
        f'caps={PARENT_DOUBLE_CAP:,}/{PARAMETER_CAP:,}, '
        f'aggregate={actual_total:,}'
    )

    small_candidate_cfg = copy.deepcopy(candidate_cfg)
    small_candidate_cfg['stft_cfg'].update({'n_fft': 30, 'win_size': 30})
    small_candidate_cfg['model_cfg']['pitch_candidates'] = 8
    small_baseline_cfg = copy.deepcopy(small_candidate_cfg)
    for key in tuple(small_baseline_cfg['model_cfg']):
        if key.startswith('asymmetric_polar_zip_refine_'):
            small_baseline_cfg['model_cfg'].pop(key)
    torch.manual_seed(505)
    small_baseline = MambaSEUNet(small_baseline_cfg).to(TEST_DEVICE)
    torch.manual_seed(505)
    small_candidate = MambaSEUNet(small_candidate_cfg).to(TEST_DEVICE)
    small_baseline_state = small_baseline.state_dict()
    small_candidate_state = small_candidate.state_dict()
    assert small_baseline_state.keys() <= small_candidate_state.keys()
    for key, value in small_baseline_state.items():
        assert torch.equal(value, small_candidate_state[key]), key
    noisy_mag_source = torch.rand(1, 16, 8, device=TEST_DEVICE) + 0.1
    noisy_phase_source = torch.randn(1, 16, 8, device=TEST_DEVICE)
    cotangents = (
        torch.randn(1, 16, 8, device=TEST_DEVICE),
        torch.randn(1, 16, 8, device=TEST_DEVICE),
        torch.randn(1, 16, 8, 2, device=TEST_DEVICE),
    )

    def forward_vjp(model):
        noisy_mag = noisy_mag_source.clone().requires_grad_(True)
        noisy_phase = noisy_phase_source.clone().requires_grad_(True)
        outputs = model(noisy_mag, noisy_phase)
        objective = sum(
            (output * cotangent).sum()
            for output, cotangent in zip(outputs, cotangents)
        )
        gradients = torch.autograd.grad(objective, (noisy_mag, noisy_phase))
        return outputs, gradients

    baseline_outputs, baseline_vjp = forward_vjp(small_baseline)
    candidate_outputs, candidate_vjp = forward_vjp(small_candidate)
    output_max_error = 0.0
    for baseline_value, candidate_value in zip(baseline_outputs, candidate_outputs):
        output_max_error = max(
            output_max_error,
            (baseline_value - candidate_value).abs().max().item(),
        )
        assert torch.allclose(
            baseline_value, candidate_value, atol=1e-6, rtol=1e-5
        )
        assert torch.isfinite(candidate_value).all()
    vjp_max_error = 0.0
    for baseline_value, candidate_value in zip(baseline_vjp, candidate_vjp):
        vjp_max_error = max(
            vjp_max_error,
            (baseline_value - candidate_value).abs().max().item(),
        )
        assert torch.allclose(
            baseline_value, candidate_value, atol=1e-6, rtol=1e-5
        )
        assert torch.isfinite(candidate_value).all()
    required_aux = {
        'base_complex', 'delta_log_mag', 'applied_delta_magnitude', 'rotation',
        'outer_mag_gate', 'outer_phase_gate', 'ri_residual',
        'ri_residual_energy_gate', 'ri_residual_applied',
    }
    assert required_aux <= small_candidate.latest_aux.keys()
    print(
        'PASS generator identity integration and finite forward/input VJP; '
        f'max errors output={output_max_error:.3e}, vjp={vjp_max_error:.3e}'
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--require-native-cuda',
        action='store_true',
        help='Fail instead of falling back to structural CPU stubs.',
    )
    parser.add_argument(
        '--allow-structural-only',
        action='store_true',
        help='Permit a labeled structural-only success when CUDA kernels are absent.',
    )
    args = parser.parse_args()
    if args.require_native_cuda and args.allow_structural_only:
        parser.error('Choose native CUDA or structural-only mode, not both.')
    _install_import_stubs()
    _configure_runtime(args.require_native_cuda)
    _assert_config_control()
    refiner = _assert_switches_and_contract()
    _assert_directional_alignment()
    _assert_bidirectional_exchange(refiner)
    _assert_shapes_identity_and_gradients(refiner)
    _assert_checkpoint_parity()
    _assert_state_rng_parameters_and_generator()
    if STRUCTURAL_STUB:
        print(
            'LIMITATION structural CPU stubs used: parameter tensors and local '
            'autograd paths were checked, but this is NOT native selective-scan '
            'or real CUDA validation.'
        )
        if not args.allow_structural_only:
            raise SystemExit(
                'STRUCTURAL CHECK INCOMPLETE: rerun with --require-native-cuda '
                'for Gate 0, or explicitly use --allow-structural-only.'
            )
        print('STRUCTURAL CHECK PASS; NATIVE CUDA GATE 0 STILL REQUIRED')
    else:
        import torch

        torch.cuda.synchronize(TEST_DEVICE)
        peak_mib = torch.cuda.max_memory_allocated(TEST_DEVICE) / (1024 ** 2)
        print(
            f'PASS native selective-scan CUDA execution; device={TEST_DEVICE}, '
            f'Gate 0 peak allocated={peak_mib:.1f} MiB'
        )
        print('NATIVE CUDA GATE 0 PASS')


if __name__ == '__main__':
    main()
