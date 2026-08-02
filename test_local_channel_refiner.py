import math
import sys
import types
from unittest import mock

import torch
import torch.nn as nn
import yaml

if not hasattr(nn, 'RMSNorm'):
    nn.RMSNorm = nn.LayerNorm


class _IdentityMamba(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return x


# The Windows environment lacks selective-scan and Transformers. The production
# Mamba blocks are replaced below, so import-only package stubs are sufficient.
if 'mamba_ssm' not in sys.modules:
    mamba_package = types.ModuleType('mamba_ssm')
    mamba_modules = types.ModuleType('mamba_ssm.modules')
    mamba_simple = types.ModuleType('mamba_ssm.modules.mamba_simple')
    mamba_simple.Mamba = _IdentityMamba
    mamba_simple.Block = _IdentityMamba
    mamba_models = types.ModuleType('mamba_ssm.models')
    mixer = types.ModuleType('mamba_ssm.models.mixer_seq_simple')
    mixer._init_weights = lambda *args, **kwargs: None
    mamba_ops = types.ModuleType('mamba_ssm.ops')
    mamba_triton = types.ModuleType('mamba_ssm.ops.triton')
    layernorm = types.ModuleType('mamba_ssm.ops.triton.layernorm')
    layernorm.RMSNorm = nn.LayerNorm
    sys.modules['mamba_ssm'] = mamba_package
    sys.modules['mamba_ssm.modules'] = mamba_modules
    sys.modules['mamba_ssm.modules.mamba_simple'] = mamba_simple
    sys.modules['mamba_ssm.models'] = mamba_models
    sys.modules['mamba_ssm.models.mixer_seq_simple'] = mixer
    sys.modules['mamba_ssm.ops'] = mamba_ops
    sys.modules['mamba_ssm.ops.triton'] = mamba_triton
    sys.modules['mamba_ssm.ops.triton.layernorm'] = layernorm

import models.generator as generator_module
from models.generator import MambaSEUNet, MultiScaleLocalChannelRefiner


def test_zero_dense_scales_match_parallel_branches():
    torch.manual_seed(7)
    refiner = MultiScaleLocalChannelRefiner(channels=8, strip_kernel=7)
    features = torch.randn(2, 8, 13, 11)
    projected = refiner.input_projection(refiner.pre_norm(features))

    local_3x3, temporal, frequency = refiner._compute_branches(projected)

    assert torch.equal(local_3x3, refiner.local_3x3(projected))
    assert torch.equal(temporal, refiner.temporal_strip(projected))
    assert torch.equal(frequency, refiner.frequency_strip(projected))


def test_initial_weights_gain_and_residual_identity():
    torch.manual_seed(11)
    refiner = MultiScaleLocalChannelRefiner(channels=8, strip_kernel=7)
    features = torch.randn(2, 8, 13, 11)

    expected_weight = torch.full((3,), 1.0 / math.sqrt(3.0))
    assert torch.allclose(refiner._branch_weights(), expected_weight, atol=1e-7)

    projected = refiner.input_projection(refiner.pre_norm(features))
    branches = refiner._compute_branches(projected)
    local = sum(
        weight * branch
        for weight, branch in zip(refiner._branch_weights(), branches)
    )
    update = refiner.output_projection(local)
    assert torch.equal(refiner._channel_gain(update), torch.ones_like(update[:, :, :1, :1]))

    with torch.no_grad():
        refiner.residual_scale.zero_()
    assert torch.equal(refiner(features), features)


def test_refiner_gradients_and_detached_diagnostics():
    torch.manual_seed(13)
    refiner = MultiScaleLocalChannelRefiner(
        channels=16,
        strip_kernel=7,
        initial_scale=0.05,
        dense_initial_scale=0.0,
    )
    features = torch.randn(2, 16, 24, 20, requires_grad=True)

    output = refiner(features)
    output.square().mean().backward()

    assert output.shape == features.shape
    assert torch.isfinite(output).all()
    assert torch.isfinite(features.grad).all()
    assert features.grad.abs().sum() > 0
    assert math.isclose(
        torch.tanh(refiner.residual_scale).item(), 0.05, rel_tol=0.0, abs_tol=1e-6
    )

    parameters = (
        refiner.input_projection.weight,
        refiner.local_3x3.weight,
        refiner.temporal_strip.weight,
        refiner.frequency_strip.weight,
        refiner.output_projection[1].weight,
        refiner.channel_attention.weight,
        refiner.dense_residual_scales,
        refiner.branch_logits,
        refiner.residual_scale,
    )
    for parameter in parameters:
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0

    diagnostics = refiner.latest_diagnostics
    assert diagnostics['dense_scales'].shape == (3,)
    assert diagnostics['branch_weights'].shape == (3,)
    for value in diagnostics.values():
        assert not value.requires_grad
        assert torch.isfinite(value).all()


def test_stabilization_preserves_constructor_rng_consumption():
    torch.manual_seed(17)
    MultiScaleLocalChannelRefiner(channels=16, strip_kernel=7)
    stabilized_rng_state = torch.random.get_rng_state()

    torch.manual_seed(17)
    nn.GroupNorm(num_groups=1, num_channels=16)
    nn.Conv2d(16, 16, 1, bias=False)
    nn.Conv2d(16, 16, 3, padding=1, groups=16, bias=False)
    nn.Conv2d(16, 16, (7, 1), padding=(3, 0), groups=16, bias=False)
    nn.Conv2d(16, 16, (1, 7), padding=(0, 3), groups=16, bias=False)
    nn.Sequential(nn.SiLU(), nn.Conv2d(16, 16, 1, bias=False))
    nn.Conv1d(1, 1, kernel_size=3, padding=1, bias=False)
    original_rng_state = torch.random.get_rng_state()

    assert torch.equal(stabilized_rng_state, original_rng_state)


def test_stub_generator_diagnostics_and_parameter_budget():
    with open(
        'recipes/Mamba-SEUNet/Mamba-SEUNet-mini-3090.yaml', encoding='utf-8'
    ) as config_file:
        cfg = yaml.safe_load(config_file)

    with (
        mock.patch.object(generator_module, 'TMambaBlock', _IdentityMamba),
        mock.patch.object(generator_module, 'FMambaBlock', _IdentityMamba),
        mock.patch.object(generator_module, 'TFMambaBlock', _IdentityMamba),
    ):
        model = MambaSEUNet(cfg).eval()

    stabilization_parameters = sum(
        refiner.dense_residual_scales.numel() + refiner.branch_logits.numel()
        for refiner in model.local_channel_refiners.values()
    )
    assert len(model.local_channel_refiners) == 12
    assert stabilization_parameters == 72
    assert stabilization_parameters <= 128

    frames = 8
    frequency_bins = cfg['stft_cfg']['n_fft'] // 2 + 1
    magnitude = torch.rand(1, frequency_bins, frames)
    phase = torch.rand_like(magnitude) * (2.0 * torch.pi) - torch.pi
    with torch.no_grad():
        outputs = model(magnitude, phase)

    assert all(torch.isfinite(output).all() for output in outputs)
    aux = model.latest_aux
    assert aux['local_channel_scales'].shape == (12,)
    assert aux['local_channel_dense_scales'].shape == (12, 3)
    assert aux['local_channel_branch_weights'].shape == (12, 3)
    assert aux['local_channel_channel_gain'].shape == (12,)
    assert aux['local_channel_update_ratio'].shape == (12,)
    assert torch.allclose(
        aux['local_channel_scales'], torch.full((12,), 0.05), atol=1e-6
    )
    assert torch.count_nonzero(aux['local_channel_dense_scales']) == 0
    assert torch.allclose(
        aux['local_channel_branch_weights'],
        torch.full((12, 3), 1.0 / math.sqrt(3.0)),
        atol=1e-7,
    )
    assert torch.equal(
        aux['local_channel_channel_gain'], torch.ones(12)
    )
    for key in (
        'local_channel_scales',
        'local_channel_dense_scales',
        'local_channel_branch_weights',
        'local_channel_channel_gain',
        'local_channel_update_ratio',
    ):
        assert not aux[key].requires_grad
        assert torch.isfinite(aux[key]).all()


def test_refiner_rejects_invalid_settings():
    for kernel in (1, 4):
        try:
            MultiScaleLocalChannelRefiner(16, strip_kernel=kernel)
        except ValueError:
            pass
        else:
            raise AssertionError(f'Expected strip_kernel={kernel} to be rejected.')

    for scale in (0.0, 1.0):
        try:
            MultiScaleLocalChannelRefiner(16, initial_scale=scale)
        except ValueError:
            pass
        else:
            raise AssertionError(f'Expected initial_scale={scale} to be rejected.')

    for scale in (-1.0, 1.0):
        try:
            MultiScaleLocalChannelRefiner(16, dense_initial_scale=scale)
        except ValueError:
            pass
        else:
            raise AssertionError(
                f'Expected dense_initial_scale={scale} to be rejected.'
            )


if __name__ == '__main__':
    test_zero_dense_scales_match_parallel_branches()
    test_initial_weights_gain_and_residual_identity()
    test_refiner_gradients_and_detached_diagnostics()
    test_stabilization_preserves_constructor_rng_consumption()
    test_stub_generator_diagnostics_and_parameter_budget()
    test_refiner_rejects_invalid_settings()
    print('Stabilized TF-LCA tests passed.')
