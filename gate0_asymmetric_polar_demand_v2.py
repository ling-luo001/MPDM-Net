"""Gate 0 for the evidence-conditioned asymmetric polar demand refiner.

This checks configuration, model contracts, gradient routing, checkpoint parity,
and small end-to-end generator execution. Structural stubs are explicitly
labeled and never count as native CUDA acceptance.
"""

import argparse
import copy
import json
import os
import pathlib

import gate0_asymmetric_polar_ziprefine as legacy_gate


ROOT = pathlib.Path(__file__).resolve().parent
CANDIDATE_RECIPE = (
    ROOT / 'recipes' / 'RD-Asymmetric-Polar-Demand-V2'
    / 'RD-Asymmetric-Polar-Demand-V2.yaml'
)
REFERENCE_RECIPE = (
    ROOT / 'recipes' / 'RD-Asymmetric-Polar-Anchor-Dense'
    / 'RD-Asymmetric-Polar-Anchor-Dense.yaml'
)


def _load_yaml(path):
    import yaml

    with path.open('r', encoding='utf-8') as stream:
        return yaml.safe_load(stream)


def _candidate_cfg():
    return _load_yaml(CANDIDATE_RECIPE)


def _baseline_cfg():
    cfg = _candidate_cfg()
    for key in tuple(cfg['model_cfg']):
        if key.startswith('asymmetric_polar_zip_refine_'):
            cfg['model_cfg'].pop(key)
    return cfg


def _small_cfg(enabled=True, checkpointing=False):
    cfg = _candidate_cfg() if enabled else _baseline_cfg()
    cfg['stft_cfg'].update({'n_fft': 30, 'win_size': 30})
    cfg['model_cfg']['pitch_candidates'] = 8
    if enabled:
        cfg['model_cfg'][
            'asymmetric_polar_zip_refine_activation_checkpointing'
        ] = checkpointing
    return cfg


def _make_evidence(refiner, noisy, base, requires_grad=False):
    import torch

    batch, _, frames, bins = noisy.shape
    encoded_bins = (bins + 1) // 2
    evidence = {
        'coarse_complex': torch.randn_like(base),
        'base_minus_coarse': torch.randn_like(base),
        'harmonic_prior': torch.rand(
            batch, 1, frames, bins, device=noisy.device
        ),
        'voicing_map': torch.rand(
            batch, 1, frames, bins, device=noisy.device
        ),
        'restoration_gates': torch.rand(
            batch, 2, frames, bins, device=noisy.device
        ),
        'mag_final': torch.randn(
            batch, refiner.evidence_specs['mag_final'][0], frames,
            encoded_bins, device=noisy.device,
        ),
        'restore_final': torch.randn(
            batch, refiner.evidence_specs['restore_final'][0], frames,
            encoded_bins, device=noisy.device,
        ),
        'suppress_bottleneck': torch.randn(
            batch, refiner.evidence_specs['suppress_bottleneck'][0],
            frames // 4, encoded_bins // 4, device=noisy.device,
        ),
        'restore_bottleneck': torch.randn(
            batch, refiner.evidence_specs['restore_bottleneck'][0],
            frames // 4, encoded_bins // 4, device=noisy.device,
        ),
    }
    if requires_grad:
        evidence = {
            name: value.detach().requires_grad_(True)
            for name, value in evidence.items()
        }
    return evidence


def _assert_config_and_manifests():
    candidate = _candidate_cfg()
    reference = _load_yaml(REFERENCE_RECIPE)
    assert candidate['data_cfg'] == reference['data_cfg']
    assert candidate['stft_cfg'] == reference['stft_cfg']
    assert candidate['training_cfg']['batch_size'] == 2
    assert candidate['training_cfg']['segment_size'] == 30600
    assert candidate['training_cfg']['use_PCS400'] is False
    assert candidate['model_cfg']['asymmetric_polar_zip_refine_enabled'] is True
    assert candidate['model_cfg']['asymmetric_polar_zip_refine_persistent_backbone'] is True
    assert candidate['model_cfg']['asymmetric_polar_zip_refine_oneway_anchor'] is False

    def load_json(relative):
        return json.loads((ROOT / relative).read_text(encoding='utf-8'))

    valid_clean = load_json(candidate['data_cfg']['valid_clean_json'])
    valid_noisy = load_json(candidate['data_cfg']['valid_noisy_json'])
    mini_clean = load_json('data/mini_val_clean_list.json')
    mini_noisy = load_json('data/mini_val_noisy_list.json')
    assert len(valid_clean) == len(valid_noisy) == 824
    assert len(mini_clean) == len(mini_noisy) == 165
    assert set(mini_clean) <= set(valid_clean)
    assert set(mini_noisy) <= set(valid_noisy)
    print('PASS data/STFT controls; validation=824 and mini validation=165 subset')


def _assert_refiner_contract(device):
    import torch
    from models.asymmetric_polar_zip_refine import AsymmetricPolarZipRefine

    refiner = AsymmetricPolarZipRefine(_small_cfg()).to(device).train()
    assert refiner.stage_modes == ('full', 'down', 'half', 'up')
    assert refiner.evidence_channels == 20
    assert [stage.interaction_position for stage in refiner.paired_stages] == [
        'none', 'compressed', 'compressed', 'full_resolution'
    ]
    assert refiner.compressed_mag_dense_bridge is not None

    noisy = torch.randn(1, 2, 8, 16, device=device, requires_grad=True)
    base = torch.randn(1, 2, 8, 16, device=device, requires_grad=True)
    evidence = _make_evidence(refiner, noisy, base, requires_grad=True)
    maps = refiner.build_evidence_input(noisy, base, evidence)
    assert maps.shape == (1, 20, 8, 16)
    output, aux = refiner(noisy, base, evidence)
    assert output.shape == base.shape and torch.isfinite(output).all()
    relative_initial_change = (
        (output - base).float().norm()
        / base.float().norm().clamp_min(1e-6)
    ).item()
    assert relative_initial_change < 0.05, relative_initial_change
    required_aux = {
        'base_complex', 'coarse_complex', 'raw_mag_multiplicative',
        'applied_mag_multiplicative', 'raw_mag_additive',
        'applied_mag_additive', 'corrected_magnitude',
        'magnitude_reference', 'raw_phase_delta', 'applied_phase_delta',
        'polar_complex', 'mag_demand_gate', 'phase_demand_gate',
        'ri_demand_gate', 'ri_residual_raw', 'ri_residual_applied',
        'ri_residual_ratio', 'refined_complex', 'asymmetric_context_scales',
    }
    assert required_aux <= aux.keys()
    for gate_name in ('mag_demand_gate', 'phase_demand_gate', 'ri_demand_gate'):
        gate = aux[gate_name]
        assert torch.all((gate > 0.0) & (gate < 1.0))
        assert 0.12 < gate.mean().item() < 0.26

    objective = output.square().mean()
    gradients = torch.autograd.grad(
        objective, (noisy, base, *evidence.values()), allow_unused=False
    )
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert all(gradient.abs().sum() > 0 for gradient in gradients)
    print(
        'PASS 20-map contract, nine evidence gradients, demand gates, finite '
        f'near-identity output; relative change={relative_initial_change:.3e}'
    )
    return refiner


def _assert_spectral_hole_recovery(device):
    import torch
    from models.asymmetric_polar_zip_refine import AsymmetricPolarZipRefine

    refiner = AsymmetricPolarZipRefine(_small_cfg()).to(device).eval()
    with torch.no_grad():
        refiner.delta_log_mag_head.weight.zero_()
        refiner.delta_log_mag_head.bias.zero_()
        refiner.delta_add_mag_head.weight.zero_()
        refiner.delta_add_mag_head.bias.fill_(1.0)
        refiner.mag_demand_head.weight.zero_()
        refiner.mag_demand_head.bias.fill_(4.0)
        refiner.phase_delta_head.weight.zero_()
        refiner.phase_delta_head.bias.zero_()
        refiner.ri_residual_head[-1].weight.zero_()
        refiner.ri_residual_head[-1].bias.zero_()
    noisy = torch.ones(1, 2, 8, 16, device=device)
    base = torch.zeros_like(noisy)
    evidence = _make_evidence(refiner, noisy, base)
    evidence['coarse_complex'].zero_()
    evidence['base_minus_coarse'].zero_()
    with torch.no_grad():
        output, aux = refiner(noisy, base, evidence)
    assert aux['corrected_magnitude'].min().item() > 0.1
    assert torch.linalg.vector_norm(output, dim=1).min().item() > 0.1
    print('PASS additive magnitude path restores exactly zero base bins')


def _assert_soft_vjp(device):
    import torch
    from models.generator import apply_asymmetric_polar_zip_refiner_soft_vjp

    torch.manual_seed(902)
    noisy = torch.randn(1, 2, 4, 4, device=device, requires_grad=True)
    base = torch.randn(1, 2, 4, 4, device=device, requires_grad=True)
    evidence = {
        f'evidence_{index}': torch.randn(
            1, index + 1, 2, 2, device=device, requires_grad=True
        )
        for index in range(9)
    }

    class ToyRefiner:
        def __call__(self, noisy_complex, base_complex, evidence_values):
            signal = 0.2 * noisy_complex + 0.7 * base_complex
            for index, value in enumerate(evidence_values.values(), start=1):
                signal = signal + (0.01 * index) * value.mean()
            return torch.sin(signal), {}

    refiner = ToyRefiner()
    cotangent = torch.randn_like(base)
    direct, _ = refiner(noisy, base, evidence)
    direct_gradients = torch.autograd.grad(
        (direct * cotangent).sum(),
        (noisy, base, *evidence.values()),
        retain_graph=True,
    )
    alpha = 0.3
    blended, _ = apply_asymmetric_polar_zip_refiner_soft_vjp(
        refiner, noisy, base, evidence=evidence, alpha=alpha
    )
    blended_gradients = torch.autograd.grad(
        (blended * cotangent).sum(), (noisy, base, *evidence.values())
    )
    assert torch.equal(direct, blended)
    expected = (
        direct_gradients[0],
        alpha * direct_gradients[1] + (1.0 - alpha) * cotangent,
        *(alpha * gradient for gradient in direct_gradients[2:]),
    )
    names = ('noisy', 'base', *evidence.keys())
    errors = {
        name: (actual - target).abs().max().item()
        for name, actual, target in zip(names, blended_gradients, expected)
    }
    max_error = max(errors.values())
    assert max_error < 1e-6, errors
    print(
        f'PASS alpha={alpha:g} full-evidence soft VJP algebra; '
        f'max error={max_error:.3e}'
    )


def _assert_checkpoint_parity(device):
    import torch
    import models.asymmetric_polar_zip_refine as refiner_module
    from models.asymmetric_polar_zip_refine import AsymmetricPolarZipRefine

    disabled = AsymmetricPolarZipRefine(
        _small_cfg(checkpointing=False)
    ).to(device).train()
    enabled = AsymmetricPolarZipRefine(
        _small_cfg(checkpointing=True)
    ).to(device).train()
    enabled.load_state_dict(disabled.state_dict())
    noisy = torch.randn(1, 2, 8, 16, device=device)
    base = torch.randn_like(noisy)
    evidence = _make_evidence(disabled, noisy, base)
    calls = []
    real_checkpoint = refiner_module.checkpoint

    def tracked(function, *arguments, **kwargs):
        calls.append(kwargs.copy())
        return real_checkpoint(function, *arguments, **kwargs)

    refiner_module.checkpoint = tracked
    try:
        enabled_output, _ = enabled(noisy, base, evidence)
    finally:
        refiner_module.checkpoint = real_checkpoint
    disabled_output, _ = disabled(noisy, base, evidence)
    assert len(calls) == 4
    assert all(call == {'use_reentrant': False} for call in calls)
    assert torch.equal(enabled_output, disabled_output)
    calls.clear()
    refiner_module.checkpoint = tracked
    try:
        enabled.eval()(noisy, base, evidence)
    finally:
        refiner_module.checkpoint = real_checkpoint
    assert not calls
    print('PASS four-stage checkpoint parity and inference bypass')


def _assert_generator_integration(device):
    import torch
    from models.generator import MambaSEUNet

    baseline_cfg = _baseline_cfg()
    candidate_cfg = _candidate_cfg()
    candidate_cfg['model_cfg'][
        'asymmetric_polar_zip_refine_activation_checkpointing'
    ] = False
    torch.manual_seed(903)
    baseline = MambaSEUNet(baseline_cfg)
    baseline_rng = torch.random.get_rng_state().clone()
    torch.manual_seed(903)
    candidate = MambaSEUNet(candidate_cfg)
    candidate_rng = torch.random.get_rng_state().clone()
    baseline_state = baseline.state_dict()
    candidate_state = candidate.state_dict()
    assert baseline_state.keys() <= candidate_state.keys()
    for key, value in baseline_state.items():
        assert torch.equal(value, candidate_state[key]), key
    assert torch.equal(baseline_rng, candidate_rng)

    small_candidate = MambaSEUNet(
        _small_cfg(enabled=True, checkpointing=False)
    ).to(device)

    noisy_mag = torch.rand(1, 16, 8, device=device) + 0.1
    noisy_phase = torch.randn(1, 16, 8, device=device)
    outputs = small_candidate(noisy_mag, noisy_phase)
    assert [tuple(value.shape) for value in outputs] == [
        (1, 16, 8), (1, 16, 8), (1, 16, 8, 2)
    ]
    assert all(torch.isfinite(value).all() for value in outputs)
    assert 'parent_base_complex' in small_candidate.latest_aux
    assert 'refiner_coarse_complex' in small_candidate.latest_aux
    parent_parameters = sum(parameter.numel() for parameter in baseline.parameters())
    total_parameters = sum(parameter.numel() for parameter in candidate.parameters())
    ratio = total_parameters / parent_parameters
    assert ratio <= 2.1, ratio
    print(
        f'PASS generator integration/RNG isolation; parent={parent_parameters:,}, '
        f'total={total_parameters:,}, ratio={ratio:.3f}x'
    )


def _assert_training_contract(device):
    import tempfile

    import torch
    from models.discriminator import MetricDiscriminator
    from models.generator import MambaSEUNet
    from train import (
        asymmetric_refiner_losses,
        clip_generator_gradients,
        load_parent_initialization,
        setup_optimizers,
        update_refiner_anchor_alpha,
    )

    cfg = _small_cfg(enabled=True, checkpointing=False)
    generator = MambaSEUNet(cfg).to(device).train()
    discriminator = MetricDiscriminator().to(device)
    optim_g, _ = setup_optimizers((generator, discriminator), cfg)
    parameter_groups = {
        id(parameter): group
        for group in optim_g.param_groups
        for parameter in group['params']
    }
    assert len(parameter_groups) == sum(
        1 for parameter in generator.parameters() if parameter.requires_grad
    )
    base_lr = cfg['training_cfg']['learning_rate']
    for name, parameter in generator.named_parameters():
        if not parameter.requires_grad:
            continue
        group = parameter_groups[id(parameter)]
        if getattr(parameter, '_no_weight_decay', False):
            assert group['weight_decay'] == 0.0, name
        if 'asymmetric_polar_zip_refiner.mag_demand_head' in name:
            assert group['lr'] == base_lr * 1.5, name

    alpha = update_refiner_anchor_alpha(generator, 1000, cfg)
    assert abs(alpha - 0.05) < 1e-8
    noisy_mag = torch.rand(1, 16, 8, device=device) + 0.1
    noisy_phase = torch.randn(1, 16, 8, device=device)
    _, _, generated_complex = generator(noisy_mag, noisy_phase)
    clean_complex = generated_complex.detach() + 0.01 * torch.randn_like(
        generated_complex
    )
    clean_magnitude = torch.linalg.vector_norm(clean_complex, dim=-1)
    losses = asymmetric_refiner_losses(
        generator, clean_magnitude, clean_complex
    )
    assert all(torch.isfinite(value) for value in losses.values())
    objective = sum(losses.values())
    optim_g.zero_grad(set_to_none=True)
    objective.backward()
    norms = clip_generator_gradients(generator, cfg)
    assert norms['parent'] > 0.0 and norms['refiner'] > 0.0
    assert generator.asymmetric_polar_zip_refiner.mag_stem[0].weight.grad is not None
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for name, parameter in generator.named_parameters()
        if 'asymmetric_polar_zip_refiner.' not in name
    )
    optim_g.step()

    source = MambaSEUNet(cfg).to(device)
    target = MambaSEUNet(cfg).to(device)
    with torch.no_grad():
        for name, parameter in source.named_parameters():
            parameter.fill_(
                0.25 if 'asymmetric_polar_zip_refiner.' not in name else 0.75
            )
        source.asymmetric_polar_zip_refine_anchor_alpha.fill_(0.75)
    target_anchor_before = (
        target.asymmetric_polar_zip_refine_anchor_alpha.detach().clone()
    )
    target_refiner_before = {
        name: value.detach().clone()
        for name, value in target.state_dict().items()
        if name.startswith('asymmetric_polar_zip_refiner.')
    }
    with tempfile.TemporaryDirectory() as directory:
        checkpoint_path = os.path.join(directory, 'parent_source.pth')
        torch.save({'generator': source.state_dict()}, checkpoint_path)
        load_parent_initialization(target, checkpoint_path, device, cfg)
    target_state = target.state_dict()
    for name, expected in target_refiner_before.items():
        assert torch.equal(target_state[name], expected), name
    assert torch.equal(
        target_state['asymmetric_polar_zip_refine_anchor_alpha'],
        target_anchor_before,
    )
    assert torch.equal(
        target_state['mag_encoder.dense_conv_1.0.weight'],
        source.state_dict()['mag_encoder.dense_conv_1.0.weight'],
    )
    print(
        'PASS optimizer partition, direct refiner losses, soft-anchor schedule, '
        'separate finite gradients, parent-only warm start'
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--require-native-cuda', action='store_true')
    parser.add_argument('--allow-structural-only', action='store_true')
    args = parser.parse_args()
    if args.require_native_cuda and args.allow_structural_only:
        parser.error('Choose native CUDA or structural-only mode, not both.')

    legacy_gate._install_import_stubs()
    legacy_gate._configure_runtime(args.require_native_cuda)
    device = legacy_gate.TEST_DEVICE
    _assert_config_and_manifests()
    _assert_refiner_contract(device)
    _assert_spectral_hole_recovery(device)
    _assert_soft_vjp(device)
    _assert_checkpoint_parity(device)
    _assert_generator_integration(device)
    _assert_training_contract(device)

    if legacy_gate.STRUCTURAL_STUB:
        print('LIMITATION structural stubs used; native selective-scan is unverified')
        if not args.allow_structural_only:
            raise SystemExit(
                'STRUCTURAL CHECK INCOMPLETE: rerun on native CUDA, or pass '
                '--allow-structural-only for a labeled structural result.'
            )
        print('STRUCTURAL CHECK PASS; NATIVE CUDA GATE 0 STILL REQUIRED')
    else:
        import torch

        torch.cuda.synchronize(device)
        peak_mib = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        print(f'NATIVE CUDA GATE 0 PASS; peak allocated={peak_mib:.1f} MiB')


if __name__ == '__main__':
    main()
