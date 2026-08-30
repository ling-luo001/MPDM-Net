"""Gate 0 for Demand-V3 Balanced Residual Routing.

Windows structural stubs are labeled and never count as native CUDA acceptance.
"""

import argparse
import json
import os
import pathlib
import tempfile

import gate0_asymmetric_polar_ziprefine as legacy_gate


ROOT = pathlib.Path(__file__).resolve().parent
CANDIDATE_RECIPE = (
    ROOT / 'recipes' / 'RD-Asymmetric-Polar-Demand-V3'
    / 'RD-Asymmetric-Polar-Demand-V3.yaml'
)
V2_RECIPE = (
    ROOT / 'recipes' / 'RD-Asymmetric-Polar-Demand-V2'
    / 'RD-Asymmetric-Polar-Demand-V2.yaml'
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


def _make_evidence(refiner, noisy, base):
    import torch

    batch, _, frames, bins = noisy.shape
    encoded_bins = (bins + 1) // 2
    return {
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


def _assert_config_and_data():
    candidate = _candidate_cfg()
    v2 = _load_yaml(V2_RECIPE)
    assert candidate['data_cfg'] == v2['data_cfg']
    assert candidate['stft_cfg'] == v2['stft_cfg']
    assert candidate['training_cfg']['batch_size'] == 2
    assert candidate['training_cfg']['segment_size'] == 30600
    assert candidate['training_cfg']['lr_decay'] == 0.98
    assert candidate['training_cfg']['ema'] == {
        'enabled': True, 'decay': 0.999
    }
    assert candidate['model_cfg'][
        'asymmetric_polar_zip_refine_anchor_schedule_epochs'
    ] == 24.0

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
    print('PASS unchanged data/STFT controls; validation=824, mini=165 subset')


def _assert_pure_cross_and_upskip(device):
    import torch
    import torch.nn as nn
    from models.asymmetric_polar_zip_refine import (
        _ProjectedCrossInteraction,
        _UpSkipTransition,
    )

    class SeparableScans(nn.Module):
        def forward(self, mag_value, phase_value):
            return 2.0 * mag_value, 3.0 * phase_value

    torch.manual_seed(3001)
    interaction = _ProjectedCrossInteraction(6, 4, 5, d_state=4).to(device)
    interaction.cross = SeparableScans()
    mag = torch.randn(1, 6, 4, 5, device=device, requires_grad=True)
    phase = torch.randn(1, 4, 4, 5, device=device, requires_grad=True)
    mag_output, phase_output = interaction(mag, phase)
    mag_residual = mag_output - mag
    phase_residual = phase_output - phase
    mag_self, mag_cross = torch.autograd.grad(
        mag_residual.sum(), (mag, phase), retain_graph=True
    )
    phase_cross, phase_self = torch.autograd.grad(
        phase_residual.sum(), (mag, phase)
    )
    assert mag_self.abs().max().item() < 1e-7
    assert phase_self.abs().max().item() < 1e-7
    assert mag_cross.abs().sum().item() > 0.0
    assert phase_cross.abs().sum().item() > 0.0
    assert torch.allclose(
        interaction.mag_layer_scale.values(),
        torch.full((6,), 0.03, device=device),
    )
    assert interaction.mag_layer_scale.max_scale == 0.25
    assert interaction.phase_layer_scale.max_scale == 0.25

    transition = _UpSkipTransition(6).to(device)
    with torch.no_grad():
        transition.layer_scale.logit.zero_()
    compressed = torch.randn(1, 6, 4, 4, device=device)
    skip = torch.randn(1, 6, 8, 8, device=device)
    assert torch.equal(transition(compressed, skip), skip)
    print('PASS behavioral pure-cross routing and zero-scale UpSkip identity')


def _assert_ri_budget(device):
    import torch
    from models.asymmetric_polar_zip_refine import AsymmetricPolarZipRefine

    refiner = AsymmetricPolarZipRefine(_small_cfg()).to(device).eval()
    assert not any(
        name.endswith('ri_residual_ratio_logit')
        for name, _ in refiner.named_parameters()
    )
    noisy = torch.full((1, 2, 8, 16), 4.0, device=device)
    base = torch.full_like(noisy, 0.5)
    evidence = _make_evidence(refiner, noisy, base)
    evidence['coarse_complex'].fill_(1.0)
    with torch.no_grad():
        _, aux = refiner(noisy, base, evidence)
    trusted = torch.maximum(
        torch.linalg.vector_norm(base, dim=1, keepdim=True),
        torch.linalg.vector_norm(
            evidence['coarse_complex'], dim=1, keepdim=True
        ),
    )
    noisy_mag = torch.linalg.vector_norm(noisy, dim=1, keepdim=True)
    trusted = trusted + refiner.magnitude_floor * noisy_mag.mean(
        dim=(2, 3), keepdim=True
    ).clamp_min(refiner.eps)
    expected_reference = trusted + 0.25 * torch.relu(noisy_mag - trusted)
    assert torch.allclose(aux['trusted_magnitude_reference'], trusted)
    assert torch.allclose(aux['magnitude_reference'], expected_reference)
    assert aux['ri_residual_ratio'].item() == 0.25
    budget = 0.25 * aux['magnitude_reference'] * aux['ri_demand_gate']
    assert torch.all(aux['ri_residual_applied'].abs() <= budget + 1e-7)
    print('PASS trusted magnitude reference and fixed per-bin RI 0.25 budget')


def _assert_schedules_and_optimizer(device):
    import torch
    from models.discriminator import MetricDiscriminator
    from models.generator import MambaSEUNet
    from train import (
        _generator_parameter_groups,
        refiner_intermediate_loss_multipliers,
        setup_optimizers,
        update_refiner_anchor_alpha,
    )

    cfg = _small_cfg()
    generator = MambaSEUNet(cfg).to(device)
    alpha_a = update_refiner_anchor_alpha(
        generator, 10, cfg, fractional_epoch=12.0
    )
    alpha_b = update_refiner_anchor_alpha(
        generator, 100000, cfg, fractional_epoch=12.0
    )
    assert alpha_a == alpha_b == 0.5
    assert update_refiner_anchor_alpha(
        generator, 0, cfg, fractional_epoch=24.0
    ) == 1.0

    v2 = _load_yaml(V2_RECIPE)
    assert 'asymmetric_polar_zip_refine_anchor_schedule_epochs' not in v2['model_cfg']
    v2_generator = MambaSEUNet(v2).to(device)
    assert v2_generator.asymmetric_polar_zip_refiner.ri_residual_ratio == 0.25
    assert abs(update_refiner_anchor_alpha(generator, 1000, v2) - 0.05) < 1e-8
    del v2_generator

    at_10 = refiner_intermediate_loss_multipliers(10.0, cfg)
    at_20 = refiner_intermediate_loss_multipliers(20.0, cfg)
    at_30 = refiner_intermediate_loss_multipliers(30.0, cfg)
    assert all(value == 1.0 for value in at_10.values())
    assert abs(at_20['magnitude'] - 0.65) < 1e-8
    assert abs(at_20['phase'] - (2.0 / 3.0)) < 1e-8
    assert at_30 == {
        'base_complex': 0.0,
        'magnitude': 0.30,
        'phase': 1.0 / 3.0,
        'polar_complex': 0.0,
        'ri_residual': 0.0,
        'demand': 0.0,
    }

    discriminator = MetricDiscriminator().to(device)
    optim_g, _ = setup_optimizers((generator, discriminator), cfg)
    groups = {
        id(parameter): group
        for group in optim_g.param_groups
        for parameter in group['params']
    }
    assert len(groups) == sum(p.requires_grad for p in generator.parameters())
    base_lr = cfg['training_cfg']['learning_rate']
    saw_gate_weight = saw_layer_scale = saw_ri_body = saw_ri_head = False
    for name, parameter in generator.named_parameters():
        if not parameter.requires_grad:
            continue
        group = groups[id(parameter)]
        if 'asymmetric_polar_zip_refiner.' in name:
            assert group['lr'] == base_lr * 0.75, name
        else:
            assert group['lr'] == base_lr, name
        if parameter.ndim > 1 and 'gate' in name.lower():
            assert group['weight_decay'] > 0.0, name
            saw_gate_weight = True
        if 'layer_scale.logit' in name:
            assert group['weight_decay'] == 0.0, name
            saw_layer_scale = True
        if any(token in name.lower() for token in (
            'x_proj_weight', 'dt_projs_weight', 'dt_projs_bias', 'a_logs'
        )):
            assert group['weight_decay'] == 0.0, name
        if 'ri_residual_head.1.weight' in name:
            assert group['group_name'].startswith('refiner_body_'), name
            saw_ri_body = True
        if 'ri_residual_head.3.weight' in name:
            assert group['group_name'].startswith('refiner_head_'), name
            saw_ri_head = True
    assert saw_gate_weight and saw_layer_scale and saw_ri_body and saw_ri_head
    assert _generator_parameter_groups(generator, cfg)
    print('PASS epoch schedules and exact LR/weight-decay parameter grouping')


def _assert_ema_roundtrip(device):
    import torch
    import torch.nn as nn
    import train as train_module

    model = nn.Sequential(nn.Linear(3, 4), nn.BatchNorm1d(4)).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    parameter_ids = tuple(id(parameter) for parameter in model.parameters())
    ema = train_module.GeneratorEMA(model, decay=0.5)
    initial = {name: value.clone() for name, value in ema.shadow.items()}
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(2.0)
    raw = {name: value.detach().clone() for name, value in model.state_dict().items()}

    real_ddp = train_module.DistributedDataParallel

    class FakeDDP(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

    train_module.DistributedDataParallel = FakeDDP
    try:
        ema.update(FakeDDP(model))
    finally:
        train_module.DistributedDataParallel = real_ddp
    for name, value in ema.shadow.items():
        if torch.is_floating_point(value):
            assert torch.allclose(value, 0.5 * initial[name] + 0.5 * raw[name])

    with ema.average_parameters(model):
        for name, value in model.state_dict().items():
            assert torch.equal(value, ema.shadow[name])
    for name, value in model.state_dict().items():
        assert torch.equal(value, raw[name])
    assert tuple(id(parameter) for parameter in model.parameters()) == parameter_ids
    assert all(
        id(parameter) in parameter_ids
        for group in optimizer.param_groups for parameter in group['params']
    )

    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, 'g_ema_roundtrip.pth')
        torch.save({
            'generator': model.state_dict(),
            'generator_ema': ema.state_dict(),
        }, path)
        checkpoint = torch.load(path, map_location=device)
        restored_model = nn.Sequential(
            nn.Linear(3, 4), nn.BatchNorm1d(4)
        ).to(device)
        restored_model.load_state_dict(checkpoint['generator'])
        restored_ema = train_module.GeneratorEMA(restored_model, decay=0.1)
        restored_ema.load_state_dict(checkpoint['generator_ema'])
    assert restored_ema.num_updates == 1
    assert restored_ema.decay == 0.5
    for name, value in ema.shadow.items():
        assert torch.equal(value, restored_ema.shadow[name])
    print('PASS EMA single/DDP update, swap/restore, optimizer identity, roundtrip')


def _assert_parent_structure_and_gradients(device):
    import torch
    from models.generator import MambaSEUNet

    torch.manual_seed(3002)
    baseline = MambaSEUNet(_baseline_cfg())
    baseline_rng = torch.random.get_rng_state().clone()
    torch.manual_seed(3002)
    candidate = MambaSEUNet(_candidate_cfg())
    candidate_rng = torch.random.get_rng_state().clone()
    baseline_state = baseline.state_dict()
    candidate_state = candidate.state_dict()
    assert baseline_state.keys() <= candidate_state.keys()
    for name, value in baseline_state.items():
        assert torch.equal(value, candidate_state[name]), name
    assert torch.equal(baseline_rng, candidate_rng)
    parent_parameters = sum(p.numel() for p in baseline.parameters())
    total_parameters = sum(p.numel() for p in candidate.parameters())
    assert total_parameters <= 3_600_000, total_parameters

    small = MambaSEUNet(_small_cfg()).to(device).train()
    small.set_asymmetric_refiner_anchor_alpha(1.0)
    noisy_mag = (torch.rand(1, 16, 8, device=device) + 0.1).requires_grad_()
    noisy_phase = torch.randn(1, 16, 8, device=device, requires_grad=True)
    outputs = small(noisy_mag, noisy_phase)
    objective = sum(output.float().square().mean() for output in outputs)
    objective.backward()
    parent_grad = sum(
        parameter.grad.detach().abs().sum().item()
        for name, parameter in small.named_parameters()
        if 'asymmetric_polar_zip_refiner.' not in name
        and parameter.grad is not None
    )
    refiner_grad = sum(
        parameter.grad.detach().abs().sum().item()
        for name, parameter in small.named_parameters()
        if 'asymmetric_polar_zip_refiner.' in name
        and parameter.grad is not None
    )
    assert all(torch.isfinite(output).all() for output in outputs)
    assert torch.isfinite(noisy_mag.grad).all()
    assert torch.isfinite(noisy_phase.grad).all()
    assert parent_grad > 0.0 and refiner_grad > 0.0
    print(
        f'PASS 7bc parent structure, finite forward/backward and gradients; '
        f'parent={parent_parameters:,}, total={total_parameters:,}'
    )
    return parent_parameters, total_parameters


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
    _assert_config_and_data()
    _assert_pure_cross_and_upskip(device)
    _assert_ri_budget(device)
    _assert_schedules_and_optimizer(device)
    _assert_ema_roundtrip(device)
    _assert_parent_structure_and_gradients(device)

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
