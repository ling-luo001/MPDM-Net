# Reference: https://github.com/RoyChao19477/SEMamba/train.py
# Reference: https://github.com/yxlu-0102/MP-SENet/blob/main/train.py

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import math
import os
import time
import argparse
import json
import yaml
from contextlib import contextmanager
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DistributedSampler, DataLoader
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
import sys

from dataloaders.dataloader_vctk import VCTKDemandDataset, Val_Dataset
from models.stfts import mag_phase_stft, mag_phase_istft
from models.generator import MambaSEUNet
from models.loss import pesq_score, phase_losses
from models.discriminator import MetricDiscriminator, batch_pesq
from utils.util import (
    load_checkpoint, load_ckpts, load_optimizer_states, save_checkpoint,
    build_env, load_config, initialize_seed, 
    print_gpu_info, log_model_info, initialize_process_group,
)

torch.backends.cudnn.benchmark = True


MINI_DATA_CFG = {
    'train_clean_json': 'data/mini_train_clean_list.json',
    'train_noisy_json': 'data/mini_train_noisy_list.json',
    'valid_clean_json': 'data/mini_val_clean_list.json',
    'valid_noisy_json': 'data/mini_val_noisy_list.json',
}


def _unwrap_model(model):
    return model.module if isinstance(model, DistributedDataParallel) else model


class GeneratorEMA:
    """EMA shadow state that never replaces raw optimizer parameters."""

    def __init__(self, model, decay=0.999):
        if not 0.0 <= decay < 1.0:
            raise ValueError('EMA decay must be in [0, 1).')
        self.decay = float(decay)
        self.num_updates = 0
        core = _unwrap_model(model)
        self.shadow = {
            name: value.detach().clone()
            for name, value in core.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model):
        current = _unwrap_model(model).state_dict()
        if current.keys() != self.shadow.keys():
            raise RuntimeError('EMA/model state keys differ during update.')
        for name, value in current.items():
            shadow_value = self.shadow[name]
            if torch.is_floating_point(shadow_value):
                shadow_value.mul_(self.decay).add_(
                    value.detach(), alpha=1.0 - self.decay
                )
            else:
                shadow_value.copy_(value.detach())
        self.num_updates += 1

    def state_dict(self):
        return {
            'decay': self.decay,
            'num_updates': self.num_updates,
            'shadow': {
                name: value.detach().clone()
                for name, value in self.shadow.items()
            },
        }

    def load_state_dict(self, state):
        if not isinstance(state, dict) or 'shadow' not in state:
            raise ValueError('EMA checkpoint must contain a shadow state.')
        incoming = state['shadow']
        if incoming.keys() != self.shadow.keys():
            raise RuntimeError('EMA checkpoint/model state keys differ.')
        for name, value in incoming.items():
            target = self.shadow[name]
            if target.shape != value.shape:
                raise RuntimeError(
                    f'EMA tensor shape mismatch for {name}: '
                    f'{tuple(value.shape)} != {tuple(target.shape)}.'
                )
            target.copy_(value.detach().to(device=target.device, dtype=target.dtype))
        self.decay = float(state.get('decay', self.decay))
        if not 0.0 <= self.decay < 1.0:
            raise ValueError('EMA checkpoint decay must be in [0, 1).')
        self.num_updates = int(state.get('num_updates', 0))

    @contextmanager
    def average_parameters(self, model):
        core = _unwrap_model(model)
        raw_state = {
            name: value.detach().clone()
            for name, value in core.state_dict().items()
        }
        core.load_state_dict(self.shadow, strict=True)
        try:
            yield core
        finally:
            core.load_state_dict(raw_state, strict=True)


def harmonic_generation_losses(generator, clean_mag, clean_com):
    """Auxiliary objectives for suppression, source analysis, and restoration."""
    aux = generator.latest_aux
    required_keys = {
        'coarse_complex',
        'pitch_posterior',
        'voicing',
        'harmonic_prior',
        'harmonic_residual',
        'aperiodic_residual',
    }
    missing_keys = required_keys.difference(aux)
    if missing_keys:
        raise RuntimeError(f'Missing generator auxiliary outputs: {sorted(missing_keys)}')

    loss_coarse_complex = F.mse_loss(clean_com, aux['coarse_complex']) * 2
    with torch.no_grad():
        clean_pitch, _, clean_pitch_confidence = generator.harmonic_analysis(clean_mag)
        clean_voicing_target = clean_pitch_confidence.pow(
            generator.voicing_confidence_power
        )

    predicted_pitch = aux['pitch_posterior'].clamp_min(1e-8)
    loss_pitch = F.kl_div(
        predicted_pitch.log(), clean_pitch, reduction='none'
    ).sum(dim=-1).mean()
    predicted_voicing = aux['voicing'].squeeze(1).squeeze(-1)
    loss_voicing = F.mse_loss(predicted_voicing, clean_voicing_target)

    harmonic_prior = (
        aux['harmonic_prior'].squeeze(1).permute(0, 2, 1).unsqueeze(-1).detach()
    )
    harmonic_residual = aux['harmonic_residual'].permute(0, 3, 2, 1)
    aperiodic_residual = aux['aperiodic_residual'].permute(0, 3, 2, 1)
    harmonic_leakage = (
        harmonic_residual.abs() * (1.0 - harmonic_prior)
    ).mean()
    aperiodic_leakage = (
        aperiodic_residual.abs() * harmonic_prior
    ).mean()
    loss_harmonic_support = harmonic_leakage + aperiodic_leakage
    return {
        'coarse_complex': loss_coarse_complex,
        'pitch': loss_pitch,
        'voicing': loss_voicing,
        'harmonic_support': loss_harmonic_support,
        'voicing_target_mean': clean_voicing_target.mean(),
    }


def asymmetric_refiner_losses(generator, clean_mag, clean_com):
    """Directly supervise the residual demand and each serial correction step."""
    refiner = getattr(generator, 'asymmetric_polar_zip_refiner', None)
    if refiner is None:
        zero = clean_mag.new_zeros(())
        return {
            'base_complex': zero,
            'magnitude': zero,
            'phase': zero,
            'polar_complex': zero,
            'ri_residual': zero,
            'demand': zero,
        }

    aux = generator.latest_aux
    required_keys = {
        'base_complex',
        'parent_base_complex',
        'corrected_magnitude',
        'applied_phase_delta',
        'polar_complex',
        'mag_demand_gate',
        'phase_demand_gate',
        'ri_demand_gate',
        'ri_residual_applied',
    }
    missing = required_keys.difference(aux)
    if missing:
        raise RuntimeError(
            f'Missing asymmetric-refiner auxiliary outputs: {sorted(missing)}'
        )
    if clean_com.ndim != 4 or clean_com.shape[-1] != 2:
        raise ValueError(
            f'clean_com must have shape [B, F, T, 2], got {tuple(clean_com.shape)}.'
        )

    eps = float(getattr(refiner, 'eps', 1e-6))
    clean_complex = clean_com.permute(0, 3, 2, 1).contiguous()
    clean_magnitude = clean_mag.permute(0, 2, 1).unsqueeze(1).contiguous()
    base_complex = aux['base_complex']
    parent_base_complex = aux['parent_base_complex']
    base_magnitude = torch.linalg.vector_norm(
        base_complex, dim=1, keepdim=True
    )
    polar_complex = aux['polar_complex']
    polar_magnitude = torch.linalg.vector_norm(
        polar_complex, dim=1, keepdim=True
    )

    loss_base_complex = F.mse_loss(parent_base_complex, clean_complex) * 2
    loss_magnitude = F.smooth_l1_loss(
        aux['corrected_magnitude'], clean_magnitude
    )

    base_real, base_imag = torch.chunk(base_complex, 2, dim=1)
    clean_real, clean_imag = torch.chunk(clean_complex, 2, dim=1)
    target_phase_delta = torch.atan2(
        base_real * clean_imag - base_imag * clean_real,
        base_real * clean_real + base_imag * clean_imag,
    ).detach()
    phase_confidence = (
        2.0 * base_magnitude * clean_magnitude
        / (base_magnitude.square() + clean_magnitude.square() + eps)
    ).detach()
    phase_error = 1.0 - torch.cos(
        aux['applied_phase_delta'] - target_phase_delta
    )
    loss_phase = (
        phase_error * phase_confidence
    ).sum() / phase_confidence.sum().clamp_min(eps)

    loss_polar_complex = F.mse_loss(polar_complex, clean_complex) * 2
    remaining_residual = (clean_complex - polar_complex).detach()
    loss_ri_residual = F.smooth_l1_loss(
        aux['ri_residual_applied'], remaining_residual
    )

    magnitude_demand_target = (
        (clean_magnitude - base_magnitude).abs()
        / (clean_magnitude + base_magnitude + eps)
    ).detach().clamp(0.0, 1.0)
    phase_demand_target = (
        target_phase_delta.abs() / torch.pi * phase_confidence
    ).detach().clamp(0.0, 1.0)
    ri_demand_target = (
        torch.linalg.vector_norm(remaining_residual, dim=1, keepdim=True)
        / (clean_magnitude + polar_magnitude + eps)
    ).detach().clamp(0.0, 1.0)

    def binary_demand_loss(prediction, target):
        return F.binary_cross_entropy(
            prediction.clamp(eps, 1.0 - eps), target
        )

    loss_demand = (
        binary_demand_loss(aux['mag_demand_gate'], magnitude_demand_target)
        + binary_demand_loss(aux['phase_demand_gate'], phase_demand_target)
        + binary_demand_loss(aux['ri_demand_gate'], ri_demand_target)
    ) / 3.0
    return {
        'base_complex': loss_base_complex,
        'magnitude': loss_magnitude,
        'phase': loss_phase,
        'polar_complex': loss_polar_complex,
        'ri_residual': loss_ri_residual,
        'demand': loss_demand,
    }


def _is_no_weight_decay_parameter(name, parameter):
    """Keep SSM dynamics, normalization, and residual routing out of decay."""
    if parameter.ndim <= 1 or getattr(parameter, '_no_weight_decay', False):
        return True
    lowered = name.lower()
    return (
        lowered.endswith('bias')
        or '.norm' in lowered
        or 'normalization' in lowered
        or 'x_proj_weight' in lowered
        or 'dt_projs_weight' in lowered
        or 'a_log' in lowered
        or 'a_logs' in lowered
    )


def _is_refiner_head_parameter(name):
    marker = 'asymmetric_polar_zip_refiner.'
    relative = name.split(marker, 1)[-1]
    head_modules = (
        'delta_log_mag_head.',
        'delta_add_mag_head.',
        'phase_delta_head.',
        'mag_demand_head.',
        'phase_demand_head.',
        'ri_residual_head.3.',
        'ri_demand_head.',
    )
    return relative.startswith(head_modules)


def _generator_parameter_groups(generator, cfg):
    optimizer_cfg = cfg['training_cfg'].get('optimizer', {})
    base_lr = float(cfg['training_cfg']['learning_rate'])
    weight_decay = float(optimizer_cfg.get('weight_decay', 0.01))
    parent_lr_scale = float(optimizer_cfg.get('parent_lr_scale', 1.0))
    refiner_lr_scale = float(optimizer_cfg.get('refiner_lr_scale', 1.0))
    head_lr_scale = float(optimizer_cfg.get('refiner_head_lr_scale', 1.0))
    if min(parent_lr_scale, refiner_lr_scale, head_lr_scale) <= 0.0:
        raise ValueError('All generator optimizer learning-rate scales must be positive.')
    if weight_decay < 0.0:
        raise ValueError('training_cfg.optimizer.weight_decay must be non-negative.')

    grouped = {}
    seen = set()
    for name, parameter in generator.named_parameters():
        if not parameter.requires_grad:
            continue
        parameter_id = id(parameter)
        if parameter_id in seen:
            raise RuntimeError(f'Duplicate generator parameter in optimizer: {name}')
        seen.add(parameter_id)

        is_refiner = 'asymmetric_polar_zip_refiner.' in name
        if is_refiner and _is_refiner_head_parameter(name):
            role = 'refiner_head'
            lr_scale = head_lr_scale
        elif is_refiner:
            role = 'refiner_body'
            lr_scale = refiner_lr_scale
        else:
            role = 'parent'
            lr_scale = parent_lr_scale
        decay = not _is_no_weight_decay_parameter(name, parameter)
        key = (role, decay)
        grouped.setdefault(key, []).append(parameter)

    expected = {id(parameter) for parameter in generator.parameters() if parameter.requires_grad}
    if seen != expected:
        raise RuntimeError('Generator optimizer parameter partition is incomplete.')

    groups = []
    for (role, decay), parameters in grouped.items():
        lr_scale = {
            'parent': parent_lr_scale,
            'refiner_body': refiner_lr_scale,
            'refiner_head': head_lr_scale,
        }[role]
        groups.append({
            'params': parameters,
            'lr': base_lr * lr_scale,
            'initial_lr': base_lr * lr_scale,
            'lr_scale': lr_scale,
            'weight_decay': weight_decay if decay else 0.0,
            'group_name': f"{role}_{'decay' if decay else 'no_decay'}",
        })
    return groups


def setup_optimizers(models, cfg):
    """Set up optimizers for the models."""
    generator, discriminator = models
    learning_rate = cfg['training_cfg']['learning_rate']
    betas = (cfg['training_cfg']['adam_b1'], cfg['training_cfg']['adam_b2'])

    generator_groups = _generator_parameter_groups(generator, cfg)
    discriminator_weight_decay = float(
        cfg['training_cfg'].get('optimizer', {}).get(
            'discriminator_weight_decay', 0.01
        )
    )
    optim_g = optim.AdamW(generator_groups, lr=learning_rate, betas=betas)
    optim_d = optim.AdamW(
        discriminator.parameters(),
        lr=learning_rate,
        betas=betas,
        weight_decay=discriminator_weight_decay,
    )

    return optim_g, optim_d

def setup_schedulers(optimizers, cfg, last_epoch):
    """Set up learning rate schedulers."""
    optim_g, optim_d = optimizers
    lr_decay = cfg['training_cfg']['lr_decay']

    scheduler_g = optim.lr_scheduler.ExponentialLR(optim_g, gamma=lr_decay)
    scheduler_d = optim.lr_scheduler.ExponentialLR(optim_d, gamma=lr_decay)
    if last_epoch >= 0:
        scheduler_g.last_epoch = last_epoch
        scheduler_d.last_epoch = last_epoch

    return scheduler_g, scheduler_d


def update_refiner_anchor_alpha(generator, steps, cfg, fractional_epoch=None):
    """Blend identity and joint-refiner VJPs without changing forward values."""
    if getattr(generator, 'asymmetric_polar_zip_refiner', None) is None:
        return 1.0
    model_cfg = cfg['model_cfg']
    start = float(
        model_cfg.get('asymmetric_polar_zip_refine_anchor_alpha_start', 0.0)
    )
    end = float(
        model_cfg.get('asymmetric_polar_zip_refine_anchor_alpha_end', 1.0)
    )
    if not 0.0 <= start <= 1.0 or not 0.0 <= end <= 1.0:
        raise ValueError('Refiner anchor alpha endpoints must be in [0, 1].')
    schedule_epochs = model_cfg.get(
        'asymmetric_polar_zip_refine_anchor_schedule_epochs'
    )
    if schedule_epochs is not None:
        schedule_epochs = float(schedule_epochs)
        if schedule_epochs < 0.0:
            raise ValueError('Refiner anchor schedule epochs must be non-negative.')
        if fractional_epoch is None:
            raise ValueError('Fractional epoch is required by the anchor schedule.')
        progress = (
            1.0 if schedule_epochs == 0.0
            else min(1.0, max(0.0, float(fractional_epoch)) / schedule_epochs)
        )
    else:
        schedule_steps = int(
            model_cfg.get('asymmetric_polar_zip_refine_anchor_schedule_steps', 20000)
        )
        if schedule_steps < 0:
            raise ValueError('Refiner anchor schedule steps must be non-negative.')
        progress = 1.0 if schedule_steps == 0 else min(1.0, steps / schedule_steps)
    alpha = start + (end - start) * progress
    generator.set_asymmetric_refiner_anchor_alpha(alpha)
    return alpha


def refiner_intermediate_loss_multipliers(fractional_epoch, cfg):
    """Epoch-based V3 direct-supervision taper with a legacy all-one fallback."""
    names = (
        'base_complex', 'magnitude', 'phase', 'polar_complex',
        'ri_residual', 'demand',
    )
    schedule = cfg['training_cfg'].get('refiner_intermediate_schedule')
    if schedule is None:
        return {name: 1.0 for name in names}
    full_until = float(schedule.get('full_weight_until_epoch', 10.0))
    taper_until = float(schedule.get('taper_until_epoch', 30.0))
    if full_until < 0.0 or taper_until <= full_until:
        raise ValueError('Invalid refiner intermediate supervision epoch bounds.')
    final = {
        'base_complex': float(schedule.get('final_base_complex', 0.0)),
        'magnitude': float(schedule.get('final_magnitude', 0.30)),
        'phase': float(schedule.get('final_phase', 1.0 / 3.0)),
        'polar_complex': float(schedule.get('final_polar_complex', 0.0)),
        'ri_residual': float(schedule.get('final_ri_residual', 0.0)),
        'demand': float(schedule.get('final_demand', 0.0)),
    }
    if any(not 0.0 <= value <= 1.0 for value in final.values()):
        raise ValueError('Refiner intermediate supervision multipliers must be in [0, 1].')
    epoch = max(0.0, float(fractional_epoch))
    if epoch <= full_until:
        blend = 1.0
    elif epoch >= taper_until:
        blend = 0.0
    else:
        progress = (epoch - full_until) / (taper_until - full_until)
        blend = 0.5 * (1.0 + math.cos(math.pi * progress))
    return {
        name: final[name] + (1.0 - final[name]) * blend
        for name in names
    }


def clip_generator_gradients(generator, cfg):
    """Clip the established parent and new refiner independently."""
    training_cfg = cfg['training_cfg']
    default_max_norm = float(training_cfg.get('max_grad_norm', 5.0))
    parent_max_norm = float(training_cfg.get('parent_max_grad_norm', default_max_norm))
    refiner_max_norm = float(training_cfg.get('refiner_max_grad_norm', default_max_norm))
    parent_parameters = []
    refiner_parameters = []
    for name, parameter in generator.named_parameters():
        if parameter.grad is None:
            continue
        if 'asymmetric_polar_zip_refiner.' in name:
            refiner_parameters.append(parameter)
        else:
            parent_parameters.append(parameter)

    norms = {'parent': 0.0, 'refiner': 0.0}
    for key, parameters, max_norm in (
        ('parent', parent_parameters, parent_max_norm),
        ('refiner', refiner_parameters, refiner_max_norm),
    ):
        if not parameters:
            continue
        if max_norm > 0.0:
            norm = torch.nn.utils.clip_grad_norm_(
                parameters, max_norm, error_if_nonfinite=True
            )
        else:
            norm = torch.linalg.vector_norm(torch.stack([
                parameter.grad.detach().float().norm(2)
                for parameter in parameters
            ]), 2)
        norms[key] = float(norm.detach().item())
    return norms


def load_parent_initialization(generator, checkpoint_path, device, cfg):
    """Warm-start only the established parent while leaving the refiner new."""
    checkpoint = load_checkpoint(checkpoint_path, device)
    state_dict = checkpoint.get('generator', checkpoint)
    refiner_prefix = 'asymmetric_polar_zip_refiner.'
    refiner_buffer = 'asymmetric_polar_zip_refine_anchor_alpha'
    parent_state_dict = {
        key: value for key, value in state_dict.items()
        if not key.startswith(refiner_prefix) and key != refiner_buffer
    }
    incompatible = generator.load_state_dict(parent_state_dict, strict=False)
    unexpected = list(incompatible.unexpected_keys)
    disallowed_missing = [
        key for key in incompatible.missing_keys
        if not key.startswith(refiner_prefix) and key != refiner_buffer
    ]
    if unexpected or disallowed_missing:
        raise RuntimeError(
            'Parent initialization is not architecture-compatible: '
            f'missing={disallowed_missing}, unexpected={unexpected}'
        )
    optimizer_cfg = cfg['training_cfg'].setdefault('optimizer', {})
    optimizer_cfg['parent_lr_scale'] = float(
        optimizer_cfg.get('warm_start_parent_lr_scale', 0.2)
    )
    print(
        f'Initialized parent from {checkpoint_path}; '
        f'new refiner tensors={len(incompatible.missing_keys)}, '
        f'parent_lr_scale={optimizer_cfg["parent_lr_scale"]:g}',
        flush=True,
    )


def create_val_dataset(cfg, train=True, split=True, device='cuda:0'):
    """Create dataset based on cfguration."""
    clean_json = cfg['data_cfg']['train_clean_json'] if train else cfg['data_cfg']['valid_clean_json']
    noisy_json = cfg['data_cfg']['train_noisy_json'] if train else cfg['data_cfg']['valid_noisy_json']
    shuffle = (cfg['env_setting']['num_gpus'] <= 1) if train else False
    pcs = cfg['training_cfg']['use_PCS400'] if train else False

    return Val_Dataset(
        clean_json=clean_json,
        noisy_json=noisy_json,
        sampling_rate=cfg['stft_cfg']['sampling_rate'],
        segment_size=cfg['training_cfg']['segment_size'],
        n_fft=cfg['stft_cfg']['n_fft'],
        hop_size=cfg['stft_cfg']['hop_size'],
        win_size=cfg['stft_cfg']['win_size'],
        compress_factor=cfg['model_cfg']['compress_factor'],
        split=split,
        n_cache_reuse=0,
        shuffle=shuffle,
        device=device,
        pcs=pcs
    )


def create_dataset(cfg, train=True, split=True, device='cuda:0'):
    """Create dataset based on cfguration."""
    clean_json = cfg['data_cfg']['train_clean_json'] if train else cfg['data_cfg']['valid_clean_json']
    noisy_json = cfg['data_cfg']['train_noisy_json'] if train else cfg['data_cfg']['valid_noisy_json']
    shuffle = (cfg['env_setting']['num_gpus'] <= 1) if train else False
    pcs = cfg['training_cfg']['use_PCS400'] if train else False
    
    return VCTKDemandDataset(
        clean_json=clean_json,
        noisy_json=noisy_json,
        sampling_rate=cfg['stft_cfg']['sampling_rate'],
        segment_size=cfg['training_cfg']['segment_size'],
        n_fft=cfg['stft_cfg']['n_fft'],
        hop_size=cfg['stft_cfg']['hop_size'],
        win_size=cfg['stft_cfg']['win_size'],
        compress_factor=cfg['model_cfg']['compress_factor'],
        split=split,
        n_cache_reuse=0,
        shuffle=shuffle,
        device=device,
        pcs=pcs
    )

def create_dataloader(dataset, cfg, train=True):
    """Create dataloader based on dataset and configuration."""
    if cfg['env_setting']['num_gpus'] > 1:
        sampler = DistributedSampler(dataset)
        sampler.set_epoch(cfg['training_cfg']['training_epochs'])
        batch_size = (cfg['training_cfg']['batch_size'] // cfg['env_setting']['num_gpus']) if train else 1
    else:
        sampler = None
        batch_size = cfg['training_cfg']['batch_size'] if train else 1
    num_workers = cfg['env_setting']['num_workers']

    return DataLoader(
        dataset,
        num_workers=num_workers,
        shuffle=(sampler is None) and train,
        sampler=sampler,
        batch_size=batch_size,
        pin_memory=True,
        drop_last=True
    )


def validate_generator(
    generator, validation_loader, cfg, device, n_fft, hop_size, win_size,
    compress_factor,
):
    """Run the existing validation membership and metrics for one model state."""
    generator.eval()
    torch.cuda.empty_cache()
    audios_r, audios_g = [], []
    val_mag_err_tot = 0.0
    val_pha_err_tot = 0.0
    val_com_err_tot = 0.0
    with torch.no_grad():
        for j, batch in enumerate(validation_loader):
            clean_audio, clean_mag, clean_pha, clean_com, noisy_audio = batch
            clean_audio = clean_audio.to(device, non_blocking=True)
            clean_mag = clean_mag.to(device, non_blocking=True)
            clean_pha = clean_pha.to(device, non_blocking=True)
            clean_com = clean_com.to(device, non_blocking=True)
            noisy_audio = noisy_audio.to(device, non_blocking=True)
            orig_size = noisy_audio.size(1)

            if noisy_audio.size(1) >= cfg['training_cfg']['segment_size']:
                last_segment_size = (
                    noisy_audio.size(1) % cfg['training_cfg']['segment_size']
                )
                if last_segment_size > 0:
                    last_segment = noisy_audio[
                        :, -cfg['training_cfg']['segment_size']:
                    ]
                    noisy_audio = noisy_audio[:, :-last_segment_size]
                    segments = list(torch.split(
                        noisy_audio, cfg['training_cfg']['segment_size'], dim=1
                    ))
                    segments.append(last_segment)
                    reshape_last = True
                else:
                    segments = torch.split(
                        noisy_audio, cfg['training_cfg']['segment_size'], dim=1
                    )
                    reshape_last = False
            else:
                padded_zeros = torch.zeros(
                    1,
                    cfg['training_cfg']['segment_size'] - noisy_audio.size(1),
                    device=device,
                )
                segments = [torch.cat((noisy_audio, padded_zeros), dim=1)]
                reshape_last = False

            processed_segments = []
            for segment_index, segment in enumerate(segments):
                noisy_amp, noisy_pha, _ = mag_phase_stft(
                    segment, n_fft, hop_size, win_size, compress_factor
                )
                amp_g, pha_g, _ = generator(
                    noisy_amp.to(device, non_blocking=True),
                    noisy_pha.to(device, non_blocking=True),
                )
                audio_g = mag_phase_istft(
                    amp_g, pha_g, n_fft, hop_size, win_size, compress_factor
                ).squeeze()
                if reshape_last and segment_index == len(segments) - 2:
                    audio_g = audio_g[:- (
                        cfg['training_cfg']['segment_size'] - last_segment_size
                    )]
                processed_segments.append(audio_g)

            audio_g = torch.cat(processed_segments, dim=-1)[:orig_size]
            mag_g, pha_g, com_g = mag_phase_stft(
                audio_g, n_fft, hop_size, win_size, compress_factor
            )
            mag_g = mag_g.to(device, non_blocking=True).squeeze()
            pha_g = pha_g.to(device, non_blocking=True).unsqueeze(0)
            com_g = com_g.to(device, non_blocking=True)
            clean_mag = clean_mag.squeeze()
            clean_com = clean_com.squeeze()
            audios_r += torch.split(clean_audio, 1, dim=0)
            audios_g += torch.split(audio_g.unsqueeze(0), 1, dim=0)
            val_mag_err_tot += F.mse_loss(clean_mag, mag_g).item()
            val_ip_err, val_gd_err, val_iaf_err = phase_losses(
                clean_pha, pha_g, cfg
            )
            val_pha_err_tot += (val_ip_err + val_gd_err + val_iaf_err).item()
            val_com_err_tot += F.mse_loss(clean_com, com_g).item()

    batches = j + 1
    return {
        'pesq': pesq_score(audios_r, audios_g, cfg).item(),
        'magnitude': val_mag_err_tot / batches,
        'phase': val_pha_err_tot / batches,
        'complex': val_com_err_tot / batches,
    }


def train(rank, args, cfg):
    num_gpus = cfg['env_setting']['num_gpus']
    n_fft, hop_size, win_size = cfg['stft_cfg']['n_fft'], cfg['stft_cfg']['hop_size'], cfg['stft_cfg']['win_size']
    compress_factor = cfg['model_cfg']['compress_factor']
    batch_size = cfg['training_cfg']['batch_size'] // cfg['env_setting']['num_gpus']
    if num_gpus >= 1:
        initialize_process_group(cfg, rank)
        device = torch.device('cuda:{:d}'.format(rank))
    else:
        raise RuntimeError("Mamba needs GPU acceleration")

    generator = MambaSEUNet(cfg).to(device)
    discriminator = MetricDiscriminator().to(device)

    if rank == 0:
        log_model_info(rank, generator, args.exp_path)

    state_dict_g, state_dict_do, steps, last_epoch = load_ckpts(args, device)
    if args.init_parent_checkpoint is not None and state_dict_g is not None:
        raise ValueError(
            '--init_parent_checkpoint cannot be combined with checkpoint resume.'
        )
    if state_dict_g is not None:
        generator.load_state_dict(state_dict_g['generator'], strict=False)
        discriminator.load_state_dict(state_dict_do['discriminator'], strict=False)
    elif args.init_parent_checkpoint is not None:
        load_parent_initialization(
            generator, args.init_parent_checkpoint, device, cfg
        )

    ema_cfg = cfg['training_cfg'].get('ema', {})
    ema_enabled = bool(ema_cfg.get('enabled', False))
    generator_ema = (
        GeneratorEMA(generator, decay=float(ema_cfg.get('decay', 0.999)))
        if ema_enabled else None
    )
    if generator_ema is not None and state_dict_g is not None:
        ema_state = state_dict_g.get('generator_ema')
        if ema_state is not None:
            generator_ema.load_state_dict(ema_state)
        else:
            warnings.warn(
                'Resume checkpoint has no EMA state; EMA starts from raw generator.',
                UserWarning,
            )

    if num_gpus > 1 and torch.cuda.is_available():
        generator = DistributedDataParallel(generator, device_ids=[rank]).to(device)
        discriminator = DistributedDataParallel(discriminator, device_ids=[rank]).to(device)
    generator_core = generator.module if isinstance(
        generator, DistributedDataParallel
    ) else generator

    # Create optimizer and schedulers
    optimizers = setup_optimizers((generator, discriminator), cfg)
    load_optimizer_states(optimizers, state_dict_do, cfg, args.resume_lr)
    # # 确保 optimizers 的顺序是 (generator_optim, discriminator_optim)
    # optim_g, optim_d = optimizers  # 显式解包，避免直接使用索引
    #
    # # 加载优化器状态（如果有）
    # if state_dict_do is not None:
    #     # 过滤函数
    #     def _filter_optim_state(optimizer, old_state_dict):
    #         new_state = optimizer.state_dict()
    #         current_params = {id(p): name for name, p in optimizer.named_parameters()}
    #         filtered_state = {"state": {}, "param_groups": new_state["param_groups"]}
    #         for param_id, param_state in old_state_dict["state"].items():
    #             if param_id in current_params:
    #                 filtered_state["state"][param_id] = param_state
    #         return filtered_state
    #
    #     # 加载生成器优化器状态
    #     filtered_g = _filter_optim_state(optim_g, state_dict_do["optim_g"])
    #     optim_g.load_state_dict(filtered_g)
    #
    #     # 加载判别器优化器状态
    #     filtered_d = _filter_optim_state(optim_d, state_dict_do["optim_d"])
    #     optim_d.load_state_dict(filtered_d)
    #
    # 创建学习率调度器
    # scheduler_g, scheduler_d = setup_schedulers((optim_g, optim_d), cfg, last_epoch)
    optim_g, optim_d = optimizers
    scheduler_g, scheduler_d = setup_schedulers(optimizers, cfg, last_epoch)
    max_grad_norm = cfg['training_cfg'].get('max_grad_norm', 5.0)

    # Create trainset and train_loader
    trainset = create_dataset(cfg, train=True, split=True, device=device)
    train_loader = create_dataloader(trainset, cfg, train=True)

    # Create validset and validation_loader if rank is 0
    if rank == 0:
        validset = create_val_dataset(cfg, train=False, split=False, device=device)
        validation_loader = create_dataloader(validset, cfg, train=False)
        sw = SummaryWriter(os.path.join(args.exp_path, 'logs'))

    generator.train()
    discriminator.train()

    best_pesq, best_pesq_step = 0.0, 0
    for epoch in range(max(0, last_epoch), cfg['training_cfg']['training_epochs']):
        if rank == 0:
            start = time.time()
            print("Epoch: {}".format(epoch+1))

        for i, batch in enumerate(train_loader):
            if args.max_steps is not None and steps >= args.max_steps:
                if rank == 0:
                    sw.close()
                    print(f'Reached max_steps={args.max_steps}; training stopped cleanly.')
                return
            if rank == 0:
                start_b = time.time()
            if len(batch) == 7:
                clean_audio, clean_mag, clean_pha, clean_com, noisy_audio, noisy_mag, noisy_pha = batch
            else:
                clean_audio, clean_mag, clean_pha, clean_com, noisy_mag, noisy_pha = batch
            clean_audio = torch.autograd.Variable(clean_audio.to(device, non_blocking=True))
            clean_mag = torch.autograd.Variable(clean_mag.to(device, non_blocking=True))
            clean_pha = torch.autograd.Variable(clean_pha.to(device, non_blocking=True))
            clean_com = torch.autograd.Variable(clean_com.to(device, non_blocking=True))
            noisy_mag = torch.autograd.Variable(noisy_mag.to(device, non_blocking=True))
            noisy_pha = torch.autograd.Variable(noisy_pha.to(device, non_blocking=True))
            one_labels = torch.ones(batch_size).to(device, non_blocking=True)

            fractional_epoch = epoch + i / max(1, len(train_loader))
            anchor_alpha = update_refiner_anchor_alpha(
                generator_core, steps, cfg, fractional_epoch=fractional_epoch
            )
            mag_g, pha_g, com_g = generator(noisy_mag, noisy_pha)

            audio_g = mag_phase_istft(mag_g, pha_g, n_fft, hop_size, win_size, compress_factor)
            audio_list_r, audio_list_g = list(clean_audio.cpu().numpy()), list(audio_g.detach().cpu().numpy())
            batch_pesq_score = batch_pesq(audio_list_r, audio_list_g, cfg)

            # Discriminator
            # ------------------------------------------------------- #
            optim_d.zero_grad()
            metric_r = discriminator(clean_mag, clean_mag)
            metric_g = discriminator(clean_mag, mag_g.detach())
            loss_disc_r = F.mse_loss(one_labels, metric_r.flatten())
            
            if batch_pesq_score is not None:
                loss_disc_g = F.mse_loss(batch_pesq_score.to(device), metric_g.flatten())
            else:
                loss_disc_g = 0
            
            loss_disc_all = loss_disc_r + loss_disc_g
            
            loss_disc_all.backward()
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    discriminator.parameters(),
                    max_grad_norm,
                    error_if_nonfinite=True
                )
            optim_d.step()
            # ------------------------------------------------------- #
            
            # Generator
            # ------------------------------------------------------- #
            optim_g.zero_grad()

            # Reference: https://github.com/yxlu-0102/MP-SENet/blob/main/train.py
            # L2 Magnitude Loss
            loss_mag = F.mse_loss(clean_mag, mag_g)
            # Anti-wrapping Phase Loss
            loss_ip, loss_gd, loss_iaf = phase_losses(clean_pha, pha_g, cfg)
            loss_pha = loss_ip + loss_gd + loss_iaf
            # L2 Complex Loss
            loss_com = F.mse_loss(clean_com, com_g) * 2
            # Time Loss
            loss_time = F.l1_loss(clean_audio, audio_g)
            # Metric Loss
            metric_g = discriminator(clean_mag, mag_g)
            loss_metric = F.mse_loss(metric_g.flatten(), one_labels)
            # Consistancy Loss
            _, _, rec_com = mag_phase_stft(audio_g, n_fft, hop_size, win_size, compress_factor, addeps=True)
            loss_con = F.mse_loss(com_g, rec_com) * 2
            auxiliary_losses = harmonic_generation_losses(
                generator_core, clean_mag, clean_com
            )
            refiner_losses = asymmetric_refiner_losses(
                generator_core, clean_mag, clean_com
            )
            loss_cfg = cfg['training_cfg']['loss']
            refiner_loss_multipliers = refiner_intermediate_loss_multipliers(
                fractional_epoch, cfg
            )

            loss_gen_all = (
                loss_metric * loss_cfg['metric'] +
                loss_mag * loss_cfg['magnitude'] +
                loss_pha * loss_cfg['phase'] +
                loss_com * loss_cfg['complex'] +
                loss_time * loss_cfg['time'] +
                loss_con * loss_cfg['consistancy'] +
                auxiliary_losses['coarse_complex'] * loss_cfg['coarse_complex'] +
                auxiliary_losses['pitch'] * loss_cfg['pitch'] +
                auxiliary_losses['voicing'] * loss_cfg['voicing'] +
                auxiliary_losses['harmonic_support'] * loss_cfg['harmonic_support'] +
                refiner_losses['base_complex'] * loss_cfg.get('refiner_base_complex', 0.0)
                * refiner_loss_multipliers['base_complex'] +
                refiner_losses['magnitude'] * loss_cfg.get('refiner_magnitude', 0.0)
                * refiner_loss_multipliers['magnitude'] +
                refiner_losses['phase'] * loss_cfg.get('refiner_phase', 0.0)
                * refiner_loss_multipliers['phase'] +
                refiner_losses['polar_complex'] * loss_cfg.get('refiner_polar_complex', 0.0)
                * refiner_loss_multipliers['polar_complex'] +
                refiner_losses['ri_residual'] * loss_cfg.get('refiner_ri_residual', 0.0)
                * refiner_loss_multipliers['ri_residual'] +
                refiner_losses['demand'] * loss_cfg.get('refiner_demand', 0.0)
                * refiner_loss_multipliers['demand']
            )

            loss_gen_all.backward()
            gradient_norms = clip_generator_gradients(generator_core, cfg)
            optim_g.step()
            if generator_ema is not None:
                generator_ema.update(generator_core)
            # ------------------------------------------------------- #

            validation_due = (
                steps % cfg['env_setting']['validation_interval'] == 0
                and steps != 0
            )
            if num_gpus > 1 and validation_due:
                torch.distributed.barrier()

            if rank == 0:
                # STDOUT logging
                if steps % cfg['env_setting']['stdout_interval'] == 0:
                    with torch.no_grad():
                        metric_error = F.mse_loss(metric_g.flatten(), one_labels).item()
                        mag_error = F.mse_loss(clean_mag, mag_g).item()
                        ip_error, gd_error, iaf_error = phase_losses(clean_pha, pha_g, cfg)
                        pha_error = (loss_ip + loss_gd + loss_iaf).item()
                        com_error = F.mse_loss(clean_com, com_g).item()
                        time_error = F.l1_loss(clean_audio, audio_g).item()
                        con_error = F.mse_loss( com_g, rec_com ).item()
                        coarse_error = auxiliary_losses['coarse_complex'].item()
                        pitch_error = auxiliary_losses['pitch'].item()
                        voicing_error = auxiliary_losses['voicing'].item()
                        support_error = auxiliary_losses['harmonic_support'].item()
                        aux_snapshot = generator_core.latest_aux
                        pitch_peak = aux_snapshot['pitch_posterior'].amax(dim=-1).mean().item()
                        voicing_mean = aux_snapshot['voicing'].mean().item()
                        voicing_target_mean = auxiliary_losses[
                            'voicing_target_mean'
                        ].item()
                        deep_filter_activity = (
                            aux_snapshot['deep_filter_coefficients'].abs().mean().item()
                        )
                        generated_activity = (
                            aux_snapshot['complex_residual_applied'].abs().mean().item()
                        )
                        dense_bridge_scale = (
                            aux_snapshot['dense_bridge_scales'].abs().mean().item()
                        )
                        transition_residual_scale = (
                            aux_snapshot['transition_residual_scales'].abs().mean().item()
                        )

                        print(
                            'Steps : {:d}, Gen Loss: {:4.3f}, Disc Loss: {:4.3f}, Metric Loss: {:4.3f}, '
                            'Mag Loss: {:4.3f}, Pha Loss: {:4.3f}, Com Loss: {:4.3f}, Time Loss: {:4.3f}, '
                            'Cons Loss: {:4.3f}, Coarse Loss: {:4.3f}, Pitch Loss: {:4.3f}, '
                            'Voice Loss: {:4.3f}, Support Loss: {:4.3f}, s/b : {:4.3f}'.format(
                                steps, loss_gen_all, loss_disc_all, metric_error, mag_error,
                                pha_error, com_error, time_error, con_error, coarse_error,
                                pitch_error, voicing_error, support_error, time.time() - start_b
                            ), flush=True
                        )
                        print(
                            'Residual-dense diagnostics - Pitch peak: {:4.3f}, Voicing: {:4.3f}, '
                            'Voice target: {:4.3f}, '
                            'Deep-filter activity: {:4.3f}, Generated activity: {:4.3f}, '
                            'Dense bridge scale: {:4.3f}, Transition scale: {:4.3f}'.format(
                                pitch_peak, voicing_mean, voicing_target_mean,
                                deep_filter_activity, generated_activity,
                                dense_bridge_scale, transition_residual_scale
                            ),
                            flush=True
                        )
                        if getattr(generator_core, 'asymmetric_polar_zip_refiner', None) is not None:
                            print(
                                'Asymmetric refiner optimization - Anchor alpha: {:4.3f}, '
                                'Parent grad: {:4.3f}, Refiner grad: {:4.3f}, '
                                'Base/Polar/RI: {:4.3f}/{:4.3f}/{:4.3f}'.format(
                                    anchor_alpha,
                                    gradient_norms['parent'],
                                    gradient_norms['refiner'],
                                    refiner_losses['base_complex'].item(),
                                    refiner_losses['polar_complex'].item(),
                                    refiner_losses['ri_residual'].item(),
                                ),
                                flush=True,
                            )

                # Checkpointing
                if steps % cfg['env_setting']['checkpoint_interval'] == 0 and steps != 0:
                    exp_name = f"{args.exp_path}/g_{steps:08d}.pth"
                    save_checkpoint(
                        exp_name,
                        {
                            'generator': generator_core.state_dict(),
                            **(
                                {'generator_ema': generator_ema.state_dict()}
                                if generator_ema is not None else {}
                            ),
                        }
                    )
                    exp_name = f"{args.exp_path}/do_{steps:08d}.pth"
                    save_checkpoint(
                        exp_name,
                        {
                            'discriminator': (discriminator.module if num_gpus > 1 else discriminator).state_dict(),
                            'optim_g': optim_g.state_dict(),
                            'optim_d': optim_d.state_dict(),
                            'steps': steps,
                            'epoch': epoch
                        }
                    )

                # Tensorboard summary logging
                if steps % cfg['env_setting']['summary_interval'] == 0:
                    sw.add_scalar("Training/Generator Loss", loss_gen_all, steps)
                    sw.add_scalar("Training/Discriminator Loss", loss_disc_all, steps)
                    sw.add_scalar("Training/Metric Loss", metric_error, steps)
                    sw.add_scalar("Training/Magnitude Loss", mag_error, steps)
                    sw.add_scalar("Training/Phase Loss", pha_error, steps)
                    sw.add_scalar("Training/Complex Loss", com_error, steps)
                    sw.add_scalar("Training/Time Loss", time_error, steps)
                    sw.add_scalar("Training/Consistancy Loss", con_error, steps)
                    sw.add_scalar(
                        "Training/Coarse Complex Loss",
                        auxiliary_losses['coarse_complex'].item(),
                        steps
                    )
                    sw.add_scalar(
                        "Training/Pitch KL Loss", auxiliary_losses['pitch'].item(), steps
                    )
                    sw.add_scalar(
                        "Training/Voicing Loss", auxiliary_losses['voicing'].item(), steps
                    )
                    sw.add_scalar(
                        "Training/Harmonic Support Loss",
                        auxiliary_losses['harmonic_support'].item(),
                        steps
                    )
                    aux_snapshot = generator_core.latest_aux
                    sw.add_scalar(
                        "Training/Pitch Posterior Peak",
                        aux_snapshot['pitch_posterior'].amax(dim=-1).mean().item(),
                        steps
                    )
                    sw.add_scalar(
                        "Training/Voicing Mean", aux_snapshot['voicing'].mean().item(), steps
                    )
                    sw.add_scalar(
                        "Training/Voicing Target Mean",
                        auxiliary_losses['voicing_target_mean'].item(),
                        steps
                    )
                    sw.add_scalar(
                        "Training/Deep Filter Activity",
                        aux_snapshot['deep_filter_coefficients'].abs().mean().item(),
                        steps
                    )
                    sw.add_scalar(
                        "Training/Generated Residual Activity",
                        aux_snapshot['complex_residual_applied'].abs().mean().item(),
                        steps
                    )
                    sw.add_scalar(
                        "Training/Dense Bridge Scale",
                        aux_snapshot['dense_bridge_scales'].abs().mean().item(),
                        steps
                    )
                    sw.add_scalar(
                        "Training/Transition Residual Scale",
                        aux_snapshot['transition_residual_scales'].abs().mean().item(),
                        steps
                    )
                    if 'asymmetric_mag_stage_scales' in aux_snapshot:
                        for loss_name, value in refiner_losses.items():
                            sw.add_scalar(
                                f"Training/Asymmetric Direct {loss_name.replace('_', ' ').title()} Loss",
                                value.item(),
                                steps,
                            )
                        interaction_scales = torch.cat((
                            aux_snapshot[
                                'asymmetric_interaction_mag_scales'
                            ].reshape(-1),
                            aux_snapshot[
                                'asymmetric_interaction_phase_scales'
                            ].reshape(-1),
                        ))
                        applied_delta_abs = aux_snapshot[
                            'applied_mag_multiplicative'
                        ].detach().abs().float().reshape(-1)
                        sw.add_scalar(
                            "Training/Asymmetric Mag Stage Scale",
                            aux_snapshot['asymmetric_mag_stage_scales'].abs().mean().item(),
                            steps
                        )
                        sw.add_scalar(
                            "Training/Asymmetric Phase Stage Scale",
                            aux_snapshot['asymmetric_phase_stage_scales'].abs().mean().item(),
                            steps
                        )
                        sw.add_scalar(
                            "Training/Asymmetric Interaction Scale",
                            interaction_scales.abs().mean().item(),
                            steps
                        )
                        upskip_scales = torch.cat((
                            aux_snapshot['asymmetric_upskip_mag_scale'],
                            aux_snapshot['asymmetric_upskip_phase_scale'],
                        ))
                        sw.add_scalar(
                            "Training/Asymmetric UpSkip Scale",
                            upskip_scales.abs().mean().item(),
                            steps,
                        )
                        sw.add_scalar(
                            "Training/Asymmetric Dense Bridge Scale",
                            aux_snapshot['asymmetric_dense_bridge_scale'].item(),
                            steps
                        )
                        sw.add_scalar(
                            "Training/Asymmetric Context Scale",
                            aux_snapshot['asymmetric_context_scales'].abs().mean().item(),
                            steps
                        )
                        sw.add_scalar(
                            "Training/Asymmetric Magnitude Demand",
                            aux_snapshot['mag_demand_gate'].mean().item(),
                            steps
                        )
                        sw.add_scalar(
                            "Training/Asymmetric Phase Demand",
                            aux_snapshot['phase_demand_gate'].mean().item(),
                            steps
                        )
                        sw.add_scalar(
                            "Training/Asymmetric RI Demand",
                            aux_snapshot['ri_demand_gate'].mean().item(),
                            steps
                        )
                        sw.add_scalar(
                            "Training/Asymmetric Applied Log-Mag Delta Activity",
                            applied_delta_abs.mean().item(),
                            steps
                        )
                        for quantile, suffix in ((0.5, 'P50'), (0.9, 'P90'), (0.99, 'P99')):
                            sw.add_scalar(
                                f"Training/Asymmetric Applied Log-Mag Delta {suffix}",
                                torch.quantile(applied_delta_abs, quantile).item(),
                                steps
                            )
                        sw.add_scalar(
                            "Training/Asymmetric Additive Magnitude Activity",
                            aux_snapshot['applied_mag_additive'].abs().mean().item(),
                            steps
                        )
                        sw.add_scalar(
                            "Training/Asymmetric Phase Delta Activity",
                            aux_snapshot['applied_phase_delta'].abs().mean().item(),
                            steps
                        )
                        sw.add_scalar(
                            "Training/Asymmetric RI Residual Activity",
                            aux_snapshot['ri_residual_applied'].abs().mean().item(),
                            steps
                        )
                        sw.add_scalar(
                            "Training/Asymmetric RI Residual Ratio",
                            aux_snapshot['ri_residual_ratio'].item(),
                            steps
                        )
                        sw.add_scalar(
                            "Training/Asymmetric Anchor Alpha", anchor_alpha, steps
                        )
                        for loss_name, multiplier in refiner_loss_multipliers.items():
                            sw.add_scalar(
                                f"Training/Asymmetric Direct {loss_name.replace('_', ' ').title()} Multiplier",
                                multiplier,
                                steps,
                            )
                        sw.add_scalar(
                            "Training/Parent Gradient Norm",
                            gradient_norms['parent'],
                            steps
                        )
                        sw.add_scalar(
                            "Training/Refiner Gradient Norm",
                            gradient_norms['refiner'],
                            steps
                        )

                # If NaN happend in training period, RaiseError
                if torch.isnan(loss_gen_all).any():
                    raise ValueError("NaN values found in loss_gen_all")

                # Validation
                if validation_due:
                    generator.eval()
                    torch.cuda.empty_cache()
                    audios_r, audios_g = [], []
                    val_mag_err_tot = 0
                    val_pha_err_tot = 0
                    val_com_err_tot = 0
                    with torch.no_grad():
                        for j, batch in enumerate(validation_loader):
                            clean_audio, clean_mag, clean_pha, clean_com, noisy_audio = batch # [B, 1, F, T], F = nfft // 2+ 1, T = nframes
                            clean_audio = torch.autograd.Variable(clean_audio.to(device, non_blocking=True))
                            clean_mag = torch.autograd.Variable(clean_mag.to(device, non_blocking=True))
                            clean_pha = torch.autograd.Variable(clean_pha.to(device, non_blocking=True))
                            clean_com = torch.autograd.Variable(clean_com.to(device, non_blocking=True))
                            noisy_audio = torch.autograd.Variable(noisy_audio.to(device, non_blocking=True))

                            orig_size = noisy_audio.size(1)

                            # 判断是否需要补零
                            if noisy_audio.size(1) >= cfg['training_cfg']['segment_size']:
                                num_segments = noisy_audio.size(1) // cfg['training_cfg']['segment_size']
                                last_segment_size = noisy_audio.size(1) % cfg['training_cfg']['segment_size']
                                if last_segment_size > 0:
                                    last_segment = noisy_audio[:, -cfg['training_cfg']['segment_size']:]
                                    noisy_audio = noisy_audio[:, :-last_segment_size]
                                    segments = torch.split(noisy_audio, cfg['training_cfg']['segment_size'], dim=1)
                                    segments = list(segments)
                                    segments.append(last_segment)
                                    reshapelast = 1
                                else:
                                    segments = torch.split(noisy_audio, cfg['training_cfg']['segment_size'], dim=1)
                                    reshapelast = 0

                            else:
                                # 如果语音长度小于一个segment_size，则直接补零
                                padded_zeros = torch.zeros(1, cfg['training_cfg']['segment_size'] - noisy_audio.size(1)).to(device)
                                noisy_audio = torch.cat((noisy_audio, padded_zeros), dim=1)
                                segments = [noisy_audio]
                                reshapelast = 0

                            # 处理每个语音切片并连接结果
                            processed_segments = []
                            audio_g = []

                            for i, segment in enumerate(segments):

                                noisy_amp, noisy_pha, noisy_com = mag_phase_stft(segment, n_fft, hop_size, win_size,
                                                                               compress_factor)
                                amp_g, pha_g, com_g = generator(noisy_amp.to(device, non_blocking=True), noisy_pha.to(device, non_blocking=True))
                                audio_g = mag_phase_istft(amp_g, pha_g, n_fft, hop_size, win_size, compress_factor)

                                audio_g = audio_g.squeeze()
                                if reshapelast == 1 and i == len(segments) - 2:
                                    audio_g = audio_g[:-(cfg['training_cfg']['segment_size'] - last_segment_size)]
                                    # print(orig_size)

                                processed_segments.append(audio_g)

                            # 将所有处理后的片段连接成一个完整的语音

                            processed_audio = torch.cat(processed_segments, dim=-1)

                            # 裁切末尾部分，保留noisy_wav长度的部分
                            audio_g = processed_audio[:orig_size]

                            mag_g, pha_g, com_g = mag_phase_stft(audio_g, n_fft, hop_size, win_size,
                                                               compress_factor)

                            mag_g = torch.autograd.Variable(mag_g.to(device, non_blocking=True))
                            pha_g = torch.autograd.Variable(pha_g.to(device, non_blocking=True))

                            com_g = torch.autograd.Variable(com_g.to(device, non_blocking=True))

                            mag_g = mag_g.squeeze()
                            pha_g = torch.unsqueeze(pha_g, dim=0)

                            # com_g = com_g.squeeze()
                            clean_mag = clean_mag.squeeze()
                            # clean_pha = clean_pha.squeeze()

                            clean_com = clean_com.squeeze()
                            audios_r += torch.split(clean_audio, 1, dim=0)  # [1, T] * B
                            # print(clean_audio.size())
                            # # print(len(audios_r))
                            audio_g = torch.unsqueeze(audio_g, dim=0)
                            audios_g += torch.split(audio_g, 1, dim=0)


                            val_mag_err_tot += F.mse_loss(clean_mag, mag_g).item()
                            val_ip_err, val_gd_err, val_iaf_err = phase_losses(clean_pha, pha_g, cfg)
                            val_pha_err_tot += (val_ip_err + val_gd_err + val_iaf_err).item()
                            val_com_err_tot += F.mse_loss(clean_com, com_g).item()

                        val_mag_err = val_mag_err_tot / (j+1)
                        val_pha_err = val_pha_err_tot / (j+1)
                        val_com_err = val_com_err_tot / (j+1)
                        val_pesq_score = pesq_score(audios_r, audios_g, cfg).item()
                        print('Steps : {:d}, PESQ Score: {:4.3f}, s/b : {:4.3f}'.
                                format(steps, val_pesq_score, time.time() - start_b))
                        sw.add_scalar(
                            "Validation/Raw/PESQ Score", val_pesq_score, steps
                        )
                        sw.add_scalar(
                            "Validation/Raw/Magnitude Loss", val_mag_err, steps
                        )
                        sw.add_scalar(
                            "Validation/Raw/Phase Loss", val_pha_err, steps
                        )
                        sw.add_scalar(
                            "Validation/Raw/Complex Loss", val_com_err, steps
                        )

                    generator.train()

                    selection_name = 'Raw'
                    selected_pesq = val_pesq_score
                    selected_mag = val_mag_err
                    selected_phase = val_pha_err
                    selected_complex = val_com_err
                    if generator_ema is not None:
                        with generator_ema.average_parameters(generator_core):
                            ema_metrics = validate_generator(
                                generator, validation_loader, cfg, device,
                                n_fft, hop_size, win_size, compress_factor,
                            )
                        ema_metric_tags = {
                            'pesq': 'PESQ Score',
                            'magnitude': 'Magnitude Loss',
                            'phase': 'Phase Loss',
                            'complex': 'Complex Loss',
                        }
                        for metric_name, value in ema_metrics.items():
                            sw.add_scalar(
                                f'Validation/EMA/{ema_metric_tags[metric_name]}',
                                value,
                                steps,
                            )
                        print(
                            f'Steps : {steps:d}, EMA PESQ Score: '
                            f'{ema_metrics["pesq"]:4.3f}, s/b : '
                            f'{time.time() - start_b:4.3f}'
                        )
                        selection_name = 'EMA'
                        selected_pesq = ema_metrics['pesq']
                        selected_mag = ema_metrics['magnitude']
                        selected_phase = ema_metrics['phase']
                        selected_complex = ema_metrics['complex']
                        generator.train()

                    # Keep the established tags as the score-selection stream so
                    # existing result-audit scripts continue to see V3 runs.
                    sw.add_scalar("Validation/PESQ Score", selected_pesq, steps)
                    sw.add_scalar("Validation/Magnitude Loss", selected_mag, steps)
                    sw.add_scalar("Validation/Phase Loss", selected_phase, steps)
                    sw.add_scalar("Validation/Complex Loss", selected_complex, steps)

                    # Print best validation PESQ score in terminal
                    if selected_pesq >= best_pesq:
                        best_pesq = selected_pesq
                        best_pesq_step = steps
                    print(
                        f'valid[{selection_name}]: PESQ {selected_pesq}, '
                        f'Mag_loss {selected_mag}, Phase_loss {selected_phase}. '
                        f'Best_PESQ: {best_pesq} at step {best_pesq_step}'
                    )

            if num_gpus > 1 and validation_due:
                torch.distributed.barrier()
            steps += 1

        scheduler_g.step()
        scheduler_d.step()
        
        if rank == 0:
            print('Time taken for epoch {} is {} sec\n'.format(epoch + 1, int(time.time() - start)))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_folder', default='exp')
    parser.add_argument(
        '--exp_name', default='rd_asymmetric_polar_demand_v3_full_seed1234'
    )
    parser.add_argument(
        '--config',
        default=(
            'recipes/RD-Asymmetric-Polar-Demand-V3/'
            'RD-Asymmetric-Polar-Demand-V3.yaml'
        ),
    )
    parser.add_argument('--resume_from', default=None,
                        help='Optional checkpoint directory to load from while saving into exp_folder/exp_name.')
    parser.add_argument('--resume_step', type=int, default=None,
                        help='Optional checkpoint step to load from resume_from. Defaults to the latest step.')
    parser.add_argument('--resume_lr', type=float, default=None,
                        help='Optional effective optimizer learning rate after loading a checkpoint.')
    parser.add_argument(
        '--init_parent_checkpoint', default=None,
        help=(
            'Optional Residual-Dense generator checkpoint used only to initialize '
            'the parent; training steps and optimizer state start fresh.'
        ),
    )
    parser.add_argument('--mini', action='store_true',
                        help='Use the repository mini train/validation JSON lists.')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override training_cfg.training_epochs for this run.')
    parser.add_argument('--max_steps', type=int, default=None,
                        help='Stop before processing this global training step.')
    args = parser.parse_args()

    if args.epochs is not None and args.epochs <= 0:
        parser.error('--epochs must be a positive integer')
    if args.max_steps is not None and args.max_steps <= 0:
        parser.error('--max_steps must be a positive integer')

    cfg = load_config(args.config)
    if args.epochs is not None:
        cfg['training_cfg']['training_epochs'] = args.epochs
    if args.mini:
        cfg['data_cfg'].update(MINI_DATA_CFG)
    seed = cfg['env_setting']['seed']
    num_gpus = cfg['env_setting']['num_gpus']
    available_gpus = torch.cuda.device_count()

    if num_gpus > available_gpus:
        warnings.warn(
            f"Warning: The actual number of available GPUs ({available_gpus}) is less than the .yaml config ({num_gpus}). Auto reset to num_gpu = {available_gpus}",
            UserWarning
        )
        cfg['env_setting']['num_gpus'] = available_gpus
        num_gpus = available_gpus
        time.sleep(5)
        

    initialize_seed(seed)
    args.exp_path = os.path.join(args.exp_folder, args.exp_name)
    build_env(args.config, 'config.yaml', args.exp_path)
    if args.mini:
        resolved_config_path = os.path.join(args.exp_path, 'config.yaml')
        with open(resolved_config_path, 'w', encoding='utf-8') as config_file:
            yaml.safe_dump(cfg, config_file, sort_keys=False, allow_unicode=True)
        print('Mini dataset mode enabled.')

    if torch.cuda.is_available():
        num_available_gpus = torch.cuda.device_count()
        print(f"Number of GPUs available: {num_available_gpus}")
        print_gpu_info(num_available_gpus, cfg)
    else:
        print("CUDA is not available.")

    if num_gpus > 1:
        mp.spawn(train, nprocs=num_gpus, args=(args, cfg))
    else:
        train(0, args, cfg)

if __name__ == '__main__':
    main()
