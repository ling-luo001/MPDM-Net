import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class LayerNorm2d(nn.Module):
    """Channel-wise layer normalization for a time-frequency feature map."""

    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        variance = (x - mean).square().mean(dim=1, keepdim=True)
        x = (x - mean) * torch.rsqrt(variance + self.eps)
        return x * self.weight + self.bias


class AxisNAFBlock(nn.Module):
    """Activation-free restoration block with separate time/frequency filters."""

    def __init__(self, channels, axis_kernel_size=7, dropout=0.0):
        super().__init__()
        if axis_kernel_size % 2 == 0:
            raise ValueError('axis_kernel_size must be odd')

        expanded = channels * 2
        padding = axis_kernel_size // 2
        self.norm1 = LayerNorm2d(channels)
        self.in_project = nn.Conv2d(channels, expanded, 1)
        self.time_filter = nn.Conv2d(
            expanded,
            expanded,
            kernel_size=(axis_kernel_size, 1),
            padding=(padding, 0),
            groups=expanded,
        )
        self.frequency_filter = nn.Conv2d(
            expanded,
            expanded,
            kernel_size=(1, axis_kernel_size),
            padding=(0, padding),
            groups=expanded,
        )
        self.channel_scale = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, 1),
        )
        self.out_project = nn.Conv2d(channels, channels, 1)
        self.dropout1 = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        self.norm2 = LayerNorm2d(channels)
        self.ffn_in = nn.Conv2d(channels, expanded, 1)
        self.ffn_out = nn.Conv2d(channels, channels, 1)
        self.dropout2 = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        # A new block starts as an identity map and opts into both residuals.
        self.spatial_scale = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.channel_residual_scale = nn.Parameter(torch.zeros(1, channels, 1, 1))

    @staticmethod
    def simple_gate(x):
        first, second = x.chunk(2, dim=1)
        return first * second

    def forward(self, x):
        residual = self.in_project(self.norm1(x))
        temporal = self.time_filter(residual)
        spectral = self.frequency_filter(residual)
        residual = self.simple_gate(0.5 * (temporal + spectral))
        residual = residual * self.channel_scale(residual)
        residual = self.dropout1(self.out_project(residual))
        x = x + residual * self.spatial_scale

        residual = self.simple_gate(self.ffn_in(self.norm2(x)))
        residual = self.dropout2(self.ffn_out(residual))
        return x + residual * self.channel_residual_scale


class DenseFeatureProjector(nn.Module):
    """Compress concatenated features with an activation-free gated MLP."""

    def __init__(self, input_channels, output_channels, compression=0.5):
        super().__init__()
        hidden_channels = max(int(output_channels * compression), 16)
        self.norm = LayerNorm2d(input_channels)
        self.in_project = nn.Conv2d(input_channels, hidden_channels * 2, 1)
        self.out_project = nn.Conv2d(hidden_channels, output_channels, 1)

    def forward(self, x):
        first, second = self.in_project(self.norm(x)).chunk(2, dim=1)
        return self.out_project(first * second)


class ResidualDenseNAFStage(nn.Module):
    """NAF stage with compressed dense reuse and a bounded long residual."""

    def __init__(
        self,
        channels,
        count,
        axis_kernel_size,
        dropout,
        dense_compression=0.5,
        stage_gain_limit=0.25,
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                AxisNAFBlock(
                    channels,
                    axis_kernel_size=axis_kernel_size,
                    dropout=dropout,
                )
                for _ in range(count)
            ]
        )
        self.dense_fusions = nn.ModuleList(
            [
                DenseFeatureProjector(
                    channels * (index + 1),
                    channels,
                    compression=dense_compression,
                )
                for index in range(1, count)
            ]
        )
        self.dense_scales = nn.ParameterList(
            [nn.Parameter(torch.zeros(())) for _ in range(count - 1)]
        )
        self.stage_gain_delta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.stage_gain_limit = float(stage_gain_limit)

    def forward(self, x):
        anchor = x
        features = [x]
        current = x
        for index, block in enumerate(self.blocks):
            if index > 0:
                dense_update = self.dense_fusions[index - 1](
                    torch.cat(features, dim=1)
                )
                current = current + torch.tanh(
                    self.dense_scales[index - 1]
                ) * dense_update
            current = block(current)
            features.append(current)

        gain = 1.0 + self.stage_gain_limit * torch.tanh(self.stage_gain_delta)
        return anchor + gain * (current - anchor)

    def mean_dense_scale(self):
        if not self.dense_scales:
            return self.stage_gain_delta.new_zeros(())
        return torch.stack(
            [torch.tanh(scale).abs() for scale in self.dense_scales]
        ).mean()

    def mean_stage_gain(self):
        gain = 1.0 + self.stage_gain_limit * torch.tanh(self.stage_gain_delta)
        return gain.mean()


class ResidualDownsample(nn.Module):
    """Learned strided transition with a gated anti-aliased shortcut."""

    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.body = nn.Conv2d(input_channels, output_channels, 2, stride=2)
        self.shortcut = nn.Sequential(
            nn.AvgPool2d(2, stride=2),
            nn.Conv2d(input_channels, output_channels, 1, bias=False),
        )
        self.residual_scale = nn.Parameter(torch.zeros(()))

    def forward(self, x):
        return self.body(x) + torch.tanh(self.residual_scale) * self.shortcut(x)


class ResidualUpsample(nn.Module):
    """Pixel-shuffle transition with a gated interpolation shortcut."""

    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(input_channels, output_channels * 4, 1, bias=False),
            nn.PixelShuffle(2),
        )
        self.shortcut = nn.Conv2d(input_channels, output_channels, 1, bias=False)
        self.residual_scale = nn.Parameter(torch.zeros(()))

    def forward(self, x):
        shortcut = F.interpolate(
            self.shortcut(x), scale_factor=2.0, mode='bilinear', align_corners=False
        )
        return self.body(x) + torch.tanh(self.residual_scale) * shortcut


class ResidualDenseSkipFusion(nn.Module):
    """Preserve the baseline skip fusion and add gated residual/dense paths."""

    def __init__(self, channels, dense_compression=0.5):
        super().__init__()
        self.base_fusion = nn.Conv2d(channels * 2, channels, 1)
        self.dense_update = DenseFeatureProjector(
            channels * 3, channels, compression=dense_compression
        )
        self.shortcut_scale = nn.Parameter(torch.zeros(()))
        self.dense_scale = nn.Parameter(torch.zeros(()))

    def forward(self, decoder, encoder):
        base = self.base_fusion(torch.cat([decoder, encoder], dim=1))
        shortcut = 0.5 * (decoder + encoder)
        dense_update = self.dense_update(
            torch.cat([decoder, encoder, base], dim=1)
        )
        return (
            base
            + torch.tanh(self.shortcut_scale) * shortcut
            + torch.tanh(self.dense_scale) * dense_update
        )


class MultiScalePromptContext(nn.Module):
    """Densely aggregate encoder scales before estimating degradation prompts."""

    def __init__(self, level1, level2, level3, dense_compression=0.5):
        super().__init__()
        self.level1_project = nn.Conv2d(level1, level3, 1, bias=False)
        self.level2_project = nn.Conv2d(level2, level3, 1, bias=False)
        self.dense_update = DenseFeatureProjector(
            level3 * 3, level3, compression=dense_compression
        )
        self.residual_scale = nn.Parameter(torch.zeros(()))

    def forward(self, bottleneck, encoder1, encoder2):
        output_size = bottleneck.shape[-2:]
        level1_context = self.level1_project(
            F.adaptive_avg_pool2d(encoder1, output_size)
        )
        level2_context = self.level2_project(
            F.adaptive_avg_pool2d(encoder2, output_size)
        )
        dense_update = self.dense_update(
            torch.cat([bottleneck, level1_context, level2_context], dim=1)
        )
        return bottleneck + torch.tanh(self.residual_scale) * dense_update


class DenseOutputBridge(nn.Module):
    """Recover shallow detail through a gated full-resolution dense bridge."""

    def __init__(self, channels, dense_compression=0.5):
        super().__init__()
        self.dense_update = DenseFeatureProjector(
            channels * 4, channels, compression=dense_compression
        )
        self.residual_scale = nn.Parameter(torch.zeros(()))

    def forward(self, target, refinement_input, encoder, intro):
        update = self.dense_update(
            torch.cat([target, refinement_input, encoder, intro], dim=1)
        )
        return target + torch.tanh(self.residual_scale) * update


class TriGranularPromptEstimator(nn.Module):
    """Infer utterance, temporal, and spectral degradation descriptors."""

    def __init__(self, input_channels, prompt_channels, prompt_count):
        super().__init__()
        self.norm = LayerNorm2d(input_channels)
        self.project = nn.Conv2d(input_channels, prompt_channels, 1)
        self.global_logits = nn.Linear(prompt_channels, prompt_count)
        self.temporal_logits = nn.Conv1d(prompt_channels, prompt_count, 1)
        self.spectral_logits = nn.Conv1d(prompt_channels, prompt_count, 1)

        # These two heads make the latent prompt measurable during training.
        self.temporal_noise_profile = nn.Conv1d(prompt_channels, 1, 1)
        self.spectral_noise_profile = nn.Conv1d(prompt_channels, 1, 1)

    def forward(self, x):
        feature = self.project(self.norm(x))
        global_descriptor = feature.mean(dim=(2, 3))
        temporal_descriptor = feature.mean(dim=3)
        spectral_descriptor = feature.mean(dim=2)

        global_weights = torch.softmax(self.global_logits(global_descriptor), dim=1)
        temporal_weights = torch.softmax(self.temporal_logits(temporal_descriptor), dim=1)
        spectral_weights = torch.softmax(self.spectral_logits(spectral_descriptor), dim=1)
        temporal_profile = self.temporal_noise_profile(temporal_descriptor).squeeze(1)
        spectral_profile = self.spectral_noise_profile(spectral_descriptor).squeeze(1)
        return {
            'global_weights': global_weights,
            'temporal_weights': temporal_weights,
            'spectral_weights': spectral_weights,
            'temporal_noise_log_ratio': temporal_profile,
            'spectral_noise_log_ratio': spectral_profile,
        }


class PromptModulation(nn.Module):
    """Build a scale-specific prompt map and apply bounded FiLM modulation."""

    def __init__(self, channels, prompt_count, modulation_limit=0.25):
        super().__init__()
        self.global_basis = nn.Parameter(torch.empty(prompt_count, channels))
        self.temporal_basis = nn.Parameter(torch.empty(prompt_count, channels))
        self.spectral_basis = nn.Parameter(torch.empty(prompt_count, channels))
        self.prompt_norm = LayerNorm2d(channels)
        self.to_affine = nn.Conv2d(channels, channels * 2, 1)
        self.modulation_limit = float(modulation_limit)

        nn.init.normal_(self.global_basis, std=0.02)
        nn.init.normal_(self.temporal_basis, std=0.02)
        nn.init.normal_(self.spectral_basis, std=0.02)
        nn.init.normal_(self.to_affine.weight, std=1e-3)
        nn.init.zeros_(self.to_affine.bias)

    def forward(self, x, prompt):
        global_prompt = torch.einsum(
            'bk,kc->bc', prompt['global_weights'], self.global_basis
        ).unsqueeze(-1).unsqueeze(-1)

        temporal_prompt = torch.einsum(
            'bkt,kc->bct', prompt['temporal_weights'], self.temporal_basis
        )
        temporal_prompt = F.interpolate(
            temporal_prompt, size=x.shape[2], mode='linear', align_corners=False
        ).unsqueeze(-1)

        spectral_prompt = torch.einsum(
            'bkf,kc->bcf', prompt['spectral_weights'], self.spectral_basis
        )
        spectral_prompt = F.interpolate(
            spectral_prompt, size=x.shape[3], mode='linear', align_corners=False
        ).unsqueeze(-2)

        prompt_map = self.prompt_norm(global_prompt + temporal_prompt + spectral_prompt)
        gain, bias = self.to_affine(prompt_map).chunk(2, dim=1)
        limit = self.modulation_limit
        return x * (1.0 + limit * torch.tanh(gain)) + limit * torch.tanh(bias)


def _make_stage(
    channels,
    count,
    axis_kernel_size,
    dropout,
    dense_compression,
    stage_gain_limit,
):
    return ResidualDenseNAFStage(
        channels,
        count,
        axis_kernel_size=axis_kernel_size,
        dropout=dropout,
        dense_compression=dense_compression,
        stage_gain_limit=stage_gain_limit,
    )


class PromptNAFSEUNet(nn.Module):
    """
    Residual-dense single-tower restoration with tri-granular prompts.

    The model estimates one complex residual from normalized noisy
    real/imaginary spectra and applies no external source prior. Gated dense
    paths improve feature reuse while preserving the original prompt mechanism.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        model_cfg = cfg['model_cfg']
        width = int(model_cfg.get('prompt_width', 48))
        prompt_count = int(model_cfg.get('prompt_count', 6))
        prompt_channels = int(model_cfg.get('prompt_channels', 64))
        block_counts = model_cfg.get('prompt_naf_blocks', [2, 2, 6, 2, 2, 2])
        if len(block_counts) != 6 or any(int(value) < 1 for value in block_counts):
            raise ValueError('prompt_naf_blocks must contain six positive integers')
        block_counts = [int(value) for value in block_counts]
        axis_kernel_size = int(model_cfg.get('axis_kernel_size', 7))
        dropout = float(model_cfg.get('prompt_dropout', 0.0))
        dense_compression = float(model_cfg.get('dense_compression', 0.5))
        stage_gain_limit = float(model_cfg.get('stage_gain_limit', 0.25))
        if not 0.0 < dense_compression <= 1.0:
            raise ValueError('dense_compression must be in (0, 1]')
        if not 0.0 <= stage_gain_limit <= 1.0:
            raise ValueError('stage_gain_limit must be in [0, 1]')

        self.phase_eps = float(model_cfg.get('phase_eps', 1e-3))
        self.residual_limit = float(model_cfg.get('residual_limit', 2.0))
        self.residual_floor_ratio = float(model_cfg.get('residual_floor_ratio', 0.1))

        level1, level2, level3 = width, width * 2, width * 4
        self.intro = nn.Conv2d(3, level1, 3, padding=1)
        self.encoder_level1 = _make_stage(
            level1,
            block_counts[0],
            axis_kernel_size,
            dropout,
            dense_compression,
            stage_gain_limit,
        )
        self.down1 = ResidualDownsample(level1, level2)
        self.encoder_level2 = _make_stage(
            level2,
            block_counts[1],
            axis_kernel_size,
            dropout,
            dense_compression,
            stage_gain_limit,
        )
        self.down2 = ResidualDownsample(level2, level3)

        self.prompt_context = MultiScalePromptContext(
            level1, level2, level3, dense_compression=dense_compression
        )
        self.prompt_estimator = TriGranularPromptEstimator(
            level3, prompt_channels, prompt_count
        )
        self.bottleneck_prompt = PromptModulation(level3, prompt_count)
        self.bottleneck = _make_stage(
            level3,
            block_counts[2],
            axis_kernel_size,
            dropout,
            dense_compression,
            stage_gain_limit,
        )

        self.up2 = ResidualUpsample(level3, level2)
        self.fuse_level2 = ResidualDenseSkipFusion(
            level2, dense_compression=dense_compression
        )
        self.decoder_level2_prompt = PromptModulation(level2, prompt_count)
        self.decoder_level2 = _make_stage(
            level2,
            block_counts[3],
            axis_kernel_size,
            dropout,
            dense_compression,
            stage_gain_limit,
        )

        self.up1 = ResidualUpsample(level2, level1)
        self.fuse_level1 = ResidualDenseSkipFusion(
            level1, dense_compression=dense_compression
        )
        self.decoder_level1_prompt = PromptModulation(level1, prompt_count)
        self.decoder_level1 = _make_stage(
            level1,
            block_counts[4],
            axis_kernel_size,
            dropout,
            dense_compression,
            stage_gain_limit,
        )

        self.refinement_prompt = PromptModulation(level1, prompt_count)
        self.refinement = _make_stage(
            level1,
            block_counts[5],
            axis_kernel_size,
            dropout,
            dense_compression,
            stage_gain_limit,
        )
        self.output_dense_bridge = DenseOutputBridge(
            level1, dense_compression=dense_compression
        )
        self.complex_residual_head = nn.Conv2d(level1, 2, 3, padding=1)
        nn.init.zeros_(self.complex_residual_head.weight)
        nn.init.zeros_(self.complex_residual_head.bias)

        self.latest_aux = {}

    @staticmethod
    def _pad_to_multiple(x, multiple=4):
        pad_time = (multiple - x.shape[2] % multiple) % multiple
        pad_frequency = (multiple - x.shape[3] % multiple) % multiple
        if pad_time == 0 and pad_frequency == 0:
            return x
        return F.pad(x, (0, pad_frequency, 0, pad_time), mode='reflect')

    @staticmethod
    def _entropy(weights, dim=1):
        return -(weights.clamp_min(1e-8).log() * weights).sum(dim=dim).mean()

    def forward(self, noisy_mag, noisy_pha):
        if noisy_mag.ndim != 3 or noisy_pha.ndim != 3:
            raise ValueError('Expected noisy_mag and noisy_pha with shape [B, F, T]')
        if noisy_mag.shape != noisy_pha.shape:
            raise ValueError(
                f'Input shapes differ: {tuple(noisy_mag.shape)} vs {tuple(noisy_pha.shape)}'
            )
        if not torch.isfinite(noisy_mag).all() or not torch.isfinite(noisy_pha).all():
            raise RuntimeError('Input spectrum contains NaN/Inf')

        frequency_bins, time_frames = noisy_mag.shape[1], noisy_mag.shape[2]
        noisy_mag_4d = rearrange(noisy_mag, 'b f t -> b 1 t f')
        noisy_pha_4d = rearrange(noisy_pha, 'b f t -> b 1 t f')
        noisy_real_4d = noisy_mag_4d * torch.cos(noisy_pha_4d)
        noisy_imag_4d = noisy_mag_4d * torch.sin(noisy_pha_4d)

        utterance_scale = noisy_mag_4d.mean(dim=(2, 3), keepdim=True).clamp_min(1e-4)
        network_input = torch.cat(
            [
                noisy_real_4d / utterance_scale,
                noisy_imag_4d / utterance_scale,
                torch.log1p(noisy_mag_4d / utterance_scale),
            ],
            dim=1,
        )
        network_input = self._pad_to_multiple(network_input)

        intro = self.intro(network_input)
        encoder1 = self.encoder_level1(intro)
        encoder2 = self.encoder_level2(self.down1(encoder1))
        bottleneck = self.down2(encoder2)

        bottleneck = self.prompt_context(bottleneck, encoder1, encoder2)
        prompt = self.prompt_estimator(bottleneck)
        bottleneck = self.bottleneck_prompt(bottleneck, prompt)
        bottleneck = self.bottleneck(bottleneck)

        decoder2 = self.up2(bottleneck)
        decoder2 = self.fuse_level2(decoder2, encoder2)
        decoder2 = self.decoder_level2_prompt(decoder2, prompt)
        decoder2 = self.decoder_level2(decoder2)

        decoder1 = self.up1(decoder2)
        decoder1 = self.fuse_level1(decoder1, encoder1)
        decoder1 = self.decoder_level1_prompt(decoder1, prompt)
        decoder1 = self.decoder_level1(decoder1)
        refinement_input = decoder1
        decoder1 = self.refinement_prompt(decoder1, prompt)
        decoder1 = self.refinement(decoder1)
        decoder1 = self.output_dense_bridge(
            decoder1, refinement_input, encoder1, intro
        )

        raw_residual = self.complex_residual_head(decoder1)
        raw_residual = raw_residual[:, :, :time_frames, :frequency_bins]
        residual_reference = noisy_mag_4d + self.residual_floor_ratio * utterance_scale
        complex_residual = (
            self.residual_limit * torch.tanh(raw_residual) * residual_reference
        )
        enhanced_real_4d = noisy_real_4d + complex_residual[:, :1]
        enhanced_imag_4d = noisy_imag_4d + complex_residual[:, 1:]

        enhanced_real = rearrange(enhanced_real_4d.squeeze(1), 'b t f -> b f t')
        enhanced_imag = rearrange(enhanced_imag_4d.squeeze(1), 'b t f -> b f t')
        denoised_mag = torch.sqrt(
            torch.clamp(enhanced_real.square() + enhanced_imag.square(), min=1e-12)
        )
        phase_floor = torch.full_like(enhanced_real, self.phase_eps)
        phase_real = torch.where(
            denoised_mag.detach() < self.phase_eps, phase_floor, enhanced_real
        )
        pred_pha = torch.atan2(enhanced_imag, phase_real)
        denoised_com = torch.stack([enhanced_real, enhanced_imag], dim=-1)

        temporal_profile = F.interpolate(
            prompt['temporal_noise_log_ratio'].unsqueeze(1),
            size=time_frames,
            mode='linear',
            align_corners=False,
        ).squeeze(1)
        spectral_profile = F.interpolate(
            prompt['spectral_noise_log_ratio'].unsqueeze(1),
            size=frequency_bins,
            mode='linear',
            align_corners=False,
        ).squeeze(1)
        temporal_weights = F.interpolate(
            prompt['temporal_weights'],
            size=time_frames,
            mode='linear',
            align_corners=False,
        )
        spectral_weights = F.interpolate(
            prompt['spectral_weights'],
            size=frequency_bins,
            mode='linear',
            align_corners=False,
        )
        prompt_entropy = (
            self._entropy(prompt['global_weights'])
            + self._entropy(prompt['temporal_weights'])
            + self._entropy(prompt['spectral_weights'])
        ) / 3.0

        stages = (
            self.encoder_level1,
            self.encoder_level2,
            self.bottleneck,
            self.decoder_level2,
            self.decoder_level1,
            self.refinement,
        )
        dense_connection_scales = torch.stack(
            [stage.mean_dense_scale() for stage in stages]
        )
        stage_residual_gains = torch.stack(
            [stage.mean_stage_gain() for stage in stages]
        )
        transition_residual_scales = torch.stack(
            [
                torch.tanh(module.residual_scale)
                for module in (self.down1, self.down2, self.up2, self.up1)
            ]
        )
        skip_residual_scales = torch.stack(
            [
                torch.tanh(self.fuse_level2.shortcut_scale),
                torch.tanh(self.fuse_level2.dense_scale),
                torch.tanh(self.fuse_level1.shortcut_scale),
                torch.tanh(self.fuse_level1.dense_scale),
            ]
        )

        if not torch.isfinite(denoised_com).all():
            raise RuntimeError('Enhanced complex spectrum contains NaN/Inf')
        self.latest_aux = {
            'complex_residual': rearrange(
                complex_residual, 'b c t f -> b f t c'
            ),
            'global_prompt_weights': prompt['global_weights'],
            'temporal_prompt_weights': temporal_weights,
            'spectral_prompt_weights': spectral_weights,
            'temporal_noise_log_ratio': temporal_profile,
            'spectral_noise_log_ratio': spectral_profile,
            'prompt_entropy': prompt_entropy,
            'dense_connection_scales': dense_connection_scales.detach(),
            'stage_residual_gains': stage_residual_gains.detach(),
            'transition_residual_scales': transition_residual_scales.detach(),
            'skip_residual_scales': skip_residual_scales.detach(),
            'prompt_context_scale': torch.tanh(
                self.prompt_context.residual_scale
            ).detach(),
            'output_dense_scale': torch.tanh(
                self.output_dense_bridge.residual_scale
            ).detach(),
        }
        return denoised_mag, pred_pha, denoised_com


# Keep the historical import name used by train.py and inference scripts.
MambaSEUNet = PromptNAFSEUNet
