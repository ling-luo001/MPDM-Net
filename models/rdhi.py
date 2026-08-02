import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RestorationDemandHistogramInteraction(nn.Module):
    """Mix restoration tokens through demand-local and cross-bin residuals."""

    def __init__(
        self,
        channels,
        bins=8,
        heads=4,
        local_initial_scale=0.01,
        summary_initial_scale=0.0,
    ):
        super().__init__()
        if channels <= 0:
            raise ValueError('channels must be positive.')
        if bins <= 0:
            raise ValueError('bins must be positive.')
        if heads <= 0 or channels % heads != 0:
            raise ValueError('heads must be positive and divide channels.')
        for name, scale in (
            ('local_initial_scale', local_initial_scale),
            ('summary_initial_scale', summary_initial_scale),
        ):
            if not -1.0 < scale < 1.0:
                raise ValueError(f'{name} must be in (-1, 1).')

        self.channels = int(channels)
        self.bins = int(bins)
        self.heads = int(heads)
        self.token_norm = nn.LayerNorm(self.channels)
        self.bin_attention = nn.MultiheadAttention(
            self.channels,
            self.heads,
            dropout=0.0,
            batch_first=True,
        )
        self.summary_norm = nn.LayerNorm(self.channels)
        self.summary_mixer = nn.MultiheadAttention(
            self.channels,
            self.heads,
            dropout=0.0,
            batch_first=True,
        )
        self.output_projection = nn.Linear(self.channels, self.channels, bias=False)
        nn.init.eye_(self.output_projection.weight)
        self.local_residual_scale = nn.Parameter(
            torch.tensor(math.atanh(float(local_initial_scale)), dtype=torch.float32)
        )
        self.summary_residual_scale = nn.Parameter(
            torch.tensor(math.atanh(float(summary_initial_scale)), dtype=torch.float32)
        )
        self.latest_diagnostics = {}

    @property
    def effective_local_scale(self):
        return torch.tanh(self.local_residual_scale)

    @property
    def effective_summary_scale(self):
        return torch.tanh(self.summary_residual_scale)

    @property
    def effective_scale(self):
        """Compatibility alias for readers of the original RDHI scale."""
        return self.effective_local_scale

    @staticmethod
    def _sort_by_demand(tokens, demand):
        if tokens.ndim != 3 or demand.ndim != 2:
            raise ValueError('Expected tokens [B, N, C] and demand [B, N].')
        if tokens.shape[:2] != demand.shape:
            raise ValueError('Token and demand batch/token dimensions must match.')

        sort_indices = torch.argsort(demand.detach(), dim=1, stable=True)
        sorted_tokens = torch.gather(
            tokens,
            1,
            sort_indices.unsqueeze(-1).expand(-1, -1, tokens.shape[-1]),
        )
        inverse_indices = torch.empty_like(sort_indices)
        original_indices = torch.arange(
            tokens.shape[1], device=tokens.device, dtype=sort_indices.dtype
        ).unsqueeze(0).expand_as(sort_indices)
        inverse_indices.scatter_(1, sort_indices, original_indices)
        return sorted_tokens, sort_indices, inverse_indices

    @staticmethod
    def _restore_order(sorted_tokens, inverse_indices):
        return torch.gather(
            sorted_tokens,
            1,
            inverse_indices.unsqueeze(-1).expand(-1, -1, sorted_tokens.shape[-1]),
        )

    @staticmethod
    def _masked_rms(tensor, valid):
        weights = valid.unsqueeze(-1).to(tensor.dtype)
        denominator = (weights.sum() * tensor.shape[-1]).clamp_min(1.0)
        return ((tensor.square() * weights).sum() / denominator).sqrt()

    def _forward_tokens(self, tokens, demand, padding_value=0.0):
        batch, num_tokens, channels = tokens.shape
        sorted_raw, sort_indices, inverse_indices = self._sort_by_demand(
            tokens, demand
        )
        sorted_demand = torch.gather(demand.detach(), 1, sort_indices)

        active_bins = min(self.bins, num_tokens)
        bin_size = math.ceil(num_tokens / active_bins)
        padded_tokens = active_bins * bin_size
        pad_count = padded_tokens - num_tokens
        if pad_count:
            sorted_raw = F.pad(
                sorted_raw, (0, 0, 0, pad_count), value=float(padding_value)
            )
            sorted_demand = F.pad(sorted_demand, (0, pad_count))

        valid = torch.arange(padded_tokens, device=tokens.device) < num_tokens
        valid = valid.view(1, active_bins, bin_size).expand(batch, -1, -1)
        binned_raw = sorted_raw.view(batch, active_bins, bin_size, channels)
        flat_raw = binned_raw.reshape(batch * active_bins, bin_size, channels)
        padding_mask = ~valid.reshape(batch * active_bins, bin_size)

        normalized_raw = self.token_norm(flat_raw)
        local_delta, _ = self.bin_attention(
            normalized_raw,
            normalized_raw,
            normalized_raw,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        local_delta = local_delta.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        local_delta = local_delta.view(batch, active_bins, bin_size, channels)

        local_residual = self.effective_local_scale * local_delta
        local_tokens = binned_raw + local_residual
        valid_weights = valid.unsqueeze(-1).to(local_tokens.dtype)
        summaries = (local_tokens * valid_weights).sum(dim=2)
        summaries = summaries / valid_weights.sum(dim=2).clamp_min(1.0)
        normalized_summaries = self.summary_norm(summaries)
        summary_delta, _ = self.summary_mixer(
            normalized_summaries,
            normalized_summaries,
            normalized_summaries,
            need_weights=False,
        )
        summary_residual = (
            self.effective_summary_scale * summary_delta.unsqueeze(2)
        ).expand(-1, -1, bin_size, -1)

        projected_local = self.output_projection(local_residual)
        projected_summary = self.output_projection(summary_residual)
        combined = (projected_local + projected_summary).masked_fill(
            ~valid.unsqueeze(-1), 0.0
        )
        sorted_output = binned_raw + combined
        sorted_output = sorted_output.reshape(batch, padded_tokens, channels)
        sorted_output = sorted_output[:, :num_tokens]
        output = self._restore_order(sorted_output, inverse_indices)

        binned_demand = sorted_demand.view(batch, active_bins, bin_size)
        demand_max = binned_demand.masked_fill(~valid, float('-inf')).amax(dim=2)
        demand_min = binned_demand.masked_fill(~valid, float('inf')).amin(dim=2)
        reference_rms = self._masked_rms(binned_raw, valid).clamp_min(1e-8)
        self.latest_diagnostics = {
            'local_scale': self.effective_local_scale.detach(),
            'summary_scale': self.effective_summary_scale.detach(),
            'local_update_ratio': (
                self._masked_rms(projected_local, valid) / reference_rms
            ).detach(),
            'summary_update_ratio': (
                self._masked_rms(projected_summary, valid) / reference_rms
            ).detach(),
            'bin_demand_span_mean': (demand_max - demand_min).mean().detach(),
        }
        return output

    def _compute_output(self, x, demand, padding_value=0.0):
        if x.ndim != 4 or demand.ndim != 4:
            raise ValueError('Expected x [B, C, T, F] and demand [B, 1, T, F].')
        if x.shape[0] != demand.shape[0] or x.shape[2:] != demand.shape[2:]:
            raise ValueError('Feature and demand spatial dimensions must match.')
        if x.shape[1] != self.channels or demand.shape[1] != 1:
            raise ValueError('Unexpected feature channels or demand channels.')

        batch, channels, time, frequency = x.shape
        num_tokens = time * frequency
        if num_tokens == 0:
            raise ValueError('RDHI requires at least one token.')

        tokens = x.permute(0, 2, 3, 1).reshape(batch, num_tokens, channels)
        demand_tokens = demand.reshape(batch, num_tokens)
        output = self._forward_tokens(tokens, demand_tokens, padding_value)
        return output.view(batch, time, frequency, channels).permute(0, 3, 1, 2)

    def padding_utilization(self, num_tokens):
        if num_tokens <= 0:
            raise ValueError('num_tokens must be positive.')
        active_bins = min(self.bins, int(num_tokens))
        padded_tokens = active_bins * math.ceil(int(num_tokens) / active_bins)
        return float(num_tokens) / float(padded_tokens)

    def forward(self, x, demand):
        return self._compute_output(x, demand)
