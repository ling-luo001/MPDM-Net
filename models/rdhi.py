import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RestorationDemandHistogramInteraction(nn.Module):
    """Mix bottleneck tokens grouped by their Stage-1 restoration demand."""

    def __init__(self, channels, bins=8, heads=4, initial_scale=0.05):
        super().__init__()
        if channels <= 0:
            raise ValueError('channels must be positive.')
        if bins <= 0:
            raise ValueError('bins must be positive.')
        if heads <= 0 or channels % heads != 0:
            raise ValueError('heads must be positive and divide channels.')
        if not -1.0 < initial_scale < 1.0:
            raise ValueError('initial_scale must be in (-1, 1).')

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
        self.residual_scale = nn.Parameter(
            torch.tensor(math.atanh(float(initial_scale)), dtype=torch.float32)
        )

    @property
    def effective_scale(self):
        return torch.tanh(self.residual_scale)

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

    def _compute_update(self, x, demand):
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
        tokens = self.token_norm(tokens)
        demand_tokens = demand.reshape(batch, num_tokens)
        sorted_tokens, _, inverse_indices = self._sort_by_demand(
            tokens, demand_tokens
        )

        active_bins = min(self.bins, num_tokens)
        bin_size = math.ceil(num_tokens / active_bins)
        padded_tokens = active_bins * bin_size
        pad_count = padded_tokens - num_tokens
        if pad_count:
            sorted_tokens = F.pad(sorted_tokens, (0, 0, 0, pad_count))

        valid = torch.arange(padded_tokens, device=x.device) < num_tokens
        valid = valid.view(1, active_bins, bin_size).expand(batch, -1, -1)
        binned_tokens = sorted_tokens.view(
            batch, active_bins, bin_size, channels
        )
        flat_bins = binned_tokens.reshape(batch * active_bins, bin_size, channels)
        padding_mask = ~valid.reshape(batch * active_bins, bin_size)

        local_update, _ = self.bin_attention(
            flat_bins,
            flat_bins,
            flat_bins,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        local_update = local_update.masked_fill(padding_mask.unsqueeze(-1), 0.0)
        local_update = local_update.view(batch, active_bins, bin_size, channels)

        valid_weights = valid.unsqueeze(-1).to(local_update.dtype)
        summaries = (local_update * valid_weights).sum(dim=2)
        summaries = summaries / valid_weights.sum(dim=2).clamp_min(1.0)
        normalized_summaries = self.summary_norm(summaries)
        summary_update, _ = self.summary_mixer(
            normalized_summaries,
            normalized_summaries,
            normalized_summaries,
            need_weights=False,
        )

        combined = local_update + summary_update.unsqueeze(2)
        combined = self.output_projection(combined)
        combined = combined.masked_fill(~valid.unsqueeze(-1), 0.0)
        sorted_update = combined.reshape(batch, padded_tokens, channels)
        sorted_update = sorted_update[:, :num_tokens]
        update = self._restore_order(sorted_update, inverse_indices)
        return update.view(batch, time, frequency, channels).permute(0, 3, 1, 2)

    def padding_utilization(self, num_tokens):
        if num_tokens <= 0:
            raise ValueError('num_tokens must be positive.')
        active_bins = min(self.bins, int(num_tokens))
        padded_tokens = active_bins * math.ceil(int(num_tokens) / active_bins)
        return float(num_tokens) / float(padded_tokens)

    def forward(self, x, demand):
        update = self._compute_update(x, demand)
        return x + self.effective_scale * update
