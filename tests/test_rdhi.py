import copy
import importlib
import sys
import types
import unittest
from unittest import mock

import torch
import torch.nn as nn

from models.rdhi import RestorationDemandHistogramInteraction


class _IdentityBlock(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return x


class RDHITest(unittest.TestCase):
    def test_sort_inverse_shape_stability_and_non_divisible_bins(self):
        module = RestorationDemandHistogramInteraction(8, bins=3, heads=2)
        tokens = torch.arange(2 * 10 * 8, dtype=torch.float32).view(2, 10, 8)
        demand = torch.tensor([
            [0.3, 0.1, 0.1, 0.8, 0.2, 0.5, 0.5, 0.9, 0.0, 0.4],
            [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
        ])

        sorted_tokens, indices, inverse = module._sort_by_demand(tokens, demand)
        restored = module._restore_order(sorted_tokens, inverse)
        output = module(
            tokens.transpose(1, 2).view(2, 8, 2, 5),
            demand.view(2, 1, 2, 5),
        )

        self.assertEqual(sorted_tokens.shape, tokens.shape)
        self.assertTrue(torch.equal(restored, tokens))
        self.assertEqual(indices[0, 1].item(), 1)
        self.assertEqual(indices[0, 2].item(), 2)
        self.assertEqual(output.shape, (2, 8, 2, 5))
        self.assertTrue(torch.isfinite(output).all())
        self.assertAlmostEqual(module.padding_utilization(10), 10.0 / 12.0)

    def test_zero_scales_are_strict_identity(self):
        module = RestorationDemandHistogramInteraction(
            8,
            bins=4,
            heads=2,
            local_initial_scale=0.0,
            summary_initial_scale=0.0,
        )
        x = torch.randn(1, 8, 2, 4)
        demand = torch.rand(1, 1, 2, 4)

        self.assertTrue(torch.equal(module(x, demand), x))
        self.assertEqual(module.effective_local_scale.item(), 0.0)
        self.assertEqual(module.effective_summary_scale.item(), 0.0)

    def test_local_path_and_summary_scale_then_summary_parameters_get_gradients(self):
        torch.manual_seed(0)
        module = RestorationDemandHistogramInteraction(
            8,
            bins=3,
            heads=2,
            local_initial_scale=0.01,
            summary_initial_scale=0.0,
        )
        x = torch.randn(2, 8, 2, 5)
        demand = torch.rand(2, 1, 2, 5)

        module(x, demand).square().mean().backward()
        local_grad = module.bin_attention.in_proj_weight.grad
        summary_scale_grad = module.summary_residual_scale.grad
        summary_parameter_grad = module.summary_mixer.in_proj_weight.grad
        self.assertIsNotNone(local_grad)
        self.assertGreater(local_grad.abs().sum().item(), 0.0)
        self.assertIsNotNone(summary_scale_grad)
        self.assertGreater(summary_scale_grad.abs().item(), 0.0)
        self.assertIsNotNone(summary_parameter_grad)
        self.assertEqual(summary_parameter_grad.abs().sum().item(), 0.0)

        with torch.no_grad():
            for parameter in module.parameters():
                if parameter.grad is not None:
                    parameter.add_(parameter.grad, alpha=-0.1)
        self.assertNotEqual(module.effective_summary_scale.item(), 0.0)
        for parameter in module.parameters():
            parameter.grad = None
        module(x, demand).square().mean().backward()
        summary_parameter_grad = module.summary_mixer.in_proj_weight.grad
        self.assertIsNotNone(summary_parameter_grad)
        self.assertTrue(torch.isfinite(summary_parameter_grad).all())
        self.assertGreater(summary_parameter_grad.abs().sum().item(), 0.0)

    def test_padding_content_does_not_change_valid_output(self):
        torch.manual_seed(1)
        module = RestorationDemandHistogramInteraction(8, bins=3, heads=2).eval()
        x = torch.randn(2, 8, 2, 5)
        demand = torch.rand(2, 1, 2, 5)

        with torch.no_grad():
            zero_padded = module._compute_output(x, demand, padding_value=0.0)
            large_padded = module._compute_output(x, demand, padding_value=123.0)

        self.assertTrue(torch.allclose(zero_padded, large_padded, atol=1e-6, rtol=1e-6))

    def test_parameter_increment_is_bounded_and_projection_is_identity(self):
        legacy_parameters_at_48_channels = 21313
        module = RestorationDemandHistogramInteraction(48, bins=8, heads=4)
        parameters = sum(parameter.numel() for parameter in module.parameters())

        self.assertEqual(parameters - legacy_parameters_at_48_channels, 1)
        self.assertLessEqual(parameters - legacy_parameters_at_48_channels, 4)
        self.assertTrue(
            torch.equal(module.output_projection.weight, torch.eye(48))
        )

    def test_generator_forward_shape_finite_and_diagnostics(self):
        fake_mamba_block = types.ModuleType('models.mamba_block')
        fake_mamba_block.TFMambaBlock = _IdentityBlock
        fake_mamba_block.TMambaBlock = _IdentityBlock
        fake_mamba_block.FMambaBlock = _IdentityBlock
        sys.modules.pop('models.generator', None)
        with mock.patch.dict(
            sys.modules, {'models.mamba_block': fake_mamba_block}
        ):
            generator_module = importlib.import_module('models.generator')

        def simple_rearrange(tensor, pattern, **axes_lengths):
            if pattern == 'b f t -> b t f':
                return tensor.permute(0, 2, 1)
            if pattern == 'b f t -> b 1 t f':
                return tensor.permute(0, 2, 1).unsqueeze(1)
            if pattern == 'b t f -> b f t':
                return tensor.permute(0, 2, 1)
            if pattern == 'b c t f -> b f t c':
                return tensor.permute(0, 3, 2, 1)
            raise ValueError(f'Unexpected test rearrange pattern: {pattern}')

        generator_module.rearrange = simple_rearrange
        generator_module.MagDecoder.forward.__globals__['rearrange'] = simple_rearrange

        cfg = {
            'stft_cfg': {'sampling_rate': 16000, 'n_fft': 30},
            'model_cfg': {
                'hid_feature': 4,
                'dense_channel': 4,
                'compress_factor': 0.3,
                'beta': 2.0,
                'num_tfmamba': 1,
                'num_mid_pairs': 1,
                'input_channel': 2,
                'output_channel': 1,
                'rdhi_enabled': True,
                'rdhi_bins': 8,
                'rdhi_heads': 4,
                'rdhi_local_initial_scale': 0.01,
                'rdhi_summary_initial_scale': 0.0,
                'rdhi_eps': 1e-6,
            },
        }
        patches = (
            mock.patch.object(generator_module, 'TFMambaBlock', _IdentityBlock),
            mock.patch.object(generator_module, 'TMambaBlock', _IdentityBlock),
            mock.patch.object(generator_module, 'FMambaBlock', _IdentityBlock),
            mock.patch.object(generator_module, 'Patch_Embed_stage', _IdentityBlock),
        )
        with patches[0], patches[1], patches[2], patches[3]:
            cfg_without_rdhi = copy.deepcopy(cfg)
            cfg_without_rdhi['model_cfg']['rdhi_enabled'] = False
            torch.manual_seed(7)
            model_without_rdhi = generator_module.MambaSEUNet(
                cfg_without_rdhi
            ).eval()
            torch.manual_seed(7)
            model = generator_module.MambaSEUNet(copy.deepcopy(cfg)).eval()
            rdhi_free_state = model_without_rdhi.state_dict()
            rdhi_state = model.state_dict()
            for name, value in rdhi_free_state.items():
                self.assertTrue(
                    torch.equal(value, rdhi_state[name]),
                    msg=f'RDHI changed original-layer initialization: {name}',
                )
            magnitude = torch.rand(1, 16, 16)
            phase = torch.rand_like(magnitude) * (2.0 * torch.pi) - torch.pi
            with mock.patch.object(
                model.rdhi, 'forward', wraps=model.rdhi.forward
            ) as rdhi_forward:
                with torch.no_grad():
                    enhanced_mag, enhanced_phase, enhanced_complex = model(
                        magnitude, phase
                    )

        self.assertEqual(rdhi_forward.call_count, 1)
        self.assertEqual(enhanced_mag.shape, (1, 16, 16))
        self.assertEqual(enhanced_phase.shape, (1, 16, 16))
        self.assertEqual(enhanced_complex.shape, (1, 16, 16, 2))
        self.assertTrue(torch.isfinite(enhanced_mag).all())
        self.assertTrue(torch.isfinite(enhanced_phase).all())
        self.assertTrue(torch.isfinite(enhanced_complex).all())
        expected_diagnostics = {
            'rdhi_scale',
            'rdhi_local_scale',
            'rdhi_summary_scale',
            'rdhi_local_update_ratio',
            'rdhi_summary_update_ratio',
            'rdhi_bin_demand_span_mean',
            'rdhi_demand_mean',
            'rdhi_padding_utilization',
        }
        self.assertTrue(expected_diagnostics.issubset(model.latest_aux))
        self.assertAlmostEqual(
            model.latest_aux['rdhi_local_scale'].item(), 0.01, places=6
        )
        self.assertEqual(model.latest_aux['rdhi_summary_scale'].item(), 0.0)
        for name in expected_diagnostics:
            self.assertFalse(model.latest_aux[name].requires_grad)


if __name__ == '__main__':
    unittest.main()
