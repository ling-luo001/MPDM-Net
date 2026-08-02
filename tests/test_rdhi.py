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
    def test_sort_inverse_shape_and_stability(self):
        module = RestorationDemandHistogramInteraction(8, bins=3, heads=2)
        tokens = torch.arange(2 * 10 * 8, dtype=torch.float32).view(2, 10, 8)
        demand = torch.tensor([
            [0.3, 0.1, 0.1, 0.8, 0.2, 0.5, 0.5, 0.9, 0.0, 0.4],
            [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
        ])

        sorted_tokens, indices, inverse = module._sort_by_demand(tokens, demand)
        restored = module._restore_order(sorted_tokens, inverse)

        self.assertEqual(sorted_tokens.shape, tokens.shape)
        self.assertTrue(torch.equal(restored, tokens))
        self.assertEqual(indices[0, 1].item(), 1)
        self.assertEqual(indices[0, 2].item(), 2)

    def test_non_divisible_bins_are_finite(self):
        module = RestorationDemandHistogramInteraction(8, bins=3, heads=2)
        x = torch.randn(2, 8, 2, 5)
        demand = torch.rand(2, 1, 2, 5)

        output = module(x, demand)

        self.assertEqual(output.shape, x.shape)
        self.assertTrue(torch.isfinite(output).all())
        self.assertAlmostEqual(module.padding_utilization(10), 10.0 / 12.0)

    def test_identity_and_effective_scale(self):
        module = RestorationDemandHistogramInteraction(
            8, bins=4, heads=2, initial_scale=0.05
        )
        x = torch.randn(1, 8, 2, 4)
        demand = torch.rand(1, 1, 2, 4)
        update = module._compute_update(x, demand)
        output = module(x, demand)

        self.assertAlmostEqual(module.effective_scale.item(), 0.05, places=6)
        self.assertTrue(torch.allclose(output, x + 0.05 * update, atol=1e-6))
        with torch.no_grad():
            module.residual_scale.zero_()
        self.assertTrue(torch.equal(module(x, demand), x))

    def test_finite_nonzero_gradients(self):
        torch.manual_seed(0)
        module = RestorationDemandHistogramInteraction(8, bins=3, heads=2)
        x = torch.randn(2, 8, 2, 5, requires_grad=True)
        demand = torch.rand(2, 1, 2, 5)

        module(x, demand).square().mean().backward()

        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertGreater(x.grad.abs().sum().item(), 0.0)
        parameter_gradients = [
            parameter.grad for parameter in module.parameters()
            if parameter.grad is not None
        ]
        self.assertTrue(parameter_gradients)
        self.assertTrue(all(torch.isfinite(grad).all() for grad in parameter_gradients))
        self.assertGreater(sum(grad.abs().sum().item() for grad in parameter_gradients), 0.0)

    def test_generator_forward_shape_and_finite(self):
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
            'stft_cfg': {
                'sampling_rate': 16000,
                'n_fft': 30,
            },
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
                'rdhi_initial_scale': 0.05,
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
            model = generator_module.MambaSEUNet(copy.deepcopy(cfg)).eval()
            magnitude = torch.rand(1, 16, 16)
            phase = torch.rand_like(magnitude) * (2.0 * torch.pi) - torch.pi
            with torch.no_grad():
                enhanced_mag, enhanced_phase, enhanced_complex = model(
                    magnitude, phase
                )

        self.assertEqual(enhanced_mag.shape, (1, 16, 16))
        self.assertEqual(enhanced_phase.shape, (1, 16, 16))
        self.assertEqual(enhanced_complex.shape, (1, 16, 16, 2))
        self.assertTrue(torch.isfinite(enhanced_mag).all())
        self.assertTrue(torch.isfinite(enhanced_phase).all())
        self.assertTrue(torch.isfinite(enhanced_complex).all())
        self.assertIn('rdhi_scale', model.latest_aux)
        self.assertIn('rdhi_demand_mean', model.latest_aux)


if __name__ == '__main__':
    unittest.main()
