import unittest

import torch

from models.wavelet_fusion import (
    DirectionalSubbandExchange,
    ResidualDenseDirectionalExchange,
    WaveletSubbandCrossFusion,
    haar_dwt2,
    haar_idwt2,
)


class HaarWaveletTest(unittest.TestCase):
    def test_even_and_odd_shapes_reconstruct_exactly(self):
        for shape in ((2, 3, 8, 10), (1, 2, 7, 9), (1, 2, 1, 5)):
            with self.subTest(shape=shape):
                value = torch.randn(*shape, requires_grad=True)
                bands, output_size = haar_dwt2(value)
                reconstructed = haar_idwt2(bands, output_size)
                self.assertEqual(reconstructed.shape, value.shape)
                self.assertTrue(
                    torch.allclose(reconstructed, value, atol=1e-6, rtol=1e-6)
                )
                reconstructed.square().mean().backward()
                self.assertTrue(torch.isfinite(value.grad).all())


class WaveletSubbandCrossFusionTest(unittest.TestCase):
    def test_zero_start_is_identity(self):
        module = WaveletSubbandCrossFusion(channels=4, levels=2)
        magnitude = torch.randn(2, 4, 15, 17)
        phase = torch.randn(2, 4, 15, 17)
        enhanced_magnitude, enhanced_phase = module(magnitude, phase)
        self.assertTrue(
            torch.allclose(enhanced_magnitude, magnitude, atol=1e-6, rtol=1e-6)
        )
        self.assertTrue(
            torch.allclose(enhanced_phase, phase, atol=1e-6, rtol=1e-6)
        )

    def test_cross_update_has_finite_gradients(self):
        module = WaveletSubbandCrossFusion(channels=4, levels=2)
        for submodule in module.modules():
            if hasattr(submodule, "raw_magnitude_scale"):
                submodule.raw_magnitude_scale.data.fill_(0.1)
                submodule.raw_phase_scale.data.fill_(0.1)

        magnitude = torch.randn(2, 4, 15, 17, requires_grad=True)
        phase = torch.randn(2, 4, 15, 17, requires_grad=True)
        enhanced_magnitude, enhanced_phase = module(magnitude, phase)
        loss = enhanced_magnitude.square().mean() + enhanced_phase.square().mean()
        loss.backward()
        self.assertTrue(torch.isfinite(magnitude.grad).all())
        self.assertTrue(torch.isfinite(phase.grad).all())
        self.assertTrue(
            all(
                parameter.grad is None or torch.isfinite(parameter.grad).all()
                for parameter in module.parameters()
            )
        )

    def test_dense_context_update_has_finite_gradients(self):
        module = ResidualDenseDirectionalExchange(
            channels=4,
            kernel_size=(5, 1),
            dense_depth=3,
        )
        module.dense_adapter.raw_magnitude_scale.data.fill_(0.1)
        module.dense_adapter.raw_phase_scale.data.fill_(0.1)
        magnitude = torch.randn(2, 4, 9, 7, requires_grad=True)
        phase = torch.randn(2, 4, 9, 7, requires_grad=True)
        context_magnitude = torch.randn(2, 4, 9, 7, requires_grad=True)
        context_phase = torch.randn(2, 4, 9, 7, requires_grad=True)
        enhanced_magnitude, enhanced_phase = module(
            magnitude,
            phase,
            context_magnitude,
            context_phase,
        )
        loss = enhanced_magnitude.square().mean() + enhanced_phase.square().mean()
        loss.backward()
        for tensor in (magnitude, phase, context_magnitude, context_phase):
            self.assertTrue(torch.isfinite(tensor.grad).all())

    def test_dense_adapter_preserves_baseline_rng_sequence(self):
        torch.manual_seed(19)
        baseline = DirectionalSubbandExchange(4, (3, 3))
        baseline_next_random = torch.rand(8)

        torch.manual_seed(19)
        optimized = ResidualDenseDirectionalExchange(4, (3, 3))
        optimized_next_random = torch.rand(8)

        for key, value in baseline.state_dict().items():
            self.assertTrue(
                torch.equal(value, optimized.base_exchange.state_dict()[key])
            )
        self.assertTrue(torch.equal(baseline_next_random, optimized_next_random))


if __name__ == "__main__":
    unittest.main()
