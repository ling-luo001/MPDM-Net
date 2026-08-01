import math

import torch

from models.generator import MultiScaleLocalChannelRefiner


def test_refiner_preserves_shape_and_backpropagates():
    torch.manual_seed(7)
    refiner = MultiScaleLocalChannelRefiner(
        channels=16, strip_kernel=7, initial_scale=0.1
    )
    features = torch.randn(2, 16, 24, 20, requires_grad=True)

    output = refiner(features)
    assert output.shape == features.shape
    assert torch.isfinite(output).all()
    assert math.isclose(
        torch.tanh(refiner.residual_scale).item(), 0.1, rel_tol=0.0, abs_tol=1e-6
    )

    output.square().mean().backward()
    assert torch.isfinite(features.grad).all()
    for branch in (
        refiner.local_3x3,
        refiner.temporal_strip,
        refiner.frequency_strip,
        refiner.channel_attention,
    ):
        assert branch.weight.grad is not None
        assert torch.isfinite(branch.weight.grad).all()
        assert branch.weight.grad.abs().sum() > 0


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


if __name__ == '__main__':
    test_refiner_preserves_shape_and_backpropagates()
    test_refiner_rejects_invalid_settings()
    print('Multi-scale local-channel refiner tests passed.')
