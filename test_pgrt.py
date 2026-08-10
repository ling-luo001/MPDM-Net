import math

import torch

from models.pgrt import (
    ConservativeSoftSplat2D,
    PGRTInteraction,
    compute_analytic_phase_field,
)


def test_analytic_constant_and_linear_phase_fields():
    freq_bins, frames = 7, 9
    magnitude = torch.ones(1, freq_bins, frames)
    constant_phase = torch.full_like(magnitude, 0.37)
    offsets, confidence = compute_analytic_phase_field(
        magnitude,
        constant_phase,
        target_size=(5, 4),
    )
    assert offsets.shape == (1, 2, 5, 4)
    assert confidence.shape == (1, 1, 5, 4)
    assert torch.equal(offsets, torch.zeros_like(offsets))

    time_slope = 0.2
    frequency_slope = -0.15
    time = torch.arange(frames).view(1, 1, frames)
    frequency = torch.arange(freq_bins).view(1, freq_bins, 1)
    linear_phase = time_slope * time + frequency_slope * frequency
    offsets, _ = compute_analytic_phase_field(
        magnitude,
        linear_phase,
        target_size=(frames, freq_bins),
    )
    expected_time = torch.full_like(offsets[:, 0], -frequency_slope / math.pi)
    expected_frequency = torch.full_like(offsets[:, 1], time_slope / math.pi)
    torch.testing.assert_close(offsets[:, 0], expected_time, atol=1e-6, rtol=0.0)
    torch.testing.assert_close(offsets[:, 1], expected_frequency, atol=1e-6, rtol=0.0)

    n_fft, hop_size = 16, 4
    frame = torch.arange(frames).view(1, 1, frames)
    frequency = torch.arange(freq_bins).view(1, freq_bins, 1)
    carrier = 2.0 * math.pi * frequency * frame * hop_size / n_fft
    offsets, _ = compute_analytic_phase_field(
        magnitude,
        carrier + time_slope * frame + frequency_slope * frequency,
        target_size=(frames, freq_bins),
        n_fft=n_fft,
        hop_size=hop_size,
    )
    torch.testing.assert_close(offsets[:, 0], expected_time, atol=4e-6, rtol=0.0)
    torch.testing.assert_close(offsets[:, 1], expected_frequency, atol=4e-6, rtol=0.0)

    varying_magnitude = torch.ones_like(magnitude)
    varying_magnitude[:, 0] = 1e-4
    _, reliability = compute_analytic_phase_field(
        varying_magnitude,
        linear_phase,
        target_size=(frames, freq_bins),
    )
    assert reliability[:, :, :, 0].mean() < reliability[:, :, :, 1:].mean()
    assert offsets.abs().amax() <= 1.0

    wrapped_increments = torch.tensor(
        [[[0.0, math.pi - 0.1, 0.0]]]
    )
    offsets, _ = compute_analytic_phase_field(
        torch.ones_like(wrapped_increments),
        wrapped_increments,
        target_size=(3, 1),
    )
    assert offsets[0, 1, 1, 0].abs() > 0.95


def test_soft_splat_zero_offset_identity_and_source_weights():
    torch.manual_seed(0)
    splat = ConservativeSoftSplat2D()
    features = torch.randn(2, 3, 4, 5)
    offsets = torch.zeros(2, 2, 4, 5)
    output = splat(features, offsets)
    assert torch.equal(output, features)
    weights = splat.source_weights(offsets)
    assert torch.equal(weights.sum(dim=1), torch.ones_like(weights[:, 0]))


def test_soft_splat_mass_conservation_and_boundaries():
    for boundary_scale in (0.8, 20.0):
        torch.manual_seed(1)
        splat = ConservativeSoftSplat2D()
        features = torch.rand(2, 4, 5, 6)
        offsets = boundary_scale * torch.randn(2, 2, 5, 6)
        output = splat(features, offsets)
        torch.testing.assert_close(
            output.sum(dim=(-2, -1)),
            features.sum(dim=(-2, -1)),
            atol=2e-5,
            rtol=2e-6,
        )
        weights = splat.source_weights(offsets)
        torch.testing.assert_close(
            weights.sum(dim=1),
            torch.ones_like(weights[:, 0]),
            atol=1e-7,
            rtol=0.0,
        )


def test_soft_splat_adjoint_inner_product():
    torch.manual_seed(2)
    splat = ConservativeSoftSplat2D()
    source = torch.randn(2, 3, 4, 6, dtype=torch.float64)
    target = torch.randn_like(source)
    offsets = 0.7 * torch.randn(2, 2, 4, 6, dtype=torch.float64)
    lhs = (splat(source, offsets) * target).sum()
    rhs = (source * splat.adjoint(target, offsets)).sum()
    torch.testing.assert_close(lhs, rhs, atol=1e-11, rtol=1e-11)


def test_soft_splat_feature_and_offset_gradients_are_finite():
    torch.manual_seed(3)
    splat = ConservativeSoftSplat2D()
    features = torch.randn(2, 3, 4, 5, requires_grad=True)
    offsets = (0.3 * torch.randn(2, 2, 4, 5)).requires_grad_()
    target = torch.randn_like(features)
    loss = (splat(features, offsets) * target).square().mean()
    loss = loss + (splat.adjoint(target, offsets) * features).mean()
    loss.backward()
    assert features.grad is not None and torch.isfinite(features.grad).all()
    assert offsets.grad is not None and torch.isfinite(offsets.grad).all()
    assert offsets.grad.abs().sum() > 0.0


def test_pgrt_zero_injection_matches_base_value_and_jacobian_exactly():
    torch.manual_seed(4)
    module = PGRTInteraction(channels=2, num_stages=3, hidden=4).double()
    shape = (1, 2, 2, 2)
    offsets = torch.zeros(1, 2, 2, 2, dtype=torch.float64)
    confidence = torch.full((1, 1, 2, 2), 0.8, dtype=torch.float64)
    mag = torch.randn(shape, dtype=torch.float64, requires_grad=True)
    phase = torch.randn(shape, dtype=torch.float64, requires_grad=True)

    def combined_output(mag_input, phase_input):
        base_mag = 1.25 * mag_input + 0.2 * phase_input
        base_phase = -0.4 * mag_input + 0.75 * phase_input
        output_mag, output_phase = module(
            mag_input,
            phase_input,
            offsets,
            confidence,
            stage_index=1,
            base_outputs=(base_mag, base_phase),
        )
        return torch.cat((output_mag.flatten(), output_phase.flatten()))

    actual = combined_output(mag, phase)
    expected = torch.cat(
        ((1.25 * mag + 0.2 * phase).flatten(), (-0.4 * mag + 0.75 * phase).flatten())
    )
    assert torch.equal(actual, expected)

    actual_jacobian = torch.autograd.functional.jacobian(combined_output, (mag, phase))

    def base_only(mag_input, phase_input):
        return torch.cat(
            (
                (1.25 * mag_input + 0.2 * phase_input).flatten(),
                (-0.4 * mag_input + 0.75 * phase_input).flatten(),
            )
        )

    base_jacobian = torch.autograd.functional.jacobian(base_only, (mag, phase))
    assert torch.equal(actual_jacobian[0], base_jacobian[0])
    assert torch.equal(actual_jacobian[1], base_jacobian[1])


def test_pgrt_nonzero_scale_trains_shared_predictor_and_refiner():
    torch.manual_seed(5)
    module = PGRTInteraction(channels=4, num_stages=2, hidden=6)
    with torch.no_grad():
        module.stage_branch_scales[1].copy_(torch.tensor((0.2, -0.15)))

    mag = torch.randn(2, 4, 4, 5, requires_grad=True)
    phase = torch.randn_like(mag, requires_grad=True)
    analytic_offsets = 0.2 * torch.tanh(torch.randn(2, 2, 4, 5))
    analytic_confidence = torch.full((2, 1, 4, 5), 0.7)
    base_outputs = (0.6 * mag, 0.8 * phase)
    output_mag, output_phase, diagnostics = module(
        mag,
        phase,
        analytic_offsets,
        analytic_confidence,
        stage_index=1,
        base_outputs=base_outputs,
        return_diagnostics=True,
    )
    loss = output_mag.square().mean() + output_phase.square().mean()
    loss.backward()

    predictor_grads = [
        parameter.grad for parameter in module.field_predictor.parameters()
        if parameter.requires_grad
    ]
    refiner_grads = [
        parameter.grad for parameter in module.refiner.parameters()
        if parameter.requires_grad
    ]
    assert all(gradient is not None and torch.isfinite(gradient).all()
               for gradient in predictor_grads)
    assert all(gradient is not None and torch.isfinite(gradient).all()
               for gradient in refiner_grads)
    assert sum(gradient.abs().sum() for gradient in predictor_grads) > 0.0
    assert sum(gradient.abs().sum() for gradient in refiner_grads) > 0.0
    assert module.stage_branch_scales.grad is not None
    assert module.stage_branch_scales.grad[1].abs().sum() > 0.0
    assert diagnostics["source_weight_error"] <= 1e-6
    assert diagnostics["branch_scales"].abs().amax() <= 0.25
    assert diagnostics["offsets"].abs().amax() <= 1.0

    default_module = PGRTInteraction()
    parameter_count = sum(parameter.numel() for parameter in default_module.parameters())
    assert parameter_count < 120_000
    assert default_module.refiner.mag_output.weight.detach().abs().sum() > 0.0
    assert default_module.refiner.phase_output.weight.detach().abs().sum() > 0.0


if __name__ == "__main__":
    tests = sorted(
        (name, value)
        for name, value in globals().copy().items()
        if name.startswith("test_") and callable(value)
    )
    for test_name, test_function in tests:
        test_function()
        print("PASS", test_name)
