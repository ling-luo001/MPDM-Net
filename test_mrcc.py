import unittest

import torch
import yaml

from models.mrcc import MRCCRefiner


def _config(enabled=True, compact=True):
    with open("recipes/Mamba-SEUNet/MRCC-MPDM-v1-mini.yaml", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    cfg["model_cfg"]["mrcc_enabled"] = enabled
    if compact:
        cfg["model_cfg"].update(
            mrcc_proposer_width=8,
            mrcc_proposer_depth=1,
            mrcc_condition_dim=8,
        )
    return cfg


def _inputs(batch=1, frames=16):
    torch.manual_seed(17)
    noisy_magnitude = torch.rand(batch, 256, frames) + 0.1
    noisy_phase = torch.rand(batch, 256, frames) * (2.0 * torch.pi) - torch.pi
    baseline_magnitude = 0.8 * noisy_magnitude + 0.02
    baseline_phase = noisy_phase + 0.1 * torch.tanh(noisy_phase)
    baseline_complex = torch.stack(
        (
            baseline_magnitude * torch.cos(baseline_phase),
            baseline_magnitude * torch.sin(baseline_phase),
        ),
        dim=-1,
    )
    return noisy_magnitude, noisy_phase, baseline_magnitude, baseline_phase, baseline_complex


class MRCCTests(unittest.TestCase):
    def test_disabled_path_is_exact_identity(self):
        module = MRCCRefiner(_config(enabled=False))
        inputs = _inputs()
        outputs = module(*inputs)
        self.assertIs(outputs[0], inputs[2])
        self.assertIs(outputs[1], inputs[3])
        self.assertIs(outputs[2], inputs[4])
        self.assertEqual(module.last_diagnostics, {})

    def test_zero_gain_contract_and_natural_staged_gradients(self):
        module = MRCCRefiner(_config())
        inputs = _inputs()
        outputs = module(*inputs)

        self.assertEqual(outputs[0].shape, inputs[2].shape)
        self.assertEqual(outputs[1].shape, inputs[3].shape)
        self.assertEqual(outputs[2].shape, inputs[4].shape)
        self.assertTrue(all(torch.isfinite(value).all() for value in outputs))
        self.assertTrue(torch.allclose(outputs[0], inputs[2], atol=2.0e-6, rtol=2.0e-6))
        phase_error = torch.atan2(
            torch.sin(outputs[1] - inputs[3]), torch.cos(outputs[1] - inputs[3])
        ).abs().amax()
        self.assertLessEqual(float(phase_error), 2.0e-6)
        self.assertTrue(torch.allclose(outputs[2], inputs[4], atol=2.0e-6, rtol=2.0e-6))

        loss = outputs[0].square().mean() + outputs[2].square().mean()
        loss.backward()
        self.assertTrue(torch.isfinite(module.correction_gains.grad).all())
        self.assertTrue((module.correction_gains.grad.abs() > 0.0).all())
        proposer_gradients = [
            parameter.grad
            for parameter in module.proposer.parameters()
            if parameter.grad is not None
        ]
        self.assertTrue(proposer_gradients)
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in proposer_gradients))
        self.assertEqual(
            sum(float(gradient.abs().sum()) for gradient in proposer_gradients), 0.0
        )

    def test_small_nonzero_gains_activate_finite_proposer_gradients(self):
        module = MRCCRefiner(_config())
        with torch.no_grad():
            module.correction_gains.copy_(torch.tensor([0.01, -0.01]))
        outputs = module(*_inputs())
        loss = outputs[0].square().mean() + outputs[2].square().mean()
        loss.backward()

        proposer_gradients = [
            parameter.grad
            for parameter in module.proposer.parameters()
            if parameter.grad is not None
        ]
        self.assertTrue(proposer_gradients)
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in proposer_gradients))
        self.assertGreater(
            sum(float(gradient.abs().sum()) for gradient in proposer_gradients), 0.0
        )

    def test_proposal_and_reliability_bounds(self):
        module = MRCCRefiner(_config())
        torch.manual_seed(23)
        noisy = torch.randn(2, 2, 17, 19)
        baseline = noisy + 0.2 * torch.randn_like(noisy)
        correction, reliability, proposed, ratio = module._propose(noisy, baseline, 0)

        self.assertTrue(torch.equal(correction, torch.zeros_like(correction)))
        self.assertLessEqual(float(ratio.amax()), 1.0 + 1.0e-6)
        self.assertLess(float(reliability.amax()), module.reliability_ceiling)
        self.assertGreater(float(reliability.amin()), module.reliability_floor)
        self.assertTrue(torch.isfinite(proposed).all())

    def test_two_iteration_consensus_objective_does_not_increase(self):
        module = MRCCRefiner(_config())
        with torch.no_grad():
            module.correction_gains.copy_(torch.tensor([0.15, -0.1]))
        module(*_inputs())
        objectives = [
            float(module.last_diagnostics[name])
            for name in ("objective_initial", "objective_iter1", "objective_iter2")
        ]
        self.assertGreater(objectives[0], 0.0)
        self.assertLessEqual(objectives[1], objectives[0] + 1.0e-10)
        self.assertLessEqual(objectives[2], objectives[1] + 1.0e-10)
        self.assertLess(objectives[2], objectives[0])
        self.assertGreaterEqual(float(module.last_diagnostics["reliability_min"]), 0.05)
        self.assertLessEqual(float(module.last_diagnostics["reliability_max"]), 0.95)
        self.assertLessEqual(
            float(module.last_diagnostics["proposed_correction_ratio_max"]), 1.0 + 1.0e-6
        )

    def test_frequency_placement_changes_full_tf_consensus(self):
        module = MRCCRefiner(_config())
        torch.manual_seed(29)
        waveform_length = 512
        target_waveform = torch.randn(1, waveform_length)
        proposals = [
            module._stft_ri(target_waveform, resolution)
            for resolution in module.aux_resolutions
        ]
        reliability_a = [
            torch.full_like(proposal[:, :1], 0.2) for proposal in proposals
        ]
        reliability_b = [value.clone() for value in reliability_a]
        reliability_a[0][:, :, 4, :] = 0.8
        reliability_b[0][:, :, 20, :] = 0.8
        reliability_a[1][:, :, 7, :] = 0.8
        reliability_b[1][:, :, 31, :] = 0.8

        for first, second in zip(reliability_a, reliability_b):
            self.assertTrue(torch.equal(first.mean(dim=2), second.mean(dim=2)))
        correction_a, objectives_a, _ = module._consensus(
            proposals, reliability_a, waveform_length
        )
        correction_b, objectives_b, _ = module._consensus(
            proposals, reliability_b, waveform_length
        )

        self.assertGreater(float((correction_a - correction_b).abs().amax()), 1.0e-6)
        self.assertLessEqual(float(objectives_a[-1]), float(objectives_a[0]))
        self.assertLessEqual(float(objectives_b[-1]), float(objectives_b[0]))


if __name__ == "__main__":
    unittest.main()
