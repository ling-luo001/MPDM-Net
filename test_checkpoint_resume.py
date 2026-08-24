import os
import tempfile
import unittest
from types import SimpleNamespace

import torch

from utils.util import load_ckpts, load_optimizer_states, scan_checkpoint_pair


class ScanCheckpointPairTest(unittest.TestCase):
    def _touch(self, directory, filename):
        path = os.path.join(directory, filename)
        with open(path, 'wb'):
            pass

    def test_selects_latest_step_with_both_checkpoints(self):
        with tempfile.TemporaryDirectory() as checkpoint_dir:
            for filename in (
                'g_00000002.pth', 'do_00000002.pth',
                'g_00000010.pth', 'do_00000010.pth',
                'g_00000020.pth', 'do_00000030.pth',
            ):
                self._touch(checkpoint_dir, filename)

            cp_g, cp_do = scan_checkpoint_pair(checkpoint_dir)

            self.assertEqual(os.path.basename(cp_g), 'g_00000010.pth')
            self.assertEqual(os.path.basename(cp_do), 'do_00000010.pth')

    def test_returns_none_when_no_pair_exists(self):
        with tempfile.TemporaryDirectory() as checkpoint_dir:
            self._touch(checkpoint_dir, 'g_00000010.pth')
            self._touch(checkpoint_dir, 'do_00000020.pth')

            args = SimpleNamespace(
                resume_from=checkpoint_dir,
                resume_step=None,
                exp_path='unused',
            )
            self.assertEqual(load_ckpts(args, 'cpu'), (None, None, 0, -1))


class OptimizerGroupResumeTest(unittest.TestCase):
    def _optimizer_pair(self):
        parent = torch.nn.Parameter(torch.ones(()))
        refiner = torch.nn.Parameter(torch.ones(()))
        discriminator = torch.nn.Parameter(torch.ones(()))
        generator_optimizer = torch.optim.AdamW([
            {
                'params': [parent],
                'lr': 1e-3,
                'initial_lr': 1e-3,
                'lr_scale': 1.0,
            },
            {
                'params': [refiner],
                'lr': 2e-4,
                'initial_lr': 2e-4,
                'lr_scale': 0.2,
            },
        ])
        discriminator_optimizer = torch.optim.AdamW(
            [discriminator], lr=1e-3
        )
        return generator_optimizer, discriminator_optimizer

    def test_resume_preserves_learning_rate_scales(self):
        source_g, source_d = self._optimizer_pair()
        source_g.param_groups[0]['lr'] = 5e-4
        source_g.param_groups[1]['lr'] = 1e-4
        state = {
            'optim_g': source_g.state_dict(),
            'optim_d': source_d.state_dict(),
        }
        target = self._optimizer_pair()
        load_optimizer_states(
            target,
            state,
            cfg={'training_cfg': {'learning_rate': 2e-3}},
        )
        self.assertAlmostEqual(target[0].param_groups[0]['lr'], 1e-3)
        self.assertAlmostEqual(target[0].param_groups[1]['lr'], 2e-4)
        self.assertAlmostEqual(
            target[0].param_groups[1]['lr']
            / target[0].param_groups[0]['lr'],
            0.2,
        )

    def test_resume_lr_is_scaled_per_group(self):
        source = self._optimizer_pair()
        state = {
            'optim_g': source[0].state_dict(),
            'optim_d': source[1].state_dict(),
        }
        target = self._optimizer_pair()
        load_optimizer_states(
            target,
            state,
            cfg={'training_cfg': {'learning_rate': 2e-3}},
            resume_lr=4e-4,
        )
        self.assertAlmostEqual(target[0].param_groups[0]['lr'], 4e-4)
        self.assertAlmostEqual(target[0].param_groups[1]['lr'], 8e-5)


if __name__ == '__main__':
    unittest.main()
