import os
import tempfile
import unittest
from types import SimpleNamespace

from utils.util import load_ckpts, scan_checkpoint_pair


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


if __name__ == '__main__':
    unittest.main()
