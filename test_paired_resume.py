import os
import tempfile
import types
import unittest

import torch

from utils.util import find_latest_paired_checkpoints, load_ckpts


class PairedResumeTests(unittest.TestCase):
    def test_latest_numeric_complete_pair_wins_over_unmatched_files(self):
        with tempfile.TemporaryDirectory() as checkpoint_dir:
            for filename in (
                "g_00000009.pth",
                "do_00000009.pth",
                "g_00000100.pth",
                "do_00000080.pth",
                "g_notastep.pth",
            ):
                open(os.path.join(checkpoint_dir, filename), "wb").close()
            generator, training = find_latest_paired_checkpoints(checkpoint_dir)
            self.assertEqual(os.path.basename(generator), "g_00000009.pth")
            self.assertEqual(os.path.basename(training), "do_00000009.pth")

    def test_load_ckpts_uses_the_latest_pair(self):
        with tempfile.TemporaryDirectory() as checkpoint_dir:
            torch.save({"generator": {"paired": torch.tensor(1)}}, os.path.join(checkpoint_dir, "g_00000012.pth"))
            torch.save({"steps": 12, "epoch": 3}, os.path.join(checkpoint_dir, "do_00000012.pth"))
            torch.save({"generator": {"unmatched": torch.tensor(1)}}, os.path.join(checkpoint_dir, "g_00000099.pth"))

            generator, training, next_step, epoch = load_ckpts(
                types.SimpleNamespace(exp_path=checkpoint_dir), torch.device("cpu")
            )
            self.assertIn("paired", generator["generator"])
            self.assertEqual(training["steps"], 12)
            self.assertEqual(next_step, 13)
            self.assertEqual(epoch, 3)


if __name__ == "__main__":
    unittest.main()
