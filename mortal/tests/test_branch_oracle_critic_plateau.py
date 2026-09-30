import unittest

from scripts.branch_oracle_critic_plateau import replay_plateau_history


class TestBranchOracleCriticPlateau(unittest.TestCase):
    def test_improvements_preserve_shared_constant_lr_trunk(self):
        history = [
            {"step": 10000, "metric": 3.30},
            {"step": 40000, "metric": 3.29},
            {"step": 80000, "metric": 3.28},
            {"step": 100000, "metric": 3.281},
        ]

        replay = replay_plateau_history(
            history,
            peak=5e-5,
            factor=0.5,
            patience_steps=40000,
            threshold=0.0,
            min_lr=1e-6,
        )

        self.assertEqual(0, replay["num_reductions"])
        self.assertEqual(80000, replay["last_improvement_step"])
        self.assertEqual(5e-5, replay["plateau_lr"])

    def test_stale_history_reduces_at_patience_boundary(self):
        history = [
            {"step": 80000, "metric": 3.28},
            {"step": 110000, "metric": 3.281},
            {"step": 120000, "metric": 3.282},
        ]

        replay = replay_plateau_history(
            history,
            peak=5e-5,
            factor=0.5,
            patience_steps=40000,
            threshold=0.0,
            min_lr=1e-6,
        )

        self.assertEqual(1, replay["num_reductions"])
        self.assertEqual(120000, replay["last_improvement_step"])
        self.assertEqual(2.5e-5, replay["plateau_lr"])
        self.assertEqual("reduce_lr", replay["observations"][-1]["action"])


if __name__ == "__main__":
    unittest.main()
