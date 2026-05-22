import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import mortal.eval.oracle_eval as oracle_eval


class DummyStat:
    def __init__(self, avg_rank, avg_pt):
        self.avg_rank = avg_rank
        self._avg_pt = avg_pt
        self.rank_1_rate = 0.25
        self.rank_2_rate = 0.25
        self.rank_3_rate = 0.25
        self.rank_4_rate = 0.25
        self.agari_rate = 0.22
        self.houjuu_rate = 0.11
        self.riichi_rate = 0.18
        self.fuuro_rate = 0.14
        self.avg_point_per_round = 12.0

    def avg_pt(self, _rule):
        return self._avg_pt


class DummyTestPlayer:
    def __init__(self):
        self.calls = []

    def test_play(self, seed_count, mortal, dqn, device, *, search_runtime_bundle=None, oracle_input_mode='zero', oracle_guiding_keep_prob=1.0):
        self.calls.append((seed_count, oracle_input_mode))
        if oracle_input_mode == 'true':
            return DummyStat(2.1, 35.0)
        if oracle_input_mode == 'shuffled':
            return DummyStat(2.8, -20.0)
        return DummyStat(2.4, 10.0)


class DummyMortal:
    def __init__(self, is_oracle):
        self.is_oracle = is_oracle


class OracleEvalTests(unittest.TestCase):
    def test_evaluate_dependency_modes_uses_precomputed_zero(self):
        test_player = DummyTestPlayer()
        results = oracle_eval.evaluate_oracle_dependency_modes(
            test_player,
            DummyMortal(is_oracle=True),
            dqn=None,
            device='cpu',
            seed_count=64,
            precomputed_zero={'avg_rank': 2.4, 'avg_pt': 10.0, 'agari_rate': 0.2, 'houjuu_rate': 0.1},
        )
        self.assertEqual(2, len(test_player.calls))
        self.assertEqual(25.0, results['modes']['true']['delta_vs_zero_avg_pt'])
        self.assertEqual(-30.0, results['modes']['shuffled']['delta_vs_zero_avg_pt'])


if __name__ == '__main__':
    unittest.main()
