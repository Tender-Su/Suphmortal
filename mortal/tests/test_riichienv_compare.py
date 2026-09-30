import unittest

from mortal.eval.riichienv_compare import (
    ActorCriticNetwork,
    infer_riichippo_spec,
    summarize_games,
)


class RiichiPPOCheckpointTest(unittest.TestCase):
    def test_infers_model_spec_and_roundtrips_state_dict(self):
        model = ActorCriticNetwork(
            in_channels=74,
            num_actions=82,
            conv_channels=32,
            num_blocks=2,
            fc_dim=64,
            tile_dim=34,
        )
        state = model.state_dict()
        spec = infer_riichippo_spec(state)
        restored = ActorCriticNetwork(**spec)
        restored.load_state_dict(state, strict=True)

        self.assertEqual(74, spec['in_channels'])
        self.assertEqual(82, spec['num_actions'])
        self.assertEqual(2, spec['num_blocks'])
        self.assertEqual(34, spec['tile_dim'])


class SummarizeGamesTest(unittest.TestCase):
    def test_uses_duplicate_seed_sets_for_confidence_interval(self):
        games = [
            {'seed': 10, 'challenger_rank': rank}
            for rank in (1, 2, 3, 4)
        ] + [
            {'seed': 11, 'challenger_rank': rank}
            for rank in (1, 1, 2, 2)
        ]
        summary = summarize_games(games)

        self.assertEqual(2, summary['sets'])
        self.assertEqual(8, summary['games'])
        self.assertEqual([3, 3, 1, 1], summary['rankings'])
        self.assertEqual('duplicate_seed_set', summary['ci_unit'])
        self.assertAlmostEqual(2.0, summary['avg_rank'])
        self.assertAlmostEqual(33.75, summary['avg_pt'])


if __name__ == '__main__':
    unittest.main()
