import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import sl_selection as s05


class Stage05SelectionTests(unittest.TestCase):
    def test_selection_scenario_factor_retuned_to_dense_search_value(self):
        self.assertAlmostEqual(0.20, s05.SELECTION_SCENARIO_FACTOR)

    def test_action_quality_score_uses_new_chi_exact_component(self):
        metrics = {
            'chi_decision_balanced_bce': 0.50,
            'chi_decision_pos_count': 2048,
            'chi_decision_neg_count': 2048,
            'chi_exact_nll': 1.00,
            'chi_exact_count': 2048,
        }
        expected = -(
            s05.ACTION_SCORE_WEIGHTS['chi_decision_balanced_bce'] * (0.50 / 1.50) * (2048 / (2048 + 1024))
            + s05.ACTION_SCORE_WEIGHTS['chi_exact_nll'] * (1.00 / 2.00) * (2048 / (2048 + 384))
        )
        self.assertAlmostEqual(expected, s05.action_quality_score(metrics))

    def test_action_quality_score_recomputes_when_raw_metrics_exist(self):
        metrics = {
            'action_quality_score': 123.0,
            'discard_nll': 1.0,
            'discard_count': 16384,
        }
        self.assertLess(s05.action_quality_score(metrics), 0.0)
        self.assertNotEqual(123.0, s05.action_quality_score(metrics))

    def test_action_quality_score_falls_back_to_explicit_value_without_components(self):
        self.assertEqual(1.23, s05.action_quality_score({'action_quality_score': 1.23}))

    def test_scenario_quality_score_shrinks_low_count_slices(self):
        high_count = {
            'discard_push_fold_extreme_nll': 0.40,
            'discard_push_fold_extreme_count': 5000,
        }
        low_count = {
            'discard_push_fold_extreme_nll': 0.40,
            'discard_push_fold_extreme_count': 20,
        }
        self.assertLess(s05.scenario_quality_score(high_count), s05.scenario_quality_score(low_count))

    def test_scenario_quality_score_uses_new_p0_p1_p2_slices(self):
        metrics = {
            'riichi_decision_role_dealer_balanced_bce': 0.20,
            'riichi_decision_role_dealer_pos_count': 2048,
            'riichi_decision_role_dealer_neg_count': 2048,
            'riichi_decision_gap_up_close_2k_balanced_bce': 0.30,
            'riichi_decision_gap_up_close_2k_pos_count': 2048,
            'riichi_decision_gap_up_close_2k_neg_count': 2048,
            'discard_push_fold_core_nll': 0.40,
            'discard_push_fold_core_count': 2048,
        }
        expected = -(
            s05.SCENARIO_SCORE_WEIGHTS['riichi_decision_role_dealer_balanced_bce'] * (0.20 / 1.20) * (2048 / (2048 + 256))
            + s05.SCENARIO_SCORE_WEIGHTS['riichi_decision_gap_up_close_2k_balanced_bce'] * (0.30 / 1.30) * (2048 / (2048 + 256))
            + s05.SCENARIO_SCORE_WEIGHTS['discard_push_fold_core_nll'] * (0.40 / 1.40) * (2048 / (2048 + 384))
        )
        self.assertAlmostEqual(expected, s05.scenario_quality_score(metrics))

    def test_scenario_quality_score_falls_back_to_explicit_value_without_raw_components(self):
        self.assertEqual(
            1.23,
            s05.scenario_quality_score({
                'scenario_quality_score': 1.23,
                s05.SCENARIO_SCORE_VERSION_FIELD: s05.SCENARIO_SCORE_VERSION,
            }),
        )

    def test_scenario_quality_score_ignores_stale_explicit_value_without_matching_version(self):
        self.assertTrue(
            math.isinf(
                s05.scenario_quality_score({'scenario_quality_score': 1.23})
            )
        )

    def test_scenario_quality_score_recomputes_when_raw_metrics_exist_even_if_cached_version_is_stale(self):
        metrics = {
            'scenario_quality_score': 123.0,
            s05.SCENARIO_SCORE_VERSION_FIELD: 'stale-version',
            'discard_push_fold_core_nll': 0.40,
            'discard_push_fold_core_count': 2048,
        }
        self.assertLess(s05.scenario_quality_score(metrics), 0.0)
        self.assertNotEqual(123.0, s05.scenario_quality_score(metrics))

    def test_selection_tiebreak_uses_combined_selection_score(self):
        left = {
            'discard_nll': 0.30,
            'discard_count': 20000,
            'discard_push_fold_core_nll': 0.60,
            'discard_push_fold_core_count': 4000,
        }
        right = {
            'discard_nll': 0.31,
            'discard_count': 20000,
            'discard_push_fold_core_nll': 0.20,
            'discard_push_fold_core_count': 4000,
        }
        self.assertGreater(
            s05.selection_tiebreak_key(left, recent_loss=0.5, old_regression_loss=0.5),
            s05.selection_tiebreak_key(right, recent_loss=0.5, old_regression_loss=0.5),
        )

    def test_selection_tiebreak_does_not_recompare_scenario_and_action_after_selection_score(self):
        left = {
            'action_quality_score': -0.2000,
            'scenario_quality_score': -0.1000,
            s05.SCENARIO_SCORE_VERSION_FIELD: s05.SCENARIO_SCORE_VERSION,
        }
        right = {
            'action_quality_score': -0.2200,
            'scenario_quality_score': 0.0,
            s05.SCENARIO_SCORE_VERSION_FIELD: s05.SCENARIO_SCORE_VERSION,
        }
        self.assertAlmostEqual(
            s05.selection_quality_score(left),
            s05.selection_quality_score(right),
        )
        left_key = s05.selection_tiebreak_key(left, recent_loss=0.5, old_regression_loss=0.5)
        right_key = s05.selection_tiebreak_key(right, recent_loss=0.5, old_regression_loss=0.5)
        self.assertAlmostEqual(left_key[0], right_key[0])
        self.assertEqual(
            left_key[1:],
            right_key[1:],
        )

    def test_selection_tiebreak_ignores_accuracy_metrics(self):
        base = {
            'action_quality_score': -0.20,
            'scenario_quality_score': -0.10,
            s05.SCENARIO_SCORE_VERSION_FIELD: s05.SCENARIO_SCORE_VERSION,
        }
        with_acc = {
            **base,
            'discard_top1_acc': 0.99,
            'riichi_decision_balanced_acc': 0.99,
            'agari_decision_balanced_acc': 0.99,
            'chi_decision_balanced_acc': 0.99,
            'pon_decision_balanced_acc': 0.99,
            'kan_decision_balanced_acc': 0.99,
            'rank_acc': 0.99,
        }
        self.assertEqual(
            s05.selection_tiebreak_key(base, recent_loss=0.5, old_regression_loss=0.6),
            s05.selection_tiebreak_key(with_acc, recent_loss=0.5, old_regression_loss=0.6),
        )

    def test_scenario_score_weights_cover_requested_new_slice_families(self):
        required = {
            'riichi_decision_role_dealer_balanced_bce',
            'riichi_decision_role_nondealer_balanced_bce',
            'riichi_decision_rank_1_balanced_bce',
            'riichi_decision_rank_4_balanced_bce',
            'riichi_decision_pressure_calm_balanced_bce',
            'agari_decision_turn_late_balanced_bce',
            'kan_decision_gap_close_4k_balanced_bce',
            'riichi_decision_gap_up_close_2k_balanced_bce',
            'riichi_decision_gap_down_close_2k_balanced_bce',
            'riichi_decision_pressure_multi_threat_balanced_bce',
            'riichi_decision_all_last_target_escape_fourth_balanced_bce',
            'riichi_decision_opp_any_tenpai_balanced_bce',
            'discard_all_last_yes_nll',
            'discard_pressure_threat_nll',
            'discard_gap_close_2k_nll',
            'discard_push_fold_core_nll',
            'discard_push_fold_extreme_nll',
        }
        self.assertTrue(required.issubset(set(s05.SCENARIO_SCORE_WEIGHTS)))
        self.assertIn('chi_exact_nll', s05.ACTION_SCORE_WEIGHTS)

    def test_scenario_quality_breakdown_returns_none_for_missing(self):
        breakdown = s05.scenario_quality_breakdown({})
        self.assertIn('discard_push_fold_extreme_nll', breakdown)
        self.assertIsNone(breakdown['discard_push_fold_extreme_nll'])
        self.assertTrue(math.isinf(s05.scenario_quality_score({})))


if __name__ == '__main__':
    unittest.main()
