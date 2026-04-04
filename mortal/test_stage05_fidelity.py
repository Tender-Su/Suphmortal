import unittest
import json
import tempfile
from pathlib import Path
import sys
from unittest.mock import patch
import math

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_stage05_fidelity as fidelity


def make_candidate(name, *, cfg_overrides=None, meta=None):
    return fidelity.CandidateSpec(
        arm_name=name,
        scheduler_profile='plateau',
        curriculum_profile='curriculum',
        weight_profile='weights',
        window_profile='window',
        cfg_overrides={} if cfg_overrides is None else cfg_overrides,
        meta={'stage': 'P1'} if meta is None else meta,
    )


def make_ranking_entry(
    candidate,
    *,
    valid=True,
    eligible=True,
    full_recent_loss=1.0,
    full_recent_metrics=None,
    old_regression_metrics=None,
):
    return {
        'arm_name': candidate.arm_name,
        'candidate_meta': candidate.meta,
        'scheduler_profile': candidate.scheduler_profile,
        'curriculum_profile': candidate.curriculum_profile,
        'weight_profile': candidate.weight_profile,
        'window_profile': candidate.window_profile,
        'cfg_overrides': candidate.cfg_overrides,
        'valid': valid,
        'eligible': eligible,
        'full_recent_loss': full_recent_loss,
        'full_recent_metrics': {} if full_recent_metrics is None else full_recent_metrics,
        'old_regression_metrics': {} if old_regression_metrics is None else old_regression_metrics,
    }


def make_current_p1_search_space(**overrides):
    payload = {
        'p1_mainline_stages': list(fidelity.P1_MAINLINE_STAGES),
        'p1_backlog_stages': list(fidelity.P1_BACKLOG_STAGES),
        'p1_ablation_policy': fidelity.P1_ABLATION_POLICY,
        'p1_ablation_note': fidelity.P1_ABLATION_NOTE,
        'calibration_protocol_arms': list(fidelity.P1_CALIBRATION_DEFAULT_PROTOCOL_ARMS),
        'protocol_decide_coordinate_mode': fidelity.P1_PROTOCOL_DECIDE_COORDINATE_MODE,
        'protocol_decide_total_budget_ratios': list(fidelity.P1_PROTOCOL_DECIDE_TOTAL_BUDGET_RATIOS),
        'protocol_decide_mixes': fidelity.current_protocol_decide_mix_payload(),
        'calibration_mode': fidelity.P1_CALIBRATION_DEFAULT_MODE,
        'inherited_single_head_source': fidelity.P1_SINGLE_HEAD_CALIBRATION_SOURCE,
        'protocol_decide_progressive_ambiguity_mode': fidelity.P1_PROTOCOL_DECIDE_PROGRESSIVE_AMBIGUITY_MODE,
        'protocol_decide_progressive_gap_threshold': fidelity.P1_PROTOCOL_DECIDE_PROGRESSIVE_GAP_THRESHOLD,
        'protocol_decide_progressive_noise_margin_mult': fidelity.P1_PROGRESSIVE_NOISE_MARGIN_MULT,
        'budget_ratio_digits': fidelity.P1_BUDGET_RATIO_DIGITS,
        'aux_weight_digits': fidelity.P1_AUX_WEIGHT_DIGITS,
        'winner_refine_center_mode': fidelity.P1_WINNER_REFINE_CENTER_MODE,
        'winner_refine_center_keep': fidelity.P1_WINNER_REFINE_CENTER_KEEP,
        'winner_refine_center_arm_names': fidelity.current_p1_winner_refine_center_arm_payload(),
        'selection_policy': {
            'policy_loss_epsilon': fidelity.P1_POLICY_LOSS_EPSILON,
            'old_regression_policy_loss_epsilon': fidelity.P1_OLD_REGRESSION_POLICY_EPSILON,
        },
    }
    payload.update(overrides)
    return payload


class Stage05FidelityCacheTests(unittest.TestCase):
    def test_p1_snapshot_uses_current_defaults_detects_old_protocol_decide_grid(self):
        self.assertTrue(
            fidelity.p1_snapshot_uses_current_defaults(
                {
                    'search_space': make_current_p1_search_space()
                }
            )
        )
        self.assertFalse(
            fidelity.p1_snapshot_uses_current_defaults(
                {
                    'search_space': make_current_p1_search_space(
                        protocol_decide_total_budget_ratios=[0.08, 0.11, 0.14]
                    )
                }
            )
        )
        self.assertFalse(
            fidelity.p1_snapshot_uses_current_defaults(
                {
                    'search_space': make_current_p1_search_space(
                        protocol_decide_mixes=[
                            {'name': 'anchor', 'rank_share': 0.43, 'opp_share': 0.21, 'danger_share': 0.36},
                            {'name': 'rank_lean', 'rank_share': 0.53, 'opp_share': 0.16, 'danger_share': 0.31},
                        ]
                    )
                }
            )
        )

    def test_p1_snapshot_uses_current_defaults_rejects_nondefault_calibration_protocol_arms(self):
        self.assertFalse(
            fidelity.p1_snapshot_uses_current_defaults(
                {
                    'search_space': make_current_p1_search_space(
                        calibration_protocol_arms=['nondefault_protocol_arm']
                    )
                }
            )
        )

    def test_p1_snapshot_uses_current_defaults_rejects_legacy_winner_refine_count_key(self):
        self.assertFalse(
            fidelity.p1_snapshot_uses_current_defaults(
                {
                    'search_space': make_current_p1_search_space(
                        winner_refine_centers=2
                    )
                }
            )
        )

    def test_apply_protocol_decide_progressive_settings_defaults_to_flip_or_gap(self):
        search_space = fidelity.apply_protocol_decide_progressive_settings({})

        self.assertEqual(
            fidelity.P1_PROTOCOL_DECIDE_PROGRESSIVE_AMBIGUITY_MODE,
            search_space['protocol_decide_progressive_ambiguity_mode'],
        )
        self.assertEqual(
            fidelity.P1_PROTOCOL_DECIDE_PROGRESSIVE_GAP_THRESHOLD,
            search_space['protocol_decide_progressive_gap_threshold'],
        )
        self.assertEqual(
            fidelity.P1_PROGRESSIVE_NOISE_MARGIN_MULT,
            search_space['protocol_decide_progressive_noise_margin_mult'],
        )

    def test_apply_protocol_decide_progressive_settings_preserves_explicit_legacy_mode(self):
        search_space = fidelity.apply_protocol_decide_progressive_settings(
            {
                'protocol_decide_progressive_ambiguity_mode': fidelity.P1_PROGRESSIVE_AMBIGUITY_MODE_LEGACY,
            }
        )

        self.assertEqual(
            fidelity.P1_PROGRESSIVE_AMBIGUITY_MODE_LEGACY,
            search_space['protocol_decide_progressive_ambiguity_mode'],
        )
        self.assertIsNone(search_space['protocol_decide_progressive_gap_threshold'])

    def test_p1_snapshot_uses_current_defaults_rejects_nondefault_single_head_source(self):
        self.assertFalse(
            fidelity.p1_snapshot_uses_current_defaults(
                {
                    'search_space': make_current_p1_search_space(
                        inherited_single_head_source='historical baseline'
                    )
                }
            )
        )

    def test_update_results_doc_shows_protocol_winner_before_final_p1_winner(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / 'fidelity_run'
            run_dir.mkdir()
            results_path = Path(tmp_dir) / 'stage05-fidelity-results.md'
            state = {
                'status': 'stopped_after_p1_protocol_decide',
                'p1': {},
                'final_conclusion': {
                    'p1_protocol_winner': 'C_A2x_cosine_broad_to_recent_strong_24m_12m',
                },
            }

            with patch.object(fidelity, 'RESULTS_DOC_PATH', results_path):
                fidelity.update_results_doc(run_dir, state)

            text = results_path.read_text(encoding='utf-8')
            self.assertIn(
                '- P1 协议 winner：`C_A2x_cosine_broad_to_recent_strong_24m_12m`',
                text,
            )
            self.assertIn('- P1 最终总胜者：`TBD`', text)
            self.assertIn('- P0 下游种子 top4：`TBD`', text)

    def test_p1_calibration_defaults_to_single_seed(self):
        self.assertEqual([0], fidelity.P1_CALIBRATION_SEED_OFFSETS)
        self.assertEqual('combo_only', fidelity.P1_CALIBRATION_DEFAULT_MODE)
        self.assertEqual(
            ['C_A2y_cosine_broad_to_recent_strong_12m_6m'],
            list(fidelity.P1_CALIBRATION_DEFAULT_PROTOCOL_ARMS),
        )

    def test_lock_belongs_to_running_process_uses_recorded_start_time(self):
        payload = {
            'pid': 4321,
            'process_start_unix_ms': 123_456,
            'created_at': '2026-03-20 10:00:00',
        }

        with (
            patch.object(fidelity, 'process_is_alive', return_value=True),
            patch.object(fidelity, 'process_start_unix_ms', return_value=123_456),
        ):
            self.assertTrue(fidelity.lock_belongs_to_running_process(payload))

        with (
            patch.object(fidelity, 'process_is_alive', return_value=True),
            patch.object(fidelity, 'process_start_unix_ms', return_value=123_999),
        ):
            self.assertFalse(fidelity.lock_belongs_to_running_process(payload))

    def test_lock_belongs_to_running_process_rejects_legacy_lock_after_pid_reuse(self):
        payload = {
            'pid': 4321,
            'created_at': '2026-03-20 10:00:00',
        }

        with (
            patch.object(fidelity, 'process_is_alive', return_value=True),
            patch.object(
                fidelity,
                'process_start_unix_ms',
                return_value=fidelity.parse_ts_to_unix_ms('2026-03-20 10:05:00'),
            ),
        ):
            self.assertFalse(fidelity.lock_belongs_to_running_process(payload))

    def test_acquire_run_lock_reclaims_stale_lock_when_identity_check_fails(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir)
            lock_path = run_dir / 'run.lock.json'
            lock_path.write_text(
                json.dumps({'pid': 2468, 'run_name': 'demo', 'created_at': '2026-03-20 10:00:00'}, ensure_ascii=False, indent=2),
                encoding='utf-8',
                newline='\n',
            )

            with (
                patch.object(
                    fidelity,
                    'build_run_lock_payload',
                    return_value={
                        'pid': 1357,
                        'run_name': 'demo',
                        'created_at': '2026-03-20 10:10:00',
                        'process_start_unix_ms': 777,
                    },
                ),
                patch.object(fidelity, 'lock_belongs_to_running_process', return_value=False),
            ):
                acquired = fidelity.acquire_run_lock(run_dir, 'demo')

            self.assertEqual(lock_path, acquired)
            payload = json.loads(lock_path.read_text(encoding='utf-8'))
            self.assertEqual(1357, payload['pid'])
            self.assertEqual(777, payload['process_start_unix_ms'])

    def test_acquire_run_lock_rejects_matching_live_lock(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir)
            lock_path = run_dir / 'run.lock.json'
            lock_path.write_text(
                json.dumps({'pid': 2468, 'run_name': 'demo', 'created_at': '2026-03-20 10:00:00'}, ensure_ascii=False, indent=2),
                encoding='utf-8',
                newline='\n',
            )

            with patch.object(fidelity, 'lock_belongs_to_running_process', return_value=True):
                with self.assertRaisesRegex(RuntimeError, 'already active under pid=2468'):
                    fidelity.acquire_run_lock(run_dir, 'demo')

    def test_reset_state_for_stop_flags_clears_stale_results(self):
        for kwargs in (
            {
                'stop_after_p0': True,
                'stop_after_p1_calibration': False,
                'stop_after_p1_protocol_decide': False,
                'stop_after_p1_winner_refine': False,
            },
            {
                'stop_after_p0': False,
                'stop_after_p1_calibration': True,
                'stop_after_p1_protocol_decide': False,
                'stop_after_p1_winner_refine': False,
            },
            {
                'stop_after_p0': False,
                'stop_after_p1_calibration': False,
                'stop_after_p1_protocol_decide': True,
                'stop_after_p1_winner_refine': False,
            },
            {
                'stop_after_p0': False,
                'stop_after_p1_calibration': False,
                'stop_after_p1_protocol_decide': False,
                'stop_after_p1_winner_refine': True,
            },
        ):
            state = {
                'started_at': '2026-03-20 09:00:00',
                'p0': {'round3': {'ranking': []}, 'winner': 'old_p0'},
                'p1': {'winner': 'old_p1', 'protocol_decide_round': {'ranking': []}},
                'formal': {'status': 'completed'},
                'final_conclusion': {
                    'p0_winner': 'old_p0',
                    'p1_winner': 'old_p1',
                    'formal_status': 'completed',
                },
            }

            fidelity.reset_state_for_stop_flags(state, **kwargs)

            self.assertEqual({'started_at': '2026-03-20 09:00:00'}, state)

    def test_build_p1_protocol_decide_candidates_keeps_ce_anchor_and_all_three_grid(self):
        protocol = fidelity.CandidateSpec(
            arm_name='C_A2x_cosine_broad_to_recent_strong_24m_12m',
            scheduler_profile='cosine',
            curriculum_profile='broad_to_recent',
            weight_profile='strong',
            window_profile='24m_12m',
            cfg_overrides={},
            meta={'protocol_arm': 'C_A2x_cosine_broad_to_recent_strong_24m_12m'},
        )
        calibration = {
            'rank_effective_base': 0.05121494575615136,
            'opp_weight_per_budget_unit': 0.052,
            'danger_weight_per_budget_unit': 0.18,
            'triple_combo_factor': 0.902,
            'joint_combo_factor': 0.968,
            'protocol_triple_combo_factors': {
                'C_A2x_cosine_broad_to_recent_strong_24m_12m': 0.902,
            },
            'protocol_joint_combo_factors': {
                'C_A2x_cosine_broad_to_recent_strong_24m_12m': 0.968,
            },
        }

        candidates = fidelity.build_p1_protocol_decide_candidates([protocol], calibration)

        self.assertEqual(
            1 + len(fidelity.P1_PROTOCOL_DECIDE_TOTAL_BUDGET_RATIOS) * len(fidelity.P1_PROTOCOL_DECIDE_MIXES),
            len(candidates),
        )
        self.assertEqual('ce_only', candidates[0].meta['aux_family'])
        all_three = [candidate for candidate in candidates if candidate.meta['aux_family'] == 'all_three']
        self.assertEqual(
            len(fidelity.P1_PROTOCOL_DECIDE_TOTAL_BUDGET_RATIOS) * len(fidelity.P1_PROTOCOL_DECIDE_MIXES),
            len(all_three),
        )
        self.assertTrue(all(candidate.meta['applied_combo_mode'] == 'triple' for candidate in all_three))
        self.assertTrue(all(abs(candidate.meta['applied_combo_factor'] - 0.902) < 1e-9 for candidate in all_three))
        self.assertEqual(
            {
                f'{mix_name}_{int(round(total_budget_ratio * 100)):02d}'
                for total_budget_ratio in fidelity.P1_PROTOCOL_DECIDE_TOTAL_BUDGET_RATIOS
                for mix_name, _, _, _ in fidelity.P1_PROTOCOL_DECIDE_MIXES
            },
            {candidate.meta['candidate_name'] for candidate in all_three},
        )
        by_name = {candidate.meta['candidate_name']: candidate for candidate in all_three}
        self.assertAlmostEqual(0.0456, by_name['opp_lean_12'].meta['effective_rank_scale'])
        self.assertAlmostEqual(0.00214, by_name['opp_lean_12'].meta['effective_opp_weight'], places=5)
        self.assertAlmostEqual(0.00743, by_name['opp_lean_12'].meta['effective_danger_weight'], places=5)
        self.assertAlmostEqual(0.00111, by_name['rank_lean_12'].meta['effective_opp_weight'], places=5)

    def test_build_p1_calibration_candidates_combo_only_uses_only_pairwise_and_triple_probes(self):
        protocol = fidelity.CandidateSpec(
            arm_name='C_A2y_cosine_broad_to_recent_strong_12m_6m',
            scheduler_profile='cosine',
            curriculum_profile='broad_to_recent',
            weight_profile='strong',
            window_profile='12m_6m',
            cfg_overrides={},
            meta={'protocol_arm': 'C_A2y_cosine_broad_to_recent_strong_12m_6m'},
        )
        candidates = fidelity.build_p1_calibration_candidates(
            [protocol],
            0.05,
            calibration_mode='combo_only',
        )

        self.assertEqual(4, len(candidates))
        self.assertEqual(
            ['rank_opp_probe', 'rank_danger_probe', 'opp_danger_probe', 'triple_probe'],
            [candidate.meta['calibration_role'] for candidate in candidates],
        )
        self.assertTrue(all(candidate.meta['calibration_mode'] == 'combo_only' for candidate in candidates))

    def test_build_p1_winner_refine_and_ablation_candidates_follow_new_round_shapes(self):
        protocol = fidelity.CandidateSpec(
            arm_name='C_A2x_cosine_broad_to_recent_strong_24m_12m',
            scheduler_profile='cosine',
            curriculum_profile='broad_to_recent',
            weight_profile='strong',
            window_profile='24m_12m',
            cfg_overrides={},
            meta={'protocol_arm': 'C_A2x_cosine_broad_to_recent_strong_24m_12m'},
        )
        calibration = {
            'rank_effective_base': 0.05121494575615136,
            'opp_weight_per_budget_unit': 0.052,
            'danger_weight_per_budget_unit': 0.18,
            'triple_combo_factor': 0.902,
            'joint_combo_factor': 0.968,
            'protocol_triple_combo_factors': {
                'C_A2x_cosine_broad_to_recent_strong_24m_12m': 0.902,
            },
            'protocol_joint_combo_factors': {
                'C_A2x_cosine_broad_to_recent_strong_24m_12m': 0.968,
            },
        }
        center = fidelity.make_p1_effective_triplet_candidate(
            protocol,
            calibration=calibration,
            rank_scale=0.052,
            opp_weight=0.001,
            danger_weight=0.008,
            stage='P1_protocol_decide_round',
            coordinate_name='anchor_12',
        )

        refine_candidates = fidelity.build_p1_winner_refine_candidates(
            [protocol],
            calibration,
            [center],
        )
        ablation_candidates = fidelity.build_p1_ablation_candidates(
            [protocol],
            calibration,
            center,
        )

        self.assertGreaterEqual(len(refine_candidates), 7)
        self.assertTrue(all(candidate.meta['aux_family'] == 'all_three' for candidate in refine_candidates))
        drop_rank = next(candidate for candidate in ablation_candidates if candidate.meta['aux_family'] == 'drop_rank')
        self.assertEqual('opp_danger', drop_rank.meta['applied_combo_mode'])
        self.assertAlmostEqual(0.968, drop_rank.meta['applied_combo_factor'])
        all_three_ablation = next(
            candidate for candidate in ablation_candidates if candidate.meta['aux_family'] == 'all_three'
        )
        self.assertAlmostEqual(
            center.meta['effective_opp_weight'],
            all_three_ablation.meta['effective_opp_weight'],
            places=5,
        )
        self.assertAlmostEqual(
            center.meta['effective_danger_weight'],
            all_three_ablation.meta['effective_danger_weight'],
            places=5,
        )
        self.assertAlmostEqual(
            center.meta['effective_opp_weight'],
            drop_rank.meta['effective_opp_weight'],
            places=5,
        )
        self.assertAlmostEqual(
            center.meta['effective_danger_weight'],
            drop_rank.meta['effective_danger_weight'],
            places=5,
        )
        self.assertEqual(
            {'ce_only', 'all_three', 'drop_rank', 'drop_opp', 'drop_danger'},
            {candidate.meta['aux_family'] for candidate in ablation_candidates},
        )
        self.assertEqual(
            len(refine_candidates),
            len({candidate.arm_name for candidate in refine_candidates}),
        )

    def test_build_p1_protocol_decide_candidates_use_unique_arm_names(self):
        protocol = fidelity.CandidateSpec(
            arm_name='C_A2x_cosine_broad_to_recent_strong_24m_12m',
            scheduler_profile='cosine',
            curriculum_profile='broad_to_recent',
            weight_profile='strong',
            window_profile='24m_12m',
            cfg_overrides={},
            meta={'protocol_arm': 'C_A2x_cosine_broad_to_recent_strong_24m_12m'},
        )
        calibration = {
            'rank_effective_base': 0.05121494575615136,
            'opp_weight_per_budget_unit': 0.052,
            'danger_weight_per_budget_unit': 0.18,
            'triple_combo_factor': 0.902,
            'joint_combo_factor': 0.968,
            'protocol_triple_combo_factors': {
                'C_A2x_cosine_broad_to_recent_strong_24m_12m': 0.902,
            },
            'protocol_joint_combo_factors': {
                'C_A2x_cosine_broad_to_recent_strong_24m_12m': 0.968,
            },
        }

        candidates = fidelity.build_p1_protocol_decide_candidates([protocol], calibration)

        self.assertEqual(
            len(candidates),
            len({candidate.arm_name for candidate in candidates}),
        )

    def test_build_p1_protocol_decide_candidates_legacy_search_space_preserves_budget_names(self):
        protocol = fidelity.CandidateSpec(
            arm_name='C_A2x_cosine_broad_to_recent_strong_24m_12m',
            scheduler_profile='cosine',
            curriculum_profile='broad_to_recent',
            weight_profile='strong',
            window_profile='24m_12m',
            cfg_overrides={},
            meta={'protocol_arm': 'C_A2x_cosine_broad_to_recent_strong_24m_12m'},
        )
        calibration = {
            'rank_effective_base': 0.05121494575615136,
            'opp_weight_per_budget_unit': 0.052,
            'danger_weight_per_budget_unit': 0.18,
            'triple_combo_factor': 0.902,
            'joint_combo_factor': 0.968,
            'protocol_triple_combo_factors': {
                'C_A2x_cosine_broad_to_recent_strong_24m_12m': 0.902,
            },
            'protocol_joint_combo_factors': {
                'C_A2x_cosine_broad_to_recent_strong_24m_12m': 0.968,
            },
        }
        legacy_search_space = {
            'protocol_decide_total_budget_ratios': [0.12],
            'protocol_decide_mixes': [
                {'name': 'opp_lean', 'rank_share': 0.38, 'opp_share': 0.31, 'danger_share': 0.31},
            ],
            'winner_refine_center_mode': 'explicit_arm_names',
            'winner_refine_center_protocol_arm': protocol.arm_name,
            'winner_refine_center_arm_names': [
                f'{protocol.arm_name}__W_r00516_o000135_d000804',
            ],
        }

        candidates = fidelity.build_p1_protocol_decide_candidates(
            [protocol],
            calibration,
            search_space=legacy_search_space,
        )

        all_three = [candidate for candidate in candidates if candidate.meta['aux_family'] == 'all_three']
        self.assertEqual(1, len(all_three))
        self.assertTrue(all_three[0].arm_name.endswith('__W_r00456_o000214_d000743'))
        self.assertEqual('effective_weights', all_three[0].meta['coordinate_space'])

    def test_build_p1_protocol_decide_candidates_normalizes_custom_mix_and_uses_pairwise_factor(self):
        protocol = fidelity.CandidateSpec(
            arm_name='C_A2x_cosine_broad_to_recent_strong_24m_12m',
            scheduler_profile='cosine',
            curriculum_profile='broad_to_recent',
            weight_profile='strong',
            window_profile='24m_12m',
            cfg_overrides={},
            meta={'protocol_arm': 'C_A2x_cosine_broad_to_recent_strong_24m_12m'},
        )
        calibration = {
            'rank_effective_base': 0.05121494575615136,
            'opp_weight_per_budget_unit': 0.052,
            'danger_weight_per_budget_unit': 0.18,
            'triple_combo_factor': 0.902,
            'rank_danger_combo_factor': 0.811,
            'protocol_triple_combo_factors': {
                'C_A2x_cosine_broad_to_recent_strong_24m_12m': 0.902,
            },
            'protocol_rank_danger_combo_factors': {
                'C_A2x_cosine_broad_to_recent_strong_24m_12m': 0.811,
            },
        }
        custom_search_space = {
            'protocol_decide_coordinate_mode': fidelity.P1_PROTOCOL_DECIDE_COORDINATE_MODE,
            'protocol_decide_total_budget_ratios': [0.12],
            'protocol_decide_mixes': [
                {'name': 'rank_danger_corner', 'rank_share': 2.0, 'opp_share': 0.0, 'danger_share': 2.0},
            ],
        }

        candidates = fidelity.build_p1_protocol_decide_candidates(
            [protocol],
            calibration,
            search_space=custom_search_space,
        )

        all_three = [candidate for candidate in candidates if candidate.meta['aux_family'] == 'all_three']
        self.assertEqual(1, len(all_three))
        candidate = all_three[0]
        self.assertEqual('rank_danger', candidate.meta['applied_combo_mode'])
        self.assertAlmostEqual(0.811, candidate.meta['applied_combo_factor'])
        self.assertAlmostEqual(0.06, candidate.meta['rank_budget_ratio'], places=4)
        self.assertAlmostEqual(0.0, candidate.meta['opp_budget_ratio'], places=6)
        self.assertAlmostEqual(0.06, candidate.meta['danger_budget_ratio'], places=4)

    def test_build_p1_protocol_decide_candidates_respects_persisted_precision(self):
        protocol = fidelity.CandidateSpec(
            arm_name='C_A2x_cosine_broad_to_recent_strong_24m_12m',
            scheduler_profile='cosine',
            curriculum_profile='broad_to_recent',
            weight_profile='strong',
            window_profile='24m_12m',
            cfg_overrides={},
            meta={'protocol_arm': 'C_A2x_cosine_broad_to_recent_strong_24m_12m'},
        )
        calibration = {
            'rank_effective_base': 0.05121494575615136,
            'opp_weight_per_budget_unit': 0.052,
            'danger_weight_per_budget_unit': 0.18,
            'triple_combo_factor': 0.902,
            'protocol_triple_combo_factors': {
                'C_A2x_cosine_broad_to_recent_strong_24m_12m': 0.902,
            },
        }
        search_space = {
            'protocol_decide_coordinate_mode': fidelity.P1_PROTOCOL_DECIDE_COORDINATE_MODE,
            'protocol_decide_total_budget_ratios': [0.12],
            'protocol_decide_mixes': [
                {'name': 'opp_lean', 'rank_share': 0.38, 'opp_share': 0.31, 'danger_share': 0.31},
            ],
            'budget_ratio_digits': 3,
            'aux_weight_digits': 3,
        }

        with fidelity.temporary_search_precision(budget_ratio_digits=3, aux_weight_digits=3):
            expected = fidelity.build_p1_protocol_decide_candidates(
                [protocol],
                calibration,
                search_space=search_space,
            )
        with fidelity.temporary_search_precision(budget_ratio_digits=5, aux_weight_digits=6):
            rebuilt = fidelity.build_p1_protocol_decide_candidates(
                [protocol],
                calibration,
                search_space=search_space,
            )

        self.assertEqual(
            [candidate.arm_name for candidate in expected],
            [candidate.arm_name for candidate in rebuilt],
        )

    def test_select_p1_protocol_centers_excludes_ce_only_anchor(self):
        protocol_arm = 'C_A2x_cosine_broad_to_recent_strong_24m_12m'
        ce_only = make_candidate(
            'ce_only_anchor',
            meta={
                'protocol_arm': protocol_arm,
                'aux_family': 'ce_only',
                'rank_budget_ratio': 0.0,
                'opp_budget_ratio': 0.0,
                'danger_budget_ratio': 0.0,
            },
        )
        all_three = make_candidate(
            'all_three_anchor',
            meta={
                'protocol_arm': protocol_arm,
                'aux_family': 'all_three',
                'rank_budget_ratio': 0.06,
                'opp_budget_ratio': 0.03,
                'danger_budget_ratio': 0.05,
            },
        )

        centers = fidelity.select_p1_protocol_centers(
            [
                make_ranking_entry(ce_only, valid=True, full_recent_loss=0.90),
                make_ranking_entry(all_three, valid=True, full_recent_loss=0.91),
            ],
            protocol_arm=protocol_arm,
            keep=2,
        )

        self.assertEqual(['all_three_anchor'], [candidate.arm_name for candidate in centers])

    def test_select_p1_protocol_centers_supports_explicit_arm_names(self):
        protocol_arm = fidelity.P1_WINNER_REFINE_PROTOCOL_ARM
        first = make_candidate(
            'center_a',
            meta={
                'protocol_arm': protocol_arm,
                'aux_family': 'all_three',
                'rank_budget_ratio': 0.04,
                'opp_budget_ratio': 0.03,
                'danger_budget_ratio': 0.03,
            },
        )
        second = make_candidate(
            'center_b',
            meta={
                'protocol_arm': protocol_arm,
                'aux_family': 'all_three',
                'rank_budget_ratio': 0.03,
                'opp_budget_ratio': 0.01,
                'danger_budget_ratio': 0.04,
            },
        )
        third = make_candidate(
            'center_c',
            meta={
                'protocol_arm': protocol_arm,
                'aux_family': 'all_three',
                'rank_budget_ratio': 0.05,
                'opp_budget_ratio': 0.02,
                'danger_budget_ratio': 0.04,
            },
        )

        centers = fidelity.select_p1_protocol_centers(
            [
                make_ranking_entry(first, valid=True, full_recent_loss=0.90),
                make_ranking_entry(second, valid=True, full_recent_loss=0.91),
                make_ranking_entry(third, valid=True, full_recent_loss=0.92),
            ],
            protocol_arm=protocol_arm,
            explicit_arm_names=[third.arm_name, first.arm_name],
        )

        self.assertEqual([third.arm_name, first.arm_name], [candidate.arm_name for candidate in centers])

    def test_build_p1_winner_refine_candidates_ignores_non_all_three_centers(self):
        protocol = fidelity.CandidateSpec(
            arm_name='C_A2x_cosine_broad_to_recent_strong_24m_12m',
            scheduler_profile='cosine',
            curriculum_profile='broad_to_recent',
            weight_profile='strong',
            window_profile='24m_12m',
            cfg_overrides={},
            meta={'protocol_arm': 'C_A2x_cosine_broad_to_recent_strong_24m_12m'},
        )
        calibration = {
            'rank_effective_base': 0.05121494575615136,
            'opp_weight_per_budget_unit': 0.052,
            'danger_weight_per_budget_unit': 0.18,
            'triple_combo_factor': 0.902,
            'joint_combo_factor': 0.968,
            'protocol_triple_combo_factors': {
                'C_A2x_cosine_broad_to_recent_strong_24m_12m': 0.902,
            },
            'protocol_joint_combo_factors': {
                'C_A2x_cosine_broad_to_recent_strong_24m_12m': 0.968,
            },
        }
        ce_only_center = fidelity.make_p1_budget_candidate(
            protocol,
            calibration=calibration,
            rank_budget_ratio=0.0,
            opp_budget_ratio=0.0,
            danger_budget_ratio=0.0,
            stage='P1_protocol_decide_round',
            family='ce_only',
        )
        all_three_center = fidelity.make_p1_triplet_candidate(
            protocol,
            calibration=calibration,
            total_budget_ratio=0.14,
            rank_share=0.43,
            opp_share=0.21,
            danger_share=0.36,
            stage='P1_protocol_decide_round',
            mix_name='anchor',
        )

        refine_candidates = fidelity.build_p1_winner_refine_candidates(
            [protocol],
            calibration,
            [ce_only_center, all_three_center],
        )

        self.assertTrue(refine_candidates)
        self.assertEqual(
            {all_three_center.arm_name},
            {candidate.meta['source_arm'] for candidate in refine_candidates},
        )
        self.assertTrue(
            all(candidate.meta['rank_budget_ratio'] > 0 for candidate in refine_candidates)
        )

    def test_build_p1_winner_refine_candidates_respects_persisted_precision(self):
        protocol = fidelity.CandidateSpec(
            arm_name='C_A2x_cosine_broad_to_recent_strong_24m_12m',
            scheduler_profile='cosine',
            curriculum_profile='broad_to_recent',
            weight_profile='strong',
            window_profile='24m_12m',
            cfg_overrides={},
            meta={'protocol_arm': 'C_A2x_cosine_broad_to_recent_strong_24m_12m'},
        )
        calibration = {
            'rank_effective_base': 0.05121494575615136,
            'opp_weight_per_budget_unit': 0.052,
            'danger_weight_per_budget_unit': 0.18,
            'triple_combo_factor': 0.902,
            'protocol_triple_combo_factors': {
                'C_A2x_cosine_broad_to_recent_strong_24m_12m': 0.902,
            },
        }
        search_space = {
            'budget_ratio_digits': 3,
            'aux_weight_digits': 3,
        }
        with fidelity.temporary_search_precision(budget_ratio_digits=3, aux_weight_digits=3):
            center = fidelity.make_p1_effective_triplet_candidate(
                protocol,
                calibration=calibration,
                rank_scale=0.046,
                opp_weight=0.002,
                danger_weight=0.007,
                stage='P1_protocol_decide_round',
                coordinate_name='opp_lean_12',
            )
            expected = fidelity.build_p1_winner_refine_candidates(
                [protocol],
                calibration,
                [center],
                search_space=search_space,
            )
        with fidelity.temporary_search_precision(budget_ratio_digits=5, aux_weight_digits=6):
            rebuilt = fidelity.build_p1_winner_refine_candidates(
                [protocol],
                calibration,
                [center],
                search_space=search_space,
            )

        self.assertEqual(
            [candidate.arm_name for candidate in expected],
            [candidate.arm_name for candidate in rebuilt],
        )

    def test_build_p1_ablation_candidates_respects_persisted_precision(self):
        protocol = fidelity.CandidateSpec(
            arm_name='C_A2x_cosine_broad_to_recent_strong_24m_12m',
            scheduler_profile='cosine',
            curriculum_profile='broad_to_recent',
            weight_profile='strong',
            window_profile='24m_12m',
            cfg_overrides={},
            meta={'protocol_arm': 'C_A2x_cosine_broad_to_recent_strong_24m_12m'},
        )
        calibration = {
            'rank_effective_base': 0.05121494575615136,
            'opp_weight_per_budget_unit': 0.052,
            'danger_weight_per_budget_unit': 0.18,
            'triple_combo_factor': 0.902,
            'protocol_triple_combo_factors': {
                'C_A2x_cosine_broad_to_recent_strong_24m_12m': 0.902,
            },
        }
        search_space = {
            'budget_ratio_digits': 3,
            'aux_weight_digits': 3,
        }
        with fidelity.temporary_search_precision(budget_ratio_digits=3, aux_weight_digits=3):
            refine_winner = fidelity.make_p1_effective_triplet_candidate(
                protocol,
                calibration=calibration,
                rank_scale=0.046,
                opp_weight=0.002,
                danger_weight=0.007,
                stage='P1_winner_refine_round',
                coordinate_name='winner_refine_center',
            )
            expected = fidelity.build_p1_ablation_candidates(
                [protocol],
                calibration,
                refine_winner,
                search_space=search_space,
            )
        with fidelity.temporary_search_precision(budget_ratio_digits=5, aux_weight_digits=6):
            rebuilt = fidelity.build_p1_ablation_candidates(
                [protocol],
                calibration,
                refine_winner,
                search_space=search_space,
            )

        self.assertEqual(
            [candidate.arm_name for candidate in expected],
            [candidate.arm_name for candidate in rebuilt],
        )

    def test_run_p1_uses_winner_refine_as_mainline_final_compare(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir)
            protocol_arm = 'C_A2x_cosine_broad_to_recent_strong_24m_12m'
            protocol = fidelity.CandidateSpec(
                arm_name=protocol_arm,
                scheduler_profile='cosine',
                curriculum_profile='broad_to_recent',
                weight_profile='strong',
                window_profile='24m_12m',
                cfg_overrides={},
                meta={'protocol_arm': protocol_arm},
            )
            refine_winner = fidelity.CandidateSpec(
                arm_name='refine_winner',
                scheduler_profile=protocol.scheduler_profile,
                curriculum_profile=protocol.curriculum_profile,
                weight_profile=protocol.weight_profile,
                window_profile=protocol.window_profile,
                cfg_overrides={},
                meta={
                    'protocol_arm': protocol_arm,
                    'aux_family': 'all_three',
                    'rank_budget_ratio': 0.05,
                    'opp_budget_ratio': 0.04,
                    'danger_budget_ratio': 0.03,
                },
            )
            calibration = {
                'budget_ratios': [0.0, 0.25],
                'rank_effective_base': 0.05121494575615136,
                'opp_weight_per_budget_unit': 0.052,
                'danger_weight_per_budget_unit': 0.18,
                'triple_combo_factor': 0.902,
                'joint_combo_factor': 0.968,
                'protocol_triple_combo_factors': {protocol_arm: 0.902},
                'protocol_joint_combo_factors': {protocol_arm: 0.968},
            }
            state = {'final_conclusion': {}}

            with (
                patch.object(
                    fidelity.ab,
                    'build_eval_splits',
                    return_value={
                        'monitor_recent_files': ['monitor.json.gz'],
                        'full_recent_files': ['recent.json.gz'],
                        'old_regression_files': ['old.json.gz'],
                    },
                ),
                patch.object(fidelity, 'rank_weight_mean_for_files', return_value=0.05),
                patch.object(fidelity, 'update_results_doc'),
                patch.object(
                    fidelity,
                    'build_p1_calibration_candidates',
                    return_value=['calibration_candidate'],
                ),
                patch.object(
                    fidelity,
                    'derive_p1_budget_calibration',
                    return_value=dict(calibration),
                ),
                patch.object(
                    fidelity,
                    'execute_round_progressive_multiseed',
                    return_value={
                        'round_name': 'p1_protocol_decide_round',
                        'ranking': [
                            make_ranking_entry(
                                refine_winner,
                                valid=True,
                                full_recent_loss=0.91,
                            ),
                        ],
                    },
                ),
                patch.object(
                    fidelity,
                    'build_p1_protocol_compare',
                    return_value=[
                        {
                            'arm_name': protocol_arm,
                            'candidate_meta': {'protocol_arm': protocol_arm},
                        }
                    ],
                ),
                patch.object(
                    fidelity,
                    'execute_round_multiseed',
                    side_effect=[
                        {'round_name': 'p1_calibration', 'ranking': []},
                        {
                            'round_name': 'p1_winner_refine_round',
                            'ranking': [
                                make_ranking_entry(
                                    refine_winner,
                                    valid=True,
                                    full_recent_loss=0.90,
                                ),
                            ],
                        },
                    ],
                ) as execute_round_multiseed,
                patch.object(
                    fidelity,
                    'winner_refine_center_selection_from_search_space',
                    return_value={'keep': None, 'explicit_arm_names': (refine_winner.arm_name,)},
                ),
                patch.object(fidelity, 'build_p1_ablation_candidates') as build_p1_ablation_candidates,
            ):
                winner, finalists = fidelity.run_p1(
                    run_dir,
                    {'control': {'version': 4}},
                    {'202501': ['dummy.json.gz']},
                    123,
                    [protocol],
                    state,
                )

        self.assertEqual(refine_winner.arm_name, winner.arm_name)
        self.assertEqual([refine_winner.arm_name], [candidate.arm_name for candidate in finalists])
        self.assertEqual('p1_final_compare', state['p1']['final_compare']['round_name'])
        self.assertEqual(refine_winner.arm_name, state['p1']['winner'])
        self.assertEqual('winner_refine_mainline', state['p1']['winner_source'])
        self.assertEqual(refine_winner.arm_name, state['final_conclusion']['p1_winner'])
        self.assertEqual('winner_refine_mainline', state['final_conclusion']['p1_winner_source'])
        self.assertEqual(fidelity.P1_ABLATION_POLICY, state['final_conclusion']['p1_ablation_policy'])
        self.assertEqual(2, execute_round_multiseed.call_count)
        build_p1_ablation_candidates.assert_not_called()

    def test_build_p0_candidates_can_filter_to_cosine_only(self):
        candidates = fidelity.build_p0_candidates(scheduler_profiles={'cosine'})

        self.assertEqual(18, len(candidates))
        self.assertTrue(all(candidate.scheduler_profile == 'cosine' for candidate in candidates))

    def test_build_p0_round0_survivors_keeps_all_valid_cosine_candidates(self):
        candidates = [
            make_candidate('arm_a', meta={'stage': 'P0'}),
            make_candidate('arm_b', meta={'stage': 'P0'}),
            make_candidate('arm_c', meta={'stage': 'P0'}),
        ]
        candidate_index = {candidate.arm_name: candidate for candidate in candidates}
        ranking = [
            make_ranking_entry(candidates[0], valid=True),
            make_ranking_entry(candidates[1], valid=False),
            make_ranking_entry(candidates[2], valid=True),
        ]

        survivors = fidelity.build_p0_round0_survivors(
            ranking,
            candidate_index,
            candidate_subset='cosine_only',
        )

        self.assertEqual(['arm_a', 'arm_c'], [candidate.arm_name for candidate in survivors])

    def test_build_p0_round0_survivors_falls_back_to_ranked_cosine_entries_when_all_invalid(self):
        candidates = [
            make_candidate('arm_a', meta={'stage': 'P0'}),
            make_candidate('arm_b', meta={'stage': 'P0'}),
        ]
        candidate_index = {candidate.arm_name: candidate for candidate in candidates}
        ranking = [
            make_ranking_entry(candidates[0], valid=False),
            make_ranking_entry(candidates[1], valid=False),
        ]

        survivors = fidelity.build_p0_round0_survivors(
            ranking,
            candidate_index,
            candidate_subset='cosine_only',
        )

        self.assertEqual(['arm_a', 'arm_b'], [candidate.arm_name for candidate in survivors])

    def test_build_p0_round1_survivors_keeps_loss_band_with_top8_floor(self):
        candidates = [make_candidate(f'arm_{idx:02d}', meta={'stage': 'P0'}) for idx in range(10)]
        ranking = [
            make_ranking_entry(candidate, full_recent_loss=0.500 + idx * 0.001)
            for idx, candidate in enumerate(candidates)
        ]

        survivors = fidelity.build_p0_round1_survivors(
            ranking,
            loss_epsilon=0.003,
            min_keep=8,
        )

        self.assertEqual(
            ['arm_00', 'arm_01', 'arm_02', 'arm_03', 'arm_04', 'arm_05', 'arm_06', 'arm_07'],
            [candidate.arm_name for candidate in survivors],
        )

    def test_build_p0_round1_survivors_can_cap_max_keep(self):
        candidates = [make_candidate(f'arm_{idx:02d}', meta={'stage': 'P0'}) for idx in range(12)]
        ranking = [
            make_ranking_entry(candidate, full_recent_loss=0.500 + idx * 0.0002)
            for idx, candidate in enumerate(candidates)
        ]

        survivors = fidelity.build_p0_round1_survivors(
            ranking,
            loss_epsilon=0.003,
            min_keep=8,
            max_keep=9,
        )

        self.assertEqual(
            ['arm_00', 'arm_01', 'arm_02', 'arm_03', 'arm_04', 'arm_05', 'arm_06', 'arm_07', 'arm_08'],
            [candidate.arm_name for candidate in survivors],
        )

    def test_build_p0_round1_survivors_keeps_two_when_max_keep_is_one(self):
        candidates = [make_candidate(f'arm_{idx:02d}', meta={'stage': 'P0'}) for idx in range(3)]
        ranking = [
            make_ranking_entry(candidate, full_recent_loss=0.500 + idx * 0.001)
            for idx, candidate in enumerate(candidates)
        ]

        survivors = fidelity.build_p0_round1_survivors(
            ranking,
            min_keep=3,
            max_keep=1,
        )

        self.assertEqual(
            ['arm_00', 'arm_01'],
            [candidate.arm_name for candidate in survivors],
        )

    def test_build_p0_round1_survivors_backfills_second_entry_when_only_one_valid(self):
        candidates = [make_candidate(f'arm_{idx:02d}', meta={'stage': 'P0'}) for idx in range(3)]
        ranking = [
            make_ranking_entry(candidates[0], valid=True, full_recent_loss=0.500),
            make_ranking_entry(candidates[1], valid=False, full_recent_loss=0.700),
            make_ranking_entry(candidates[2], valid=False, full_recent_loss=0.900),
        ]

        survivors = fidelity.build_p0_round1_survivors(
            ranking,
            min_keep=1,
        )

        self.assertEqual(
            ['arm_00', 'arm_01'],
            [candidate.arm_name for candidate in survivors],
        )

    def test_build_p0_round1_survivors_backfills_to_min_keep_after_transient_failures(self):
        candidates = [make_candidate(f'arm_{idx:02d}', meta={'stage': 'P0'}) for idx in range(10)]
        ranking = [
            make_ranking_entry(candidate, valid=idx < 3, full_recent_loss=0.500 + idx * 0.001)
            for idx, candidate in enumerate(candidates)
        ]

        survivors = fidelity.build_p0_round1_survivors(
            ranking,
            min_keep=8,
        )

        self.assertEqual(
            ['arm_00', 'arm_01', 'arm_02', 'arm_03', 'arm_04', 'arm_05', 'arm_06', 'arm_07'],
            [candidate.arm_name for candidate in survivors],
        )

    def test_run_formal_prepares_checkpoint_pack_and_marks_formal_1v3_pending(self):
        winner = fidelity.CandidateSpec(
            arm_name='C_A2y_cosine_broad_to_recent_strong_12m_6m__B_r000_o000_d000',
            scheduler_profile='cosine',
            curriculum_profile='broad_to_recent',
            weight_profile='strong',
            window_profile='12m_6m',
            cfg_overrides={'aux': {'rank_weight': 0.03}},
            meta={
                'stage': 'P1',
                'protocol_arm': 'C_A2y_cosine_broad_to_recent_strong_12m_6m',
            },
        )
        finalized_result = {
            'winner': 'best_acc',
            'offline_checkpoint_winner': 'best_acc',
            'shortlist_checkpoint_types': ['best_loss', 'best_acc', 'best_rank'],
            'selected_protocol_arm': 'C_A2y_cosine_broad_to_recent_strong_12m_6m',
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir)
            state: dict[str, object] = {}
            with (
                patch.object(fidelity.formal, 'apply_formal_defaults'),
                patch.object(fidelity.ab, 'merge_dict', return_value={'merged': True}) as merge_mock,
                patch.object(fidelity.ab, 'run_ab6_checkpoint', return_value={'winner': 'best_acc'}) as run_mock,
                patch.object(fidelity.formal, 'finalize_formal_result', return_value=finalized_result) as finalize_mock,
                patch.object(fidelity, 'atomic_write_json') as write_mock,
                patch.object(fidelity, 'update_results_doc') as update_mock,
            ):
                fidelity.run_formal(
                    run_dir,
                    base_cfg={'base': 'cfg'},
                    grouped={'202603': ['a.json.gz']},
                    winner=winner,
                    seed=123,
                    formal_step_scale=5.0,
                    state=state,
                )

        merge_mock.assert_called_once_with({'base': 'cfg'}, winner.cfg_overrides)
        run_mock.assert_called_once_with(
            {'merged': True},
            {'202603': ['a.json.gz']},
            seed=123,
            scheduler_profile='cosine',
            curriculum_profile='broad_to_recent',
            weight_profile='strong',
            window_profile='12m_6m',
            step_scale=5.0,
            ab_name=f'{run_dir.name}_formal',
        )
        finalize_mock.assert_called_once_with(
            {'merged': True},
            {'winner': 'best_acc'},
            protocol_arm='C_A2y_cosine_broad_to_recent_strong_12m_6m',
        )
        self.assertEqual('completed', state['formal']['status'])
        self.assertEqual('best_acc', state['formal']['offline_checkpoint_winner'])
        self.assertEqual(['best_loss', 'best_acc', 'best_rank'], state['formal']['shortlist_checkpoint_types'])
        self.assertEqual(['best_loss', 'best_acc', 'best_rank'], state['formal']['checkpoint_pack_types'])
        self.assertEqual('pending', state['formal_1v3']['status'])
        self.assertEqual('completed', state['final_conclusion']['formal_train_status'])
        self.assertEqual('pending', state['final_conclusion']['formal_1v3_status'])
        self.assertEqual('pending_1v3', state['final_conclusion']['formal_status'])
        write_mock.assert_called_once_with(run_dir / 'state.json', state)
        update_mock.assert_called_once_with(run_dir, state)

    def test_load_cached_p0_rounds_upto_round2_can_skip_round1(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            run_dir = tmp_path / 'run'
            run_dir.mkdir()
            candidates = [make_candidate(f'arm_{idx:02d}', meta={'stage': 'P0'}) for idx in range(10)]
            ranking = [
                make_ranking_entry(candidate, full_recent_loss=0.500 + idx * 0.001)
                for idx, candidate in enumerate(candidates)
            ]

            round0_payload = {'round_name': 'p0_round0', 'ranking': ranking}
            round2_payload = {'round_name': 'p0_round2', 'ranking': ranking[:8]}

            def fake_load_cached_round(path, signature, legacy_matcher=None):
                if path.name == 'p0_round0.json':
                    return round0_payload
                if path.name == 'p0_round2.json':
                    return round2_payload
                return None

            with (
                patch.object(fidelity, 'build_p0_candidates', return_value=candidates),
                patch.object(
                    fidelity.ab,
                    'build_eval_splits',
                    return_value={
                        'monitor_recent_files': ['a'],
                        'full_recent_files': ['b'],
                        'old_regression_files': [],
                    },
                ),
                patch.object(fidelity, 'load_cached_round_if_valid', side_effect=fake_load_cached_round),
                patch.object(fidelity, 'load_revalidated_round_if_valid', return_value=None),
            ):
                loaded = fidelity.load_cached_p0_rounds_upto_round2(
                    run_dir,
                    base_cfg={'base': 'cfg'},
                    grouped={},
                    seed=7,
                    skip_round1=True,
                )

            self.assertIsNotNone(loaded)
            assert loaded is not None
            round0, round1, round2 = loaded
            self.assertEqual('p0_round0', round0['round_name'])
            self.assertEqual('p0_round1', round1['round_name'])
            self.assertEqual('skipped', round1['status'])
            self.assertEqual('direct_round2_top8', round1['reason'])
            self.assertEqual('p0_round2', round2['round_name'])

    def test_load_cached_round_if_valid_accepts_legacy_p0_round_when_structure_matches(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / 'p0_round0.json'
            candidates = [make_candidate('arm_a', meta={'stage': 'P0'}), make_candidate('arm_b', meta={'stage': 'P0'})]
            payload = {
                'round_name': 'p0_round0',
                'ab_name': 'demo_p0_r0',
                'seed': 123,
                'step_scale': 0.5,
                'evaluated_arms': 2,
                'eval_split_counts': {
                    'monitor_recent_files': 3,
                    'full_recent_files': 2,
                    'old_regression_files': 1,
                },
                'ranking': [
                    make_ranking_entry(candidates[0]),
                    make_ranking_entry(candidates[1]),
                ],
                'round_signature': 'legacy-signature',
            }
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8', newline='\n')

            loaded = fidelity.load_cached_round_if_valid(
                path,
                expected_signature='new-signature',
                legacy_matcher=lambda item: fidelity.legacy_round_payload_matches(
                    item,
                    round_name='p0_round0',
                    ab_name='demo_p0_r0',
                    expected_candidates=candidates,
                    seed=123,
                    step_scale=0.5,
                    eval_splits={
                        'monitor_recent_files': ['a', 'b', 'c'],
                        'full_recent_files': ['d', 'e'],
                        'old_regression_files': ['f'],
                    },
                ),
            )

            self.assertIsNotNone(loaded)
            self.assertTrue(loaded['legacy_cache_accepted'])

    def test_load_cached_round_if_valid_rejects_legacy_p0_round_when_candidates_drift(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / 'p0_round0.json'
            payload = {
                'round_name': 'p0_round0',
                'ab_name': 'demo_p0_r0',
                'seed': 123,
                'step_scale': 0.5,
                'evaluated_arms': 2,
                'eval_split_counts': {
                    'monitor_recent_files': 3,
                    'full_recent_files': 2,
                    'old_regression_files': 1,
                },
                'ranking': [
                    make_ranking_entry(make_candidate('arm_a', meta={'stage': 'P0'})),
                    make_ranking_entry(make_candidate('arm_b', meta={'stage': 'P0'})),
                ],
                'round_signature': 'legacy-signature',
            }
            path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8', newline='\n')

            loaded = fidelity.load_cached_round_if_valid(
                path,
                expected_signature='new-signature',
                legacy_matcher=lambda item: fidelity.legacy_round_payload_matches(
                    item,
                    round_name='p0_round0',
                    ab_name='demo_p0_r0',
                    expected_candidates=[
                        make_candidate('arm_a', meta={'stage': 'P0'}),
                        make_candidate('arm_c', meta={'stage': 'P0'}),
                    ],
                    seed=123,
                    step_scale=0.5,
                    eval_splits={
                        'monitor_recent_files': ['a', 'b', 'c'],
                        'full_recent_files': ['d', 'e'],
                        'old_regression_files': ['f'],
                    },
                ),
            )

            self.assertIsNone(loaded)

    def test_load_revalidated_round_if_valid_adopts_summary(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            run_dir = tmp_path / 'run'
            run_dir.mkdir()
            ab_root = tmp_path / 'ab'
            source_dir = ab_root / 'demo_p0_r0'
            source_dir.mkdir(parents=True)
            candidates = [
                make_candidate('arm_a', meta={'stage': 'P0'}),
                make_candidate('arm_b', meta={'stage': 'P0'}),
            ]
            payload = {
                'source_dir': str(source_dir),
                'checkpoint_kind': 'best_loss',
                'phase_name': 'final',
                'evaluated_arms': 2,
                'ranking': [
                    make_ranking_entry(candidates[0], full_recent_loss=0.1),
                    make_ranking_entry(candidates[1], full_recent_loss=0.2),
                ],
            }
            (source_dir / 'revalidated_best_loss_final_round.json').write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding='utf-8',
                newline='\n',
            )

            with patch.object(fidelity.ab, 'AB_ROOT', ab_root):
                loaded = fidelity.load_revalidated_round_if_valid(
                    run_dir=run_dir,
                    round_name='p0_round0',
                    ab_name='demo_p0_r0',
                    ab_root_name='demo_p0_r0',
                    expected_signature='sig',
                    expected_candidates=candidates,
                    seed=123,
                    step_scale=0.5,
                    eval_splits={
                        'monitor_recent_files': ['a', 'b'],
                        'full_recent_files': ['c'],
                        'old_regression_files': [],
                    },
                )

            self.assertIsNotNone(loaded)
            assert loaded is not None
            self.assertEqual('p0_round0', loaded['round_name'])
            self.assertEqual('demo_p0_r0', loaded['ab_name'])
            self.assertEqual('sig', loaded['round_signature'])
            self.assertTrue(loaded['revalidated_cache_accepted'])
            self.assertEqual(0.1, loaded['best_loss'])
            self.assertTrue((run_dir / 'p0_round0.json').exists())

    def test_load_cached_p0_rounds_upto_round2_accepts_revalidated_summaries(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            run_dir = tmp_path / 'run'
            run_dir.mkdir()
            ab_root = tmp_path / 'ab'
            candidates = [
                make_candidate('arm_a', meta={'stage': 'P0'}),
                make_candidate('arm_b', meta={'stage': 'P0'}),
            ]
            ranking = [
                make_ranking_entry(candidates[0], full_recent_loss=0.1),
                make_ranking_entry(candidates[1], full_recent_loss=0.2),
            ]
            for suffix in ('p0_r0', 'p0_r1', 'p0_r2'):
                source_dir = ab_root / f'run_{suffix}'
                source_dir.mkdir(parents=True)
                payload = {
                    'source_dir': str(source_dir),
                    'checkpoint_kind': 'best_loss',
                    'phase_name': 'final',
                    'evaluated_arms': 2,
                    'ranking': ranking,
                }
                (source_dir / 'revalidated_best_loss_final_round.json').write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2),
                    encoding='utf-8',
                    newline='\n',
                )

            eval_splits = {
                'monitor_recent_files': ['a', 'b'],
                'full_recent_files': ['c'],
                'old_regression_files': [],
            }
            with (
                patch.object(fidelity.ab, 'AB_ROOT', ab_root),
                patch.object(fidelity, 'build_p0_candidates', return_value=candidates),
                patch.object(fidelity.ab, 'build_eval_splits', return_value=eval_splits),
            ):
                loaded = fidelity.load_cached_p0_rounds_upto_round2(
                    run_dir,
                    base_cfg={'base': 'cfg'},
                    grouped={},
                    seed=7,
                )

            self.assertIsNotNone(loaded)
            assert loaded is not None
            round0, round1, round2 = loaded
            self.assertTrue(round0['revalidated_cache_accepted'])
            self.assertTrue(round1['revalidated_cache_accepted'])
            self.assertTrue(round2['revalidated_cache_accepted'])
            self.assertEqual(['arm_a', 'arm_b'], [entry['arm_name'] for entry in round2['ranking']])

    def test_load_cached_p0_rounds_upto_round2_prefers_authoritative_revalidated_rounds(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            run_dir = tmp_path / 'demo'
            run_dir.mkdir()
            ab_root = tmp_path / 'ab'
            candidates = [make_candidate('arm_a', meta={'stage': 'P0'})]
            ranking = [
                make_ranking_entry(candidates[0], full_recent_loss=0.1),
            ]
            for round_name, suffix in (
                ('p0_round0', 'p0_r0'),
                ('p0_round1', 'p0_r1'),
                ('p0_round2', 'p0_r2'),
            ):
                source_dir = ab_root / f'demo_{suffix}'
                source_dir.mkdir(parents=True)
                payload = {
                    'round_name': 'revalidated_best_loss_final',
                    'source_dir': str(source_dir),
                    'checkpoint_kind': 'best_loss',
                    'phase_name': 'final',
                    'evaluated_arms': 1,
                    'ranking': ranking,
                }
                (source_dir / 'revalidated_best_loss_final_round.json').write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2),
                    encoding='utf-8',
                    newline='\n',
                )

            eval_splits = {
                'monitor_recent_files': ['a', 'b'],
                'full_recent_files': ['c'],
                'old_regression_files': [],
            }
            with (
                patch.object(fidelity.ab, 'AB_ROOT', ab_root),
                patch.object(fidelity, 'build_p0_candidates', return_value=candidates),
                patch.object(fidelity.ab, 'build_eval_splits', return_value=eval_splits),
            ):
                loaded = fidelity.load_cached_p0_rounds_upto_round2(
                    run_dir,
                    base_cfg={'unused': True},
                    grouped={'unused': []},
                    seed=123,
                )

            self.assertIsNotNone(loaded)
            assert loaded is not None
            self.assertEqual('p0_round0', loaded[0]['round_name'])
            self.assertTrue(loaded[0]['revalidated_cache_accepted'])
            self.assertEqual('p0_round2', loaded[2]['round_name'])
            self.assertTrue(loaded[2]['revalidated_cache_accepted'])

    def test_load_cached_p0_rounds_upto_round2_rejects_authoritative_revalidated_rounds_when_candidate_subset_changes(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            run_dir = tmp_path / 'demo'
            run_dir.mkdir()
            ab_root = tmp_path / 'ab'
            cosine_candidates = [
                make_candidate('arm_cos_a', meta={'stage': 'P0'}),
                make_candidate('arm_cos_b', meta={'stage': 'P0'}),
            ]
            stale_candidates = cosine_candidates + [make_candidate('arm_plateau', meta={'stage': 'P0'})]
            stale_ranking = [
                make_ranking_entry(candidate, full_recent_loss=0.1 + idx * 0.1)
                for idx, candidate in enumerate(stale_candidates)
            ]
            for suffix in ('p0_r0', 'p0_r1', 'p0_r2'):
                source_dir = ab_root / f'demo_{suffix}'
                source_dir.mkdir(parents=True)
                payload = {
                    'round_name': 'revalidated_best_loss_final',
                    'source_dir': str(source_dir),
                    'checkpoint_kind': 'best_loss',
                    'phase_name': 'final',
                    'evaluated_arms': len(stale_ranking),
                    'ranking': stale_ranking,
                }
                (source_dir / 'revalidated_best_loss_final_round.json').write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2),
                    encoding='utf-8',
                    newline='\n',
                )

            eval_splits = {
                'monitor_recent_files': ['a', 'b'],
                'full_recent_files': ['c'],
                'old_regression_files': [],
            }
            with (
                patch.object(fidelity.ab, 'AB_ROOT', ab_root),
                patch.object(fidelity, 'build_p0_candidates', return_value=cosine_candidates),
                patch.object(fidelity.ab, 'build_eval_splits', return_value=eval_splits),
            ):
                loaded = fidelity.load_cached_p0_rounds_upto_round2(
                    run_dir,
                    base_cfg={'unused': True},
                    grouped={'unused': []},
                    seed=123,
                    candidate_subset='cosine_only',
                )

            self.assertIsNone(loaded)

    def test_load_cached_p0_rounds_upto_round2_rejects_authoritative_revalidated_rounds_when_round2_cap_changes(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            run_dir = tmp_path / 'demo'
            run_dir.mkdir()
            ab_root = tmp_path / 'ab'
            candidates = [make_candidate(f'arm_{idx}', meta={'stage': 'P0'}) for idx in range(3)]
            ranking = [
                make_ranking_entry(candidate, full_recent_loss=0.1 + idx * 0.1)
                for idx, candidate in enumerate(candidates)
            ]
            for suffix in ('p0_r0', 'p0_r1', 'p0_r2'):
                source_dir = ab_root / f'demo_{suffix}'
                source_dir.mkdir(parents=True)
                payload = {
                    'round_name': 'revalidated_best_loss_final',
                    'source_dir': str(source_dir),
                    'checkpoint_kind': 'best_loss',
                    'phase_name': 'final',
                    'evaluated_arms': len(ranking),
                    'ranking': ranking,
                }
                (source_dir / 'revalidated_best_loss_final_round.json').write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2),
                    encoding='utf-8',
                    newline='\n',
                )

            eval_splits = {
                'monitor_recent_files': ['a', 'b'],
                'full_recent_files': ['c'],
                'old_regression_files': [],
            }
            with (
                patch.object(fidelity.ab, 'AB_ROOT', ab_root),
                patch.object(fidelity, 'build_p0_candidates', return_value=candidates),
                patch.object(fidelity.ab, 'build_eval_splits', return_value=eval_splits),
            ):
                loaded = fidelity.load_cached_p0_rounds_upto_round2(
                    run_dir,
                    base_cfg={'unused': True},
                    grouped={'unused': []},
                    seed=123,
                    round2_max_candidates=2,
                )

            self.assertIsNone(loaded)

    def test_execute_round_invalidates_stale_summary_when_candidates_change(self):
        calls = []
        stored_json = {}

        def fake_run_arm_cached(*, candidate, **kwargs):
            calls.append(candidate.arm_name)
            return {
                'ok': True,
                'arm_name': candidate.arm_name,
                'run': {'final': {'best_loss': {}}},
            }

        def fake_summarize_entry(name, candidate, payload, *, ranking_mode='full_recent'):
            return {
                'arm_name': name,
                'candidate_meta': candidate.meta,
                'scheduler_profile': candidate.scheduler_profile,
                'curriculum_profile': candidate.curriculum_profile,
                'weight_profile': candidate.weight_profile,
                'window_profile': candidate.window_profile,
                'cfg_overrides': candidate.cfg_overrides,
                'valid': True,
                'full_recent_loss': float(len(name)),
            }

        def fake_rank_round_entries(entries, weights=None, **kwargs):
            return [{**entry, 'rank': idx} for idx, entry in enumerate(entries, start=1)]

        def fake_exists(path_obj):
            return str(path_obj) in stored_json

        def fake_write_json(path, payload):
            stored_json[str(path)] = fidelity.normalize_payload(payload)

        def fake_load_json(path):
            return stored_json[str(path)]

        run_dir = Path('X:/virtual/stage05_fidelity')
        kwargs = {
            'run_dir': run_dir,
            'round_name': 'p1a_round0',
            'ab_name': 'ab_p1a_r0',
            'base_cfg': {'control': {'version': 4}},
            'grouped': {'recent': ['train_a.json.gz']},
            'eval_splits': {
                'monitor_recent_files': ['monitor.json.gz'],
                'full_recent_files': ['recent.json.gz'],
                'old_regression_files': ['old.json.gz'],
            },
            'seed': 17,
            'step_scale': 1.0,
        }
        with (
            patch.object(Path, 'exists', fake_exists),
            patch.object(fidelity, 'atomic_write_json', side_effect=fake_write_json),
            patch.object(fidelity, 'load_json', side_effect=fake_load_json),
            patch.object(fidelity, 'run_arm_cached', side_effect=fake_run_arm_cached),
            patch.object(fidelity, 'summarize_entry', side_effect=fake_summarize_entry),
            patch.object(fidelity, 'rank_round_entries', side_effect=fake_rank_round_entries),
        ):
            first = fidelity.execute_round(
                candidates=[make_candidate('old_budget')],
                **kwargs,
            )
            second = fidelity.execute_round(
                candidates=[make_candidate('new_budget')],
                **kwargs,
            )
            third = fidelity.execute_round(
                candidates=[make_candidate('new_budget')],
                **kwargs,
            )

        self.assertEqual(['old_budget', 'new_budget'], calls)
        self.assertEqual(['old_budget'], [entry['arm_name'] for entry in first['ranking']])
        self.assertEqual(['new_budget'], [entry['arm_name'] for entry in second['ranking']])
        self.assertEqual(['new_budget'], [entry['arm_name'] for entry in third['ranking']])

    def test_run_arm_cached_invalidates_stale_arm_cache_when_signature_changes(self):
        call_count = 0
        stored_json = {}
        existing_paths = set()

        def fake_run_arm(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return {'final': {'best_loss': {'path': f'checkpoint_{call_count}.pth'}}}

        def fake_exists(path_obj):
            return str(path_obj) in existing_paths

        def fake_write_json(path, payload):
            stored_json[str(path)] = fidelity.normalize_payload(payload)
            existing_paths.add(str(path))
            existing_paths.add(str(path.parent))

        def fake_load_json(path):
            return stored_json[str(path)]

        def fake_rmtree(path_obj):
            prefix = str(path_obj)
            for path in list(existing_paths):
                if path == prefix or path.startswith(prefix + '\\') or path.startswith(prefix + '/'):
                    existing_paths.remove(path)
            for path in list(stored_json):
                if path == prefix or path.startswith(prefix + '\\') or path.startswith(prefix + '/'):
                    stored_json.pop(path)

        candidate = make_candidate('same_arm', cfg_overrides={'aux': {'danger_weight': 0.1}})
        grouped = {'recent': ['train_a.json.gz']}
        eval_splits = {
            'monitor_recent_files': ['monitor.json.gz'],
            'full_recent_files': ['recent.json.gz'],
            'old_regression_files': ['old.json.gz'],
        }

        ab_root = Path('X:/virtual/stage05_ab')
        with (
            patch.object(Path, 'exists', fake_exists),
            patch.object(fidelity, 'atomic_write_json', side_effect=fake_write_json),
            patch.object(fidelity, 'load_json', side_effect=fake_load_json),
            patch.object(fidelity.shutil, 'rmtree', side_effect=fake_rmtree),
            patch.object(fidelity.ab, 'AB_ROOT', ab_root),
            patch.object(fidelity.ab, 'run_arm', side_effect=fake_run_arm),
        ):
            first = fidelity.run_arm_cached(
                base_cfg={'control': {'version': 4}},
                grouped=grouped,
                eval_splits=eval_splits,
                candidate=candidate,
                seed=33,
                step_scale=1.0,
                ab_name='ab_p1a_r0',
            )
            second = fidelity.run_arm_cached(
                base_cfg={'control': {'version': 5}},
                grouped=grouped,
                eval_splits=eval_splits,
                candidate=candidate,
                seed=33,
                step_scale=1.0,
                ab_name='ab_p1a_r0',
            )
            third = fidelity.run_arm_cached(
                base_cfg={'control': {'version': 5}},
                grouped=grouped,
                eval_splits=eval_splits,
                candidate=candidate,
                seed=33,
                step_scale=1.0,
                ab_name='ab_p1a_r0',
            )

        self.assertEqual(2, call_count)
        self.assertEqual('checkpoint_1.pth', first['run']['final']['best_loss']['path'])
        self.assertEqual('checkpoint_2.pth', second['run']['final']['best_loss']['path'])
        self.assertEqual('checkpoint_2.pth', third['run']['final']['best_loss']['path'])

    def test_remove_tree_with_retries_retries_windows_permission_error(self):
        path = Path('X:/virtual/stage05_ab/demo_arm')
        win_err = PermissionError(32, 'sharing violation')
        win_err.winerror = 32
        calls = []

        def fake_rmtree(target):
            calls.append(str(target))
            if len(calls) == 1:
                raise win_err

        with (
            patch.object(Path, 'exists', return_value=True),
            patch.object(fidelity.shutil, 'rmtree', side_effect=fake_rmtree),
            patch.object(fidelity.time, 'sleep'),
        ):
            fidelity.remove_tree_with_retries(path)

        self.assertEqual([str(path), str(path)], calls)

    def test_round_cache_signature_keeps_legacy_shape_for_default_full_recent_rounds(self):
        candidate = make_candidate('proto_arm', meta={'stage': 'P0'})
        signature = fidelity.round_cache_signature(
            round_name='p0_round0',
            ab_name='demo_p0_r0',
            base_cfg={'control': {'version': 4}},
            grouped={'recent': ['train_a.json.gz']},
            eval_splits={
                'monitor_recent_files': ['monitor.json.gz'],
                'full_recent_files': ['recent.json.gz'],
                'old_regression_files': ['old.json.gz'],
            },
            candidates=[candidate],
            seed=17,
            step_scale=0.5,
            selector_weights=None,
        )

        legacy_signature = fidelity.stable_payload_digest(
            {
                'schema_version': 3,
                'scenario_score_version': fidelity.SCENARIO_SCORE_VERSION,
                'round_name': 'p0_round0',
                'ab_name': 'demo_p0_r0',
                'base_cfg': {'control': {'version': 4}},
                'grouped': {'recent': ['train_a.json.gz']},
                'eval_splits': {
                    'monitor_recent_files': ['monitor.json.gz'],
                    'full_recent_files': ['recent.json.gz'],
                    'old_regression_files': ['old.json.gz'],
                },
                'candidates': [fidelity.candidate_cache_payload(candidate, include_meta=True)],
                'seed': 17,
                'step_scale': 0.5,
                'selector_weights': None,
            }
        )

        self.assertEqual(legacy_signature, signature)

    def test_rank_weight_mean_for_files_disables_unused_aux_labels(self):
        context_meta = torch.tensor(
            [
                [0, 0, 0, 0, 0, 0, 10, 20],
                [0, 1, 0, 1, 0, 0, 5, 15],
            ],
            dtype=torch.int64,
        )

        with (
            patch.object(fidelity, 'SupervisedFileDatasetsIter') as dataset_cls,
            patch.object(fidelity, 'DataLoader', return_value=[(None, None, None, None, context_meta)]),
        ):
            mean = fidelity.rank_weight_mean_for_files(
                ['recent.json.gz'],
                version=4,
                file_batch_size=7,
            )

        kwargs = dataset_cls.call_args.kwargs
        self.assertFalse(kwargs['emit_opponent_state_labels'])
        self.assertFalse(kwargs['track_danger_labels'])
        expected_weights = [
            fidelity.RANK_TEMPLATE['base_weight'],
            min(
                fidelity.RANK_TEMPLATE['max_weight'],
                fidelity.RANK_TEMPLATE['base_weight']
                * fidelity.RANK_TEMPLATE['south_factor']
                * fidelity.RANK_TEMPLATE['all_last_factor'],
            ),
        ]
        self.assertAlmostEqual(sum(expected_weights) / len(expected_weights), mean, places=6)

    def test_make_p1_budget_candidate_maps_common_budget_axis(self):
        protocol = make_candidate(
            'proto_arm',
            meta={'stage': 'P0', 'protocol_arm': 'proto_arm'},
        )
        calibration = {
            'shared_rank_template_mean': 0.08,
            'rank_effective_base': 0.02,
            'opp_weight_per_budget_unit': 0.01,
            'danger_weight_per_budget_unit': 0.02,
            'joint_combo_factor': 1.0,
            'protocol_joint_combo_factors': {},
        }

        candidate = fidelity.make_p1_budget_candidate(
            protocol,
            calibration=calibration,
            rank_budget_ratio=1.5,
            opp_budget_ratio=0.5,
            danger_budget_ratio=1.0,
            stage='P1_solo_round',
        )

        self.assertEqual('proto_arm__B_r15000_o05000_d10000', candidate.arm_name)
        self.assertEqual(1.5, candidate.meta['rank_scale'])
        self.assertEqual(1.5, candidate.meta['rank_budget_ratio'])
        self.assertEqual(0.5, candidate.meta['opp_budget_ratio'])
        self.assertEqual(1.0, candidate.meta['danger_budget_ratio'])
        self.assertAlmostEqual(0.005, candidate.meta['opp_weight'], places=6)
        self.assertAlmostEqual(0.02, candidate.meta['danger_weight'], places=6)
        self.assertEqual('rank+opp+danger', candidate.meta['aux_family'])

    def test_summarize_multiseed_entry_requires_all_seeds_for_multiseed_rounds(self):
        candidate = make_candidate('proto_arm')
        valid_summary = {
            'arm_name': candidate.arm_name,
            'candidate_meta': candidate.meta,
            'scheduler_profile': candidate.scheduler_profile,
            'curriculum_profile': candidate.curriculum_profile,
            'weight_profile': candidate.weight_profile,
            'window_profile': candidate.window_profile,
            'cfg_overrides': candidate.cfg_overrides,
            'ok': True,
            'valid': True,
            'full_recent_loss': 1.0,
            'full_recent_metrics': {'discard_nll': 0.2},
            'action_quality_score': -0.2,
            'scenario_quality_score': 0.0,
            'rank_acc': 0.4,
            'old_regression_metrics': {},
            'checkpoint_path': 'ckpt_a.pth',
        }
        invalid_summary = {
            **valid_summary,
            'valid': False,
            'full_recent_loss': math.inf,
            'full_recent_metrics': {},
            'action_quality_score': float('-inf'),
            'rank_acc': -1.0,
            'checkpoint_path': None,
        }

        def fake_summarize_entry(name, candidate, payload, *, ranking_mode='full_recent'):
            return payload['summary']

        with patch.object(fidelity, 'summarize_entry', side_effect=fake_summarize_entry):
            single_seed = fidelity.summarize_multiseed_entry(
                candidate.arm_name,
                candidate,
                {'s1': {'summary': valid_summary, 'cache_path': 'seed1.json'}},
            )
            partial_two_seed = fidelity.summarize_multiseed_entry(
                candidate.arm_name,
                candidate,
                {
                    's1': {'summary': valid_summary, 'cache_path': 'seed1.json'},
                    's2': {'summary': invalid_summary, 'cache_path': 'seed2.json', 'error': 'non-finite loss'},
                },
            )

        self.assertTrue(single_seed['valid'])
        self.assertEqual(1, single_seed['valid_seed_count'])
        self.assertFalse(partial_two_seed['valid'])
        self.assertEqual(1, partial_two_seed['valid_seed_count'])
        self.assertEqual(2, partial_two_seed['seed_count'])

    def test_select_p1_family_survivors_uses_protocol_ce_baseline_not_global_eligibility(self):
        protocol_a = make_candidate(
            'proto_a',
            meta={'stage': 'P0', 'protocol_arm': 'proto_a'},
        )
        protocol_b = make_candidate(
            'proto_b',
            meta={'stage': 'P0', 'protocol_arm': 'proto_b'},
        )

        def make_entry(protocol_arm, name, family, *, loss, eligible):
            candidate = make_candidate(
                name,
                meta={
                    'stage': 'P1_solo_round',
                    'protocol_arm': protocol_arm,
                    'aux_family': family,
                },
            )
            return make_ranking_entry(candidate, eligible=eligible, full_recent_loss=loss)

        ranking = [
            make_entry('proto_b', 'proto_b__ce', 'ce_only', loss=1.000, eligible=True),
            make_entry('proto_a', 'proto_a__ce', 'ce_only', loss=1.005, eligible=False),
            make_entry('proto_a', 'proto_a__danger', 'danger', loss=1.007, eligible=False),
        ]

        survivors = fidelity.select_p1_family_survivors(ranking, [protocol_a, protocol_b])

        self.assertEqual('proto_a__ce', survivors['proto_a']['ce_only'].arm_name)
        self.assertEqual('proto_a__danger', survivors['proto_a']['danger'].arm_name)

    def test_select_p1_family_survivors_prunes_family_that_loses_to_protocol_ce_baseline(self):
        protocol = make_candidate(
            'proto_arm',
            meta={'stage': 'P0', 'protocol_arm': 'proto_arm'},
        )

        def make_entry(name, family, *, loss):
            candidate = make_candidate(
                name,
                meta={
                    'stage': 'P1_solo_round',
                    'protocol_arm': 'proto_arm',
                    'aux_family': family,
                },
            )
            return make_ranking_entry(candidate, eligible=False, full_recent_loss=loss)

        ranking = [
            make_entry('proto_arm__ce', 'ce_only', loss=1.000),
            make_entry('proto_arm__rank', 'rank', loss=1.010),
            make_entry('proto_arm__opp', 'opp', loss=1.020),
            make_entry('proto_arm__danger', 'danger', loss=1.004),
        ]

        survivors = fidelity.select_p1_family_survivors(ranking, [protocol])

        self.assertEqual('proto_arm__ce', survivors['proto_arm']['ce_only'].arm_name)
        self.assertNotIn('rank', survivors['proto_arm'])
        self.assertNotIn('opp', survivors['proto_arm'])
        self.assertNotIn('danger', survivors['proto_arm'])

    def test_summarize_entry_policy_quality_prefers_action_checkpoint_within_policy_band(self):
        candidate = make_candidate('proto_arm')
        payload = {
            'ok': True,
            'cache_path': 'seed1.json',
            'run': {
                'final': {
                    'best_loss': {
                        'path': 'best_loss.pth',
                        'last_full_recent_metrics': {
                            'loss': 0.620,
                            'policy_loss': 0.500,
                            'action_quality_score': -0.220,
                        },
                        'last_old_regression_metrics': {
                            'loss': 0.710,
                            'policy_loss': 0.610,
                        },
                    },
                    'best_acc': {
                        'path': 'best_acc.pth',
                        'last_full_recent_metrics': {
                            'loss': 0.626,
                            'policy_loss': 0.502,
                            'action_quality_score': -0.210,
                        },
                        'last_old_regression_metrics': {
                            'loss': 0.711,
                            'policy_loss': 0.611,
                        },
                    },
                }
            },
        }

        summary = fidelity.summarize_entry(
            candidate.arm_name,
            candidate,
            payload,
            ranking_mode='policy_quality',
        )

        self.assertEqual('best_acc', summary['checkpoint_type'])
        self.assertEqual(['best_acc', 'best_loss'], summary['eligible_checkpoint_types'])
        self.assertEqual('best_acc.pth', summary['checkpoint_path'])
        self.assertAlmostEqual(0.502, summary['recent_policy_loss'], places=6)
        self.assertAlmostEqual(-0.210, summary['action_quality_score'], places=6)

    def test_summarize_entry_full_recent_pins_best_loss_checkpoint(self):
        candidate = make_candidate('proto_arm')
        payload = {
            'ok': True,
            'cache_path': 'seed1.json',
            'run': {
                'final': {
                    'best_loss': {
                        'path': 'best_loss.pth',
                        'last_full_recent_metrics': {
                            'loss': 0.620,
                            'policy_loss': 0.500,
                            'discard_nll': 0.220,
                        },
                        'last_old_regression_metrics': {
                            'loss': 0.710,
                            'policy_loss': 0.610,
                        },
                    },
                    'best_acc': {
                        'path': 'best_acc.pth',
                        'last_full_recent_metrics': {
                            'loss': 0.626,
                            'policy_loss': 0.502,
                            'discard_nll': 0.210,
                        },
                        'last_old_regression_metrics': {
                            'loss': 0.711,
                            'policy_loss': 0.611,
                        },
                    },
                }
            },
        }

        summary = fidelity.summarize_entry(
            candidate.arm_name,
            candidate,
            payload,
        )

        self.assertEqual('best_loss', summary['checkpoint_type'])
        self.assertEqual(['best_loss'], summary['eligible_checkpoint_types'])
        self.assertEqual('best_loss.pth', summary['checkpoint_path'])
        self.assertAlmostEqual(0.620, summary['full_recent_loss'], places=6)
        self.assertAlmostEqual(0.220, summary['full_recent_metrics']['discard_nll'], places=6)
        self.assertAlmostEqual(0.500, summary['recent_policy_loss'], places=6)

    def test_choose_checkpoint_summary_policy_quality_ignores_off_band_old_regression_baseline(self):
        winner_type, winner_summary, eligible_types = fidelity.choose_checkpoint_summary(
            {
                'best_loss': {
                    'path': 'best_loss.pth',
                    'last_full_recent_metrics': {
                        'policy_loss': 0.500,
                        'action_quality_score': -0.220,
                    },
                    'last_old_regression_metrics': {
                        'policy_loss': 0.620,
                    },
                },
                'best_acc': {
                    'path': 'best_acc.pth',
                    'last_full_recent_metrics': {
                        'policy_loss': 0.502,
                        'action_quality_score': -0.210,
                    },
                    'last_old_regression_metrics': {
                        'policy_loss': 0.621,
                    },
                },
                'best_rank': {
                    'path': 'best_rank.pth',
                    'last_full_recent_metrics': {
                        'policy_loss': 0.510,
                        'action_quality_score': -0.050,
                    },
                    'last_old_regression_metrics': {
                        'policy_loss': 0.600,
                    },
                },
            },
            ranking_mode='policy_quality',
        )

        self.assertEqual('best_acc', winner_type)
        self.assertEqual('best_acc.pth', winner_summary['path'])
        self.assertEqual(['best_acc', 'best_loss'], eligible_types)

    def test_rank_round_entries_policy_quality_ignores_off_band_group_old_regression_baseline(self):
        def make_entry(name, *, policy_loss, old_policy_loss, action_quality_score):
            candidate = make_candidate(
                name,
                meta={'stage': 'P1_solo_round', 'protocol_arm': 'proto_arm'},
            )
            return make_ranking_entry(
                candidate,
                full_recent_loss=1.000,
                full_recent_metrics={
                    'policy_loss': policy_loss,
                    'action_quality_score': action_quality_score,
                },
                old_regression_metrics={
                    'policy_loss': old_policy_loss,
                },
            )

        ranked = fidelity.rank_round_entries(
            [
                make_entry('proto_arm__best_loss', policy_loss=0.500, old_policy_loss=0.620, action_quality_score=-0.220),
                make_entry('proto_arm__best_acc', policy_loss=0.502, old_policy_loss=0.621, action_quality_score=-0.210),
                make_entry('proto_arm__best_rank', policy_loss=0.510, old_policy_loss=0.600, action_quality_score=-0.050),
            ],
            ranking_mode='policy_quality',
            eligibility_group_key='protocol_arm',
        )

        self.assertEqual('proto_arm__best_acc', ranked[0]['arm_name'])
        self.assertEqual(['proto_arm__best_acc', 'proto_arm__best_loss'], [entry['arm_name'] for entry in ranked if entry['eligible']])
        self.assertFalse(next(entry for entry in ranked if entry['arm_name'] == 'proto_arm__best_rank')['eligible'])
        self.assertAlmostEqual(
            0.620,
            next(
                entry
                for entry in ranked
                if entry['arm_name'] == 'proto_arm__best_acc'
            )['eligibility_old_regression_loss_baseline'],
            places=6,
        )

    def test_p1_selection_policy_metadata_uses_decoupled_epsilons(self):
        metadata = fidelity.p1_selection_policy_metadata()

        self.assertEqual('policy_quality', metadata['canonical_selector'])
        self.assertEqual('recent_policy_loss', metadata['comparison_metric'])
        self.assertEqual('protocol_arm', metadata['eligibility_group_key'])
        self.assertAlmostEqual(0.003, metadata['policy_loss_epsilon'])
        self.assertAlmostEqual(0.0035, metadata['old_regression_policy_loss_epsilon'])
        self.assertAlmostEqual(0.20, metadata['selection_scenario_factor'])
        self.assertEqual(
            [
                'selection_quality_score',
                '-recent_policy_loss',
                '-old_regression_policy_loss',
            ],
            metadata['tiebreak_order'],
        )

    def test_select_p1_family_survivors_policy_quality_uses_ce_policy_and_old_regression_baseline(self):
        protocol = make_candidate(
            'proto_arm',
            meta={'stage': 'P0', 'protocol_arm': 'proto_arm'},
        )

        def make_entry(name, family, *, loss, policy_loss, old_policy_loss):
            candidate = make_candidate(
                name,
                meta={
                    'stage': 'P1_solo_round',
                    'protocol_arm': 'proto_arm',
                    'aux_family': family,
                },
            )
            return make_ranking_entry(
                candidate,
                full_recent_loss=loss,
                full_recent_metrics={'policy_loss': policy_loss},
                old_regression_metrics={'policy_loss': old_policy_loss},
            )

        ranking = [
            make_entry('proto_arm__ce', 'ce_only', loss=1.000, policy_loss=0.500, old_policy_loss=0.600),
            make_entry('proto_arm__danger', 'danger', loss=1.010, policy_loss=0.502, old_policy_loss=0.602),
            make_entry('proto_arm__opp', 'opp', loss=1.006, policy_loss=0.501, old_policy_loss=0.605),
        ]

        survivors = fidelity.select_p1_family_survivors(
            ranking,
            [protocol],
            ranking_mode='policy_quality',
        )

        self.assertEqual('proto_arm__ce', survivors['proto_arm']['ce_only'].arm_name)
        self.assertEqual('proto_arm__danger', survivors['proto_arm']['danger'].arm_name)
        self.assertNotIn('opp', survivors['proto_arm'])

    def test_run_p1_uses_policy_quality_for_search_rounds(self):
        protocol = make_candidate(
            'proto_arm',
            meta={'stage': 'P0', 'protocol_arm': 'proto_arm'},
        )
        eval_splits = {
            'monitor_recent_files': ['monitor.json.gz'],
            'full_recent_files': ['recent.json.gz'],
            'old_regression_files': ['old.json.gz'],
        }
        calibration = {
            'budget_ratios': fidelity.P1_EFFECTIVE_BUDGET_RATIOS,
            'opp_weight_per_budget_unit': 0.01,
            'danger_weight_per_budget_unit': 0.02,
            'joint_combo_factor': 1.0,
            'protocol_joint_combo_factors': {'proto_arm': 1.0},
            'mapping_mode': fidelity.P1_CALIBRATION_MAPPING_MODE,
        }
        protocol_entry = make_ranking_entry(
            make_candidate(
                'proto_arm__protocol',
                meta={'stage': 'P1_protocol_decide_round', 'protocol_arm': 'proto_arm', 'aux_family': 'all_three'},
            ),
            full_recent_loss=0.620,
            full_recent_metrics={'policy_loss': 0.500},
        )
        refine_entry = make_ranking_entry(
            make_candidate(
                'proto_arm__refine',
                meta={'stage': 'P1_winner_refine_round', 'protocol_arm': 'proto_arm', 'aux_family': 'all_three'},
            ),
            full_recent_loss=0.621,
            full_recent_metrics={'policy_loss': 0.501},
        )
        progressive_calls = []
        multiseed_calls = []
        rank_calls = []
        original_rank_round_entries = fidelity.rank_round_entries

        def fake_execute_round_progressive_multiseed(**kwargs):
            progressive_calls.append(
                (kwargs['round_name'], kwargs.get('ranking_mode'), kwargs.get('eligibility_group_key'))
            )
            return {'ranking': [protocol_entry]}

        def fake_execute_round_multiseed(**kwargs):
            multiseed_calls.append(
                (kwargs['round_name'], kwargs.get('ranking_mode'), kwargs.get('eligibility_group_key'))
            )
            if kwargs['round_name'] == 'p1_calibration':
                return {'ranking': []}
            return {'ranking': [refine_entry]}

        def tracking_rank_round_entries(entries, weights=None, **kwargs):
            rank_calls.append(kwargs)
            return original_rank_round_entries(entries, weights, **kwargs)

        with (
            patch.object(fidelity.ab, 'build_eval_splits', return_value=eval_splits),
            patch.object(fidelity, 'rank_weight_mean_for_files', return_value=0.1),
            patch.object(fidelity, 'derive_p1_budget_calibration', return_value=calibration),
            patch.object(fidelity, 'execute_round_progressive_multiseed', side_effect=fake_execute_round_progressive_multiseed),
            patch.object(fidelity, 'execute_round_multiseed', side_effect=fake_execute_round_multiseed),
            patch.object(
                fidelity,
                'winner_refine_center_selection_from_search_space',
                return_value={'keep': None, 'explicit_arm_names': (protocol_entry['arm_name'],)},
            ),
            patch.object(fidelity, 'atomic_write_json'),
            patch.object(fidelity, 'update_results_doc'),
            patch.object(fidelity, 'rank_round_entries', side_effect=tracking_rank_round_entries),
        ):
            fidelity.run_p1(
                Path('X:/virtual/stage05_fidelity'),
                {'control': {'version': 4}},
                {'2026-01': ['train_a.json.gz']},
                17,
                [protocol],
                {'started_at': '2026-03-22 00:00:00'},
            )

        self.assertIn(
            ('p1_calibration', 'policy_quality', 'protocol_arm'),
            multiseed_calls,
        )
        self.assertIn(
            ('p1_protocol_decide_round', 'policy_quality', 'protocol_arm'),
            progressive_calls,
        )
        self.assertIn(
            ('p1_winner_refine_round', 'policy_quality', 'protocol_arm'),
            multiseed_calls,
        )
        self.assertNotIn(
            ('p1_ablation_round', 'policy_quality', 'protocol_arm'),
            multiseed_calls,
        )
        self.assertEqual('policy_quality', rank_calls[-1].get('ranking_mode'))

    def test_select_p1_finalists_skips_invalid_multiseed_entries(self):
        def make_protocol_entry(name, protocol_arm, *, valid, loss):
            candidate = make_candidate(
                name,
                meta={
                    'stage': 'P1_joint_refine_round',
                    'protocol_arm': protocol_arm,
                },
            )
            return make_ranking_entry(candidate, valid=valid, full_recent_loss=loss)

        ranking = [
            make_protocol_entry('proto_a__failed', 'proto_a', valid=False, loss=math.inf),
            make_protocol_entry('proto_b__failed', 'proto_b', valid=False, loss=math.inf),
            make_protocol_entry('proto_b__winner', 'proto_b', valid=True, loss=0.95),
        ]

        finalists = fidelity.select_p1_finalists(ranking)

        self.assertEqual(['proto_b__winner'], [entry['arm_name'] for entry in finalists])

    def test_select_p1_finalists_requires_at_least_one_valid_entry(self):
        ranking = [
            make_ranking_entry(
                make_candidate(
                    'proto_a__failed',
                    meta={'stage': 'P1_joint_refine_round', 'protocol_arm': 'proto_a'},
                ),
                valid=False,
                full_recent_loss=math.inf,
            ),
        ]

        with self.assertRaisesRegex(RuntimeError, 'no valid finalists'):
            fidelity.select_p1_finalists(ranking)

    def test_round_cache_signature_changes_with_scenario_score_version(self):
        candidate = make_candidate('proto_arm')
        kwargs = {
            'round_name': 'p0_round0',
            'ab_name': 'stage05_fidelity_p0_r0',
            'base_cfg': {'control': {'version': 4}},
            'grouped': {'2026-01': ['train_a.json.gz']},
            'eval_splits': {
                'monitor_recent_files': ['monitor.json.gz'],
                'full_recent_files': ['recent.json.gz'],
                'old_regression_files': ['old.json.gz'],
            },
            'candidates': [candidate],
            'seed': 17,
            'step_scale': 0.5,
            'selector_weights': None,
        }

        with patch.object(fidelity, 'SCENARIO_SCORE_VERSION', 'scenario-v1'):
            sig_v1 = fidelity.round_cache_signature(**kwargs)
        with patch.object(fidelity, 'SCENARIO_SCORE_VERSION', 'scenario-v2'):
            sig_v2 = fidelity.round_cache_signature(**kwargs)

        self.assertNotEqual(sig_v1, sig_v2)

    def test_load_cached_p0_stage1_top4_rejects_stale_round2_signature(self):
        stored_json = {}

        def fake_exists(path_obj):
            return str(path_obj) in stored_json

        grouped = {'2026-01': ['train_a.json.gz']}
        base_cfg = {'control': {'version': 4}}
        run_dir = Path('X:/virtual/stage05_fidelity')
        seed = 17
        candidates = [make_candidate(f'arm{i}', meta={'stage': 'P0'}) for i in range(6)]

        def fake_eval_splits(grouped, split_seed, file_count):
            return {
                'monitor_recent_files': [f'monitor_{split_seed}.json.gz'],
                'full_recent_files': [f'recent_{split_seed}.json.gz'],
                'old_regression_files': [f'old_{split_seed}.json.gz'],
            }

        round0_ranking = [make_ranking_entry(candidate, full_recent_loss=1.0 + idx * 0.01) for idx, candidate in enumerate(candidates)]
        round1_ranking = [make_ranking_entry(candidate, full_recent_loss=1.0 + idx * 0.01) for idx, candidate in enumerate(candidates)]
        round2_ranking = [make_ranking_entry(candidate, full_recent_loss=1.0 + idx * 0.01) for idx, candidate in enumerate(candidates)]

        with patch.object(fidelity.ab, 'build_eval_splits', side_effect=fake_eval_splits):
            round0_signature = fidelity.round_cache_signature(
                round_name='p0_round0',
                ab_name=f'{run_dir.name}_p0_r0',
                base_cfg=base_cfg,
                grouped=grouped,
                eval_splits=fake_eval_splits(grouped, seed + 11, fidelity.ab.BASE_SCREENING['eval_files']),
                candidates=candidates,
                seed=seed + 101,
                step_scale=0.5,
                selector_weights=None,
            )
            round1_signature = fidelity.round_cache_signature(
                round_name='p0_round1',
                ab_name=f'{run_dir.name}_p0_r1',
                base_cfg=base_cfg,
                grouped=grouped,
                eval_splits=fake_eval_splits(grouped, seed + 22, fidelity.ab.BASE_SCREENING['eval_files']),
                candidates=candidates,
                seed=seed + 202,
                step_scale=1.0,
                selector_weights=None,
            )

        stored_json[str(run_dir / 'p0_round0.json')] = {'round_signature': round0_signature, 'ranking': round0_ranking}
        stored_json[str(run_dir / 'p0_round1.json')] = {'round_signature': round1_signature, 'ranking': round1_ranking}
        stored_json[str(run_dir / 'p0_round2.json')] = {'round_signature': 'stale-signature', 'ranking': round2_ranking}

        with (
            patch.object(Path, 'exists', fake_exists),
            patch.object(fidelity, 'load_json', side_effect=lambda path: stored_json[str(path)]),
            patch.object(fidelity, 'build_p0_candidates', return_value=candidates),
            patch.object(fidelity.ab, 'build_eval_splits', side_effect=fake_eval_splits),
        ):
            stage1_top4 = fidelity.load_cached_p0_stage1_top4(
                run_dir,
                {},
                base_cfg=base_cfg,
                grouped=grouped,
                seed=seed,
            )

        self.assertIsNone(stage1_top4)

    def test_select_p0_stage1_top4_uses_round2_order(self):
        candidates = [
            make_candidate(f'arm_{idx}', meta={'stage': 'P0'})
            for idx in range(6)
        ]
        ranking = [
            make_ranking_entry(candidate, full_recent_loss=0.1 + idx * 0.01)
            for idx, candidate in enumerate(candidates)
        ]

        selected = fidelity.select_p0_stage1_top4(ranking)

        self.assertEqual(
            ['arm_0', 'arm_1', 'arm_2', 'arm_3'],
            [candidate.arm_name for candidate in selected],
        )

    def test_blend_positive_calibration_values_uses_geometric_mean(self):
        blended = fidelity.blend_positive_calibration_values(
            loss_value=0.03,
            grad_value=0.012,
            fallback=0.01,
        )
        self.assertAlmostEqual(0.019, blended, places=3)

    def test_derive_p1_budget_calibration_combines_loss_and_grad_axes(self):
        calibration_round = {
            'ranking': [
                {
                    'valid': True,
                    'arm_name': 'proto__CAL_rank_only',
                    'candidate_meta': {'calibration_role': 'rank_only', 'protocol_arm': 'proto'},
                    'full_recent_metrics': {
                        'rank_aux_weight_mean': 0.1,
                        'rank_aux_raw_loss': 0.2,
                        'rank_phi_grad_rms': 0.05,
                    },
                },
                {
                    'valid': True,
                    'arm_name': 'proto__CAL_rank_opp_probe',
                    'candidate_meta': {'calibration_role': 'rank_opp_probe', 'protocol_arm': 'proto'},
                    'full_recent_metrics': {
                        'rank_aux_weight_mean': 0.09,
                        'rank_aux_raw_loss': 0.2,
                        'opponent_aux_loss': 0.03,
                        'rank_phi_grad_rms': 0.05,
                        'opponent_phi_grad_rms': 0.12,
                        'rank_opponent_phi_grad_cos': 0.5,
                    },
                },
                {
                    'valid': True,
                    'arm_name': 'proto__CAL_rank_danger_probe',
                    'candidate_meta': {'calibration_role': 'rank_danger_probe', 'protocol_arm': 'proto'},
                    'full_recent_metrics': {
                        'rank_aux_weight_mean': 0.095,
                        'rank_aux_raw_loss': 0.2,
                        'danger_aux_loss': 0.06,
                        'rank_phi_grad_rms': 0.05,
                        'danger_phi_grad_rms': 0.09,
                        'rank_danger_phi_grad_cos': 0.4,
                    },
                },
                {
                    'valid': True,
                    'arm_name': 'proto__CAL_opp_danger_probe',
                    'candidate_meta': {'calibration_role': 'opp_danger_probe', 'protocol_arm': 'proto'},
                    'full_recent_metrics': {
                        'opponent_aux_loss': 0.027,
                        'danger_aux_loss': 0.054,
                        'opponent_phi_grad_rms': 0.12,
                        'danger_phi_grad_rms': 0.09,
                        'opp_danger_phi_combo_factor': 0.8,
                    },
                },
                {
                    'valid': True,
                    'arm_name': 'proto__CAL_triple_probe',
                    'candidate_meta': {'calibration_role': 'triple_probe', 'protocol_arm': 'proto'},
                    'full_recent_metrics': {
                        'rank_aux_weight_mean': 0.085,
                        'rank_aux_raw_loss': 0.2,
                        'opponent_aux_loss': 0.028,
                        'danger_aux_loss': 0.055,
                        'rank_phi_grad_rms': 0.05,
                        'opponent_phi_grad_rms': 0.12,
                        'danger_phi_grad_rms': 0.09,
                        'rank_opponent_phi_grad_cos': 0.5,
                        'rank_danger_phi_grad_cos': 0.4,
                        'opp_danger_phi_grad_cos': 0.2,
                    },
                },
            ]
        }

        calibration = fidelity.derive_p1_budget_calibration(calibration_round)

        self.assertEqual('hybrid_loss_grad_geomean', calibration['mapping_mode'])
        self.assertAlmostEqual(0.02, calibration['rank_effective_base'], places=6)
        self.assertAlmostEqual(0.05, calibration['rank_grad_effective_base'], places=6)
        self.assertAlmostEqual(0.5, calibration['opp_effective_per_unit'], places=6)
        self.assertAlmostEqual(2.0, calibration['opp_grad_effective_per_unit'], places=6)
        self.assertAlmostEqual(1.0, calibration['danger_effective_per_unit'], places=6)
        self.assertAlmostEqual(1.5, calibration['danger_grad_effective_per_unit'], places=6)
        self.assertAlmostEqual(0.04, calibration['opp_weight_per_budget_unit_loss'], places=3)
        self.assertAlmostEqual(0.025, calibration['opp_weight_per_budget_unit_grad'], places=3)
        self.assertAlmostEqual(0.032, calibration['opp_weight_per_budget_unit'], places=3)
        self.assertAlmostEqual(0.02, calibration['danger_weight_per_budget_unit_loss'], places=3)
        self.assertAlmostEqual(0.033, calibration['danger_weight_per_budget_unit_grad'], places=3)
        self.assertAlmostEqual(0.026, calibration['danger_weight_per_budget_unit'], places=3)
        self.assertEqual(fidelity.P1_CALIBRATION_SCHEME, calibration['combo_scheme'])
        self.assertAlmostEqual(0.924, calibration['rank_opp_combo_factor'], places=3)
        self.assertAlmostEqual(0.917, calibration['rank_danger_combo_factor'], places=3)
        self.assertAlmostEqual(0.8, calibration['opp_danger_combo_factor_grad'], places=3)
        self.assertAlmostEqual(0.849, calibration['opp_danger_combo_factor'], places=3)
        self.assertAlmostEqual(0.759, calibration['triple_combo_factor_grad'], places=3)
        self.assertAlmostEqual(0.831, calibration['triple_combo_factor'], places=3)
        self.assertAlmostEqual(calibration['opp_danger_combo_factor'], calibration['joint_combo_factor'], places=6)

    def test_derive_p1_budget_calibration_combo_only_inherits_single_head_baseline(self):
        inherited = {
            'rank_effective_base': 0.0512,
            'opp_effective_per_unit': 0.899,
            'danger_effective_per_unit': 0.0492,
            'rank_grad_effective_base': 8.4e-7,
            'opp_grad_effective_per_unit': 1.8e-5,
            'danger_grad_effective_per_unit': 1.57e-6,
            'opp_weight_per_budget_unit_loss': 0.057,
            'danger_weight_per_budget_unit_loss': 0.18,
            'opp_weight_per_budget_unit_grad': 0.047,
            'danger_weight_per_budget_unit_grad': 0.18,
            'opp_weight_per_budget_unit': 0.052,
            'danger_weight_per_budget_unit': 0.18,
        }
        calibration_round = {
            'ranking': [
                {
                    'valid': True,
                    'arm_name': 'proto__CAL_rank_opp_probe',
                    'candidate_meta': {
                        'calibration_role': 'rank_opp_probe',
                        'protocol_arm': 'proto',
                        'calibration_mode': 'combo_only',
                    },
                    'full_recent_metrics': {
                        'rank_aux_weight_mean': 0.09,
                        'rank_aux_raw_loss': 0.2,
                        'opponent_aux_loss': 0.03,
                        'rank_phi_grad_rms': 0.05,
                        'opponent_phi_grad_rms': 0.12,
                        'rank_opponent_phi_grad_cos': 0.5,
                    },
                },
                {
                    'valid': True,
                    'arm_name': 'proto__CAL_rank_danger_probe',
                    'candidate_meta': {
                        'calibration_role': 'rank_danger_probe',
                        'protocol_arm': 'proto',
                        'calibration_mode': 'combo_only',
                    },
                    'full_recent_metrics': {
                        'rank_aux_weight_mean': 0.095,
                        'rank_aux_raw_loss': 0.2,
                        'danger_aux_loss': 0.06,
                        'rank_phi_grad_rms': 0.05,
                        'danger_phi_grad_rms': 0.09,
                        'rank_danger_phi_grad_cos': 0.4,
                    },
                },
                {
                    'valid': True,
                    'arm_name': 'proto__CAL_opp_danger_probe',
                    'candidate_meta': {
                        'calibration_role': 'opp_danger_probe',
                        'protocol_arm': 'proto',
                        'calibration_mode': 'combo_only',
                    },
                    'full_recent_metrics': {
                        'opponent_aux_loss': 0.027,
                        'danger_aux_loss': 0.054,
                        'opponent_phi_grad_rms': 0.12,
                        'danger_phi_grad_rms': 0.09,
                        'opp_danger_phi_combo_factor': 0.8,
                    },
                },
                {
                    'valid': True,
                    'arm_name': 'proto__CAL_triple_probe',
                    'candidate_meta': {
                        'calibration_role': 'triple_probe',
                        'protocol_arm': 'proto',
                        'calibration_mode': 'combo_only',
                    },
                    'full_recent_metrics': {
                        'rank_aux_weight_mean': 0.085,
                        'rank_aux_raw_loss': 0.2,
                        'opponent_aux_loss': 0.028,
                        'danger_aux_loss': 0.055,
                        'rank_phi_grad_rms': 0.05,
                        'opponent_phi_grad_rms': 0.12,
                        'danger_phi_grad_rms': 0.09,
                        'rank_opponent_phi_grad_cos': 0.5,
                        'rank_danger_phi_grad_cos': 0.4,
                        'opp_danger_phi_grad_cos': 0.2,
                    },
                },
            ]
        }

        calibration = fidelity.derive_p1_budget_calibration(
            calibration_round,
            inherited_single_head=inherited,
            inherited_single_head_source='test inherited baseline',
        )

        self.assertEqual('combo_only', calibration['calibration_mode'])
        self.assertIn(
            'inherits the frozen 2026-03-25 post-shape single-head calibration baseline',
            calibration['calibration_mode_note'],
        )
        self.assertTrue(calibration['inherited_single_head'])
        self.assertEqual('test inherited baseline', calibration['inherited_single_head_source'])
        self.assertAlmostEqual(0.0512, calibration['rank_effective_base'])
        self.assertAlmostEqual(0.052, calibration['opp_weight_per_budget_unit'])
        self.assertAlmostEqual(0.18, calibration['danger_weight_per_budget_unit'])
        self.assertTrue(0.75 <= calibration['rank_opp_combo_factor'] <= 1.25)
        self.assertTrue(0.75 <= calibration['rank_danger_combo_factor'] <= 1.25)
        self.assertTrue(0.75 <= calibration['opp_danger_combo_factor'] <= 1.25)
        self.assertTrue(0.75 <= calibration['triple_combo_factor'] <= 1.25)


if __name__ == '__main__':
    unittest.main()
