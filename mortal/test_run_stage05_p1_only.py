import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_stage05_p1_only as p1_only


class Stage05P1OnlyTests(unittest.TestCase):
    def test_build_protocol_candidates_defaults_to_frozen_top3(self):
        candidates = p1_only.build_protocol_candidates(list(p1_only.FROZEN_TOP3))

        self.assertEqual(list(p1_only.FROZEN_TOP3), [candidate.arm_name for candidate in candidates])
        self.assertTrue(all(candidate.meta['protocol_arm'] == candidate.arm_name for candidate in candidates))
        self.assertTrue(all(candidate.meta['selection_source'] == 'frozen_top3' for candidate in candidates))

    def test_build_p1_search_space_uses_current_winner_refine_defaults(self):
        search_space = p1_only.build_p1_search_space(
            {
                'budget_ratios': [0.0, 0.25],
                'opp_weight_per_budget_unit': 0.052,
                'danger_weight_per_budget_unit': 0.18,
            }
        )

        self.assertEqual(
            p1_only.fidelity.P1_PROTOCOL_DECIDE_PROGRESSIVE_AMBIGUITY_MODE,
            search_space['protocol_decide_progressive_ambiguity_mode'],
        )
        self.assertEqual(
            p1_only.fidelity.P1_PROTOCOL_DECIDE_PROGRESSIVE_GAP_THRESHOLD,
            search_space['protocol_decide_progressive_gap_threshold'],
        )
        self.assertEqual(
            p1_only.fidelity.P1_WINNER_REFINE_CENTER_MODE,
            search_space['winner_refine_center_mode'],
        )
        self.assertEqual(
            p1_only.fidelity.P1_PROTOCOL_DECIDE_COORDINATE_MODE,
            search_space['protocol_decide_coordinate_mode'],
        )
        self.assertEqual(
            list(p1_only.fidelity.P1_PROTOCOL_DECIDE_TOTAL_BUDGET_RATIOS),
            search_space['protocol_decide_total_budget_ratios'],
        )
        self.assertEqual(
            p1_only.fidelity.current_protocol_decide_mix_payload(),
            search_space['protocol_decide_mixes'],
        )
        self.assertEqual(
            p1_only.fidelity.P1_WINNER_REFINE_CENTER_KEEP,
            search_space['winner_refine_center_keep'],
        )
        self.assertEqual(
            list(p1_only.fidelity.P1_WINNER_REFINE_CENTER_ARM_NAMES),
            search_space['winner_refine_center_arm_names'],
        )

    def test_select_protocol_centers_excludes_ce_only_anchor(self):
        protocol_arm = p1_only.FROZEN_TOP3[0]
        ce_only = p1_only.fidelity.CandidateSpec(
            arm_name='ce_only_anchor',
            scheduler_profile='cosine',
            curriculum_profile='broad_to_recent',
            weight_profile='strong',
            window_profile='24m_12m',
            cfg_overrides={},
            meta={
                'protocol_arm': protocol_arm,
                'aux_family': 'ce_only',
                'rank_budget_ratio': 0.0,
                'opp_budget_ratio': 0.0,
                'danger_budget_ratio': 0.0,
            },
        )
        all_three = p1_only.fidelity.CandidateSpec(
            arm_name='all_three_anchor',
            scheduler_profile='cosine',
            curriculum_profile='broad_to_recent',
            weight_profile='strong',
            window_profile='24m_12m',
            cfg_overrides={},
            meta={
                'protocol_arm': protocol_arm,
                'aux_family': 'all_three',
                'rank_budget_ratio': 0.05,
                'opp_budget_ratio': 0.04,
                'danger_budget_ratio': 0.03,
            },
        )

        centers = p1_only.select_protocol_centers(
            ranked=[
                {
                    'arm_name': ce_only.arm_name,
                    'candidate_meta': ce_only.meta,
                    'scheduler_profile': ce_only.scheduler_profile,
                    'curriculum_profile': ce_only.curriculum_profile,
                    'weight_profile': ce_only.weight_profile,
                    'window_profile': ce_only.window_profile,
                    'cfg_overrides': ce_only.cfg_overrides,
                    'valid': True,
                },
                {
                    'arm_name': all_three.arm_name,
                    'candidate_meta': all_three.meta,
                    'scheduler_profile': all_three.scheduler_profile,
                    'curriculum_profile': all_three.curriculum_profile,
                    'weight_profile': all_three.weight_profile,
                    'window_profile': all_three.window_profile,
                    'cfg_overrides': all_three.cfg_overrides,
                    'valid': True,
                },
            ],
            protocol_arm=protocol_arm,
            keep=2,
        )

        self.assertEqual(['all_three_anchor'], [candidate.arm_name for candidate in centers])

    def test_build_protocol_compare_ignores_ce_only_anchor_winners(self):
        ranked = [
            {
                'arm_name': 'proto_a_ce',
                'candidate_meta': {
                    'protocol_arm': 'proto_a',
                    'aux_family': 'ce_only',
                },
                'valid': True,
                'full_recent_loss': 0.10,
                'full_recent_metrics': {
                    'policy_loss': 0.10,
                    'action_quality_score': 0.5,
                },
                'recent_policy_loss': 0.10,
                'old_regression_policy_loss': 0.30,
                'old_regression_metrics': {
                    'policy_loss': 0.30,
                },
            },
            {
                'arm_name': 'proto_b_all',
                'candidate_meta': {
                    'protocol_arm': 'proto_b',
                    'aux_family': 'all_three',
                },
                'valid': True,
                'full_recent_loss': 0.15,
                'full_recent_metrics': {
                    'policy_loss': 0.15,
                    'action_quality_score': 0.5,
                },
                'recent_policy_loss': 0.15,
                'old_regression_policy_loss': 0.30,
                'old_regression_metrics': {
                    'policy_loss': 0.30,
                },
            },
            {
                'arm_name': 'proto_a_all',
                'candidate_meta': {
                    'protocol_arm': 'proto_a',
                    'aux_family': 'all_three',
                },
                'valid': True,
                'full_recent_loss': 0.20,
                'full_recent_metrics': {
                    'policy_loss': 0.20,
                    'action_quality_score': 0.5,
                },
                'recent_policy_loss': 0.20,
                'old_regression_policy_loss': 0.30,
                'old_regression_metrics': {
                    'policy_loss': 0.30,
                },
            },
        ]

        winners = p1_only.build_protocol_compare(ranked)

        self.assertEqual(
            {'proto_a_all', 'proto_b_all'},
            {str(entry['arm_name']) for entry in winners},
        )
        self.assertTrue(
            all(
                str(entry.get('candidate_meta', {}).get('aux_family', '')) == 'all_three'
                for entry in winners
            )
        )

    def test_run_p1_only_records_calibration_then_protocol_decide(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / 'p1_only_run'
            lock_path = run_dir / 'run.lock.json'
            calibration = {
                'budget_ratios': [0.0, 0.25],
                'mapping_mode': 'hybrid_loss_grad_geomean',
                'combo_scheme': 'single_head_mapping_plus_pairwise_triple_combo',
                'opp_weight_per_budget_unit': 0.064,
                'danger_weight_per_budget_unit': 0.144,
                'rank_opp_combo_factor': 0.92,
                'rank_danger_combo_factor': 0.91,
                'opp_danger_combo_factor': 0.88,
                'triple_combo_factor': 0.84,
                'joint_combo_factor': 0.884,
                'protocol_rank_opp_combo_factors': {
                    p1_only.FROZEN_TOP3[0]: 0.91,
                },
                'protocol_rank_danger_combo_factors': {
                    p1_only.FROZEN_TOP3[0]: 0.9,
                },
                'protocol_opp_danger_combo_factors': {
                    p1_only.FROZEN_TOP3[0]: 0.89,
                },
                'protocol_triple_combo_factors': {
                    p1_only.FROZEN_TOP3[0]: 0.85,
                },
                'protocol_joint_combo_factors': {
                    p1_only.FROZEN_TOP3[0]: 0.91,
                },
            }
            solo_survivor = p1_only.build_protocol_candidate(p1_only.FROZEN_TOP3[0])

            with (
                patch.object(p1_only.fidelity, 'acquire_run_lock', return_value=lock_path),
                patch.object(p1_only.fidelity, 'release_run_lock'),
                patch.object(p1_only.fidelity, 'update_results_doc'),
                patch('builtins.print'),
                patch.object(p1_only.ab, 'build_base_config', return_value={'control': {'version': 4}}),
                patch.object(p1_only.ab, 'load_all_files', return_value=['dummy.json.gz']),
                patch.object(p1_only.ab, 'group_files_by_month', return_value={'202501': ['dummy.json.gz']}),
                patch.object(
                    p1_only.ab,
                    'build_eval_splits',
                    return_value={
                        'monitor_recent_files': ['monitor.json.gz'],
                        'full_recent_files': ['recent.json.gz'],
                        'old_regression_files': ['old.json.gz'],
                    },
                ),
                patch.object(p1_only.fidelity, 'rank_weight_mean_for_files', return_value=0.05),
                patch.object(
                    p1_only.fidelity,
                    'build_p1_calibration_candidates',
                    return_value=['calibration_candidate'],
                ),
                patch.object(
                    p1_only.fidelity,
                    'execute_round_multiseed',
                    return_value={'round_name': 'p1_calibration', 'ranking': []},
                ),
                patch.object(p1_only.fidelity, 'derive_p1_budget_calibration', return_value=dict(calibration)),
                patch.object(
                    p1_only.fidelity,
                    'build_p1_protocol_decide_candidates',
                    return_value=['protocol_candidate'],
                ),
                patch.object(
                    p1_only.fidelity,
                    'execute_round_progressive_multiseed',
                    return_value={'round_name': 'p1_protocol_decide_round', 'ranking': []},
                ),
                patch.object(
                    p1_only,
                    'build_protocol_compare',
                    return_value=[
                        {
                            'arm_name': solo_survivor.arm_name,
                            'candidate_meta': {'protocol_arm': solo_survivor.arm_name},
                        }
                    ],
                ),
            ):
                state = p1_only.run_p1_only(
                    run_dir=run_dir,
                    seed=123,
                    protocol_arms=list(p1_only.FROZEN_TOP3),
                )

        self.assertEqual('stopped_after_p1_protocol_decide', state['status'])
        self.assertEqual(list(p1_only.FROZEN_TOP3), state['selected_protocol_arms'])
        self.assertEqual(0.05, state['p1']['shared_rank_template_mean'])
        self.assertEqual('combo_only', state['p1']['calibration_mode'])
        self.assertIn(
            'inherits the frozen 2026-03-25 post-shape single-head calibration baseline',
            state['p1']['calibration_mode_note'],
        )
        self.assertEqual(
            ['C_A2y_cosine_broad_to_recent_strong_12m_6m'],
            state['p1']['calibration_protocol_arms'],
        )
        self.assertEqual(0.064, state['p1']['search_space']['opp_weight_per_budget_unit'])
        self.assertEqual('single_head_mapping_plus_pairwise_triple_combo', state['p1']['search_space']['combo_scheme'])
        self.assertEqual('combo_only', state['p1']['search_space']['calibration_mode'])
        self.assertIn(
            'inherits the frozen 2026-03-25 post-shape single-head calibration baseline',
            state['p1']['search_space']['calibration_mode_note'],
        )
        self.assertEqual(
            ['C_A2y_cosine_broad_to_recent_strong_12m_6m'],
            state['p1']['search_space']['calibration_protocol_arms'],
        )
        self.assertAlmostEqual(0.84, state['p1']['search_space']['triple_combo_factor'])
        self.assertEqual('policy_quality', state['p1']['selection_policy']['canonical_selector'])
        self.assertEqual('recent_policy_loss', state['p1']['search_space']['selection_policy']['comparison_metric'])
        self.assertEqual('protocol_arm', state['p1']['search_space']['selection_policy']['eligibility_group_key'])
        self.assertAlmostEqual(0.003, state['p1']['search_space']['selection_policy']['policy_loss_epsilon'])
        self.assertAlmostEqual(0.0035, state['p1']['search_space']['selection_policy']['old_regression_policy_loss_epsilon'])
        self.assertAlmostEqual(0.20, state['p1']['search_space']['selection_policy']['selection_scenario_factor'])
        self.assertEqual(
            ['anchor', 'rank_lean', 'opp_lean', 'danger_lean'],
            [item['name'] for item in state['p1']['search_space']['protocol_decide_mixes']],
        )
        self.assertEqual([0.09, 0.12], state['p1']['search_space']['protocol_decide_total_budget_ratios'])
        opp_lean = next(
            item for item in state['p1']['search_space']['protocol_decide_mixes']
            if item['name'] == 'opp_lean'
        )
        self.assertAlmostEqual(0.38, opp_lean['rank_share'])
        self.assertAlmostEqual(0.31, opp_lean['opp_share'])
        self.assertAlmostEqual(0.31, opp_lean['danger_share'])
        self.assertEqual(p1_only.FROZEN_TOP3[0], state['p1']['selected_protocol_arm'])
        self.assertEqual(p1_only.FROZEN_TOP3[0], state['final_conclusion']['p1_protocol_winner'])

    def test_run_p1_only_fails_when_ablation_has_no_valid_candidates(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / 'p1_only_run'
            lock_path = run_dir / 'run.lock.json'
            protocol_arm = p1_only.FROZEN_TOP3[0]
            protocol_candidate = p1_only.build_protocol_candidate(protocol_arm)
            calibration = {
                'budget_ratios': [0.0, 0.25],
                'mapping_mode': 'hybrid_loss_grad_geomean',
                'combo_scheme': 'single_head_mapping_plus_pairwise_triple_combo',
                'opp_weight_per_budget_unit': 0.064,
                'danger_weight_per_budget_unit': 0.144,
                'triple_combo_factor': 0.84,
                'joint_combo_factor': 0.884,
                'protocol_triple_combo_factors': {protocol_arm: 0.85},
                'protocol_joint_combo_factors': {protocol_arm: 0.91},
            }
            protocol_compare = [
                {
                    'arm_name': protocol_arm,
                    'candidate_meta': {'protocol_arm': protocol_arm},
                }
            ]
            refine_winner = p1_only.fidelity.CandidateSpec(
                arm_name='refine_winner',
                scheduler_profile=protocol_candidate.scheduler_profile,
                curriculum_profile=protocol_candidate.curriculum_profile,
                weight_profile=protocol_candidate.weight_profile,
                window_profile=protocol_candidate.window_profile,
                cfg_overrides={},
                meta={
                    'protocol_arm': protocol_arm,
                    'aux_family': 'all_three',
                    'rank_budget_ratio': 0.05,
                    'opp_budget_ratio': 0.04,
                    'danger_budget_ratio': 0.03,
                },
            )

            with (
                patch.object(p1_only.fidelity, 'acquire_run_lock', return_value=lock_path),
                patch.object(p1_only.fidelity, 'release_run_lock'),
                patch.object(p1_only.fidelity, 'update_results_doc'),
                patch('builtins.print'),
                patch.object(p1_only.ab, 'build_base_config', return_value={'control': {'version': 4}}),
                patch.object(p1_only.ab, 'load_all_files', return_value=['dummy.json.gz']),
                patch.object(p1_only.ab, 'group_files_by_month', return_value={'202501': ['dummy.json.gz']}),
                patch.object(
                    p1_only.ab,
                    'build_eval_splits',
                    return_value={
                        'monitor_recent_files': ['monitor.json.gz'],
                        'full_recent_files': ['recent.json.gz'],
                        'old_regression_files': ['old.json.gz'],
                    },
                ),
                patch.object(p1_only.fidelity, 'rank_weight_mean_for_files', return_value=0.05),
                patch.object(
                    p1_only.fidelity,
                    'build_p1_calibration_candidates',
                    return_value=['calibration_candidate'],
                ),
                patch.object(
                    p1_only.fidelity,
                    'build_p1_protocol_decide_candidates',
                    return_value=['protocol_candidate'],
                ),
                patch.object(
                    p1_only.fidelity,
                    'build_p1_winner_refine_candidates',
                    return_value=['winner_refine_candidate'],
                ),
                patch.object(
                    p1_only.fidelity,
                    'build_p1_ablation_candidates',
                    return_value=['ablation_candidate'],
                ),
                patch.object(
                    p1_only.fidelity,
                    'execute_round_multiseed',
                    side_effect=[
                        {'round_name': 'p1_calibration', 'ranking': []},
                        {
                            'round_name': 'p1_winner_refine_round',
                            'ranking': [
                                {
                                    'arm_name': refine_winner.arm_name,
                                    'candidate_meta': refine_winner.meta,
                                    'scheduler_profile': refine_winner.scheduler_profile,
                                    'curriculum_profile': refine_winner.curriculum_profile,
                                    'weight_profile': refine_winner.weight_profile,
                                    'window_profile': refine_winner.window_profile,
                                    'cfg_overrides': refine_winner.cfg_overrides,
                                    'valid': True,
                                }
                            ],
                        },
                        {
                            'round_name': 'p1_ablation_round',
                            'ranking': [
                                {
                                    'arm_name': 'bad_ablation',
                                    'candidate_meta': {
                                        'protocol_arm': protocol_arm,
                                        'aux_family': 'drop_rank',
                                    },
                                    'scheduler_profile': refine_winner.scheduler_profile,
                                    'curriculum_profile': refine_winner.curriculum_profile,
                                    'weight_profile': refine_winner.weight_profile,
                                    'window_profile': refine_winner.window_profile,
                                    'cfg_overrides': {},
                                    'valid': False,
                                }
                            ],
                        },
                    ],
                ),
                patch.object(p1_only.fidelity, 'derive_p1_budget_calibration', return_value=dict(calibration)),
                patch.object(
                    p1_only.fidelity,
                    'execute_round_progressive_multiseed',
                    return_value={
                        'round_name': 'p1_protocol_decide_round',
                        'ranking': [
                            {
                                'arm_name': refine_winner.arm_name,
                                'candidate_meta': refine_winner.meta,
                                'scheduler_profile': refine_winner.scheduler_profile,
                                'curriculum_profile': refine_winner.curriculum_profile,
                                'weight_profile': refine_winner.weight_profile,
                                'window_profile': refine_winner.window_profile,
                                'cfg_overrides': refine_winner.cfg_overrides,
                                'valid': True,
                            }
                        ],
                    },
                ),
                patch.object(
                    p1_only.fidelity,
                    'winner_refine_center_selection_from_search_space',
                    return_value={'keep': None, 'explicit_arm_names': (refine_winner.arm_name,)},
                ),
                patch.object(p1_only, 'build_protocol_compare', return_value=protocol_compare),
            ):
                state = p1_only.run_p1_only(
                    run_dir=run_dir,
                    seed=123,
                    protocol_arms=[protocol_arm],
                    continue_to_winner_refine=True,
                    continue_to_ablation=True,
                )

        self.assertEqual('failed', state['status'])
        self.assertIn('p1_ablation_round produced no valid candidates', state['fatal_error'])

    def test_run_p1_only_continue_to_ablation_implies_winner_refine(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / 'p1_only_run'
            lock_path = run_dir / 'run.lock.json'
            protocol_arm = p1_only.FROZEN_TOP3[0]
            protocol_candidate = p1_only.build_protocol_candidate(protocol_arm)
            calibration = {
                'budget_ratios': [0.0, 0.25],
                'mapping_mode': 'hybrid_loss_grad_geomean',
                'combo_scheme': 'single_head_mapping_plus_pairwise_triple_combo',
                'opp_weight_per_budget_unit': 0.064,
                'danger_weight_per_budget_unit': 0.144,
                'triple_combo_factor': 0.84,
                'joint_combo_factor': 0.884,
                'protocol_triple_combo_factors': {protocol_arm: 0.85},
                'protocol_joint_combo_factors': {protocol_arm: 0.91},
            }
            protocol_compare = [
                {
                    'arm_name': protocol_arm,
                    'candidate_meta': {'protocol_arm': protocol_arm},
                }
            ]
            refine_winner = p1_only.fidelity.CandidateSpec(
                arm_name='refine_winner',
                scheduler_profile=protocol_candidate.scheduler_profile,
                curriculum_profile=protocol_candidate.curriculum_profile,
                weight_profile=protocol_candidate.weight_profile,
                window_profile=protocol_candidate.window_profile,
                cfg_overrides={},
                meta={
                    'protocol_arm': protocol_arm,
                    'aux_family': 'all_three',
                    'rank_budget_ratio': 0.05,
                    'opp_budget_ratio': 0.04,
                    'danger_budget_ratio': 0.03,
                },
            )
            final_winner = p1_only.fidelity.CandidateSpec(
                arm_name='final_winner',
                scheduler_profile=protocol_candidate.scheduler_profile,
                curriculum_profile=protocol_candidate.curriculum_profile,
                weight_profile=protocol_candidate.weight_profile,
                window_profile=protocol_candidate.window_profile,
                cfg_overrides={},
                meta={
                    'protocol_arm': protocol_arm,
                    'aux_family': 'drop_danger',
                },
            )

            with (
                patch.object(p1_only.fidelity, 'acquire_run_lock', return_value=lock_path),
                patch.object(p1_only.fidelity, 'release_run_lock'),
                patch.object(p1_only.fidelity, 'update_results_doc'),
                patch('builtins.print'),
                patch.object(p1_only.ab, 'build_base_config', return_value={'control': {'version': 4}}),
                patch.object(p1_only.ab, 'load_all_files', return_value=['dummy.json.gz']),
                patch.object(p1_only.ab, 'group_files_by_month', return_value={'202501': ['dummy.json.gz']}),
                patch.object(
                    p1_only.ab,
                    'build_eval_splits',
                    return_value={
                        'monitor_recent_files': ['monitor.json.gz'],
                        'full_recent_files': ['recent.json.gz'],
                        'old_regression_files': ['old.json.gz'],
                    },
                ),
                patch.object(p1_only.fidelity, 'rank_weight_mean_for_files', return_value=0.05),
                patch.object(
                    p1_only.fidelity,
                    'build_p1_calibration_candidates',
                    return_value=['calibration_candidate'],
                ),
                patch.object(
                    p1_only.fidelity,
                    'build_p1_protocol_decide_candidates',
                    return_value=['protocol_candidate'],
                ),
                patch.object(
                    p1_only.fidelity,
                    'build_p1_winner_refine_candidates',
                    return_value=['winner_refine_candidate'],
                ),
                patch.object(
                    p1_only.fidelity,
                    'build_p1_ablation_candidates',
                    return_value=['ablation_candidate'],
                ),
                patch.object(
                    p1_only.fidelity,
                    'execute_round_multiseed',
                    side_effect=[
                        {'round_name': 'p1_calibration', 'ranking': []},
                        {
                            'round_name': 'p1_winner_refine_round',
                            'ranking': [
                                {
                                    'arm_name': refine_winner.arm_name,
                                    'candidate_meta': refine_winner.meta,
                                    'scheduler_profile': refine_winner.scheduler_profile,
                                    'curriculum_profile': refine_winner.curriculum_profile,
                                    'weight_profile': refine_winner.weight_profile,
                                    'window_profile': refine_winner.window_profile,
                                    'cfg_overrides': refine_winner.cfg_overrides,
                                    'valid': True,
                                }
                            ],
                        },
                        {
                            'round_name': 'p1_ablation_round',
                            'ranking': [
                                {
                                    'arm_name': final_winner.arm_name,
                                    'candidate_meta': final_winner.meta,
                                    'scheduler_profile': final_winner.scheduler_profile,
                                    'curriculum_profile': final_winner.curriculum_profile,
                                    'weight_profile': final_winner.weight_profile,
                                    'window_profile': final_winner.window_profile,
                                    'cfg_overrides': final_winner.cfg_overrides,
                                    'valid': True,
                                }
                            ],
                        },
                    ],
                ),
                patch.object(p1_only.fidelity, 'derive_p1_budget_calibration', return_value=dict(calibration)),
                patch.object(
                    p1_only.fidelity,
                    'execute_round_progressive_multiseed',
                    return_value={
                        'round_name': 'p1_protocol_decide_round',
                        'ranking': [
                            {
                                'arm_name': refine_winner.arm_name,
                                'candidate_meta': refine_winner.meta,
                                'scheduler_profile': refine_winner.scheduler_profile,
                                'curriculum_profile': refine_winner.curriculum_profile,
                                'weight_profile': refine_winner.weight_profile,
                                'window_profile': refine_winner.window_profile,
                                'cfg_overrides': refine_winner.cfg_overrides,
                                'valid': True,
                            }
                        ],
                    },
                ),
                patch.object(
                    p1_only.fidelity,
                    'winner_refine_center_selection_from_search_space',
                    return_value={'keep': None, 'explicit_arm_names': (refine_winner.arm_name,)},
                ),
                patch.object(p1_only, 'build_protocol_compare', return_value=protocol_compare),
            ):
                state = p1_only.run_p1_only(
                    run_dir=run_dir,
                    seed=123,
                    protocol_arms=[protocol_arm],
                    continue_to_ablation=True,
                )

        self.assertEqual('completed', state['status'])
        self.assertEqual('refine_winner', state['final_conclusion']['p1_refine_front_runner'])
        self.assertEqual('final_winner', state['final_conclusion']['p1_winner'])
        self.assertEqual('ablation_backlog', state['final_conclusion']['p1_winner_source'])

    def test_run_p1_only_continue_mode_reuses_existing_run_configuration(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / 'p1_only_run'
            run_dir.mkdir(parents=True, exist_ok=True)
            lock_path = run_dir / 'run.lock.json'
            resumed_protocol_arm = next(
                arm for arm in sorted(p1_only.PROTOCOL_ARM_MAP) if arm not in p1_only.FROZEN_TOP3
            )
            resumed_calibration_arm = next(
                arm
                for arm in sorted(p1_only.PROTOCOL_ARM_MAP)
                if arm not in set(p1_only.FROZEN_TOP3) | {resumed_protocol_arm}
            )
            resumed_seed = 777
            resumed_protocol_candidate = p1_only.build_protocol_candidate(resumed_protocol_arm)
            refine_winner = p1_only.fidelity.CandidateSpec(
                arm_name='resume_refine_winner',
                scheduler_profile=resumed_protocol_candidate.scheduler_profile,
                curriculum_profile=resumed_protocol_candidate.curriculum_profile,
                weight_profile=resumed_protocol_candidate.weight_profile,
                window_profile=resumed_protocol_candidate.window_profile,
                cfg_overrides={},
                meta={
                    'protocol_arm': resumed_protocol_arm,
                    'aux_family': 'all_three',
                    'rank_budget_ratio': 0.05,
                    'opp_budget_ratio': 0.04,
                    'danger_budget_ratio': 0.03,
                },
            )
            calibration = {
                'budget_ratios': [0.0, 0.25],
                'calibration_mode': 'full',
                'mapping_mode': 'hybrid_loss_grad_geomean',
                'combo_scheme': 'single_head_mapping_plus_pairwise_triple_combo',
                'opp_weight_per_budget_unit': 0.064,
                'danger_weight_per_budget_unit': 0.144,
                'triple_combo_factor': 0.84,
                'joint_combo_factor': 0.884,
                'protocol_triple_combo_factors': {resumed_protocol_arm: 0.85},
                'protocol_joint_combo_factors': {resumed_protocol_arm: 0.91},
            }
            existing_state = {
                'started_at': '2026-03-29 10:00:00',
                'mode': 'p1_only',
                'selected_protocol_arms': [resumed_protocol_arm],
                'p1': {
                    'protocol_arms': [resumed_protocol_arm],
                    'calibration_mode': 'full',
                    'calibration_protocol_arms': [resumed_calibration_arm],
                    'calibration': dict(calibration),
                    'search_space': {
                        'budget_ratios': calibration['budget_ratios'],
                        'calibration_mode': 'full',
                        'calibration_protocol_arms': [resumed_calibration_arm],
                    },
                    'calibration_round': {
                        'seed': resumed_seed + 404,
                    },
                    'protocol_decide_round': {
                        'round_name': 'p1_protocol_decide_round',
                        'ranking': [
                            {
                                'arm_name': refine_winner.arm_name,
                                'candidate_meta': refine_winner.meta,
                                'scheduler_profile': refine_winner.scheduler_profile,
                                'curriculum_profile': refine_winner.curriculum_profile,
                                'weight_profile': refine_winner.weight_profile,
                                'window_profile': refine_winner.window_profile,
                                'cfg_overrides': refine_winner.cfg_overrides,
                                'valid': True,
                            }
                        ],
                        'seed': resumed_seed + 505,
                    },
                    'protocol_compare': {
                        'round_name': 'p1_protocol_compare',
                        'ranking': [
                            {
                                'arm_name': 'resume_protocol_compare',
                                'candidate_meta': {'protocol_arm': resumed_protocol_arm},
                            }
                        ],
                    },
                    'selected_protocol_arm': resumed_protocol_arm,
                    'winner_refine_centers': [
                        refine_winner.arm_name,
                    ],
                },
                'final_conclusion': {
                    'p1_entry_protocols': [resumed_protocol_arm],
                    'p1_protocol_winner': resumed_protocol_arm,
                },
            }
            (run_dir / 'state.json').write_text(json.dumps(existing_state, ensure_ascii=False), encoding='utf-8')

            with (
                patch.object(p1_only.fidelity, 'acquire_run_lock', return_value=lock_path),
                patch.object(p1_only.fidelity, 'release_run_lock'),
                patch.object(p1_only.fidelity, 'update_results_doc'),
                patch('builtins.print'),
                patch.object(p1_only.ab, 'build_base_config', return_value={'control': {'version': 4}}),
                patch.object(p1_only.ab, 'load_all_files', return_value=['dummy.json.gz']),
                patch.object(p1_only.ab, 'group_files_by_month', return_value={'202501': ['dummy.json.gz']}),
                patch.object(
                    p1_only.ab,
                    'build_eval_splits',
                    return_value={
                        'monitor_recent_files': ['monitor.json.gz'],
                        'full_recent_files': ['recent.json.gz'],
                        'old_regression_files': ['old.json.gz'],
                    },
                ) as build_eval_splits,
                patch.object(p1_only.fidelity, 'rank_weight_mean_for_files', return_value=0.05),
                patch.object(
                    p1_only.fidelity,
                    'build_p1_calibration_candidates',
                    return_value=['calibration_candidate'],
                ) as build_p1_calibration_candidates,
                patch.object(
                    p1_only.fidelity,
                    'execute_round_multiseed',
                    return_value={
                        'round_name': 'p1_winner_refine_round',
                        'ranking': [
                            {
                                'arm_name': refine_winner.arm_name,
                                'candidate_meta': refine_winner.meta,
                                'scheduler_profile': refine_winner.scheduler_profile,
                                'curriculum_profile': refine_winner.curriculum_profile,
                                'weight_profile': refine_winner.weight_profile,
                                'window_profile': refine_winner.window_profile,
                                'cfg_overrides': refine_winner.cfg_overrides,
                                'valid': True,
                            }
                        ],
                        'seed': resumed_seed + 606,
                    },
                ) as execute_round_multiseed,
                patch.object(p1_only.fidelity, 'derive_p1_budget_calibration', return_value=dict(calibration)) as derive_calibration,
                patch.object(
                    p1_only.fidelity,
                    'build_p1_protocol_decide_candidates',
                    return_value=['protocol_candidate'],
                ) as build_p1_protocol_decide_candidates,
                patch.object(
                    p1_only.fidelity,
                    'execute_round_progressive_multiseed',
                    return_value={'round_name': 'p1_protocol_decide_round', 'ranking': [], 'seed': resumed_seed + 505},
                ) as execute_round_progressive_multiseed,
                patch.object(
                    p1_only,
                    'build_protocol_compare',
                    return_value=[
                        {
                            'arm_name': 'resume_protocol_compare',
                            'candidate_meta': {'protocol_arm': resumed_protocol_arm},
                        }
                    ],
                ) as build_protocol_compare,
                patch.object(
                    p1_only.fidelity,
                    'winner_refine_center_selection_from_search_space',
                    return_value={'keep': None, 'explicit_arm_names': (refine_winner.arm_name,)},
                ),
                patch.object(p1_only, 'select_protocol_centers', return_value=[refine_winner]) as select_protocol_centers,
            ):
                state = p1_only.run_p1_only(
                    run_dir=run_dir,
                    continue_to_winner_refine=True,
                )

        self.assertEqual('stopped_after_p1_winner_refine', state['status'])
        self.assertEqual('2026-03-29 10:00:00', state['started_at'])
        self.assertEqual(resumed_seed, state['seed'])
        self.assertEqual([resumed_protocol_arm], state['selected_protocol_arms'])
        self.assertEqual([resumed_protocol_arm], state['p1']['protocol_arms'])
        self.assertEqual('full', state['p1']['calibration_mode'])
        self.assertEqual([resumed_calibration_arm], state['p1']['calibration_protocol_arms'])
        self.assertEqual('full', state['p1']['search_space']['calibration_mode'])
        self.assertEqual([resumed_calibration_arm], state['p1']['search_space']['calibration_protocol_arms'])
        self.assertEqual('resume_refine_winner', state['final_conclusion']['p1_refine_front_runner'])
        self.assertEqual('resume_refine_winner', state['final_conclusion']['p1_winner'])
        self.assertEqual('winner_refine_mainline', state['final_conclusion']['p1_winner_source'])
        build_eval_splits.assert_called_once_with(
            {'202501': ['dummy.json.gz']},
            resumed_seed + 55,
            p1_only.ab.BASE_SCREENING['eval_files'],
        )
        build_p1_calibration_candidates.assert_not_called()
        derive_calibration.assert_not_called()
        build_p1_protocol_decide_candidates.assert_not_called()
        execute_round_progressive_multiseed.assert_not_called()
        build_protocol_compare.assert_not_called()
        select_protocol_centers.assert_called_once()
        execute_round_multiseed.assert_called_once()

    def test_run_p1_only_continue_to_winner_refine_drops_legacy_budget_centers_and_uses_current_defaults(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / 'p1_only_run'
            run_dir.mkdir(parents=True, exist_ok=True)
            lock_path = run_dir / 'run.lock.json'
            resumed_seed = 4321
            protocol_arm = p1_only.FROZEN_TOP3[0]
            protocol_candidate = p1_only.build_protocol_candidate(protocol_arm)
            calibration = {
                'budget_ratios': [0.0, 0.25],
                'calibration_mode': 'combo_only',
                'mapping_mode': 'hybrid_loss_grad_geomean',
                'combo_scheme': 'single_head_mapping_plus_pairwise_triple_combo',
                'opp_weight_per_budget_unit': 0.064,
                'danger_weight_per_budget_unit': 0.144,
                'triple_combo_factor': 0.84,
                'joint_combo_factor': 0.884,
                'protocol_triple_combo_factors': {protocol_arm: 0.85},
                'protocol_joint_combo_factors': {protocol_arm: 0.91},
            }
            persisted_centers = ['legacy_center_a', 'legacy_center_b']
            refine_winner = p1_only.fidelity.CandidateSpec(
                arm_name='resume_refine_winner',
                scheduler_profile=protocol_candidate.scheduler_profile,
                curriculum_profile=protocol_candidate.curriculum_profile,
                weight_profile=protocol_candidate.weight_profile,
                window_profile=protocol_candidate.window_profile,
                cfg_overrides={},
                meta={
                    'protocol_arm': protocol_arm,
                    'aux_family': 'all_three',
                    'rank_budget_ratio': 0.05,
                    'opp_budget_ratio': 0.04,
                    'danger_budget_ratio': 0.03,
                },
            )
            existing_state = {
                'started_at': '2026-03-29 10:00:00',
                'seed': resumed_seed,
                'mode': 'p1_only',
                'status': 'stopped_after_p1_protocol_decide',
                'selected_protocol_arms': [protocol_arm],
                'p1': {
                    'protocol_arms': [protocol_arm],
                    'calibration_mode': 'combo_only',
                    'calibration_protocol_arms': [protocol_arm],
                    'calibration': dict(calibration),
                    'search_space': {
                        'budget_ratios': calibration['budget_ratios'],
                        'calibration_mode': 'combo_only',
                        'calibration_protocol_arms': [protocol_arm],
                    },
                    'calibration_round': {'seed': resumed_seed + 404},
                    'protocol_decide_round': {
                        'round_name': 'p1_protocol_decide_round',
                        'ranking': [],
                        'seed': resumed_seed + 505,
                    },
                    'protocol_compare': {
                        'round_name': 'p1_protocol_compare',
                        'ranking': [
                            {
                                'arm_name': protocol_arm,
                                'candidate_meta': {'protocol_arm': protocol_arm},
                            }
                        ],
                    },
                    'selected_protocol_arm': protocol_arm,
                    'winner_refine_centers': list(persisted_centers),
                },
                'final_conclusion': {
                    'p1_entry_protocols': [protocol_arm],
                    'p1_protocol_winner': protocol_arm,
                },
            }
            (run_dir / 'state.json').write_text(json.dumps(existing_state, ensure_ascii=False), encoding='utf-8')

            with (
                patch.object(p1_only.fidelity, 'acquire_run_lock', return_value=lock_path),
                patch.object(p1_only.fidelity, 'release_run_lock'),
                patch.object(p1_only.fidelity, 'update_results_doc'),
                patch('builtins.print'),
                patch.object(p1_only.ab, 'build_base_config', return_value={'control': {'version': 4}}),
                patch.object(p1_only.ab, 'load_all_files', return_value=['dummy.json.gz']),
                patch.object(p1_only.ab, 'group_files_by_month', return_value={'202501': ['dummy.json.gz']}),
                patch.object(
                    p1_only.ab,
                    'build_eval_splits',
                    return_value={
                        'monitor_recent_files': ['monitor.json.gz'],
                        'full_recent_files': ['recent.json.gz'],
                        'old_regression_files': ['old.json.gz'],
                    },
                ),
                patch.object(p1_only.fidelity, 'rank_weight_mean_for_files', return_value=0.05),
                patch.object(p1_only.fidelity, 'build_p1_calibration_candidates') as build_p1_calibration_candidates,
                patch.object(p1_only.fidelity, 'derive_p1_budget_calibration') as derive_calibration,
                patch.object(p1_only.fidelity, 'build_p1_protocol_decide_candidates') as build_p1_protocol_decide_candidates,
                patch.object(p1_only.fidelity, 'execute_round_progressive_multiseed') as execute_round_progressive_multiseed,
                patch.object(
                    p1_only.fidelity,
                    'build_p1_winner_refine_candidates',
                    return_value=['winner_refine_candidate'],
                ) as build_p1_winner_refine_candidates,
                patch.object(
                    p1_only.fidelity,
                    'execute_round_multiseed',
                    return_value={
                        'round_name': 'p1_winner_refine_round',
                        'ranking': [
                            {
                                'arm_name': refine_winner.arm_name,
                                'candidate_meta': refine_winner.meta,
                                'scheduler_profile': refine_winner.scheduler_profile,
                                'curriculum_profile': refine_winner.curriculum_profile,
                                'weight_profile': refine_winner.weight_profile,
                                'window_profile': refine_winner.window_profile,
                                'cfg_overrides': refine_winner.cfg_overrides,
                                'valid': True,
                            }
                        ],
                        'seed': resumed_seed + 606,
                    },
                ) as execute_round_multiseed,
                patch.object(
                    p1_only.fidelity,
                    'is_budget_triplet_arm_name',
                    side_effect=lambda arm_name: str(arm_name).startswith('legacy_center'),
                ),
                patch.object(p1_only, 'select_protocol_centers', return_value=[refine_winner]) as select_protocol_centers,
            ):
                state = p1_only.run_p1_only(
                    run_dir=run_dir,
                    continue_to_winner_refine=True,
                )

        self.assertEqual('stopped_after_p1_winner_refine', state['status'])
        self.assertEqual('top_ranked_keep', state['p1']['search_space']['winner_refine_center_mode'])
        self.assertEqual(p1_only.fidelity.P1_WINNER_REFINE_CENTER_KEEP, state['p1']['search_space']['winner_refine_center_keep'])
        self.assertEqual(4, state['p1']['search_space']['budget_ratio_digits'])
        self.assertEqual(5, state['p1']['search_space']['aux_weight_digits'])
        self.assertEqual(refine_winner.arm_name, state['final_conclusion']['p1_winner'])
        self.assertEqual('winner_refine_mainline', state['final_conclusion']['p1_winner_source'])
        self.assertEqual(
            (),
            select_protocol_centers.call_args.kwargs['explicit_arm_names'],
        )
        self.assertEqual(
            p1_only.fidelity.P1_WINNER_REFINE_CENTER_KEEP,
            select_protocol_centers.call_args.kwargs['keep'],
        )
        build_p1_calibration_candidates.assert_not_called()
        derive_calibration.assert_not_called()
        build_p1_protocol_decide_candidates.assert_not_called()
        execute_round_progressive_multiseed.assert_not_called()
        build_p1_winner_refine_candidates.assert_called_once()
        self.assertNotIn(
            'winner_refine_center_arm_names',
            build_p1_winner_refine_candidates.call_args.kwargs['search_space'],
        )
        execute_round_multiseed.assert_called_once()

    def test_run_p1_only_default_flow_resumes_from_stopped_calibration(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / 'p1_only_run'
            run_dir.mkdir(parents=True, exist_ok=True)
            lock_path = run_dir / 'run.lock.json'
            protocol_arm = p1_only.FROZEN_TOP3[0]
            protocol_candidate = p1_only.build_protocol_candidate(protocol_arm)
            refine_winner = p1_only.fidelity.CandidateSpec(
                arm_name='resume_protocol_winner',
                scheduler_profile=protocol_candidate.scheduler_profile,
                curriculum_profile=protocol_candidate.curriculum_profile,
                weight_profile=protocol_candidate.weight_profile,
                window_profile=protocol_candidate.window_profile,
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
                'calibration_mode': 'combo_only',
                'mapping_mode': 'hybrid_loss_grad_geomean',
                'combo_scheme': 'single_head_mapping_plus_pairwise_triple_combo',
                'opp_weight_per_budget_unit': 0.064,
                'danger_weight_per_budget_unit': 0.144,
                'triple_combo_factor': 0.84,
                'joint_combo_factor': 0.884,
                'protocol_triple_combo_factors': {protocol_arm: 0.85},
                'protocol_joint_combo_factors': {protocol_arm: 0.91},
                'calibration_protocol_arms': [protocol_arm],
            }
            existing_state = {
                'started_at': '2026-03-29 10:00:00',
                'seed': 123,
                'mode': 'p1_only',
                'status': 'stopped_after_p1_calibration',
                'selected_protocol_arms': [protocol_arm],
                'p1': {
                    'protocol_arms': [protocol_arm],
                    'calibration_mode': 'combo_only',
                    'calibration_protocol_arms': [protocol_arm],
                    'calibration': dict(calibration),
                    'search_space': {
                        'budget_ratios': calibration['budget_ratios'],
                        'calibration_mode': 'combo_only',
                        'calibration_protocol_arms': [protocol_arm],
                    },
                    'calibration_round': {'seed': 123 + 404, 'ranking': []},
                },
                'final_conclusion': {
                    'p1_entry_protocols': [protocol_arm],
                },
            }
            (run_dir / 'state.json').write_text(json.dumps(existing_state, ensure_ascii=False), encoding='utf-8')

            with (
                patch.object(p1_only.fidelity, 'acquire_run_lock', return_value=lock_path),
                patch.object(p1_only.fidelity, 'release_run_lock'),
                patch.object(p1_only.fidelity, 'update_results_doc'),
                patch('builtins.print'),
                patch.object(p1_only.ab, 'build_base_config', return_value={'control': {'version': 4}}),
                patch.object(p1_only.ab, 'load_all_files', return_value=['dummy.json.gz']),
                patch.object(p1_only.ab, 'group_files_by_month', return_value={'202501': ['dummy.json.gz']}),
                patch.object(
                    p1_only.ab,
                    'build_eval_splits',
                    return_value={
                        'monitor_recent_files': ['monitor.json.gz'],
                        'full_recent_files': ['recent.json.gz'],
                        'old_regression_files': ['old.json.gz'],
                    },
                ),
                patch.object(p1_only.fidelity, 'rank_weight_mean_for_files', return_value=0.05),
                patch.object(
                    p1_only.fidelity,
                    'build_p1_calibration_candidates',
                    return_value=['calibration_candidate'],
                ) as build_p1_calibration_candidates,
                patch.object(
                    p1_only.fidelity,
                    'execute_round_multiseed',
                    return_value={'round_name': 'p1_calibration', 'ranking': []},
                ) as execute_round_multiseed,
                patch.object(p1_only.fidelity, 'derive_p1_budget_calibration', return_value=dict(calibration)) as derive_calibration,
                patch.object(
                    p1_only.fidelity,
                    'build_p1_protocol_decide_candidates',
                    return_value=['protocol_candidate'],
                ) as build_p1_protocol_decide_candidates,
                patch.object(
                    p1_only.fidelity,
                    'execute_round_progressive_multiseed',
                    return_value={
                        'round_name': 'p1_protocol_decide_round',
                        'ranking': [
                            {
                                'arm_name': refine_winner.arm_name,
                                'candidate_meta': refine_winner.meta,
                                'scheduler_profile': refine_winner.scheduler_profile,
                                'curriculum_profile': refine_winner.curriculum_profile,
                                'weight_profile': refine_winner.weight_profile,
                                'window_profile': refine_winner.window_profile,
                                'cfg_overrides': refine_winner.cfg_overrides,
                                'valid': True,
                            }
                        ],
                    },
                ) as execute_round_progressive_multiseed,
                patch.object(
                    p1_only,
                    'build_protocol_compare',
                    return_value=[
                        {
                            'arm_name': refine_winner.arm_name,
                            'candidate_meta': {'protocol_arm': protocol_arm},
                        }
                    ],
                ) as build_protocol_compare,
            ):
                state = p1_only.run_p1_only(
                    run_dir=run_dir,
                )

        self.assertEqual('stopped_after_p1_protocol_decide', state['status'])
        self.assertEqual(protocol_arm, state['p1']['selected_protocol_arm'])
        self.assertEqual(protocol_arm, state['final_conclusion']['p1_protocol_winner'])
        build_p1_calibration_candidates.assert_not_called()
        derive_calibration.assert_not_called()
        execute_round_multiseed.assert_not_called()
        build_p1_protocol_decide_candidates.assert_called_once()
        execute_round_progressive_multiseed.assert_called_once()
        build_protocol_compare.assert_called_once()

    def test_run_p1_only_continue_to_ablation_reuses_existing_winner_refine(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / 'p1_only_run'
            run_dir.mkdir(parents=True, exist_ok=True)
            lock_path = run_dir / 'run.lock.json'
            protocol_arm = p1_only.FROZEN_TOP3[0]
            protocol_candidate = p1_only.build_protocol_candidate(protocol_arm)
            calibration = {
                'budget_ratios': [0.0, 0.25],
                'calibration_mode': 'combo_only',
                'mapping_mode': 'hybrid_loss_grad_geomean',
                'combo_scheme': 'single_head_mapping_plus_pairwise_triple_combo',
                'opp_weight_per_budget_unit': 0.064,
                'danger_weight_per_budget_unit': 0.144,
                'triple_combo_factor': 0.84,
                'joint_combo_factor': 0.884,
                'protocol_triple_combo_factors': {protocol_arm: 0.85},
                'protocol_joint_combo_factors': {protocol_arm: 0.91},
            }
            refine_winner = p1_only.fidelity.CandidateSpec(
                arm_name='resume_refine_winner',
                scheduler_profile=protocol_candidate.scheduler_profile,
                curriculum_profile=protocol_candidate.curriculum_profile,
                weight_profile=protocol_candidate.weight_profile,
                window_profile=protocol_candidate.window_profile,
                cfg_overrides={},
                meta={
                    'protocol_arm': protocol_arm,
                    'aux_family': 'all_three',
                    'rank_budget_ratio': 0.05,
                    'opp_budget_ratio': 0.04,
                    'danger_budget_ratio': 0.03,
                },
            )
            final_winner = p1_only.fidelity.CandidateSpec(
                arm_name='resume_final_winner',
                scheduler_profile=protocol_candidate.scheduler_profile,
                curriculum_profile=protocol_candidate.curriculum_profile,
                weight_profile=protocol_candidate.weight_profile,
                window_profile=protocol_candidate.window_profile,
                cfg_overrides={},
                meta={
                    'protocol_arm': protocol_arm,
                    'aux_family': 'drop_danger',
                },
            )
            existing_state = {
                'started_at': '2026-03-29 10:00:00',
                'seed': 123,
                'mode': 'p1_only',
                'status': 'stopped_after_p1_winner_refine',
                'selected_protocol_arms': [protocol_arm],
                'p1': {
                    'protocol_arms': [protocol_arm],
                    'calibration_mode': 'combo_only',
                    'calibration_protocol_arms': [protocol_arm],
                    'calibration': dict(calibration),
                    'search_space': {
                        'budget_ratios': calibration['budget_ratios'],
                        'calibration_mode': 'combo_only',
                        'calibration_protocol_arms': [protocol_arm],
                    },
                    'calibration_round': {'seed': 123 + 404, 'ranking': []},
                    'protocol_decide_round': {
                        'round_name': 'p1_protocol_decide_round',
                        'ranking': [
                            {
                                'arm_name': refine_winner.arm_name,
                                'candidate_meta': refine_winner.meta,
                                'scheduler_profile': refine_winner.scheduler_profile,
                                'curriculum_profile': refine_winner.curriculum_profile,
                                'weight_profile': refine_winner.weight_profile,
                                'window_profile': refine_winner.window_profile,
                                'cfg_overrides': refine_winner.cfg_overrides,
                                'valid': True,
                            }
                        ],
                    },
                    'protocol_compare': {
                        'round_name': 'p1_protocol_compare',
                        'ranking': [
                            {
                                'arm_name': protocol_arm,
                                'candidate_meta': {'protocol_arm': protocol_arm},
                            }
                        ],
                    },
                    'selected_protocol_arm': protocol_arm,
                    'winner_refine_centers': [refine_winner.arm_name],
                    'winner_refine_round': {
                        'round_name': 'p1_winner_refine_round',
                        'ranking': [
                            {
                                'arm_name': refine_winner.arm_name,
                                'candidate_meta': refine_winner.meta,
                                'scheduler_profile': refine_winner.scheduler_profile,
                                'curriculum_profile': refine_winner.curriculum_profile,
                                'weight_profile': refine_winner.weight_profile,
                                'window_profile': refine_winner.window_profile,
                                'cfg_overrides': refine_winner.cfg_overrides,
                                'valid': True,
                            }
                        ],
                    },
                    'winner_refine_front_runner': refine_winner.arm_name,
                },
                'final_conclusion': {
                    'p1_entry_protocols': [protocol_arm],
                    'p1_protocol_winner': protocol_arm,
                    'p1_refine_front_runner': refine_winner.arm_name,
                },
            }
            (run_dir / 'state.json').write_text(json.dumps(existing_state, ensure_ascii=False), encoding='utf-8')

            with (
                patch.object(p1_only.fidelity, 'acquire_run_lock', return_value=lock_path),
                patch.object(p1_only.fidelity, 'release_run_lock'),
                patch.object(p1_only.fidelity, 'update_results_doc'),
                patch('builtins.print'),
                patch.object(p1_only.ab, 'build_base_config', return_value={'control': {'version': 4}}),
                patch.object(p1_only.ab, 'load_all_files', return_value=['dummy.json.gz']),
                patch.object(p1_only.ab, 'group_files_by_month', return_value={'202501': ['dummy.json.gz']}),
                patch.object(
                    p1_only.ab,
                    'build_eval_splits',
                    return_value={
                        'monitor_recent_files': ['monitor.json.gz'],
                        'full_recent_files': ['recent.json.gz'],
                        'old_regression_files': ['old.json.gz'],
                    },
                ),
                patch.object(p1_only.fidelity, 'rank_weight_mean_for_files', return_value=0.05),
                patch.object(
                    p1_only.fidelity,
                    'execute_round_progressive_multiseed',
                    return_value={'round_name': 'p1_protocol_decide_round', 'ranking': []},
                ) as execute_round_progressive_multiseed,
                patch.object(
                    p1_only.fidelity,
                    'execute_round_multiseed',
                    return_value={
                        'round_name': 'p1_ablation_round',
                        'ranking': [
                            {
                                'arm_name': final_winner.arm_name,
                                'candidate_meta': final_winner.meta,
                                'scheduler_profile': final_winner.scheduler_profile,
                                'curriculum_profile': final_winner.curriculum_profile,
                                'weight_profile': final_winner.weight_profile,
                                'window_profile': final_winner.window_profile,
                                'cfg_overrides': final_winner.cfg_overrides,
                                'valid': True,
                            }
                        ],
                    },
                ) as execute_round_multiseed,
                patch.object(
                    p1_only.fidelity,
                    'build_p1_ablation_candidates',
                    return_value=['ablation_candidate'],
                ) as build_p1_ablation_candidates,
                patch.object(p1_only, 'select_protocol_centers', return_value=[refine_winner]) as select_protocol_centers,
            ):
                state = p1_only.run_p1_only(
                    run_dir=run_dir,
                    continue_to_ablation=True,
                )

        self.assertEqual('completed', state['status'])
        self.assertEqual(refine_winner.arm_name, state['final_conclusion']['p1_refine_front_runner'])
        self.assertEqual(final_winner.arm_name, state['final_conclusion']['p1_winner'])
        self.assertEqual('ablation_backlog', state['final_conclusion']['p1_winner_source'])
        execute_round_progressive_multiseed.assert_not_called()
        select_protocol_centers.assert_not_called()
        build_p1_ablation_candidates.assert_called_once()
        execute_round_multiseed.assert_called_once()

    def test_run_p1_only_resume_completed_run_keeps_completed_status(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / 'p1_only_run'
            run_dir.mkdir(parents=True, exist_ok=True)
            lock_path = run_dir / 'run.lock.json'
            protocol_arm = p1_only.FROZEN_TOP3[0]
            protocol_candidate = p1_only.build_protocol_candidate(protocol_arm)
            refine_winner = p1_only.fidelity.CandidateSpec(
                arm_name='resume_refine_winner',
                scheduler_profile=protocol_candidate.scheduler_profile,
                curriculum_profile=protocol_candidate.curriculum_profile,
                weight_profile=protocol_candidate.weight_profile,
                window_profile=protocol_candidate.window_profile,
                cfg_overrides={},
                meta={
                    'protocol_arm': protocol_arm,
                    'aux_family': 'all_three',
                    'rank_budget_ratio': 0.05,
                    'opp_budget_ratio': 0.04,
                    'danger_budget_ratio': 0.03,
                },
            )
            final_winner = p1_only.fidelity.CandidateSpec(
                arm_name='resume_final_winner',
                scheduler_profile=protocol_candidate.scheduler_profile,
                curriculum_profile=protocol_candidate.curriculum_profile,
                weight_profile=protocol_candidate.weight_profile,
                window_profile=protocol_candidate.window_profile,
                cfg_overrides={},
                meta={
                    'protocol_arm': protocol_arm,
                    'aux_family': 'drop_danger',
                },
            )
            existing_state = {
                'started_at': '2026-03-29 10:00:00',
                'seed': 123,
                'mode': 'p1_only',
                'status': 'completed',
                'selected_protocol_arms': [protocol_arm],
                'p1': {
                    'protocol_arms': [protocol_arm],
                    'calibration_mode': 'combo_only',
                    'calibration_protocol_arms': [protocol_arm],
                    'calibration': {
                        'budget_ratios': [0.0, 0.25],
                        'calibration_mode': 'combo_only',
                        'mapping_mode': 'hybrid_loss_grad_geomean',
                        'opp_weight_per_budget_unit': 0.064,
                        'danger_weight_per_budget_unit': 0.144,
                    },
                    'search_space': {
                        'budget_ratios': [0.0, 0.25],
                        'calibration_mode': 'combo_only',
                        'calibration_protocol_arms': [protocol_arm],
                    },
                    'calibration_round': {'seed': 123 + 404, 'ranking': []},
                    'protocol_decide_round': {
                        'round_name': 'p1_protocol_decide_round',
                        'ranking': [],
                    },
                    'protocol_compare': {
                        'round_name': 'p1_protocol_compare',
                        'ranking': [
                            {
                                'arm_name': protocol_arm,
                                'candidate_meta': {'protocol_arm': protocol_arm},
                            }
                        ],
                    },
                    'selected_protocol_arm': protocol_arm,
                    'winner_refine_round': {
                        'round_name': 'p1_winner_refine_round',
                        'ranking': [
                            {
                                'arm_name': refine_winner.arm_name,
                                'candidate_meta': refine_winner.meta,
                                'scheduler_profile': refine_winner.scheduler_profile,
                                'curriculum_profile': refine_winner.curriculum_profile,
                                'weight_profile': refine_winner.weight_profile,
                                'window_profile': refine_winner.window_profile,
                                'cfg_overrides': refine_winner.cfg_overrides,
                                'valid': True,
                            }
                        ],
                    },
                    'winner_refine_front_runner': refine_winner.arm_name,
                    'ablation_round': {
                        'round_name': 'p1_ablation_round',
                        'ranking': [
                            {
                                'arm_name': final_winner.arm_name,
                                'candidate_meta': final_winner.meta,
                                'scheduler_profile': final_winner.scheduler_profile,
                                'curriculum_profile': final_winner.curriculum_profile,
                                'weight_profile': final_winner.weight_profile,
                                'window_profile': final_winner.window_profile,
                                'cfg_overrides': final_winner.cfg_overrides,
                                'valid': True,
                            }
                        ],
                    },
                    'final_compare': {
                        'round_name': 'p1_final_compare',
                        'ranking': [
                            {
                                'arm_name': final_winner.arm_name,
                                'candidate_meta': final_winner.meta,
                                'scheduler_profile': final_winner.scheduler_profile,
                                'curriculum_profile': final_winner.curriculum_profile,
                                'weight_profile': final_winner.weight_profile,
                                'window_profile': final_winner.window_profile,
                                'cfg_overrides': final_winner.cfg_overrides,
                                'valid': True,
                            }
                        ],
                    },
                    'winner': final_winner.arm_name,
                },
                'final_conclusion': {
                    'p1_entry_protocols': [protocol_arm],
                    'p1_protocol_winner': protocol_arm,
                    'p1_refine_front_runner': refine_winner.arm_name,
                    'p1_winner': final_winner.arm_name,
                },
            }
            state_path = run_dir / 'state.json'
            state_path.write_text(json.dumps(existing_state, ensure_ascii=False), encoding='utf-8')

            with (
                patch.object(p1_only.fidelity, 'acquire_run_lock', return_value=lock_path),
                patch.object(p1_only.fidelity, 'release_run_lock'),
                patch.object(p1_only.fidelity, 'update_results_doc'),
                patch('builtins.print'),
                patch.object(p1_only.ab, 'build_base_config', return_value={'control': {'version': 4}}),
                patch.object(p1_only.ab, 'load_all_files', return_value=['dummy.json.gz']),
                patch.object(p1_only.ab, 'group_files_by_month', return_value={'202501': ['dummy.json.gz']}),
                patch.object(
                    p1_only.ab,
                    'build_eval_splits',
                    return_value={
                        'monitor_recent_files': ['monitor.json.gz'],
                        'full_recent_files': ['recent.json.gz'],
                        'old_regression_files': ['old.json.gz'],
                    },
                ),
                patch.object(p1_only.fidelity, 'rank_weight_mean_for_files', return_value=0.05),
                patch.object(p1_only.fidelity, 'execute_round_progressive_multiseed') as execute_round_progressive_multiseed,
                patch.object(p1_only.fidelity, 'execute_round_multiseed') as execute_round_multiseed,
            ):
                state = p1_only.run_p1_only(run_dir=run_dir)

            persisted_state = json.loads(state_path.read_text(encoding='utf-8'))

        self.assertEqual('completed', state['status'])
        self.assertEqual('completed', persisted_state['status'])
        self.assertEqual(final_winner.arm_name, state['final_conclusion']['p1_winner'])
        execute_round_progressive_multiseed.assert_not_called()
        execute_round_multiseed.assert_not_called()

    def test_run_p1_only_retry_success_clears_stale_fatal_fields(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / 'p1_only_run'
            run_dir.mkdir(parents=True, exist_ok=True)
            lock_path = run_dir / 'run.lock.json'
            protocol_arm = p1_only.FROZEN_TOP3[0]
            existing_state = {
                'started_at': '2026-03-29 10:00:00',
                'seed': 123,
                'mode': 'p1_only',
                'status': 'failed',
                'fatal_error': 'old failure',
                'fatal_traceback': 'old traceback',
                'selected_protocol_arms': [protocol_arm],
                'p1': {
                    'protocol_arms': [protocol_arm],
                    'calibration_mode': 'combo_only',
                    'calibration_protocol_arms': [protocol_arm],
                    'calibration': {
                        'budget_ratios': [0.0, 0.25],
                        'calibration_mode': 'combo_only',
                        'mapping_mode': 'hybrid_loss_grad_geomean',
                        'opp_weight_per_budget_unit': 0.064,
                        'danger_weight_per_budget_unit': 0.144,
                    },
                    'search_space': {
                        'budget_ratios': [0.0, 0.25],
                        'calibration_mode': 'combo_only',
                        'calibration_protocol_arms': [protocol_arm],
                    },
                    'calibration_round': {'seed': 123 + 404, 'ranking': []},
                    'protocol_decide_round': {
                        'round_name': 'p1_protocol_decide_round',
                        'ranking': [],
                    },
                    'protocol_compare': {
                        'round_name': 'p1_protocol_compare',
                        'ranking': [
                            {
                                'arm_name': protocol_arm,
                                'candidate_meta': {'protocol_arm': protocol_arm},
                            }
                        ],
                    },
                    'selected_protocol_arm': protocol_arm,
                },
                'final_conclusion': {
                    'p1_entry_protocols': [protocol_arm],
                    'p1_protocol_winner': protocol_arm,
                },
            }
            state_path = run_dir / 'state.json'
            state_path.write_text(json.dumps(existing_state, ensure_ascii=False), encoding='utf-8')

            with (
                patch.object(p1_only.fidelity, 'acquire_run_lock', return_value=lock_path),
                patch.object(p1_only.fidelity, 'release_run_lock'),
                patch.object(p1_only.fidelity, 'update_results_doc'),
                patch('builtins.print'),
                patch.object(p1_only.ab, 'build_base_config', return_value={'control': {'version': 4}}),
                patch.object(p1_only.ab, 'load_all_files', return_value=['dummy.json.gz']),
                patch.object(p1_only.ab, 'group_files_by_month', return_value={'202501': ['dummy.json.gz']}),
                patch.object(
                    p1_only.ab,
                    'build_eval_splits',
                    return_value={
                        'monitor_recent_files': ['monitor.json.gz'],
                        'full_recent_files': ['recent.json.gz'],
                        'old_regression_files': ['old.json.gz'],
                    },
                ),
                patch.object(p1_only.fidelity, 'rank_weight_mean_for_files', return_value=0.05),
                patch.object(p1_only.fidelity, 'execute_round_progressive_multiseed') as execute_round_progressive_multiseed,
                patch.object(p1_only.fidelity, 'execute_round_multiseed') as execute_round_multiseed,
            ):
                state = p1_only.run_p1_only(run_dir=run_dir)

            persisted_state = json.loads(state_path.read_text(encoding='utf-8'))

        self.assertEqual('stopped_after_p1_protocol_decide', state['status'])
        self.assertNotIn('fatal_error', state)
        self.assertNotIn('fatal_traceback', state)
        self.assertNotIn('fatal_error', persisted_state)
        self.assertNotIn('fatal_traceback', persisted_state)
        execute_round_progressive_multiseed.assert_not_called()
        execute_round_multiseed.assert_not_called()

    def test_run_p1_only_continue_mode_rejects_conflicting_protocol_override(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / 'p1_only_run'
            run_dir.mkdir(parents=True, exist_ok=True)
            lock_path = run_dir / 'run.lock.json'
            resumed_protocol_arm = next(
                arm for arm in sorted(p1_only.PROTOCOL_ARM_MAP) if arm not in p1_only.FROZEN_TOP3
            )
            conflicting_protocol_arm = next(
                arm
                for arm in sorted(p1_only.PROTOCOL_ARM_MAP)
                if arm not in set(p1_only.FROZEN_TOP3) | {resumed_protocol_arm}
            )
            existing_state = {
                'started_at': '2026-03-29 10:00:00',
                'mode': 'p1_only',
                'selected_protocol_arms': [resumed_protocol_arm],
                'p1': {
                    'protocol_arms': [resumed_protocol_arm],
                    'calibration_mode': 'combo_only',
                    'calibration_protocol_arms': [resumed_protocol_arm],
                    'calibration_round': {
                        'seed': 404 + 123,
                    },
                },
                'final_conclusion': {
                    'p1_entry_protocols': [resumed_protocol_arm],
                },
            }
            state_path = run_dir / 'state.json'
            state_path.write_text(json.dumps(existing_state, ensure_ascii=False), encoding='utf-8')

            with (
                patch.object(p1_only.fidelity, 'acquire_run_lock', return_value=lock_path),
                patch.object(p1_only.fidelity, 'release_run_lock'),
            ):
                with self.assertRaisesRegex(ValueError, 'continue mode protocol_arms mismatch'):
                    p1_only.run_p1_only(
                        run_dir=run_dir,
                        protocol_arms=[conflicting_protocol_arm],
                        continue_to_winner_refine=True,
                    )

            restored_state = json.loads(state_path.read_text(encoding='utf-8'))
            self.assertEqual([resumed_protocol_arm], restored_state['selected_protocol_arms'])


if __name__ == '__main__':
    unittest.main()
