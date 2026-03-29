import tempfile
import unittest
from unittest import mock
from pathlib import Path

import run_stage05_fidelity as fidelity


def make_ranked_entry(
    arm_name: str,
    *,
    protocol_arm: str,
    aux_family: str,
    recent_policy_loss: float,
    old_regression_policy_loss: float = 0.3,
    action_quality_score: float = 0.5,
    valid: bool = True,
) -> dict[str, object]:
    return {
        'arm_name': arm_name,
        'candidate_meta': {
            'protocol_arm': protocol_arm,
            'aux_family': aux_family,
        },
        'valid': valid,
        'full_recent_loss': recent_policy_loss,
        'full_recent_metrics': {
            'action_quality_score': action_quality_score,
            'policy_loss': recent_policy_loss,
        },
        'recent_policy_loss': recent_policy_loss,
        'old_regression_policy_loss': old_regression_policy_loss,
        'old_regression_metrics': {
            'policy_loss': old_regression_policy_loss,
        },
    }


def make_protocol(protocol_arm: str) -> fidelity.CandidateSpec:
    return fidelity.CandidateSpec(
        arm_name=f'{protocol_arm}_arm',
        scheduler_profile='sched',
        curriculum_profile='curr',
        weight_profile='weight',
        window_profile='window',
        cfg_overrides={},
        meta={'protocol_arm': protocol_arm},
    )


def make_calibration_entry(
    role: str,
    *,
    protocol_arm: str,
    rank_effective: float = 0.0,
    opp_effective: float = 0.0,
    danger_effective: float = 0.0,
    calibration_mode: str = fidelity.P1_CALIBRATION_MODE_COMBO_ONLY,
) -> dict[str, object]:
    rank_aux_weight_mean = 1.0 if rank_effective > 0 else 0.0
    rank_aux_raw_loss = rank_effective if rank_effective > 0 else 0.0
    return {
        'arm_name': f'{protocol_arm}_{role}',
        'valid': True,
        'candidate_meta': {
            'protocol_arm': protocol_arm,
            'calibration_role': role,
            'calibration_mode': calibration_mode,
        },
        'full_recent_metrics': {
            'rank_aux_weight_mean': rank_aux_weight_mean,
            'rank_aux_raw_loss': rank_aux_raw_loss,
            'opponent_aux_loss': opp_effective,
            'danger_aux_loss': danger_effective,
            'rank_phi_grad_rms': 0.0,
            'opponent_phi_grad_rms': 0.0,
            'danger_phi_grad_rms': 0.0,
            'rank_opponent_phi_grad_cos': 0.0,
            'rank_danger_phi_grad_cos': 0.0,
            'opp_danger_phi_grad_cos': 0.0,
            'opp_danger_phi_combo_factor': 0.0,
        },
    }


class RunStage05FidelityTests(unittest.TestCase):
    def test_update_results_doc_keeps_normal_chinese_title(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / 'fidelity_run'
            run_dir.mkdir(parents=True, exist_ok=True)
            doc_path = Path(tmp_dir) / 'stage05-fidelity-results.md'

            with mock.patch.object(fidelity, 'RESULTS_DOC_PATH', doc_path):
                fidelity.update_results_doc(run_dir, {'status': 'running'})

            first_line = doc_path.read_text(encoding='utf-8').splitlines()[0]

        self.assertEqual('# Stage 0.5 保真版 A/B 实时结果', first_line)

    def test_protocol_decide_probe_keeps_all_three_slots(self) -> None:
        ranked = [
            make_ranked_entry(
                'proto_a_ce',
                protocol_arm='proto_a',
                aux_family='ce_only',
                recent_policy_loss=0.10,
            ),
            make_ranked_entry(
                'proto_a_all_1',
                protocol_arm='proto_a',
                aux_family='all_three',
                recent_policy_loss=0.11,
            ),
            make_ranked_entry(
                'proto_a_all_2',
                protocol_arm='proto_a',
                aux_family='all_three',
                recent_policy_loss=0.12,
            ),
            make_ranked_entry(
                'proto_a_all_3',
                protocol_arm='proto_a',
                aux_family='all_three',
                recent_policy_loss=0.13,
            ),
            make_ranked_entry(
                'proto_a_all_4',
                protocol_arm='proto_a',
                aux_family='all_three',
                recent_policy_loss=0.14,
            ),
        ]
        candidates = [
            fidelity.CandidateSpec(
                arm_name=str(entry['arm_name']),
                scheduler_profile='sched',
                curriculum_profile='curr',
                weight_profile='weight',
                window_profile='window',
                cfg_overrides={},
                meta=dict(entry['candidate_meta']),
            )
            for entry in ranked
        ]

        selected = fidelity.select_p1_protocol_decide_probe_candidates(ranked, candidates, keep=4)

        self.assertEqual(
            ['proto_a_all_1', 'proto_a_all_2', 'proto_a_all_3', 'proto_a_all_4'],
            [candidate.arm_name for candidate in selected],
        )

    def test_protocol_compare_ignores_ce_only_anchor_winners(self) -> None:
        ranked = [
            make_ranked_entry(
                'proto_a_ce',
                protocol_arm='proto_a',
                aux_family='ce_only',
                recent_policy_loss=0.10,
            ),
            make_ranked_entry(
                'proto_b_all',
                protocol_arm='proto_b',
                aux_family='all_three',
                recent_policy_loss=0.15,
            ),
            make_ranked_entry(
                'proto_a_all',
                protocol_arm='proto_a',
                aux_family='all_three',
                recent_policy_loss=0.20,
            ),
        ]

        winners = fidelity.build_p1_protocol_compare(ranked)

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

    def test_missing_default_calibration_protocol_reuses_external_protocol(self) -> None:
        protocols = [make_protocol('proto_a')]
        external_protocol = fidelity.build_p0_candidates()[0]

        with mock.patch.object(
            fidelity,
            'P1_CALIBRATION_DEFAULT_PROTOCOL_ARMS',
            (external_protocol.arm_name,),
        ):
            resolved_arms, calibration_protocols = fidelity.resolve_p1_calibration_protocol_arms(protocols, None)

        self.assertEqual([external_protocol.arm_name], resolved_arms)
        self.assertEqual([external_protocol.arm_name], [protocol.arm_name for protocol in calibration_protocols])
        self.assertEqual(
            [external_protocol.arm_name],
            [str(protocol.meta.get('protocol_arm', '')) for protocol in calibration_protocols],
        )

    def test_unknown_calibration_protocol_still_raises(self) -> None:
        protocols = [make_protocol('proto_a')]

        with self.assertRaisesRegex(
            ValueError,
            r"unknown calibration protocol arms for p1 run: \['frozen_missing'\]",
        ):
            fidelity.resolve_p1_calibration_protocol_arms(protocols, ['frozen_missing'])

    def test_combo_only_calibration_reuses_inherited_single_head_probe_values(self) -> None:
        probe_weight = fidelity.P1_CALIBRATION_PROBE_WEIGHT
        calibration_round = {
            'ranking': [
                make_calibration_entry(
                    'rank_opp_probe',
                    protocol_arm='proto_a',
                    rank_effective=1.0,
                    opp_effective=4.0,
                ),
                make_calibration_entry(
                    'rank_danger_probe',
                    protocol_arm='proto_a',
                    rank_effective=1.0,
                    danger_effective=2.0,
                ),
                make_calibration_entry(
                    'opp_danger_probe',
                    protocol_arm='proto_a',
                    opp_effective=4.0,
                    danger_effective=2.0,
                ),
                make_calibration_entry(
                    'triple_probe',
                    protocol_arm='proto_a',
                    rank_effective=1.0,
                    opp_effective=4.0,
                    danger_effective=2.0,
                ),
            ]
        }
        inherited_single_head = {
            'rank_effective_base': 1.0,
            'opp_effective_per_unit': 4.0 / probe_weight,
            'danger_effective_per_unit': 2.0 / probe_weight,
            'rank_grad_effective_base': 0.0,
            'opp_grad_effective_per_unit': 0.0,
            'danger_grad_effective_per_unit': 0.0,
            'probe_weight': probe_weight,
        }

        calibration = fidelity.derive_p1_budget_calibration(
            calibration_round,
            inherited_single_head=inherited_single_head,
            inherited_single_head_source='unit-test',
        )

        self.assertAlmostEqual(1.0, calibration['rank_opp_combo_factor_loss'])
        self.assertAlmostEqual(1.0, calibration['rank_danger_combo_factor_loss'])
        self.assertAlmostEqual(1.0, calibration['opp_danger_combo_factor_loss'])
        self.assertAlmostEqual(1.0, calibration['triple_combo_factor_loss'])
        self.assertAlmostEqual(4.0, calibration['single_head_probe_reference']['opp_effective'])
        self.assertAlmostEqual(2.0, calibration['single_head_probe_reference']['danger_effective'])

    def test_empty_combo_only_calibration_round_keeps_requested_mode(self) -> None:
        inherited_single_head = {
            'rank_effective_base': 1.25,
            'opp_effective_per_unit': 2.5,
            'danger_effective_per_unit': 3.5,
            'rank_grad_effective_base': 0.25,
            'opp_grad_effective_per_unit': 0.5,
            'danger_grad_effective_per_unit': 0.75,
            'probe_weight': fidelity.P1_CALIBRATION_PROBE_WEIGHT,
        }
        calibration_round = {
            'ranking': [
                {
                    'arm_name': 'proto_invalid',
                    'valid': False,
                    'candidate_meta': {
                        'protocol_arm': 'proto_a',
                        'calibration_role': 'rank_opp_probe',
                        'calibration_mode': fidelity.P1_CALIBRATION_MODE_COMBO_ONLY,
                    },
                    'full_recent_metrics': {},
                }
            ]
        }

        calibration = fidelity.derive_p1_budget_calibration(
            calibration_round,
            requested_calibration_mode=fidelity.P1_CALIBRATION_MODE_COMBO_ONLY,
            inherited_single_head=inherited_single_head,
            inherited_single_head_source='unit-test',
        )

        self.assertEqual(fidelity.P1_CALIBRATION_MODE_COMBO_ONLY, calibration['calibration_mode'])
        self.assertTrue(calibration['inherited_single_head'])
        self.assertEqual('unit-test', calibration['inherited_single_head_source'])
        self.assertAlmostEqual(1.25, calibration['rank_effective_base'])
        self.assertAlmostEqual(2.5, calibration['opp_effective_per_unit'])
        self.assertAlmostEqual(3.5, calibration['danger_effective_per_unit'])

    def test_full_calibration_mode_does_not_inherit_single_head_baseline(self) -> None:
        calibration_round = {
            'ranking': [
                make_calibration_entry(
                    'rank_opp_probe',
                    protocol_arm='proto_a',
                    rank_effective=1.0,
                    opp_effective=4.0,
                    calibration_mode=fidelity.P1_CALIBRATION_MODE_FULL,
                )
            ]
        }
        inherited_single_head = {
            'rank_effective_base': 9.0,
            'opp_effective_per_unit': 9.0,
            'danger_effective_per_unit': 9.0,
            'rank_grad_effective_base': 9.0,
            'opp_grad_effective_per_unit': 9.0,
            'danger_grad_effective_per_unit': 9.0,
            'opp_weight_per_budget_unit': 0.9,
            'danger_weight_per_budget_unit': 0.8,
            'probe_weight': fidelity.P1_CALIBRATION_PROBE_WEIGHT,
        }

        calibration = fidelity.derive_p1_budget_calibration(
            calibration_round,
            requested_calibration_mode=fidelity.P1_CALIBRATION_MODE_FULL,
            inherited_single_head=inherited_single_head,
            inherited_single_head_source='unit-test',
        )

        self.assertEqual(fidelity.P1_CALIBRATION_MODE_FULL, calibration['calibration_mode'])
        self.assertFalse(calibration['inherited_single_head'])
        self.assertIsNone(calibration['inherited_single_head_source'])
        self.assertAlmostEqual(0.0, calibration['rank_effective_base'])
        self.assertAlmostEqual(0.0, calibration['single_head_probe_reference']['opp_effective'])
        self.assertAlmostEqual(
            fidelity.P1_DEFAULT_DANGER_WEIGHT_PER_BUDGET,
            calibration['danger_weight_per_budget_unit'],
        )


if __name__ == '__main__':
    unittest.main()
