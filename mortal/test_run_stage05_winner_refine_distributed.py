from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_stage05_fidelity as fidelity
import run_stage05_winner_refine_distributed as distributed


def make_candidate(name: str) -> fidelity.CandidateSpec:
    return fidelity.CandidateSpec(
        arm_name=name,
        scheduler_profile='cosine',
        curriculum_profile='broad_to_recent',
        weight_profile='strong',
        window_profile='24m_12m',
        cfg_overrides={},
        meta={
            'protocol_arm': fidelity.P1_WINNER_REFINE_PROTOCOL_ARM,
            'aux_family': 'all_three',
        },
    )


def make_ranking_entry(
    name: str,
    *,
    selection_quality_score: float,
    comparison_recent_loss: float,
    eligible: bool = True,
    valid: bool = True,
) -> dict:
    return {
        'arm_name': name,
        'valid': valid,
        'eligible': eligible,
        'selection_quality_score': selection_quality_score,
        'comparison_recent_loss': comparison_recent_loss,
        'recent_policy_loss': comparison_recent_loss,
        'candidate_meta': {
            'protocol_arm': fidelity.P1_WINNER_REFINE_PROTOCOL_ARM,
            'aux_family': 'all_three',
        },
    }


class WinnerRefineDistributedTests(unittest.TestCase):
    def test_select_winner_refine_seed2_candidates_keeps_floor_and_gap_band(self):
        candidates = [make_candidate(name) for name in ('a', 'b', 'c', 'd')]
        ranking = [
            make_ranking_entry('a', selection_quality_score=0.5500, comparison_recent_loss=0.3000),
            make_ranking_entry('b', selection_quality_score=0.5497, comparison_recent_loss=0.3002),
            make_ranking_entry('c', selection_quality_score=0.5492, comparison_recent_loss=0.3005),
            make_ranking_entry('d', selection_quality_score=0.5475, comparison_recent_loss=0.3012),
        ]

        selected, details = distributed.select_winner_refine_seed2_candidates(
            ranking,
            candidates,
            min_keep=2,
            selection_gap=0.001,
            max_keep=9,
        )

        self.assertEqual(['a', 'b', 'c'], [candidate.arm_name for candidate in selected])
        self.assertEqual('eligible_then_selection_gap', details['mode'])
        self.assertEqual(3, details['candidate_count'])

    def test_select_winner_refine_seed2_candidates_prefers_eligible_pool(self):
        candidates = [make_candidate(name) for name in ('a', 'b', 'c')]
        ranking = [
            make_ranking_entry('a', selection_quality_score=0.5500, comparison_recent_loss=0.3000, eligible=True),
            make_ranking_entry('b', selection_quality_score=0.5494, comparison_recent_loss=0.3006, eligible=True),
            make_ranking_entry('c', selection_quality_score=0.5498, comparison_recent_loss=0.3003, eligible=False),
        ]

        selected, details = distributed.select_winner_refine_seed2_candidates(
            ranking,
            candidates,
            min_keep=2,
            selection_gap=0.001,
            max_keep=9,
        )

        self.assertEqual(['a', 'b'], [candidate.arm_name for candidate in selected])
        self.assertEqual('eligible', details['pool_mode'])

    def test_reset_running_tasks_for_resume_requeues_inflight_tasks(self):
        state = {
            'seed1': {
                'tasks': {
                    't1': {
                        'status': 'running',
                        'started_at': '2026-03-31 01:00:00',
                        'worker_label': 'desktop',
                        'local_result_path': 'x.json',
                        'remote_result_path': 'y.json',
                        'log_path': 'z.log',
                    }
                }
            },
            'seed2': {
                'tasks': {
                    't2': {
                        'status': 'completed',
                    }
                }
            },
        }

        distributed.reset_running_tasks_for_resume(state)

        self.assertEqual('pending', state['seed1']['tasks']['t1']['status'])
        self.assertNotIn('started_at', state['seed1']['tasks']['t1'])
        self.assertNotIn('worker_label', state['seed1']['tasks']['t1'])
        self.assertEqual('completed', state['seed2']['tasks']['t2']['status'])

    def test_update_run_state_for_dispatch_marks_stopped_after_refine(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir)
            state_path = run_dir / 'state.json'
            state_path.write_text(
                json.dumps(
                    {
                        'status': 'running_p1_winner_refine',
                        'p1': {},
                        'final_conclusion': {},
                    },
                    ensure_ascii=False,
                ),
                encoding='utf-8',
            )
            dispatch_state = {
                'stage': 'completed',
                'status': 'completed',
                'seed1': {'candidate_count': 27},
                'seed2': {'candidate_count': 5},
                'seed2_selector': {'selected_arm_names': ['arm_a', 'arm_b']},
                'winner_refine_centers': ['center_a'],
                'local_label': 'desktop',
                'remote_label': 'laptop',
            }
            final_round = {
                'round_name': 'p1_winner_refine_round',
                'ranking': [
                    {
                        'arm_name': 'arm_a',
                        'valid': True,
                    }
                ],
            }
            with patch.object(distributed.fidelity, 'update_results_doc'):
                distributed.update_run_state_for_dispatch(
                    run_dir=run_dir,
                    dispatch_state_path=run_dir / 'dispatch_state.json',
                    dispatch_state=dispatch_state,
                    front_runner='arm_a',
                    final_round=final_round,
                )

            persisted = json.loads(state_path.read_text(encoding='utf-8'))
            self.assertEqual('stopped_after_p1_winner_refine', persisted['status'])
            self.assertEqual('arm_a', persisted['final_conclusion']['p1_refine_front_runner'])
            self.assertEqual(final_round, persisted['p1']['winner_refine_round'])
            self.assertEqual(['center_a'], persisted['p1']['winner_refine_centers'])

    def test_update_run_state_for_dispatch_accepts_seed2_not_initialized_yet(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir)
            state_path = run_dir / 'state.json'
            state_path.write_text(
                json.dumps(
                    {
                        'status': 'stopped_after_p1_protocol_decide',
                        'p1': {},
                        'final_conclusion': {},
                    },
                    ensure_ascii=False,
                ),
                encoding='utf-8',
            )
            dispatch_state = {
                'stage': 'seed1',
                'status': 'running',
                'seed1': {'candidate_count': 27},
                'seed2': None,
                'seed2_selector': None,
                'winner_refine_centers': ['center_a'],
                'local_label': 'desktop',
                'remote_label': 'laptop',
            }
            with patch.object(distributed.fidelity, 'update_results_doc'):
                distributed.update_run_state_for_dispatch(
                    run_dir=run_dir,
                    dispatch_state_path=run_dir / 'dispatch_state.json',
                    dispatch_state=dispatch_state,
                )

            persisted = json.loads(state_path.read_text(encoding='utf-8'))
            dispatch_summary = persisted['p1']['winner_refine_dispatch']
            self.assertEqual('running', dispatch_summary['status'])
            self.assertEqual(27, dispatch_summary['seed1_candidate_count'])
            self.assertIsNone(dispatch_summary['seed2_candidate_count'])
            self.assertEqual([], dispatch_summary['seed2_selected_arm_names'])


if __name__ == '__main__':
    unittest.main()
