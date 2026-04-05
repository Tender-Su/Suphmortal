from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_stage05_fidelity as fidelity
import run_stage05_formal_distributed as formal_dist


def make_candidate_entry(
    arm_name: str,
    *,
    rank: int,
    protocol_arm: str = 'proto_arm',
    candidate_name: str | None = None,
    mix_name: str | None = None,
    source_arm: str | None = None,
    rank_budget_ratio: float = 0.05,
    opp_budget_ratio: float = 0.025,
    danger_budget_ratio: float = 0.04,
) -> dict:
    return {
        'arm_name': arm_name,
        'scheduler_profile': 'cosine',
        'curriculum_profile': 'broad_to_recent',
        'weight_profile': 'strong',
        'window_profile': '24m_12m',
        'cfg_overrides': {'aux': {'dummy': rank}},
        'candidate_meta': {
            'protocol_arm': protocol_arm,
            'aux_family': 'all_three',
            'candidate_name': candidate_name,
            'mix_name': mix_name,
            'source_arm': source_arm,
            'rank_budget_ratio': rank_budget_ratio,
            'opp_budget_ratio': opp_budget_ratio,
            'danger_budget_ratio': danger_budget_ratio,
        },
        'valid': True,
        'rank': rank,
    }


class RunStage05FormalDistributedTests(unittest.TestCase):
    def test_load_source_context_builds_child_run_names_from_explicit_candidates(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            fidelity_root = Path(tmp_dir)
            source_run_dir = fidelity_root / 'source_run'
            source_run_dir.mkdir(parents=True, exist_ok=True)
            state = {
                'seed': 20260329,
                'selected_protocol_arms': ['proto_arm'],
                'p1': {
                    'selected_protocol_arm': 'proto_arm',
                    'winner_refine_front_runner': 'front_runner_arm',
                    'winner_refine_round': {
                        'ranking': [
                            make_candidate_entry('arm_b', rank=2),
                            make_candidate_entry('arm_a', rank=1),
                        ]
                    },
                },
                'final_conclusion': {
                    'p1_protocol_winner': 'proto_arm',
                    'p1_refine_front_runner': 'front_runner_arm',
                },
            }
            (source_run_dir / 'state.json').write_text(json.dumps(state, ensure_ascii=False), encoding='utf-8')

            context = formal_dist.load_source_context(
                source_run_dir=source_run_dir,
                coordinator_run_name='triplet_formal_run',
                candidate_arms=['arm_a', 'arm_b'],
                formal_seed_offset=2000,
                formal_step_scale=5.0,
            )

            self.assertEqual('source_run', context['source_run_name'])
            self.assertEqual(20262329, context['formal_seed'])
            self.assertEqual('proto_arm', context['selected_protocol_arm'])
            self.assertEqual(
                ['triplet_formal_run__arm_a', 'triplet_formal_run__arm_b'],
                [payload['child_run_name'] for payload in context['candidate_payloads']],
            )

    def test_load_source_context_resolves_structural_aliases(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            fidelity_root = Path(tmp_dir)
            source_run_dir = fidelity_root / 'source_run'
            source_run_dir.mkdir(parents=True, exist_ok=True)
            center_arm = 'opp_center'
            scaled_arm = 'opp_center_scaled'
            shifted_arm = 'opp_center_shifted'
            state = {
                'seed': 20260329,
                'selected_protocol_arms': ['proto_arm'],
                'p1': {
                    'selected_protocol_arm': 'proto_arm',
                    'winner_refine_front_runner': scaled_arm,
                    'protocol_decide_round': {
                        'ranking': [
                            make_candidate_entry(
                                center_arm,
                                rank=1,
                                candidate_name='opp_lean_12',
                                mix_name='opp_lean',
                                source_arm=None,
                                rank_budget_ratio=0.0456,
                                opp_budget_ratio=0.0372,
                                danger_budget_ratio=0.0372,
                            ),
                        ]
                    },
                    'winner_refine_round': {
                        'ranking': [
                            make_candidate_entry(
                                scaled_arm,
                                rank=1,
                                candidate_name='scaled',
                                source_arm=center_arm,
                                rank_budget_ratio=0.0388,
                                opp_budget_ratio=0.0316,
                                danger_budget_ratio=0.0316,
                            ),
                            make_candidate_entry(
                                shifted_arm,
                                rank=2,
                                candidate_name='shifted',
                                source_arm=center_arm,
                                rank_budget_ratio=0.0356,
                                opp_budget_ratio=0.0370,
                                danger_budget_ratio=0.0472,
                            ),
                        ]
                    },
                },
                'final_conclusion': {
                    'p1_protocol_winner': 'proto_arm',
                    'p1_refine_front_runner': scaled_arm,
                },
            }
            (source_run_dir / 'state.json').write_text(json.dumps(state, ensure_ascii=False), encoding='utf-8')

            context = formal_dist.load_source_context(
                source_run_dir=source_run_dir,
                coordinator_run_name='triplet_formal_run',
                candidate_arms=['opp_lean*0.85', 'opp_lean(rank--/danger++)'],
                formal_seed_offset=2000,
                formal_step_scale=5.0,
            )

            self.assertEqual(scaled_arm, context['candidate_alias_to_arm']['opp_lean*0.85'])
            self.assertEqual(shifted_arm, context['candidate_alias_to_arm']['opp_lean(rank--/danger++)'])
            self.assertEqual('opp_lean*0.85', context['candidate_payloads'][0]['candidate_alias'])
            self.assertEqual('opp_lean(rank--/danger++)', context['candidate_payloads'][1]['candidate_alias'])
            self.assertEqual([scaled_arm, shifted_arm], [item['arm_name'] for item in context['candidate_payloads']])

    def test_initialize_dispatch_state_creates_one_task_per_candidate(self):
        source_context = {
            'source_run_name': 'source_run',
            'source_seed': 1,
            'selected_protocol_arm': 'proto_arm',
            'selected_protocol_arms': ['proto_arm'],
            'source_refine_front_runner': 'front_runner',
            'formal_seed': 2001,
            'formal_step_scale': 5.0,
            'candidate_alias_to_arm': {'anchor*1.0': 'arm_a', 'opp_lean*0.85': 'arm_b'},
            'candidate_arm_to_alias': {'arm_a': 'anchor*1.0', 'arm_b': 'opp_lean*0.85'},
            'candidate_payloads': [
                {
                    **fidelity.candidate_cache_payload(
                        fidelity.CandidateSpec(
                            arm_name='arm_a',
                            scheduler_profile='cosine',
                            curriculum_profile='broad_to_recent',
                            weight_profile='strong',
                            window_profile='24m_12m',
                            cfg_overrides={},
                            meta={'protocol_arm': 'proto_arm'},
                        ),
                        include_meta=True,
                    ),
                    'source_rank': 1,
                    'candidate_alias': 'anchor*1.0',
                    'child_run_name': 'triplet__arm_a',
                },
                {
                    **fidelity.candidate_cache_payload(
                        fidelity.CandidateSpec(
                            arm_name='arm_b',
                            scheduler_profile='cosine',
                            curriculum_profile='broad_to_recent',
                            weight_profile='strong',
                            window_profile='24m_12m',
                            cfg_overrides={},
                            meta={'protocol_arm': 'proto_arm'},
                        ),
                        include_meta=True,
                    ),
                    'source_rank': 2,
                    'candidate_alias': 'opp_lean*0.85',
                    'child_run_name': 'triplet__arm_b',
                },
            ],
        }

        dispatch_state = formal_dist.initialize_dispatch_state(
            run_name='triplet_formal_run',
            source_context=source_context,
            local_label='desktop',
            remote_label='laptop',
        )

        self.assertEqual('formal', dispatch_state['stage'])
        self.assertEqual(2, dispatch_state['formal']['task_count'])
        self.assertEqual({'anchor*1.0', 'opp_lean*0.85'}, set(dispatch_state['candidate_alias_to_arm']))
        self.assertEqual(
            {'arm_a', 'arm_b'},
            {task['candidate_arm'] for task in dispatch_state['formal']['tasks'].values()},
        )

    def test_execute_single_task_writes_child_state_and_result_json(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            fidelity_root = root / 'fidelity'
            ab_root = root / 'stage05_ab'
            fidelity_root.mkdir()
            ab_root.mkdir()
            coordinator_run = fidelity_root / 'triplet_formal_run'
            coordinator_run.mkdir()

            candidate = fidelity.CandidateSpec(
                arm_name='arm_a',
                scheduler_profile='cosine',
                curriculum_profile='broad_to_recent',
                weight_profile='strong',
                window_profile='24m_12m',
                cfg_overrides={'aux': {'dummy': 1}},
                meta={'protocol_arm': 'proto_arm'},
            )
            source_context = {
                'source_run_name': 'source_run',
                'source_seed': 20260329,
                'selected_protocol_arm': 'proto_arm',
                'selected_protocol_arms': ['proto_arm'],
                'source_refine_front_runner': 'front_runner',
                'formal_seed': 20262329,
                'formal_step_scale': 5.0,
                'candidate_payloads': [
                    {
                        **fidelity.candidate_cache_payload(candidate, include_meta=True),
                        'source_rank': 1,
                        'child_run_name': 'triplet_formal_run__arm_a',
                    }
                ],
            }
            dispatch_state = formal_dist.initialize_dispatch_state(
                run_name='triplet_formal_run',
                source_context=source_context,
                local_label='desktop',
                remote_label=None,
            )
            (coordinator_run / 'distributed' / 'formal_dispatch').mkdir(parents=True, exist_ok=True)
            (coordinator_run / 'distributed' / 'formal_dispatch' / 'dispatch_state.json').write_text(
                json.dumps(dispatch_state, ensure_ascii=False, indent=2),
                encoding='utf-8',
            )

            def fake_finalize_formal_result(_cfg, result, *, protocol_arm):
                checkpoint_root = (
                    ab_root
                    / 'triplet_formal_run__arm_a_formal'
                    / 'checkpoint_compare'
                    / 'phase_c'
                    / 'checkpoints'
                )
                checkpoint_root.mkdir(parents=True, exist_ok=True)
                best_loss = checkpoint_root / 'best_loss.pth'
                best_acc = checkpoint_root / 'best_acc.pth'
                best_rank = checkpoint_root / 'best_rank.pth'
                for path in (best_loss, best_acc, best_rank):
                    path.write_text('ckpt', encoding='utf-8')
                result.update(
                    {
                        'offline_checkpoint_winner': 'best_loss',
                        'shortlist_checkpoint_types': ['best_loss', 'best_acc', 'best_rank'],
                        'checkpoint_pack_types': ['best_loss', 'best_acc', 'best_rank'],
                        'candidates': {
                            'best_loss': {'path': str(best_loss)},
                            'best_acc': {'path': str(best_acc)},
                            'best_rank': {'path': str(best_rank)},
                        },
                    }
                )
                return result

            result_json = root / 'result.json'
            with (
                patch.object(formal_dist.fidelity, 'FIDELITY_ROOT', fidelity_root),
                patch.object(formal_dist.ab, 'AB_ROOT', ab_root),
                patch.object(formal_dist.ab, 'build_base_config', return_value={'supervised': {}}),
                patch.object(formal_dist.ab, 'group_files_by_month', return_value={}),
                patch.object(formal_dist.ab, 'load_all_files', return_value=[]),
                patch.object(formal_dist.ab, 'run_ab6_checkpoint', return_value={'winner': 'best_loss'}),
                patch.object(formal_dist.formal, 'finalize_formal_result', side_effect=fake_finalize_formal_result),
            ):
                payload = formal_dist.execute_single_task(
                    run_name='triplet_formal_run',
                    candidate_arm='arm_a',
                    result_json=result_json,
                    machine_label='desktop',
                )

            self.assertEqual('triplet_formal_run__arm_a', payload['child_run_name'])
            self.assertEqual('best_loss', payload['offline_checkpoint_winner'])
            child_state = json.loads((fidelity_root / 'triplet_formal_run__arm_a' / 'state.json').read_text(encoding='utf-8'))
            self.assertEqual('completed', child_state['formal']['status'])
            self.assertEqual('pending', child_state['formal_1v3']['status'])
            self.assertTrue(result_json.exists())

    def test_rewrite_repo_paths_rehomes_repo_relative_paths(self):
        remote_repo = Path(r'C:\remote\MahjongAI')
        local_repo = Path(r'C:\local\MahjongAI')
        payload = {
            'state': str(remote_repo / 'logs' / 'stage05_fidelity' / 'demo' / 'state.json'),
            'nested': {
                'best_loss': str(remote_repo / 'logs' / 'stage05_ab' / 'demo_formal' / 'best_loss.pth'),
                'outside': r'D:\dataset\file.json.gz',
            },
        }

        rewritten = formal_dist.rewrite_repo_paths(payload, remote_repo=remote_repo, local_repo=local_repo)

        self.assertEqual(
            str(local_repo / 'logs' / 'stage05_fidelity' / 'demo' / 'state.json'),
            rewritten['state'],
        )
        self.assertEqual(r'D:\dataset\file.json.gz', rewritten['nested']['outside'])


if __name__ == '__main__':
    unittest.main()
