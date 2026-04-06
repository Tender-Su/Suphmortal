from __future__ import annotations

import argparse
import json
import sys
import tempfile
import unittest
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_sl_formal_1v3_distributed as formal_1v3


def make_task_payload(
    checkpoint_type: str,
    *,
    avg_pt: float,
    avg_rank: float,
    pt_stderr: float,
    rankings: list[int] | None = None,
) -> dict:
    return {
        'candidate_arm': checkpoint_type,
        'machine_label': 'desktop',
        'round_label': 'coarse',
        'seed_key': 11,
        'payload': {
            'checkpoint_type': checkpoint_type,
            'checkpoint_path': f'C:/virtual/{checkpoint_type}.pth',
            'games': sum(rankings or [25, 25, 25, 25]),
            'rankings': rankings or [25, 25, 25, 25],
            'avg_pt': avg_pt,
            'avg_rank': avg_rank,
            'pt_stderr': pt_stderr,
            'rank_stderr': 0.01,
        },
    }


def make_worker_budgets() -> dict[str, dict]:
    return {
        'desktop': {
            'machine_label': 'desktop',
            'seed_count_per_iter': 1024,
            'games_per_iter': 4096,
            'shard_count': 4,
            'shard_seed_counts': [256, 256, 256, 256],
            'seed_count_source': 'desktop seed',
            'shard_count_source': 'desktop shard',
        },
        'laptop': {
            'machine_label': 'laptop',
            'seed_count_per_iter': 640,
            'games_per_iter': 2560,
            'shard_count': 3,
            'shard_seed_counts': [214, 213, 213],
            'seed_count_source': 'laptop seed',
            'shard_count_source': 'laptop shard',
        },
    }


class Formal1v3DistributedTests(unittest.TestCase):
    def test_load_formal_context_dedupes_identical_inference_weights(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / 'demo_run'
            ckpt_dir = Path(tmp_dir) / 'checkpoints'
            run_dir.mkdir(parents=True, exist_ok=True)
            ckpt_dir.mkdir(parents=True, exist_ok=True)

            def write_checkpoint(path: Path, *, tag: str) -> None:
                torch.save(
                    {
                        'config': {
                            'control': {'version': 4},
                            'resnet': {'conv_channels': 192, 'num_blocks': 40},
                        },
                        'mortal': {'weight': torch.tensor([1.0, 2.0])},
                        'policy_net': {'weight': torch.tensor([3.0, 4.0])},
                        'tag': tag,
                    },
                    path,
                )

            write_checkpoint(ckpt_dir / 'best_loss.pth', tag='loss')
            write_checkpoint(ckpt_dir / 'best_acc.pth', tag='acc')
            write_checkpoint(ckpt_dir / 'best_rank.pth', tag='rank')

            state = {
                'formal': {
                    'status': 'completed',
                    'shortlist_checkpoint_types': ['best_loss', 'best_acc', 'best_rank'],
                    'result': {
                        'config_snapshot': {
                            'base_cfg': {'supervised': {}},
                            'base_1v3_cfg': {
                                'challenger': {'device': 'cuda:0', 'state_file': 'C:/virtual/challenger.pth'},
                                'champion': {'device': 'cuda:0', 'state_file': 'C:/virtual/champion.pth'},
                            },
                        },
                        'selected_protocol_arm': 'proto_arm',
                        'current_primary_protocol_arm': 'proto_arm',
                        'candidates': {
                            'best_loss': {'path': str(ckpt_dir / 'best_loss.pth')},
                            'best_acc': {'path': str(ckpt_dir / 'best_acc.pth')},
                            'best_rank': {'path': str(ckpt_dir / 'best_rank.pth')},
                        },
                    },
                }
            }
            (run_dir / 'state.json').write_text(json.dumps(state, ensure_ascii=False), encoding='utf-8')

            context = formal_1v3.load_formal_context(run_dir)

        self.assertEqual(['best_loss', 'best_acc', 'best_rank'], context['shortlist_checkpoint_types'])
        self.assertEqual(['best_loss'], context['unique_shortlist_checkpoint_types'])
        self.assertEqual(['best_loss'], [candidate.arm_name for candidate in context['candidates']])
        self.assertEqual(
            ['best_loss', 'best_acc', 'best_rank'],
            context['checkpoint_dedupe']['groups']['best_loss']['members'],
        )
        self.assertEqual('best_loss', context['checkpoint_dedupe']['representative_for_checkpoint']['best_rank'])

    def test_build_task_id_includes_run_name_for_remote_uniqueness(self):
        task_id_a = formal_1v3.build_task_id(
            run_name='demo_run_a',
            stage_name='seed1',
            round_label='coarse',
            arm_name='best_acc',
            worker_label='laptop',
        )
        task_id_b = formal_1v3.build_task_id(
            run_name='demo_run_b',
            stage_name='seed1',
            round_label='coarse',
            arm_name='best_acc',
            worker_label='laptop',
        )

        self.assertEqual('seed1__coarse__best_acc__laptop__demo_run_a', task_id_a)
        self.assertEqual('seed1__coarse__best_acc__laptop__demo_run_b', task_id_b)
        self.assertNotEqual(task_id_a, task_id_b)

    def test_load_formal_context_uses_formal_checkpoint_pack(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / 'demo_run'
            ckpt_dir = Path(tmp_dir) / 'checkpoints'
            run_dir.mkdir(parents=True, exist_ok=True)
            ckpt_dir.mkdir(parents=True, exist_ok=True)

            def write_checkpoint(path: Path, *, offset: float) -> None:
                torch.save(
                    {
                        'config': {
                            'control': {'version': 4},
                            'resnet': {'conv_channels': 192, 'num_blocks': 40},
                        },
                        'mortal': {'weight': torch.tensor([1.0 + offset, 2.0])},
                        'policy_net': {'weight': torch.tensor([3.0, 4.0 + offset])},
                    },
                    path,
                )

            best_loss_path = ckpt_dir / 'best_loss.pth'
            best_acc_path = ckpt_dir / 'best_acc.pth'
            best_rank_path = ckpt_dir / 'best_rank.pth'
            write_checkpoint(best_loss_path, offset=0.0)
            write_checkpoint(best_acc_path, offset=1.0)
            write_checkpoint(best_rank_path, offset=2.0)
            state = {
                'formal': {
                    'status': 'completed',
                    'shortlist_checkpoint_types': ['best_loss', 'best_acc', 'best_rank'],
                    'result': {
                        'config_snapshot': {
                            'base_cfg': {'supervised': {}},
                            'base_1v3_cfg': {
                                'challenger': {'device': 'cuda:0', 'state_file': 'C:/virtual/challenger.pth'},
                                'champion': {'device': 'cuda:0', 'state_file': 'C:/virtual/champion.pth'},
                            },
                        },
                        'selected_protocol_arm': 'proto_arm',
                        'current_primary_protocol_arm': 'proto_arm',
                        'pending_canonical_alias_targets': ['C:/virtual/canonical/sl_canonical.pth'],
                        'candidates': {
                            'best_loss': {'path': str(best_loss_path)},
                            'best_acc': {'path': str(best_acc_path)},
                            'best_rank': {'path': str(best_rank_path)},
                        },
                    },
                }
            }
            (run_dir / 'state.json').write_text(json.dumps(state, ensure_ascii=False), encoding='utf-8')

            context = formal_1v3.load_formal_context(run_dir)

            self.assertEqual('proto_arm', context['protocol_arm'])
            self.assertEqual('proto_arm', context['current_primary_protocol_arm'])
            self.assertEqual(
                ['C:/virtual/canonical/sl_canonical.pth'],
                context['pending_canonical_alias_targets'],
            )
            self.assertEqual(['best_loss', 'best_acc', 'best_rank'], [candidate.arm_name for candidate in context['candidates']])
            self.assertEqual(str(best_acc_path), context['candidate_index']['best_acc'].meta['checkpoint_path'])
            self.assertEqual('C:/virtual/champion.pth', context['base_1v3_cfg']['champion']['state_file'])

    def test_load_formal_context_supports_manual_shortlist_config(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / 'manual_compare'
            ckpt_dir = Path(tmp_dir) / 'checkpoints'
            run_dir.mkdir(parents=True, exist_ok=True)
            ckpt_dir.mkdir(parents=True, exist_ok=True)

            def write_checkpoint(path: Path, *, offset: float) -> None:
                torch.save(
                    {
                        'config': {
                            'control': {'version': 4},
                            'resnet': {'conv_channels': 192, 'num_blocks': 40},
                        },
                        'mortal': {'weight': torch.tensor([1.0 + offset, 2.0])},
                        'policy_net': {'weight': torch.tensor([3.0, 4.0 + offset])},
                    },
                    path,
                )

            lean_path = ckpt_dir / 'opp_lean_085.pth'
            anchor_path = ckpt_dir / 'anchor_100.pth'
            write_checkpoint(lean_path, offset=0.0)
            write_checkpoint(anchor_path, offset=1.0)

            state = {
                'formal_1v3_config': {
                    'mode': 'manual_shortlist',
                    'protocol_arm': 'proto_arm',
                    'current_primary_protocol_arm': 'other_proto_arm',
                    'publish_canonical_aliases': False,
                    'config_snapshot': {
                        'base_cfg': {'supervised': {}},
                        'base_1v3_cfg': {
                            'challenger': {'device': 'cuda:0', 'state_file': 'C:/virtual/challenger.pth'},
                            'champion': {'device': 'cuda:0', 'state_file': 'C:/virtual/champion.pth'},
                        },
                    },
                    'candidates': {
                        'opp_lean*0.85': {'path': str(lean_path)},
                        'anchor*1.0': {'path': str(anchor_path)},
                    },
                }
            }
            (run_dir / 'state.json').write_text(json.dumps(state, ensure_ascii=False), encoding='utf-8')

            context = formal_1v3.load_formal_context(run_dir)

        self.assertEqual(['opp_lean*0.85', 'anchor*1.0'], context['shortlist_checkpoint_types'])
        self.assertEqual(['opp_lean*0.85', 'anchor*1.0'], [candidate.arm_name for candidate in context['candidates']])
        self.assertEqual('proto_arm', context['protocol_arm'])
        self.assertEqual('other_proto_arm', context['current_primary_protocol_arm'])
        self.assertEqual([], context['pending_canonical_alias_targets'])
        self.assertFalse(context['publish_canonical_aliases'])

    def test_initialize_dispatch_state_freezes_worker_budgets_per_task(self):
        context = {
            'run_name': 'demo_run',
            'protocol_arm': 'proto_arm',
            'config_snapshot': {
                'base_1v3_cfg': {'challenger': {}, 'champion': {}},
                'config_dir': 'C:/virtual/custom_cfg',
            },
            'shortlist_checkpoint_types': ['best_loss'],
            'checkpoint_pack_types': ['best_loss'],
            'candidates': [
                formal_1v3.build_formal_candidate('best_loss', {'path': 'C:/virtual/best_loss.pth'}, protocol_arm='proto_arm'),
            ],
        }
        worker_budgets = make_worker_budgets()

        dispatch_state = formal_1v3.initialize_dispatch_state(
            context=context,
            local_label='desktop',
            remote_label='laptop',
            worker_budgets=worker_budgets,
            coarse_iters=1,
            extra_iters=1,
            close_stderr_mult=1.0,
            close_extra_rounds=3,
            close_max_extra_rounds=5,
        )

        desktop_task_id = formal_1v3.build_task_id(
            run_name='demo_run',
            stage_name='seed1',
            round_label='coarse',
            arm_name='best_loss',
            worker_label='desktop',
        )
        laptop_task_id = formal_1v3.build_task_id(
            run_name='demo_run',
            stage_name='seed1',
            round_label='coarse',
            arm_name='best_loss',
            worker_label='laptop',
        )
        self.assertEqual(worker_budgets, dispatch_state['worker_budgets'])
        self.assertEqual(2, dispatch_state['seed1']['task_count'])
        self.assertEqual({'desktop', 'laptop'}, {task['assigned_worker_label'] for task in dispatch_state['seed1']['tasks'].values()})
        self.assertEqual(1024, dispatch_state['seed1']['tasks'][desktop_task_id]['seed_count_per_iter'])
        self.assertEqual(4, dispatch_state['seed1']['tasks'][desktop_task_id]['shard_count'])
        self.assertEqual(640, dispatch_state['seed1']['tasks'][laptop_task_id]['seed_count_per_iter'])
        self.assertEqual(3, dispatch_state['seed1']['tasks'][laptop_task_id]['shard_count'])
        self.assertEqual(0, dispatch_state['seed1']['tasks'][desktop_task_id]['seed_start_offset'])
        self.assertEqual(1024, dispatch_state['seed1']['tasks'][laptop_task_id]['seed_start_offset'])
        self.assertEqual(1664, dispatch_state['seed1']['tasks'][desktop_task_id]['seed_stride_per_iter'])
        self.assertEqual(1664, dispatch_state['seed1']['tasks'][laptop_task_id]['seed_stride_per_iter'])

    def test_query_remote_worker_budget_ignores_stderr_startup_logs(self):
        worker = formal_1v3.WorkerSpec(
            kind='remote',
            label='laptop',
            python=r'C:\Python\python.exe',
            host='mahjong-laptop',
            repo=r'C:\Users\numbe\Desktop\MahjongAI',
            ssh_key=r'C:\Users\numbe\.ssh\mahjong_laptop_ed25519',
        )
        completed = formal_1v3.subprocess.CompletedProcess(
            args=['ssh'],
            returncode=0,
            stdout='{"machine_label":"laptop","seed_count_per_iter":640}',
            stderr='oneDNN custom operations are on\n',
        )

        with patch.object(formal_1v3.subprocess, 'run', return_value=completed):
            payload = formal_1v3.query_remote_worker_budget(
                worker,
                frozen_1v3_cfg={'challenger': {'device': 'cuda:0'}},
            )

        self.assertEqual('laptop', payload['machine_label'])
        self.assertEqual(640, payload['seed_count_per_iter'])

    def test_close_call_from_ranking_uses_avg_pt_primary(self):
        close = formal_1v3.close_call_from_ranking(
            [
                {'checkpoint_type': 'best_loss', 'avg_pt': 5.0, 'avg_rank': 2.20, 'pt_stderr': 4.0},
                {'checkpoint_type': 'best_acc', 'avg_pt': 2.5, 'avg_rank': 2.25, 'pt_stderr': 3.0},
            ],
            stderr_mult=1.0,
        )
        not_close = formal_1v3.close_call_from_ranking(
            [
                {'checkpoint_type': 'best_loss', 'avg_pt': 12.0, 'avg_rank': 2.10, 'pt_stderr': 2.0},
                {'checkpoint_type': 'best_acc', 'avg_pt': 3.0, 'avg_rank': 2.30, 'pt_stderr': 2.0},
            ],
            stderr_mult=1.0,
        )

        self.assertTrue(close['triggered'])
        self.assertFalse(not_close['triggered'])

    def test_maybe_promote_seed1_to_seed2_builds_close_rounds_for_top2(self):
        context = {
            'run_dir': Path('X:/virtual/run'),
            'protocol_arm': 'proto_arm',
            'config_snapshot': {'base_1v3_cfg': {'challenger': {}, 'champion': {}}},
            'candidates': [
                formal_1v3.build_formal_candidate('best_loss', {'path': 'C:/virtual/best_loss.pth'}, protocol_arm='proto_arm'),
                formal_1v3.build_formal_candidate('best_acc', {'path': 'C:/virtual/best_acc.pth'}, protocol_arm='proto_arm'),
                formal_1v3.build_formal_candidate('best_rank', {'path': 'C:/virtual/best_rank.pth'}, protocol_arm='proto_arm'),
            ],
            'candidate_index': {},
        }
        context['candidate_index'] = {candidate.arm_name: candidate for candidate in context['candidates']}
        dispatch_state = formal_1v3.initialize_dispatch_state(
            context={
                **context,
                'run_name': 'demo_run',
                'shortlist_checkpoint_types': ['best_loss', 'best_acc', 'best_rank'],
            },
            local_label='desktop',
            remote_label='laptop',
            worker_budgets=make_worker_budgets(),
            coarse_iters=1,
            extra_iters=1,
            close_stderr_mult=1.0,
            close_extra_rounds=3,
            close_max_extra_rounds=5,
        )
        for task in dispatch_state['seed1']['tasks'].values():
            task['status'] = 'completed'

        with (
            patch.object(
                formal_1v3,
                'aggregate_task_payloads',
                return_value=[
                    {'checkpoint_type': 'best_loss', 'avg_pt': 5.0, 'avg_rank': 2.2},
                    {'checkpoint_type': 'best_acc', 'avg_pt': 4.8, 'avg_rank': 2.25},
                    {'checkpoint_type': 'best_rank', 'avg_pt': -3.0, 'avg_rank': 2.9},
                ],
            ),
            patch.object(formal_1v3, 'stage_results_from_state', return_value=[]),
            patch.object(formal_1v3, 'close_call_from_ranking', return_value={'triggered': True}),
            patch.object(formal_1v3.fidelity, 'atomic_write_json'),
            patch.object(formal_1v3, 'write_dispatch_state'),
            patch.object(formal_1v3, 'update_run_state_for_dispatch'),
        ):
            finalized = formal_1v3.maybe_promote_seed1_to_seed2(
                context=context,
                dispatch_state=dispatch_state,
                dispatch_state_path=Path('X:/virtual/dispatch_state.json'),
            )

        self.assertFalse(finalized)
        self.assertEqual('seed2', dispatch_state['stage'])
        self.assertEqual(12, len(dispatch_state['seed2']['tasks']))

    def test_maybe_promote_seed1_to_seed2_finalizes_when_not_close(self):
        context = {
            'run_dir': Path('X:/virtual/run'),
            'protocol_arm': 'proto_arm',
            'candidates': [formal_1v3.build_formal_candidate('best_loss', {'path': 'C:/virtual/best_loss.pth'}, protocol_arm='proto_arm')],
            'candidate_index': {},
            'run_name': 'demo_run',
            'config_snapshot': {'base_1v3_cfg': {'challenger': {}, 'champion': {}}},
            'shortlist_checkpoint_types': ['best_loss'],
        }
        context['candidate_index'] = {candidate.arm_name: candidate for candidate in context['candidates']}
        dispatch_state = formal_1v3.initialize_dispatch_state(
            context=context,
            local_label='desktop',
            remote_label='laptop',
            worker_budgets=make_worker_budgets(),
            coarse_iters=1,
            extra_iters=1,
            close_stderr_mult=1.0,
            close_extra_rounds=3,
            close_max_extra_rounds=5,
        )
        for task in dispatch_state['seed1']['tasks'].values():
            task['status'] = 'completed'

        with (
            patch.object(formal_1v3, 'aggregate_task_payloads', return_value=[{'checkpoint_type': 'best_loss', 'avg_pt': 5.0, 'avg_rank': 2.2}]),
            patch.object(formal_1v3, 'stage_results_from_state', return_value=[]),
            patch.object(formal_1v3, 'close_call_from_ranking', return_value={'triggered': False}),
            patch.object(formal_1v3.fidelity, 'atomic_write_json'),
            patch.object(formal_1v3, 'write_dispatch_state'),
            patch.object(formal_1v3, 'finalize_dispatch', return_value=True) as finalize_dispatch,
        ):
            finalized = formal_1v3.maybe_promote_seed1_to_seed2(
                context=context,
                dispatch_state=dispatch_state,
                dispatch_state_path=Path('X:/virtual/dispatch_state.json'),
            )

        self.assertTrue(finalized)
        finalize_dispatch.assert_called_once()

    def test_finalize_dispatch_publishes_formal_winner(self):
        context = {
            'run_dir': Path('X:/virtual/run'),
            'protocol_arm': 'proto_arm',
            'current_primary_protocol_arm': 'proto_arm',
            'shortlist_candidates': {
                'best_loss': {'path': 'C:/virtual/best_loss.pth'},
                'best_acc': {'path': 'C:/virtual/best_acc.pth'},
            },
            'pending_canonical_alias_targets': ['X:/virtual/custom_cfg/checkpoints/sl_canonical.pth'],
            'candidates': [],
            'candidate_order': {},
            'config_snapshot': {
                'base_1v3_cfg': {'challenger': {}, 'champion': {}},
                'config_dir': 'X:/virtual/custom_cfg',
            },
            'base_cfg': {'supervised': {}},
        }
        dispatch_state = {
            'seed1': {'tasks': {}},
            'seed2': None,
            'close_stderr_mult': 1.0,
            'coarse_round_summary_path': 'coarse.json',
            'status': 'running',
            'stage': 'seed1',
            'config_snapshot': {'base_1v3_cfg': {'challenger': {}, 'champion': {}}},
        }

        with (
            patch.object(formal_1v3, 'stage_results_from_state', return_value=[]),
            patch.object(
                formal_1v3,
                'aggregate_task_payloads',
                return_value=[
                    {'checkpoint_type': 'best_acc', 'avg_pt': 6.0, 'avg_rank': 2.1},
                    {'checkpoint_type': 'best_loss', 'avg_pt': 5.0, 'avg_rank': 2.2},
                ],
            ),
            patch.object(formal_1v3, 'close_call_from_ranking', return_value={'triggered': False}),
            patch.object(formal_1v3.fidelity, 'atomic_write_json'),
            patch.object(
                formal_1v3.formal,
                'publish_sl_canonical_checkpoints',
                return_value=[
                    {
                        'source': 'src',
                        'destination': 'X:/virtual/custom_cfg/checkpoints/sl_canonical.pth',
                    },
                ],
            ) as publish_mock,
            patch.object(formal_1v3, 'write_dispatch_state'),
            patch.object(formal_1v3, 'update_run_state_for_dispatch'),
        ):
            finalized = formal_1v3.finalize_dispatch(
                context=context,
                dispatch_state=dispatch_state,
                dispatch_state_path=Path('X:/virtual/dispatch_state.json'),
            )

        self.assertTrue(finalized)
        self.assertEqual('best_acc', dispatch_state['final_winner'])
        publish_mock.assert_called_once_with(
            {'supervised': {}},
            {'winner': 'best_acc', 'candidates': {'best_loss': {'path': 'C:/virtual/best_loss.pth'}, 'best_acc': {'path': 'C:/virtual/best_acc.pth'}}},
            config_dir=Path('X:/virtual/custom_cfg'),
            protocol_arm='proto_arm',
            primary_protocol_arm='proto_arm',
        )

    def test_finalize_single_unique_candidate_dispatch_short_circuits(self):
        context = {
            'run_name': 'demo_run',
            'run_dir': Path('X:/virtual/run'),
            'protocol_arm': 'proto_arm',
            'current_primary_protocol_arm': 'proto_arm',
            'shortlist_candidates': {
                'best_loss': {'path': 'C:/virtual/best_loss.pth'},
                'best_acc': {'path': 'C:/virtual/best_acc.pth'},
                'best_rank': {'path': 'C:/virtual/best_rank.pth'},
            },
            'shortlist_checkpoint_types': ['best_loss', 'best_acc', 'best_rank'],
            'unique_shortlist_checkpoint_types': ['best_loss'],
            'checkpoint_pack_types': ['best_loss', 'best_acc', 'best_rank'],
            'checkpoint_dedupe': {
                'unique_checkpoint_types': ['best_loss'],
                'representative_for_checkpoint': {
                    'best_loss': 'best_loss',
                    'best_acc': 'best_loss',
                    'best_rank': 'best_loss',
                },
                'groups': {
                    'best_loss': {
                        'representative': 'best_loss',
                        'members': ['best_loss', 'best_acc', 'best_rank'],
                        'fingerprint': 'fp',
                        'checkpoint_path': 'C:/virtual/best_loss.pth',
                    }
                },
                'duplicate_checkpoint_types': ['best_acc', 'best_rank'],
            },
            'pending_canonical_alias_targets': ['X:/virtual/custom_cfg/checkpoints/sl_canonical.pth'],
            'candidates': [
                formal_1v3.build_formal_candidate('best_loss', {'path': 'C:/virtual/best_loss.pth'}, protocol_arm='proto_arm'),
            ],
            'candidate_index': {},
            'config_snapshot': {
                'base_1v3_cfg': {'challenger': {}, 'champion': {}},
                'config_dir': 'X:/virtual/custom_cfg',
            },
            'base_cfg': {'supervised': {}},
        }
        context['candidate_index'] = {candidate.arm_name: candidate for candidate in context['candidates']}

        with (
            patch.object(formal_1v3.fidelity, 'atomic_write_json'),
            patch.object(
                formal_1v3.formal,
                'publish_sl_canonical_checkpoints',
                return_value=[
                    {
                        'source': 'src',
                        'destination': 'X:/virtual/custom_cfg/checkpoints/sl_canonical.pth',
                    },
                ],
            ) as publish_mock,
            patch.object(formal_1v3, 'write_dispatch_state'),
            patch.object(formal_1v3, 'update_run_state_for_dispatch'),
        ):
            dispatch_state = formal_1v3.finalize_single_unique_candidate_dispatch(
                context=context,
                dispatch_state_path=Path('X:/virtual/dispatch_state.json'),
                local_label='desktop',
                remote_label='laptop',
                close_stderr_mult=1.0,
            )

        self.assertEqual('completed', dispatch_state['status'])
        self.assertEqual('completed', dispatch_state['stage'])
        self.assertEqual('best_loss', dispatch_state['final_winner'])
        self.assertEqual(['best_loss'], dispatch_state['unique_shortlist_checkpoint_types'])
        self.assertEqual(
            ['best_loss', 'best_acc', 'best_rank'],
            dispatch_state['checkpoint_dedupe']['groups']['best_loss']['members'],
        )
        publish_mock.assert_called_once_with(
            {'supervised': {}},
            {
                'winner': 'best_loss',
                'candidates': {
                    'best_loss': {'path': 'C:/virtual/best_loss.pth'},
                    'best_acc': {'path': 'C:/virtual/best_acc.pth'},
                    'best_rank': {'path': 'C:/virtual/best_rank.pth'},
                },
            },
            config_dir=Path('X:/virtual/custom_cfg'),
            protocol_arm='proto_arm',
            primary_protocol_arm='proto_arm',
        )

    def test_build_remote_formal_state_payload_rewrites_manual_shortlist_paths(self):
        state = {
            'formal_1v3_config': {
                'mode': 'manual_shortlist',
                'protocol_arm': 'proto_arm',
                'publish_canonical_aliases': False,
                'config_snapshot': {
                    'base_cfg': {'supervised': {}},
                    'base_1v3_cfg': {
                        'challenger': {'state_file': 'C:/virtual/challenger.pth'},
                        'champion': {'state_file': 'C:/virtual/champion_local.pth'},
                    },
                },
                'candidates': {
                    'opp_lean*0.85': {'path': 'C:/virtual/local_lean.pth'},
                    'anchor*1.0': {'path': 'C:/virtual/local_anchor.pth'},
                },
            }
        }

        remote_state = formal_1v3.build_remote_formal_state_payload(
            state=state,
            remote_checkpoint_paths={
                'opp_lean*0.85': Path(r'D:\remote\lean.pth'),
                'anchor*1.0': Path(r'D:\remote\anchor.pth'),
            },
            remote_champion_state_file=Path(r'D:\remote\baseline.pth'),
        )

        manual_config = remote_state['formal_1v3_config']
        self.assertEqual(r'D:\remote\lean.pth', manual_config['candidates']['opp_lean*0.85']['path'])
        self.assertEqual(r'D:\remote\anchor.pth', manual_config['candidates']['anchor*1.0']['path'])
        self.assertEqual(
            r'D:\remote\baseline.pth',
            manual_config['config_snapshot']['base_1v3_cfg']['champion']['state_file'],
        )

    def test_finalize_dispatch_skips_publish_when_manual_compare_disables_alias_publish(self):
        context = {
            'run_dir': Path('X:/virtual/run'),
            'protocol_arm': 'proto_arm',
            'current_primary_protocol_arm': 'proto_arm',
            'shortlist_candidates': {
                'opp_lean*0.85': {'path': 'C:/virtual/lean.pth'},
                'anchor*1.0': {'path': 'C:/virtual/anchor.pth'},
            },
            'pending_canonical_alias_targets': [],
            'publish_canonical_aliases': False,
            'candidates': [],
            'candidate_order': {},
            'config_snapshot': {
                'base_1v3_cfg': {'challenger': {}, 'champion': {}},
                'config_dir': 'X:/virtual/custom_cfg',
            },
            'base_cfg': {'supervised': {}},
        }
        dispatch_state = {
            'seed1': {'tasks': {}},
            'seed2': None,
            'close_stderr_mult': 1.0,
            'coarse_round_summary_path': 'coarse.json',
            'status': 'running',
            'stage': 'seed1',
            'config_snapshot': {'base_1v3_cfg': {'challenger': {}, 'champion': {}}},
        }

        with (
            patch.object(formal_1v3, 'stage_results_from_state', return_value=[]),
            patch.object(
                formal_1v3,
                'aggregate_task_payloads',
                return_value=[
                    {'checkpoint_type': 'anchor*1.0', 'avg_pt': 6.0, 'avg_rank': 2.1},
                    {'checkpoint_type': 'opp_lean*0.85', 'avg_pt': 5.0, 'avg_rank': 2.2},
                ],
            ),
            patch.object(formal_1v3, 'close_call_from_ranking', return_value={'triggered': False}),
            patch.object(formal_1v3.fidelity, 'atomic_write_json'),
            patch.object(
                formal_1v3.formal,
                'publish_sl_canonical_checkpoints',
            ) as publish_mock,
            patch.object(formal_1v3, 'write_dispatch_state'),
            patch.object(formal_1v3, 'update_run_state_for_dispatch'),
        ):
            finalized = formal_1v3.finalize_dispatch(
                context=context,
                dispatch_state=dispatch_state,
                dispatch_state_path=Path('X:/virtual/dispatch_state.json'),
            )

        self.assertTrue(finalized)
        self.assertEqual('anchor*1.0', dispatch_state['final_winner'])
        publish_mock.assert_not_called()

    def test_finalize_dispatch_raises_when_expected_alias_publish_is_missing(self):
        context = {
            'run_dir': Path('X:/virtual/run'),
            'protocol_arm': 'proto_arm',
            'current_primary_protocol_arm': 'proto_arm',
            'shortlist_candidates': {
                'best_loss': {'path': 'C:/virtual/best_loss.pth'},
            },
            'pending_canonical_alias_targets': ['X:/virtual/custom_cfg/checkpoints/sl_canonical.pth'],
            'candidates': [],
            'candidate_order': {},
            'config_snapshot': {
                'base_1v3_cfg': {'challenger': {}, 'champion': {}},
                'config_dir': 'X:/virtual/custom_cfg',
            },
            'base_cfg': {'supervised': {}},
        }
        dispatch_state = {
            'seed1': {'tasks': {}},
            'seed2': None,
            'close_stderr_mult': 1.0,
            'coarse_round_summary_path': 'coarse.json',
            'status': 'running',
            'stage': 'seed1',
            'config_snapshot': {'base_1v3_cfg': {'challenger': {}, 'champion': {}}},
        }

        with (
            patch.object(formal_1v3, 'stage_results_from_state', return_value=[]),
            patch.object(
                formal_1v3,
                'aggregate_task_payloads',
                return_value=[
                    {'checkpoint_type': 'best_loss', 'avg_pt': 6.0, 'avg_rank': 2.1},
                ],
            ),
            patch.object(formal_1v3, 'close_call_from_ranking', return_value={'triggered': False}),
            patch.object(formal_1v3.fidelity, 'atomic_write_json'),
            patch.object(
                formal_1v3.formal,
                'publish_sl_canonical_checkpoints',
                return_value=[],
            ),
        ):
            with self.assertRaisesRegex(RuntimeError, 'did not publish any expected supervised canonical aliases'):
                formal_1v3.finalize_dispatch(
                    context=context,
                    dispatch_state=dispatch_state,
                    dispatch_state_path=Path('X:/virtual/dispatch_state.json'),
                )

    def test_reconcile_completed_dispatch_rejects_missing_publish_coverage(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / 'demo_run'
            dispatch_state_path = run_dir / 'distributed' / 'formal_1v3_dispatch' / 'dispatch_state.json'
            dispatch_state_path.parent.mkdir(parents=True, exist_ok=True)
            final_payload = {
                'round_name': 'formal_1v3_round',
                'protocol_arm': 'proto_arm',
                'ranking_metric': 'avg_pt_primary_avg_rank_secondary',
                'ranking': [{'checkpoint_type': 'best_acc', 'avg_pt': 6.0, 'avg_rank': 2.1}],
                'coarse_task_count': 2,
                'extra_task_count': 0,
                'close_call': {'triggered': False},
            }
            final_summary_path = run_dir / 'formal_1v3_round.json'
            final_summary_path.parent.mkdir(parents=True, exist_ok=True)
            final_summary_path.write_text(json.dumps(final_payload, ensure_ascii=False), encoding='utf-8')
            state = {
                'status': 'running_formal_1v3',
                'formal': {
                    'status': 'completed',
                    'result': {
                        'pending_canonical_alias_targets': [str(run_dir / 'checkpoints' / 'sl_canonical.pth')],
                    },
                },
                'final_conclusion': {
                    'formal_status': 'pending_1v3',
                    'formal_1v3_status': 'running',
                },
            }
            (run_dir / 'state.json').write_text(json.dumps(state, ensure_ascii=False), encoding='utf-8')
            dispatch_state = {
                'stage': 'completed',
                'status': 'completed',
                'final_round_summary_path': str(final_summary_path),
                'final_winner': 'best_acc',
                'published_canonical_checkpoints': [],
            }

            with self.assertRaisesRegex(RuntimeError, 'did not publish any expected supervised canonical aliases'):
                formal_1v3.reconcile_completed_dispatch_run_state(
                    run_dir=run_dir,
                    dispatch_state_path=dispatch_state_path,
                    dispatch_state=dispatch_state,
                )

    def test_update_run_state_for_dispatch_keeps_pending_handoff_on_failure(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / 'demo_run'
            dispatch_state_path = run_dir / 'distributed' / 'formal_1v3_dispatch' / 'dispatch_state.json'
            dispatch_state_path.parent.mkdir(parents=True, exist_ok=True)
            state = {
                'status': 'running_formal_1v3',
                'formal': {'status': 'completed'},
                'final_conclusion': {
                    'formal_status': 'pending_1v3',
                    'formal_1v3_status': 'running',
                },
            }
            (run_dir / 'state.json').write_text(json.dumps(state, ensure_ascii=False), encoding='utf-8')
            dispatch_state = {
                'stage': 'seed1',
                'status': 'failed',
                'shortlist_checkpoint_types': ['best_loss'],
                'checkpoint_pack_types': ['best_loss'],
            }

            with patch.object(formal_1v3.fidelity, 'update_results_doc') as update_results_doc:
                formal_1v3.update_run_state_for_dispatch(
                    run_dir=run_dir,
                    dispatch_state_path=dispatch_state_path,
                    dispatch_state=dispatch_state,
                )

            updated_state = json.loads((run_dir / 'state.json').read_text(encoding='utf-8'))
            self.assertEqual('failed', updated_state['status'])
            self.assertEqual('failed', updated_state['final_conclusion']['formal_1v3_status'])
            self.assertEqual('pending_1v3', updated_state['final_conclusion']['formal_status'])
            update_results_doc.assert_called_once()

    def test_run_dispatch_reconciles_completed_dispatch_state_before_return(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            fidelity_root = Path(tmp_dir)
            run_dir = fidelity_root / 'demo_run'
            dispatch_state_path = run_dir / 'distributed' / 'formal_1v3_dispatch' / 'dispatch_state.json'
            dispatch_state_path.parent.mkdir(parents=True, exist_ok=True)
            final_payload = {
                'round_name': 'formal_1v3_round',
                'protocol_arm': 'proto_arm',
                'ranking_metric': 'avg_pt_primary_avg_rank_secondary',
                'ranking': [{'checkpoint_type': 'best_acc', 'avg_pt': 6.0, 'avg_rank': 2.1}],
                'coarse_task_count': 2,
                'extra_task_count': 0,
                'close_call': {'triggered': False},
            }
            final_summary_path = run_dir / 'formal_1v3_round.json'
            final_summary_path.parent.mkdir(parents=True, exist_ok=True)
            final_summary_path.write_text(json.dumps(final_payload, ensure_ascii=False), encoding='utf-8')
            state = {
                'status': 'running_formal_1v3',
                'formal': {'status': 'completed'},
                'final_conclusion': {
                    'formal_status': 'pending_1v3',
                    'formal_1v3_status': 'running',
                },
            }
            (run_dir / 'state.json').write_text(json.dumps(state, ensure_ascii=False), encoding='utf-8')
            dispatch_state_path.write_text(
                json.dumps(
                    {
                        'stage': 'completed',
                        'status': 'completed',
                        'final_round_summary_path': str(final_summary_path),
                        'final_winner': 'best_acc',
                        'published_canonical_checkpoints': [{'source': 'src', 'destination': 'dst'}],
                    },
                    ensure_ascii=False,
                ),
                encoding='utf-8',
            )
            args = argparse.Namespace(
                run_name='demo_run',
                local_only=True,
                local_python=sys.executable,
                local_label='desktop',
                remote_host='mahjong-laptop',
                remote_repo=r'C:\Users\numbe\Desktop\MahjongAI',
                remote_python=sys.executable,
                remote_label='laptop',
                ssh_key=r'C:\Users\numbe\.ssh\mahjong_laptop_ed25519',
                remote_launch_mode='interactive_window',
                poll_seconds=0.01,
                max_attempts=1,
                coarse_iters=1,
                extra_iters=1,
                close_stderr_mult=1.0,
                close_extra_rounds=1,
                close_max_extra_rounds=1,
            )

            with (
                patch.object(formal_1v3.fidelity, 'FIDELITY_ROOT', fidelity_root),
                patch.object(formal_1v3.fidelity, 'acquire_run_lock', return_value=run_dir / '.lock'),
                patch.object(formal_1v3.fidelity, 'release_run_lock'),
                patch.object(formal_1v3.atexit, 'register'),
                patch.object(formal_1v3, 'load_formal_context', side_effect=AssertionError('completed dispatch should not reload context')),
                patch.object(formal_1v3.common_dispatch, 'build_workers', side_effect=AssertionError('completed dispatch should not build workers')),
                patch.object(formal_1v3, 'write_dispatch_state') as write_dispatch_state,
                patch.object(formal_1v3.fidelity, 'update_results_doc') as update_results_doc,
            ):
                rc = formal_1v3.run_dispatch(args)

            updated_state = json.loads((run_dir / 'state.json').read_text(encoding='utf-8'))
            self.assertEqual(0, rc)
            write_dispatch_state.assert_not_called()
            self.assertEqual('completed', updated_state['status'])
            self.assertEqual('completed', updated_state['final_conclusion']['formal_status'])
            self.assertEqual('completed', updated_state['final_conclusion']['formal_1v3_status'])
            self.assertEqual('best_acc', updated_state['final_conclusion']['formal_winner'])
            self.assertEqual(final_payload, updated_state['formal_1v3']['result'])
            self.assertEqual([{'source': 'src', 'destination': 'dst'}], updated_state['formal_1v3']['published_canonical_checkpoints'])
            update_results_doc.assert_called_once()

    def test_validate_resume_worker_labels_rejects_label_mismatch(self):
        dispatch_state = {'worker_budgets': make_worker_budgets()}
        workers = [
            formal_1v3.WorkerSpec(
                kind='local',
                label='desktop',
                python=sys.executable,
            ),
            formal_1v3.WorkerSpec(
                kind='remote',
                label='laptop-renamed',
                python=sys.executable,
                host='mahjong-laptop',
                repo=r'C:\Users\numbe\Desktop\MahjongAI',
                ssh_key=r'C:\Users\numbe\.ssh\mahjong_laptop_ed25519',
            ),
        ]

        with self.assertRaisesRegex(RuntimeError, 'worker labels do not match this resume invocation'):
            formal_1v3.validate_resume_worker_labels(
                dispatch_state=dispatch_state,
                workers=workers,
            )

    def test_sync_remote_formal_shortlist_assets_rewrites_remote_state_paths(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            repo_root = Path(tmp_dir) / 'repo'
            run_dir = repo_root / 'logs' / 'sl_fidelity' / 'demo_run'
            ckpt_dir = repo_root / 'logs' / 'sl_ab' / 'formal_demo' / 'checkpoints'
            champion_dir = repo_root / 'checkpoints'
            run_dir.mkdir(parents=True, exist_ok=True)
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            champion_dir.mkdir(parents=True, exist_ok=True)
            best_loss = ckpt_dir / 'best_loss.pth'
            best_acc = ckpt_dir / 'best_acc.pth'
            champion = champion_dir / 'baseline.pth'
            best_loss.write_text('loss', encoding='utf-8')
            best_acc.write_text('acc', encoding='utf-8')
            champion.write_text('champ', encoding='utf-8')
            state = {
                'formal': {
                    'status': 'completed',
                    'result': {
                        'config_snapshot': {
                            'base_cfg': {'supervised': {}},
                            'base_1v3_cfg': {
                                'challenger': {'device': 'cuda:0', 'state_file': str(ckpt_dir / 'placeholder.pth')},
                                'champion': {'device': 'cuda:0', 'state_file': str(champion)},
                            },
                        },
                        'selected_protocol_arm': 'proto_arm',
                        'candidates': {
                            'best_loss': {'path': str(best_loss)},
                            'best_acc': {'path': str(best_acc)},
                        },
                    },
                }
            }
            state_path = run_dir / 'state.json'
            state_path.write_text(json.dumps(state, ensure_ascii=False), encoding='utf-8')
            copied_targets: list[tuple[Path, Path]] = []
            captured_remote_state: dict | None = None

            def capture_copy(worker, *, local_path, remote_path):
                nonlocal captured_remote_state
                copied_targets.append((Path(local_path), Path(remote_path)))
                if Path(remote_path).name == 'state.json':
                    captured_remote_state = json.loads(Path(local_path).read_text(encoding='utf-8'))

            worker = formal_1v3.WorkerSpec(
                kind='remote',
                label='laptop',
                python=r'C:\Python\python.exe',
                host='mahjong-laptop',
                repo=r'D:\MahjongRemote',
                ssh_key=r'C:\Users\numbe\.ssh\mahjong_laptop_ed25519',
            )
            context = {
                'state': state,
                'state_path': state_path,
                'base_1v3_cfg': state['formal']['result']['config_snapshot']['base_1v3_cfg'],
                'shortlist_candidates': state['formal']['result']['candidates'],
            }

            with (
                patch.object(formal_1v3, 'REPO_ROOT', repo_root),
                patch.object(formal_1v3, 'copy_local_file_to_remote', side_effect=capture_copy),
                patch.object(formal_1v3.fidelity, 'ts_now', return_value='2026-04-04 12:00:00'),
            ):
                payload = formal_1v3.sync_remote_formal_shortlist_assets(worker=worker, context=context)

        self.assertEqual(r'D:\MahjongRemote\logs\sl_fidelity\demo_run\state.json', payload['state_path'])
        self.assertEqual(r'D:\MahjongRemote\logs\sl_ab\formal_demo\checkpoints\best_loss.pth', payload['checkpoints']['best_loss'])
        self.assertEqual(r'D:\MahjongRemote\logs\sl_ab\formal_demo\checkpoints\best_acc.pth', payload['checkpoints']['best_acc'])
        self.assertEqual(r'D:\MahjongRemote\checkpoints\baseline.pth', payload['config_assets']['champion_state_file'])
        self.assertIsNotNone(captured_remote_state)
        self.assertEqual(
            r'D:\MahjongRemote\logs\sl_ab\formal_demo\checkpoints\best_loss.pth',
            captured_remote_state['formal']['result']['candidates']['best_loss']['path'],
        )
        self.assertEqual(
            r'D:\MahjongRemote\logs\sl_ab\formal_demo\checkpoints\best_acc.pth',
            captured_remote_state['formal']['result']['candidates']['best_acc']['path'],
        )
        self.assertEqual(
            r'D:\MahjongRemote\checkpoints\baseline.pth',
            captured_remote_state['formal']['result']['config_snapshot']['base_1v3_cfg']['champion']['state_file'],
        )
        self.assertEqual(4, len(copied_targets))

    def test_launch_task_for_worker_uses_remote_launch_mode(self):
        worker = formal_1v3.WorkerSpec(
            kind='remote',
            label='laptop',
            python=r'C:\Python\python.exe',
            host='mahjong-laptop',
            repo=r'C:\Users\numbe\Desktop\MahjongAI',
            ssh_key=r'C:\Users\numbe\.ssh\mahjong_laptop_ed25519',
        )
        task_state = {
            'task_id': 'seed1__coarse__best_acc__laptop__demo_run',
            'candidate_arm': 'best_acc',
            'round_index': 0,
            'round_label': 'coarse',
            'seed_key': 123,
            'iters': 1,
            'seed_count_per_iter': 640,
            'shard_count': 3,
        }

        with patch.object(formal_1v3, 'launch_remote_task', return_value='remote-active') as launch_remote_task:
            active = formal_1v3.launch_task_for_worker(
                worker=worker,
                run_name='demo_run',
                task_state=task_state,
                dispatch_root=Path(r'C:\Users\numbe\Desktop\MahjongAI\logs\sl_fidelity\demo_run\distributed\formal_1v3_dispatch'),
                launch_mode='interactive_window',
            )

        self.assertEqual('remote-active', active)
        launch_remote_task.assert_called_once()
        self.assertEqual('interactive_window', launch_remote_task.call_args.kwargs['launch_mode'])

    def test_evaluate_formal_candidate_uses_frozen_seed_and_shard_budget(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = Path(tmp_dir) / 'best_loss.pth'
            checkpoint_path.write_text('ckpt', encoding='utf-8')
            context = {
                'base_1v3_cfg': {
                    'challenger': {},
                    'disable_progress_bar': True,
                },
                'run_dir': Path(tmp_dir) / 'demo_run',
            }
            candidate = formal_1v3.build_formal_candidate(
                'best_loss',
                {'path': str(checkpoint_path)},
                protocol_arm='proto_arm',
            )

            with (
                patch.object(formal_1v3.one_vs_three, 'resolve_seed_count', side_effect=AssertionError('should not resolve seed count')),
                patch.object(formal_1v3.one_vs_three, 'resolve_shard_count', side_effect=AssertionError('should not resolve shard count')),
                patch.object(formal_1v3.one_vs_three, 'load_eval_engines', return_value='eval'),
                patch.object(formal_1v3.one_vs_three, 'run_eval_once', return_value=[2, 1, 1, 0]),
            ):
                payload = formal_1v3.evaluate_formal_candidate(
                    context=context,
                    candidate=candidate,
                    round_index=0,
                    round_label='coarse',
                    seed_key=7,
                    iters=1,
                    machine_label='desktop',
                    frozen_seed_count_per_iter=8,
                    frozen_shard_count=1,
                )

        self.assertEqual(8, payload['seed_count_per_iter'])
        self.assertEqual(1, payload['shard_count'])
        self.assertEqual('dispatch_frozen seed_count=8', payload['seed_count_source'])
        self.assertEqual('dispatch_frozen shard_count=1', payload['shard_count_source'])

    def test_evaluate_formal_candidate_uses_seed_start_offset_and_global_stride(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = Path(tmp_dir) / 'best_loss.pth'
            checkpoint_path.write_text('ckpt', encoding='utf-8')
            context = {
                'base_1v3_cfg': {
                    'challenger': {},
                    'disable_progress_bar': True,
                },
                'run_dir': Path(tmp_dir) / 'demo_run',
            }
            candidate = formal_1v3.build_formal_candidate(
                'best_loss',
                {'path': str(checkpoint_path)},
                protocol_arm='proto_arm',
            )
            seen_seed_starts: list[int] = []

            def capture_run_eval_once(**kwargs):
                seen_seed_starts.append(int(kwargs['seed_start']))
                return [1, 1, 1, 1]

            with (
                patch.object(formal_1v3.one_vs_three, 'resolve_seed_count', side_effect=AssertionError('should not resolve seed count')),
                patch.object(formal_1v3.one_vs_three, 'resolve_shard_count', side_effect=AssertionError('should not resolve shard count')),
                patch.object(formal_1v3.one_vs_three, 'load_eval_engines', return_value='eval'),
                patch.object(formal_1v3.one_vs_three, 'run_eval_once', side_effect=capture_run_eval_once),
            ):
                formal_1v3.evaluate_formal_candidate(
                    context=context,
                    candidate=candidate,
                    round_index=0,
                    round_label='coarse',
                    seed_key=7,
                    iters=2,
                    machine_label='laptop',
                    frozen_seed_count_per_iter=640,
                    frozen_shard_count=1,
                    seed_start_offset=1024,
                    seed_stride_per_iter=1664,
                )

        self.assertEqual([11024, 12688], seen_seed_starts)

    def test_evaluate_formal_candidate_reuses_single_shard_eval_context_across_iters(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = Path(tmp_dir) / 'best_loss.pth'
            checkpoint_path.write_text('ckpt', encoding='utf-8')
            context = {
                'base_1v3_cfg': {
                    'challenger': {},
                    'disable_progress_bar': True,
                },
                'run_dir': Path(tmp_dir) / 'demo_run',
            }
            candidate = formal_1v3.build_formal_candidate(
                'best_loss',
                {'path': str(checkpoint_path)},
                protocol_arm='proto_arm',
            )
            fake_eval_context = object()

            with (
                patch.object(formal_1v3.one_vs_three, 'resolve_seed_count', return_value=(8, 'resolved seed')),
                patch.object(formal_1v3.one_vs_three, 'resolve_shard_count', return_value=(1, 'resolved shard')),
                patch.object(formal_1v3.one_vs_three, 'load_eval_engines', return_value=fake_eval_context) as mock_load_eval_engines,
                patch.object(formal_1v3.one_vs_three, 'run_eval_once', return_value=[1, 1, 1, 1]) as mock_run_eval_once,
                patch.object(formal_1v3.one_vs_three, 'stop_persistent_shard_workers') as mock_stop_workers,
            ):
                formal_1v3.evaluate_formal_candidate(
                    context=context,
                    candidate=candidate,
                    round_index=0,
                    round_label='coarse',
                    seed_key=7,
                    iters=3,
                    machine_label='desktop',
                )

        mock_load_eval_engines.assert_called_once()
        self.assertEqual(3, mock_run_eval_once.call_count)
        for call in mock_run_eval_once.call_args_list:
            self.assertIs(fake_eval_context, call.kwargs['eval_context'])
        mock_stop_workers.assert_called_once_with(None)

    def test_evaluate_formal_candidate_sanitizes_runtime_name_and_log_dir(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = Path(tmp_dir) / 'best_loss.pth'
            checkpoint_path.write_text('ckpt', encoding='utf-8')
            context = {
                'base_1v3_cfg': {
                    'challenger': {},
                    'disable_progress_bar': True,
                },
                'run_dir': Path(tmp_dir) / 'demo_run',
            }
            candidate = formal_1v3.build_formal_candidate(
                'opp_lean(rank--/danger++)',
                {'path': str(checkpoint_path)},
                protocol_arm='proto_arm',
            )
            seen_cfgs: list[dict] = []

            def capture_load_eval_engines(cfg):
                seen_cfgs.append(cfg)
                return 'eval'

            with (
                patch.object(formal_1v3.one_vs_three, 'resolve_seed_count', return_value=(8, 'resolved seed')),
                patch.object(formal_1v3.one_vs_three, 'resolve_shard_count', return_value=(1, 'resolved shard')),
                patch.object(formal_1v3.one_vs_three, 'load_eval_engines', side_effect=capture_load_eval_engines),
                patch.object(formal_1v3.one_vs_three, 'run_eval_once', return_value=[1, 1, 1, 1]),
                patch.object(formal_1v3.one_vs_three, 'stop_persistent_shard_workers'),
            ):
                formal_1v3.evaluate_formal_candidate(
                    context=context,
                    candidate=candidate,
                    round_index=0,
                    round_label='close/01',
                    seed_key=7,
                    iters=1,
                    machine_label='lap*top',
                )

        self.assertEqual(1, len(seen_cfgs))
        cfg = seen_cfgs[0]
        self.assertNotIn('*', cfg['challenger']['name'])
        self.assertNotIn('/', cfg['challenger']['name'])
        self.assertNotIn('*', cfg['log_dir'])
        self.assertNotIn('/danger++', cfg['log_dir'])
        self.assertIn('close_01', cfg['log_dir'])

    def test_evaluate_formal_candidate_reuses_multi_shard_worker_pool_across_iters(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = Path(tmp_dir) / 'best_loss.pth'
            checkpoint_path.write_text('ckpt', encoding='utf-8')
            context = {
                'base_1v3_cfg': {
                    'challenger': {},
                    'disable_progress_bar': True,
                },
                'run_dir': Path(tmp_dir) / 'demo_run',
            }
            candidate = formal_1v3.build_formal_candidate(
                'best_loss',
                {'path': str(checkpoint_path)},
                protocol_arm='proto_arm',
            )
            fake_pool = {'mode': 'process'}

            with (
                patch.object(formal_1v3.one_vs_three, 'resolve_seed_count', return_value=(8, 'resolved seed')),
                patch.object(formal_1v3.one_vs_three, 'resolve_shard_count', return_value=(2, 'resolved shard')),
                patch.object(formal_1v3.one_vs_three, 'start_persistent_shard_workers', return_value=fake_pool) as mock_start_workers,
                patch.object(
                    formal_1v3.one_vs_three,
                    'run_sharded_iteration_with_workers',
                    return_value=([2, 1, 1, 0], 2.0, 45.0, Path(r'C:\tmp\runtime')),
                ) as mock_run_iteration,
                patch.object(formal_1v3.one_vs_three, 'stop_persistent_shard_workers') as mock_stop_workers,
                patch.object(formal_1v3.one_vs_three, 'run_sharded_iteration') as mock_run_sharded_iteration,
            ):
                formal_1v3.evaluate_formal_candidate(
                    context=context,
                    candidate=candidate,
                    round_index=0,
                    round_label='coarse',
                    seed_key=7,
                    iters=3,
                    machine_label='desktop',
                )

        mock_start_workers.assert_called_once()
        self.assertEqual(3, mock_run_iteration.call_count)
        for call in mock_run_iteration.call_args_list:
            self.assertIs(fake_pool, call.kwargs['shard_worker_pool'])
        mock_run_sharded_iteration.assert_not_called()
        mock_stop_workers.assert_called_once_with(fake_pool)

    def test_build_task_command_args_includes_seed_schedule(self):
        task_state = {
            'candidate_arm': 'best_acc',
            'round_index': 1,
            'round_label': 'close_01',
            'seed_key': 123,
            'iters': 2,
            'seed_count_per_iter': 640,
            'shard_count': 3,
            'seed_start_offset': 1024,
            'seed_stride_per_iter': 1664,
        }

        command = formal_1v3.build_task_command_args(
            run_name='demo_run',
            task_state=task_state,
            machine_label='laptop',
        )

        self.assertIn('--seed-start-offset', command)
        self.assertIn('1024', command)
        self.assertIn('--seed-stride-per-iter', command)
        self.assertIn('1664', command)

    def test_print_status_includes_close_call_and_final_winner(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir) / 'demo_run'
            run_dir.mkdir(parents=True, exist_ok=True)
            dispatch_state_path = formal_1v3.dispatch_state_path_for_run(run_dir)
            dispatch_state_path.parent.mkdir(parents=True, exist_ok=True)
            dispatch_state_path.write_text(
                json.dumps(
                    {
                        'stage': 'completed',
                        'status': 'completed',
                        'seed1': {'tasks': {}},
                        'seed2': {'tasks': {}},
                        'worker_budgets': make_worker_budgets(),
                        'latest_close_call': {'triggered': False},
                        'final_winner': 'best_acc',
                    },
                    ensure_ascii=False,
                ),
                encoding='utf-8',
            )
            control_path = formal_1v3.dispatch_control_path_for_run(run_dir)
            control_path.write_text(json.dumps({'workers': {}}, ensure_ascii=False), encoding='utf-8')
            args = argparse.Namespace(run_name='demo_run')
            stdout = StringIO()

            with (
                patch.object(formal_1v3.fidelity, 'FIDELITY_ROOT', Path(tmp_dir)),
                patch('sys.stdout', stdout),
            ):
                formal_1v3.print_status(args)

            rendered = json.loads(stdout.getvalue())
            self.assertEqual('best_acc', rendered['final_winner'])
            self.assertEqual({'triggered': False}, rendered['close_call'])

    def test_main_run_task_prints_compact_summary(self):
        args = argparse.Namespace(
            command='run-task',
            run_name='demo_run',
            candidate_arm='best_acc',
            round_index=1,
            round_label='close_01',
            seed_key=123,
            iters=1,
            machine_label='laptop',
            seed_count_per_iter=640,
            shard_count=3,
            result_json=r'C:\tmp\result.json',
        )
        stdout = StringIO()
        payload = {
            'round_kind': formal_1v3.ROUND_KIND_FORMAL_1V3,
            'run_name': 'demo_run',
            'candidate_arm': 'best_acc',
            'round_index': 1,
            'round_label': 'close_01',
            'seed_key': 123,
            'iters': 1,
            'machine_label': 'laptop',
            'completed_at': '2026-04-04 10:00:00',
            'summary': {'valid': True, 'games': 4096, 'avg_pt': 5.0, 'avg_rank': 2.2},
        }

        with (
            patch.object(formal_1v3, 'parse_args', return_value=args),
            patch.object(formal_1v3, 'execute_single_task', return_value=payload),
            patch('sys.stdout', stdout),
        ):
            formal_1v3.main()

        rendered = json.loads(stdout.getvalue())
        self.assertEqual('best_acc', rendered['candidate_arm'])
        self.assertEqual('close_01', rendered['round_label'])
        self.assertEqual(5.0, rendered['avg_pt'])

    def test_main_resolve_budget_prints_payload(self):
        args = argparse.Namespace(command='resolve-budget', machine_label='laptop', cfg_json_b64=None)
        stdout = StringIO()

        with (
            patch.object(formal_1v3, 'parse_args', return_value=args),
            patch.object(formal_1v3.formal, 'build_formal_config_snapshot', return_value={'base_1v3_cfg': {'challenger': {'device': 'cuda:0'}}}),
            patch.object(formal_1v3, 'resolve_worker_budget_snapshot', return_value={'machine_label': 'laptop', 'seed_count_per_iter': 640}),
            patch('sys.stdout', stdout),
        ):
            formal_1v3.main()

        rendered = json.loads(stdout.getvalue())
        self.assertEqual('laptop', rendered['machine_label'])
        self.assertEqual(640, rendered['seed_count_per_iter'])


if __name__ == '__main__':
    unittest.main()
