import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_sl_ab as sl_ab
import run_sl_formal as sl_formal
import run_sl_loader_ab as sl_loader_ab


class RunStage05FormalTests(unittest.TestCase):
    def test_current_primary_protocol_arm_matches_loader_default(self):
        self.assertEqual(
            sl_loader_ab.DEFAULT_PROTOCOL_ARM,
            sl_formal.CURRENT_PRIMARY_PROTOCOL_ARM,
        )

    def test_formal_defaults_run_full_validation_every_monitor_check(self):
        original = dict(sl_ab.BASE_SCREENING)
        try:
            sl_formal.apply_formal_defaults()
            self.assertEqual(1, sl_ab.BASE_SCREENING['full_val_every_checks'])
        finally:
            sl_ab.BASE_SCREENING.clear()
            sl_ab.BASE_SCREENING.update(original)

    def test_formal_defaults_use_extended_phase_steps(self):
        self.assertEqual(
            {'phase_a': 9000, 'phase_b': 6000, 'phase_c': 3000},
            sl_formal.FORMAL_DEFAULTS['phase_steps'],
        )

    def test_sl_publish_plan_updates_canonical_seed_with_selected_winner(self):
        base_cfg = {
            'supervised': {
                'state_file': './checkpoints/sl_latest.pth',
                'best_state_file': './checkpoints/sl_canonical.pth',
                'best_loss_state_file': './checkpoints/sl_canonical.pth',
                'best_acc_state_file': './checkpoints/sl_best_acc.pth',
                'best_rank_state_file': './checkpoints/sl_best_rank.pth',
            },
        }
        payload = {
            'winner': 'best_acc',
            'candidates': {
                'latest': {'path': 'C:/virtual/source/latest.pth'},
                'best_loss': {'path': 'C:/virtual/source/best_loss.pth'},
                'best_acc': {'path': 'C:/virtual/source/best_acc.pth'},
                'best_rank': {'path': 'C:/virtual/source/best_rank.pth'},
            },
        }

        plan = sl_formal.sl_publish_plan(
            base_cfg,
            payload,
            config_dir=Path('C:/virtual/mortal'),
        )

        publish_map = {str(destination): str(source) for destination, source in plan}
        self.assertEqual(
            'C:\\virtual\\source\\best_acc.pth',
            publish_map['C:\\virtual\\mortal\\checkpoints\\sl_latest.pth'],
        )
        self.assertEqual(
            'C:\\virtual\\source\\best_acc.pth',
            publish_map['C:\\virtual\\mortal\\checkpoints\\sl_canonical.pth'],
        )
        self.assertEqual(
            'C:\\virtual\\source\\best_rank.pth',
            publish_map['C:\\virtual\\mortal\\checkpoints\\sl_best_rank.pth'],
        )

    def test_sl_publish_plan_prefers_winner_when_latest_and_canonical_paths_collide(self):
        base_cfg = {
            'supervised': {
                'state_file': './checkpoints/sl_canonical.pth',
                'best_state_file': './checkpoints/sl_canonical.pth',
                'best_loss_state_file': './checkpoints/sl_canonical.pth',
                'best_acc_state_file': './checkpoints/sl_best_acc.pth',
                'best_rank_state_file': './checkpoints/sl_best_rank.pth',
            },
        }
        payload = {
            'winner': 'best_acc',
            'candidates': {
                'latest': {'path': 'C:/virtual/source/latest.pth'},
                'best_loss': {'path': 'C:/virtual/source/best_loss.pth'},
                'best_acc': {'path': 'C:/virtual/source/best_acc.pth'},
                'best_rank': {'path': 'C:/virtual/source/best_rank.pth'},
            },
        }

        plan = sl_formal.sl_publish_plan(
            base_cfg,
            payload,
            config_dir=Path('C:/virtual/mortal'),
        )

        publish_map = {str(destination): str(source) for destination, source in plan}
        self.assertEqual(
            'C:\\virtual\\source\\best_acc.pth',
            publish_map['C:\\virtual\\mortal\\checkpoints\\sl_canonical.pth'],
        )

    def test_sl_publish_plan_keeps_split_best_loss_alias_on_winner(self):
        base_cfg = {
            'supervised': {
                'state_file': './checkpoints/sl_latest.pth',
                'best_state_file': './checkpoints/sl_canonical.pth',
                'best_loss_state_file': './checkpoints/sl_best_loss.pth',
                'best_acc_state_file': './checkpoints/sl_best_acc.pth',
                'best_rank_state_file': './checkpoints/sl_best_rank.pth',
            },
        }
        payload = {
            'winner': 'best_rank',
            'candidates': {
                'latest': {'path': 'C:/virtual/source/latest.pth'},
                'best_loss': {'path': 'C:/virtual/source/best_loss.pth'},
                'best_acc': {'path': 'C:/virtual/source/best_acc.pth'},
                'best_rank': {'path': 'C:/virtual/source/best_rank.pth'},
            },
        }

        plan = sl_formal.sl_publish_plan(
            base_cfg,
            payload,
            config_dir=Path('C:/virtual/mortal'),
        )

        publish_map = {str(destination): str(source) for destination, source in plan}
        self.assertEqual(
            'C:\\virtual\\source\\best_rank.pth',
            publish_map['C:\\virtual\\mortal\\checkpoints\\sl_canonical.pth'],
        )
        self.assertEqual(
            'C:\\virtual\\source\\best_rank.pth',
            publish_map['C:\\virtual\\mortal\\checkpoints\\sl_best_loss.pth'],
        )

    def test_publish_sl_canonical_checkpoints_copies_selected_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            source_dir = tmpdir / 'source'
            config_dir = tmpdir / 'config'
            source_dir.mkdir()
            config_dir.mkdir()

            latest = source_dir / 'latest.pth'
            best_loss = source_dir / 'best_loss.pth'
            best_acc = source_dir / 'best_acc.pth'
            best_rank = source_dir / 'best_rank.pth'
            latest.write_text('latest', encoding='utf-8')
            best_loss.write_text('best_loss', encoding='utf-8')
            best_acc.write_text('best_acc', encoding='utf-8')
            best_rank.write_text('best_rank', encoding='utf-8')

            base_cfg = {
                'supervised': {
                    'state_file': './checkpoints/sl_latest.pth',
                    'best_state_file': './checkpoints/sl_canonical.pth',
                    'best_loss_state_file': './checkpoints/sl_canonical.pth',
                    'best_acc_state_file': './checkpoints/sl_best_acc.pth',
                    'best_rank_state_file': './checkpoints/sl_best_rank.pth',
                },
            }
            payload = {
                'winner': 'best_acc',
                'candidates': {
                    'latest': {'path': str(latest)},
                    'best_loss': {'path': str(best_loss)},
                    'best_acc': {'path': str(best_acc)},
                    'best_rank': {'path': str(best_rank)},
                },
            }

            published = sl_formal.publish_sl_canonical_checkpoints(
                base_cfg,
                payload,
                config_dir=config_dir,
            )

            self.assertEqual(4, len(published))
            self.assertEqual(
                'best_acc',
                (config_dir / 'checkpoints' / 'sl_latest.pth').read_text(encoding='utf-8'),
            )
            self.assertEqual(
                'best_acc',
                (config_dir / 'checkpoints' / 'sl_canonical.pth').read_text(encoding='utf-8'),
            )
            published_sources = {
                Path(item['destination']).name: Path(item['source']).read_text(encoding='utf-8')
                for item in published
            }
            self.assertEqual('best_acc', published_sources['sl_canonical.pth'])

    def test_publish_sl_canonical_checkpoints_copies_winner_to_split_best_loss_alias(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            source_dir = tmpdir / 'source'
            config_dir = tmpdir / 'config'
            source_dir.mkdir()
            config_dir.mkdir()

            latest = source_dir / 'latest.pth'
            best_loss = source_dir / 'best_loss.pth'
            best_acc = source_dir / 'best_acc.pth'
            best_rank = source_dir / 'best_rank.pth'
            latest.write_text('latest', encoding='utf-8')
            best_loss.write_text('best_loss', encoding='utf-8')
            best_acc.write_text('best_acc', encoding='utf-8')
            best_rank.write_text('best_rank', encoding='utf-8')

            base_cfg = {
                'supervised': {
                    'state_file': './checkpoints/sl_latest.pth',
                    'best_state_file': './checkpoints/sl_canonical.pth',
                    'best_loss_state_file': './checkpoints/sl_best_loss.pth',
                    'best_acc_state_file': './checkpoints/sl_best_acc.pth',
                    'best_rank_state_file': './checkpoints/sl_best_rank.pth',
                },
            }
            payload = {
                'winner': 'best_rank',
                'candidates': {
                    'latest': {'path': str(latest)},
                    'best_loss': {'path': str(best_loss)},
                    'best_acc': {'path': str(best_acc)},
                    'best_rank': {'path': str(best_rank)},
                },
            }

            published = sl_formal.publish_sl_canonical_checkpoints(
                base_cfg,
                payload,
                config_dir=config_dir,
            )

            self.assertEqual(5, len(published))
            self.assertEqual(
                'best_rank',
                (config_dir / 'checkpoints' / 'sl_canonical.pth').read_text(encoding='utf-8'),
            )
            self.assertEqual(
                'best_rank',
                (config_dir / 'checkpoints' / 'sl_best_loss.pth').read_text(encoding='utf-8'),
            )

    def test_publish_sl_canonical_checkpoints_skips_non_frozen_primary_protocol_arm(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            source_dir = tmpdir / 'source'
            config_dir = tmpdir / 'config'
            source_dir.mkdir()
            config_dir.mkdir()

            latest = source_dir / 'latest.pth'
            best_loss = source_dir / 'best_loss.pth'
            best_acc = source_dir / 'best_acc.pth'
            best_rank = source_dir / 'best_rank.pth'
            latest.write_text('latest', encoding='utf-8')
            best_loss.write_text('best_loss', encoding='utf-8')
            best_acc.write_text('best_acc', encoding='utf-8')
            best_rank.write_text('best_rank', encoding='utf-8')

            base_cfg = {
                'supervised': {
                    'state_file': './checkpoints/sl_latest.pth',
                    'best_state_file': './checkpoints/sl_canonical.pth',
                    'best_loss_state_file': './checkpoints/sl_canonical.pth',
                    'best_acc_state_file': './checkpoints/sl_best_acc.pth',
                    'best_rank_state_file': './checkpoints/sl_best_rank.pth',
                },
            }
            payload = {
                'winner': 'best_acc',
                'candidates': {
                    'latest': {'path': str(latest)},
                    'best_loss': {'path': str(best_loss)},
                    'best_acc': {'path': str(best_acc)},
                    'best_rank': {'path': str(best_rank)},
                },
            }
            non_primary_arm = next(
                arm_name
                for arm_name in sl_formal.PROTOCOL_ARM_MAP
                if arm_name != sl_formal.CURRENT_PRIMARY_PROTOCOL_ARM
            )

            published = sl_formal.publish_sl_canonical_checkpoints(
                base_cfg,
                payload,
                config_dir=config_dir,
                protocol_arm=non_primary_arm,
                primary_protocol_arm=sl_formal.CURRENT_PRIMARY_PROTOCOL_ARM,
            )

            self.assertEqual([], published)
            self.assertFalse((config_dir / 'checkpoints' / 'sl_latest.pth').exists())
            self.assertFalse((config_dir / 'checkpoints' / 'sl_canonical.pth').exists())

    def test_publish_sl_canonical_checkpoints_uses_frozen_primary_protocol_arm(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            source_dir = tmpdir / 'source'
            config_dir = tmpdir / 'config'
            source_dir.mkdir()
            config_dir.mkdir()

            latest = source_dir / 'latest.pth'
            best_loss = source_dir / 'best_loss.pth'
            best_acc = source_dir / 'best_acc.pth'
            best_rank = source_dir / 'best_rank.pth'
            latest.write_text('latest', encoding='utf-8')
            best_loss.write_text('best_loss', encoding='utf-8')
            best_acc.write_text('best_acc', encoding='utf-8')
            best_rank.write_text('best_rank', encoding='utf-8')

            base_cfg = {
                'supervised': {
                    'state_file': './checkpoints/sl_latest.pth',
                    'best_state_file': './checkpoints/sl_canonical.pth',
                    'best_loss_state_file': './checkpoints/sl_canonical.pth',
                    'best_acc_state_file': './checkpoints/sl_best_acc.pth',
                    'best_rank_state_file': './checkpoints/sl_best_rank.pth',
                },
            }
            payload = {
                'winner': 'best_acc',
                'candidates': {
                    'latest': {'path': str(latest)},
                    'best_loss': {'path': str(best_loss)},
                    'best_acc': {'path': str(best_acc)},
                    'best_rank': {'path': str(best_rank)},
                },
            }
            frozen_primary_arm = next(
                arm_name
                for arm_name in sl_formal.PROTOCOL_ARM_MAP
                if arm_name != sl_formal.CURRENT_PRIMARY_PROTOCOL_ARM
            )

            published = sl_formal.publish_sl_canonical_checkpoints(
                base_cfg,
                payload,
                config_dir=config_dir,
                protocol_arm=frozen_primary_arm,
                primary_protocol_arm=frozen_primary_arm,
            )

            self.assertEqual(4, len(published))
            self.assertEqual(
                'best_acc',
                (config_dir / 'checkpoints' / 'sl_latest.pth').read_text(encoding='utf-8'),
            )
            self.assertEqual(
                'best_acc',
                (config_dir / 'checkpoints' / 'sl_canonical.pth').read_text(encoding='utf-8'),
            )

    def test_finalize_formal_result_adds_metadata_and_published_outputs(self):
        base_cfg = {
            'supervised': {
                'state_file': './checkpoints/sl_latest.pth',
                'best_state_file': './checkpoints/sl_canonical.pth',
                'best_loss_state_file': './checkpoints/sl_best_loss.pth',
                'best_acc_state_file': './checkpoints/sl_best_acc.pth',
                'best_rank_state_file': './checkpoints/sl_best_rank.pth',
            },
            '1v3': {
                'challenger': {
                    'device': 'cuda:0',
                    'state_file': './checkpoints/challenger.pth',
                },
                'champion': {
                    'device': 'cuda:0',
                    'state_file': './checkpoints/baseline.pth',
                },
            },
        }
        result = {
            'winner': 'best_acc',
            'candidates': {
                'latest': {'path': 'C:/virtual/source/latest.pth'},
                'best_loss': {'path': 'C:/virtual/source/best_loss.pth'},
                'best_acc': {'path': 'C:/virtual/source/best_acc.pth'},
                'best_rank': {'path': 'C:/virtual/source/best_rank.pth'},
            },
        }

        finalized = sl_formal.finalize_formal_result(
            base_cfg,
            result,
            protocol_arm=sl_formal.CURRENT_PRIMARY_PROTOCOL_ARM,
            config_path=Path('C:/virtual/mortal/config.toml'),
        )

        self.assertIs(result, finalized)
        self.assertEqual('best_acc', finalized['offline_checkpoint_winner'])
        self.assertEqual('best_acc', finalized['checkpoint_pack_winner'])
        self.assertEqual(
            ['best_loss', 'best_acc', 'best_rank'],
            finalized['shortlist_checkpoint_types'],
        )
        self.assertEqual(
            ['best_loss', 'best_acc', 'best_rank'],
            finalized['checkpoint_pack_types'],
        )
        self.assertTrue(finalized['latest_discarded'])
        self.assertTrue(finalized['publish_pending'])
        self.assertTrue(finalized['canonical_alias_sync_pending'])
        self.assertEqual(
            'C:\\virtual\\mortal\\config.toml',
            finalized['config_snapshot']['config_path'],
        )
        self.assertEqual(
            'C:\\virtual\\mortal\\checkpoints\\baseline.pth',
            finalized['config_snapshot']['base_1v3_cfg']['champion']['state_file'],
        )
        self.assertEqual(
            [
                'C:\\virtual\\mortal\\checkpoints\\sl_latest.pth',
                'C:\\virtual\\mortal\\checkpoints\\sl_best_acc.pth',
                'C:\\virtual\\mortal\\checkpoints\\sl_best_rank.pth',
                'C:\\virtual\\mortal\\checkpoints\\sl_canonical.pth',
                'C:\\virtual\\mortal\\checkpoints\\sl_best_loss.pth',
            ],
            finalized['pending_canonical_alias_targets'],
        )
        self.assertEqual(
            {'best_loss', 'best_acc', 'best_rank'},
            set(finalized['candidates']),
        )
        self.assertEqual(sl_formal.CURRENT_PRIMARY_PROTOCOL_ARM, finalized['selected_protocol_arm'])
        self.assertEqual(
            list(sl_formal.CURRENT_SUPERVISED_TOP_PROTOCOL_ARMS),
            finalized['current_supervised_top_protocol_arms'],
        )

    def test_finalize_formal_result_keeps_latest_offline_winner_but_falls_back_to_best_loss_pack_winner(self):
        base_cfg = {
            'supervised': {
                'state_file': './checkpoints/sl_latest.pth',
                'best_state_file': './checkpoints/sl_canonical.pth',
                'best_loss_state_file': './checkpoints/sl_best_loss.pth',
                'best_acc_state_file': './checkpoints/sl_best_acc.pth',
                'best_rank_state_file': './checkpoints/sl_best_rank.pth',
            },
        }
        result = {
            'winner': 'latest',
            'candidates': {
                'latest': {'path': 'C:/virtual/source/latest.pth'},
                'best_loss': {'path': 'C:/virtual/source/best_loss.pth'},
                'best_acc': {'path': 'C:/virtual/source/best_acc.pth'},
                'best_rank': {'path': 'C:/virtual/source/best_rank.pth'},
            },
        }

        finalized = sl_formal.finalize_formal_result(
            base_cfg,
            result,
            protocol_arm=sl_formal.CURRENT_PRIMARY_PROTOCOL_ARM,
            config_path=Path('C:/virtual/mortal/config.toml'),
        )

        self.assertEqual('latest', finalized['offline_checkpoint_winner'])
        self.assertEqual('best_loss', finalized['checkpoint_pack_winner'])
        self.assertEqual(
            {'best_loss', 'best_acc', 'best_rank'},
            set(finalized['candidates']),
        )
        self.assertTrue(finalized['pending_canonical_alias_targets'])

    def test_build_formal_shortlist_candidates_requires_three_formal_checkpoints(self):
        with self.assertRaisesRegex(RuntimeError, 'missing required checkpoint types'):
            sl_formal.build_formal_shortlist_candidates(
                {
                    'candidates': {
                        'best_loss': {'path': 'C:/virtual/source/best_loss.pth'},
                        'best_acc': {'path': 'C:/virtual/source/best_acc.pth'},
                    }
                }
            )

    def test_ensure_supervised_canonical_handoff_ready_blocks_pending_formal_1v3(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            alias_path = tmpdir / 'mortal' / 'checkpoints' / 'sl_canonical.pth'
            alias_path.parent.mkdir(parents=True, exist_ok=True)
            fidelity_root = tmpdir / 'logs' / 'sl_fidelity'
            run_dir = fidelity_root / 'demo_run'
            run_dir.mkdir(parents=True, exist_ok=True)
            state = {
                'formal': {
                    'status': 'completed',
                    'result': {
                        'pending_canonical_alias_targets': [str(alias_path)],
                    },
                },
                'final_conclusion': {
                    'formal_status': 'pending_1v3',
                    'formal_1v3_status': 'pending',
                },
            }
            (run_dir / 'state.json').write_text(
                json.dumps(state, ensure_ascii=False),
                encoding='utf-8',
            )

            with patch.object(sl_formal, 'SL_FIDELITY_ROOT', fidelity_root):
                with self.assertRaisesRegex(RuntimeError, 'pending supervised formal_1v3 handoff'):
                    sl_formal.ensure_supervised_canonical_handoff_ready(str(alias_path))

    def test_ensure_supervised_canonical_handoff_ready_blocks_failed_formal_1v3_before_publish(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            alias_path = tmpdir / 'mortal' / 'checkpoints' / 'sl_canonical.pth'
            alias_path.parent.mkdir(parents=True, exist_ok=True)
            fidelity_root = tmpdir / 'logs' / 'sl_fidelity'
            run_dir = fidelity_root / 'demo_run'
            run_dir.mkdir(parents=True, exist_ok=True)
            state = {
                'formal': {
                    'status': 'completed',
                    'result': {
                        'pending_canonical_alias_targets': [str(alias_path)],
                    },
                },
                'final_conclusion': {
                    'formal_status': 'pending_1v3',
                    'formal_1v3_status': 'failed',
                },
            }
            (run_dir / 'state.json').write_text(
                json.dumps(state, ensure_ascii=False),
                encoding='utf-8',
            )

            with patch.object(sl_formal, 'SL_FIDELITY_ROOT', fidelity_root):
                with self.assertRaisesRegex(RuntimeError, 'pending supervised formal_1v3 handoff'):
                    sl_formal.ensure_supervised_canonical_handoff_ready(str(alias_path))

    def test_ensure_supervised_canonical_handoff_ready_blocks_completed_formal_1v3_without_publish_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            alias_path = tmpdir / 'mortal' / 'checkpoints' / 'sl_canonical.pth'
            alias_path.parent.mkdir(parents=True, exist_ok=True)
            fidelity_root = tmpdir / 'logs' / 'sl_fidelity'
            run_dir = fidelity_root / 'demo_run'
            run_dir.mkdir(parents=True, exist_ok=True)
            state = {
                'formal': {
                    'status': 'completed',
                    'result': {
                        'pending_canonical_alias_targets': [str(alias_path)],
                    },
                },
                'formal_1v3': {
                    'published_canonical_checkpoints': [],
                },
                'final_conclusion': {
                    'formal_status': 'completed',
                    'formal_1v3_status': 'completed',
                },
            }
            (run_dir / 'state.json').write_text(
                json.dumps(state, ensure_ascii=False),
                encoding='utf-8',
            )

            with patch.object(sl_formal, 'SL_FIDELITY_ROOT', fidelity_root):
                with self.assertRaisesRegex(RuntimeError, 'pending supervised formal_1v3 handoff'):
                    sl_formal.ensure_supervised_canonical_handoff_ready(str(alias_path))

    def test_ensure_supervised_canonical_handoff_ready_ignores_older_pending_after_newer_completion(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmpdir = Path(tmp)
            alias_path = tmpdir / 'mortal' / 'checkpoints' / 'sl_canonical.pth'
            alias_path.parent.mkdir(parents=True, exist_ok=True)
            fidelity_root = tmpdir / 'logs' / 'sl_fidelity'
            old_run_dir = fidelity_root / 'old_pending_run'
            new_run_dir = fidelity_root / 'new_completed_run'
            old_run_dir.mkdir(parents=True, exist_ok=True)
            new_run_dir.mkdir(parents=True, exist_ok=True)
            old_state_path = old_run_dir / 'state.json'
            new_state_path = new_run_dir / 'state.json'
            old_state_path.write_text(
                json.dumps(
                    {
                        'formal': {
                            'status': 'completed',
                            'result': {
                                'pending_canonical_alias_targets': [str(alias_path)],
                            },
                        },
                        'final_conclusion': {
                            'formal_status': 'pending_1v3',
                            'formal_1v3_status': 'pending',
                        },
                    },
                    ensure_ascii=False,
                ),
                encoding='utf-8',
            )
            new_state_path.write_text(
                json.dumps(
                    {
                        'formal': {
                            'status': 'completed',
                            'result': {
                                'pending_canonical_alias_targets': [str(alias_path)],
                            },
                        },
                        'final_conclusion': {
                            'formal_status': 'completed',
                            'formal_1v3_status': 'completed',
                        },
                        'formal_1v3': {
                            'published_canonical_checkpoints': [
                                {'destination': str(alias_path), 'source': str(tmpdir / 'winner.pth')},
                            ],
                        },
                    },
                    ensure_ascii=False,
                ),
                encoding='utf-8',
            )
            os.utime(old_state_path, (1, 1))
            os.utime(new_state_path, (2, 2))

            with patch.object(sl_formal, 'SL_FIDELITY_ROOT', fidelity_root):
                sl_formal.ensure_supervised_canonical_handoff_ready(str(alias_path))


if __name__ == '__main__':
    unittest.main()
