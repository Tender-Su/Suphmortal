import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_stage05_ab as stage05_ab
import run_stage05_formal as stage05_formal
import run_stage05_loader_ab as stage05_loader_ab


class RunStage05FormalTests(unittest.TestCase):
    def test_current_primary_protocol_arm_matches_loader_default(self):
        self.assertEqual(
            stage05_loader_ab.DEFAULT_PROTOCOL_ARM,
            stage05_formal.CURRENT_PRIMARY_PROTOCOL_ARM,
        )

    def test_formal_defaults_run_full_validation_every_monitor_check(self):
        original = dict(stage05_ab.BASE_SCREENING)
        try:
            stage05_formal.apply_formal_defaults()
            self.assertEqual(1, stage05_ab.BASE_SCREENING['full_val_every_checks'])
        finally:
            stage05_ab.BASE_SCREENING.clear()
            stage05_ab.BASE_SCREENING.update(original)

    def test_stage05_publish_plan_updates_canonical_seed_with_selected_winner(self):
        base_cfg = {
            'supervised': {
                'state_file': './checkpoints/stage0_5_latest.pth',
                'best_state_file': './checkpoints/stage0_5_supervised.pth',
                'best_loss_state_file': './checkpoints/stage0_5_supervised.pth',
                'best_acc_state_file': './checkpoints/stage0_5_supervised_best_acc.pth',
                'best_rank_state_file': './checkpoints/stage0_5_supervised_best_rank.pth',
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

        plan = stage05_formal.stage05_publish_plan(
            base_cfg,
            payload,
            config_dir=Path('C:/virtual/mortal'),
        )

        publish_map = {str(destination): str(source) for destination, source in plan}
        self.assertEqual(
            'C:\\virtual\\source\\latest.pth',
            publish_map['C:\\virtual\\mortal\\checkpoints\\stage0_5_latest.pth'],
        )
        self.assertEqual(
            'C:\\virtual\\source\\best_acc.pth',
            publish_map['C:\\virtual\\mortal\\checkpoints\\stage0_5_supervised.pth'],
        )
        self.assertEqual(
            'C:\\virtual\\source\\best_rank.pth',
            publish_map['C:\\virtual\\mortal\\checkpoints\\stage0_5_supervised_best_rank.pth'],
        )

    def test_stage05_publish_plan_prefers_winner_when_latest_and_canonical_paths_collide(self):
        base_cfg = {
            'supervised': {
                'state_file': './checkpoints/stage0_5_supervised.pth',
                'best_state_file': './checkpoints/stage0_5_supervised.pth',
                'best_loss_state_file': './checkpoints/stage0_5_supervised.pth',
                'best_acc_state_file': './checkpoints/stage0_5_supervised_best_acc.pth',
                'best_rank_state_file': './checkpoints/stage0_5_supervised_best_rank.pth',
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

        plan = stage05_formal.stage05_publish_plan(
            base_cfg,
            payload,
            config_dir=Path('C:/virtual/mortal'),
        )

        publish_map = {str(destination): str(source) for destination, source in plan}
        self.assertEqual(
            'C:\\virtual\\source\\best_acc.pth',
            publish_map['C:\\virtual\\mortal\\checkpoints\\stage0_5_supervised.pth'],
        )

    def test_stage05_publish_plan_keeps_split_best_loss_alias_on_winner(self):
        base_cfg = {
            'supervised': {
                'state_file': './checkpoints/stage0_5_latest.pth',
                'best_state_file': './checkpoints/stage0_5_supervised.pth',
                'best_loss_state_file': './checkpoints/stage0_5_supervised_best_loss.pth',
                'best_acc_state_file': './checkpoints/stage0_5_supervised_best_acc.pth',
                'best_rank_state_file': './checkpoints/stage0_5_supervised_best_rank.pth',
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

        plan = stage05_formal.stage05_publish_plan(
            base_cfg,
            payload,
            config_dir=Path('C:/virtual/mortal'),
        )

        publish_map = {str(destination): str(source) for destination, source in plan}
        self.assertEqual(
            'C:\\virtual\\source\\best_rank.pth',
            publish_map['C:\\virtual\\mortal\\checkpoints\\stage0_5_supervised.pth'],
        )
        self.assertEqual(
            'C:\\virtual\\source\\best_rank.pth',
            publish_map['C:\\virtual\\mortal\\checkpoints\\stage0_5_supervised_best_loss.pth'],
        )

    def test_publish_stage05_canonical_checkpoints_copies_selected_outputs(self):
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
                    'state_file': './checkpoints/stage0_5_latest.pth',
                    'best_state_file': './checkpoints/stage0_5_supervised.pth',
                    'best_loss_state_file': './checkpoints/stage0_5_supervised.pth',
                    'best_acc_state_file': './checkpoints/stage0_5_supervised_best_acc.pth',
                    'best_rank_state_file': './checkpoints/stage0_5_supervised_best_rank.pth',
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

            published = stage05_formal.publish_stage05_canonical_checkpoints(
                base_cfg,
                payload,
                config_dir=config_dir,
            )

            self.assertEqual(4, len(published))
            self.assertEqual(
                'latest',
                (config_dir / 'checkpoints' / 'stage0_5_latest.pth').read_text(encoding='utf-8'),
            )
            self.assertEqual(
                'best_acc',
                (config_dir / 'checkpoints' / 'stage0_5_supervised.pth').read_text(encoding='utf-8'),
            )
            published_sources = {
                Path(item['destination']).name: Path(item['source']).read_text(encoding='utf-8')
                for item in published
            }
            self.assertEqual('best_acc', published_sources['stage0_5_supervised.pth'])

    def test_publish_stage05_canonical_checkpoints_copies_winner_to_split_best_loss_alias(self):
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
                    'state_file': './checkpoints/stage0_5_latest.pth',
                    'best_state_file': './checkpoints/stage0_5_supervised.pth',
                    'best_loss_state_file': './checkpoints/stage0_5_supervised_best_loss.pth',
                    'best_acc_state_file': './checkpoints/stage0_5_supervised_best_acc.pth',
                    'best_rank_state_file': './checkpoints/stage0_5_supervised_best_rank.pth',
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

            published = stage05_formal.publish_stage05_canonical_checkpoints(
                base_cfg,
                payload,
                config_dir=config_dir,
            )

            self.assertEqual(5, len(published))
            self.assertEqual(
                'best_rank',
                (config_dir / 'checkpoints' / 'stage0_5_supervised.pth').read_text(encoding='utf-8'),
            )
            self.assertEqual(
                'best_rank',
                (config_dir / 'checkpoints' / 'stage0_5_supervised_best_loss.pth').read_text(encoding='utf-8'),
            )

    def test_publish_stage05_canonical_checkpoints_skips_non_primary_protocol_arm(self):
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
                    'state_file': './checkpoints/stage0_5_latest.pth',
                    'best_state_file': './checkpoints/stage0_5_supervised.pth',
                    'best_loss_state_file': './checkpoints/stage0_5_supervised.pth',
                    'best_acc_state_file': './checkpoints/stage0_5_supervised_best_acc.pth',
                    'best_rank_state_file': './checkpoints/stage0_5_supervised_best_rank.pth',
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
                for arm_name in stage05_formal.PROTOCOL_ARM_MAP
                if arm_name != stage05_formal.CURRENT_PRIMARY_PROTOCOL_ARM
            )

            published = stage05_formal.publish_stage05_canonical_checkpoints(
                base_cfg,
                payload,
                config_dir=config_dir,
                protocol_arm=non_primary_arm,
            )

            self.assertEqual([], published)
            self.assertFalse((config_dir / 'checkpoints' / 'stage0_5_latest.pth').exists())
            self.assertFalse((config_dir / 'checkpoints' / 'stage0_5_supervised.pth').exists())

    def test_finalize_formal_result_adds_metadata_and_published_outputs(self):
        result = {
            'winner': 'best_acc',
            'candidates': {
                'latest': {'path': 'C:/virtual/source/latest.pth'},
                'best_loss': {'path': 'C:/virtual/source/best_loss.pth'},
                'best_acc': {'path': 'C:/virtual/source/best_acc.pth'},
                'best_rank': {'path': 'C:/virtual/source/best_rank.pth'},
            },
        }

        with patch.object(
            stage05_formal,
            'publish_stage05_canonical_checkpoints',
            return_value=[{'source': 'src', 'destination': 'dst'}],
        ) as publish_mock:
            finalized = stage05_formal.finalize_formal_result(
                {'supervised': {}},
                result,
                protocol_arm=stage05_formal.CURRENT_PRIMARY_PROTOCOL_ARM,
            )

        publish_mock.assert_called_once_with(
            {'supervised': {}},
            result,
            config_dir=None,
            protocol_arm=stage05_formal.CURRENT_PRIMARY_PROTOCOL_ARM,
        )
        self.assertIs(result, finalized)
        self.assertEqual(stage05_formal.CURRENT_PRIMARY_PROTOCOL_ARM, finalized['selected_protocol_arm'])
        self.assertEqual(
            list(stage05_formal.CURRENT_STAGE1_TOP_PROTOCOL_ARMS),
            finalized['current_stage1_top_protocol_arms'],
        )
        self.assertEqual([{'source': 'src', 'destination': 'dst'}], finalized['published_canonical_checkpoints'])


if __name__ == '__main__':
    unittest.main()
