import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, Mock

from scripts.run_sl_rl_repair_pipeline import main, sha256


class RepairPipelineTests(unittest.TestCase):
    def test_manifest_rejects_changed_input_and_preserves_pause_exit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, config, checkpoint = (root / name for name in ('source.py', 'config.toml', 'anchor.pth'))
            for file in (source, config, checkpoint):
                file.write_bytes(b'frozen')
            manifest = root / 'manifest.json'
            manifest.write_text(json.dumps({
                'source_root': str(root), 'source_sha256': {source.name: sha256(source)},
                'config': {'path': str(config), 'sha256': sha256(config)},
                'initial_checkpoint': {'path': str(checkpoint), 'sha256': sha256(checkpoint)},
            }))
            with patch('sys.argv', ['runner', '--manifest', str(manifest), '--verify-only']):
                self.assertEqual(0, main())
            with patch('sys.argv', ['runner', '--manifest', str(manifest)]), \
                 patch('scripts.run_sl_rl_repair_pipeline.subprocess.run', return_value=Mock(returncode=75)):
                self.assertEqual(75, main())
                self.assertFalse((root / 'calibration_result.json').exists())
            source.write_bytes(b'mutated')
            with patch('sys.argv', ['runner', '--manifest', str(manifest), '--verify-only']):
                with self.assertRaisesRegex(ValueError, 'source changed'):
                    main()


if __name__ == '__main__':
    unittest.main()
