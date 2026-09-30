import json
import tempfile
import unittest
from pathlib import Path

import torch

from mortal.core.artifacts import (
    atomic_output_path,
    atomic_torch_save,
    atomic_write_json,
    atomic_write_text,
    file_sha256,
    load_json,
    stable_json_digest,
)


class ArtifactTests(unittest.TestCase):
    def test_atomic_output_keeps_old_target_when_writer_fails(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / 'state.txt'
            target.write_text('old', encoding='utf-8')

            with self.assertRaisesRegex(RuntimeError, 'failed'):
                with atomic_output_path(target) as temporary:
                    temporary.write_text('new', encoding='utf-8')
                    raise RuntimeError('failed')

            self.assertEqual('old', target.read_text(encoding='utf-8'))
            self.assertEqual([target], list(target.parent.iterdir()))

    def test_json_and_torch_writes_replace_existing_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            json_path = root / 'value.json'
            state_path = root / 'state.pth'

            atomic_write_json(json_path, {'value': 1})
            atomic_write_json(json_path, {'value': 2})
            atomic_torch_save({'value': torch.tensor([1])}, state_path)
            atomic_torch_save({'value': torch.tensor([2])}, state_path)

            self.assertEqual({'value': 2}, load_json(json_path))
            loaded = torch.load(state_path, map_location='cpu', weights_only=False)
            self.assertEqual([2], loaded['value'].tolist())

    def test_text_write_creates_parent_and_normalizes_newlines(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / 'nested' / 'value.txt'

            atomic_write_text(target, 'first\nsecond\n')

            self.assertEqual('first\nsecond\n', target.read_text(encoding='utf-8'))
            self.assertEqual([target], list(target.parent.iterdir()))

    def test_digest_helpers_are_deterministic(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / 'value.json'
            target.write_text(json.dumps({'a': 1}), encoding='utf-8')

            self.assertEqual(
                stable_json_digest({'a': 1, 'b': 2}),
                stable_json_digest({'b': 2, 'a': 1}),
            )
            self.assertEqual(64, len(file_sha256(target)))


if __name__ == '__main__':
    unittest.main()
