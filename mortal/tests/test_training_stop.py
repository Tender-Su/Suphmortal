import ast
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

from mortal.core.artifacts import atomic_torch_save
import unittest

from mortal.core.training_stop import training_stop_requested


class TrainingStopTests(unittest.TestCase):
    def test_absent_empty_and_present_stop_file(self):
        self.assertFalse(training_stop_requested({}))
        self.assertFalse(training_stop_requested({'MORTAL_STOP_FILE': ''}))
        with tempfile.TemporaryDirectory() as directory:
            stop = Path(directory) / 'STOP'
            env = {'MORTAL_STOP_FILE': str(stop)}
            self.assertFalse(training_stop_requested(env))
            stop.touch()
            self.assertTrue(training_stop_requested(env))

    def test_interrupted_checkpoint_save_preserves_previous_latest(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'latest.pth'
            target.write_bytes(b'previous-checkpoint')
            def interrupted_save(payload, temporary):
                self.assertEqual(Path(temporary).parent, target.parent)
                self.assertNotEqual(Path(temporary), target)
                Path(temporary).write_bytes(b'incomplete-new-checkpoint')
                raise RuntimeError('interrupted save')
            with patch.dict('sys.modules', {'torch': SimpleNamespace(save=interrupted_save)}):
                with self.assertRaisesRegex(RuntimeError, 'interrupted save'):
                    atomic_torch_save({}, target)
            self.assertEqual(target.read_bytes(), b'previous-checkpoint')
            self.assertEqual(list(target.parent.iterdir()), [target])

    def test_completed_checkpoint_atomically_replaces_latest(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'latest.pth'
            target.write_bytes(b'previous')
            def save(payload, temporary):
                self.assertEqual(target.read_bytes(), b'previous')
                Path(temporary).write_bytes(b'complete-new')
            with patch.dict('sys.modules', {'torch': SimpleNamespace(save=save)}):
                atomic_torch_save({}, target)
            self.assertEqual(target.read_bytes(), b'complete-new')
            self.assertEqual(list(target.parent.iterdir()), [target])

    def test_online_stop_is_checked_at_accumulation_boundary(self):
        path = Path(__file__).resolve().parents[1] / 'online' / 'train_online.py'
        tree = ast.parse(path.read_text(encoding='utf-8'))
        checks = [node for node in ast.walk(tree) if isinstance(node, ast.If)
                  and ast.unparse(node.test) == 'idx % opt_step_every == 0']
        self.assertTrue(any('stop_after_checkpoint_if_requested()' in ast.unparse(node)
                            for node in checks))
        persist = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
                       and node.name == 'persist_live_training_state')
        self.assertIn('atomic_torch_save(state, state_file)', ast.unparse(persist))
        self.assertNotIn('torch.save(', ast.unparse(persist))
        main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
        code = ast.unparse(main)
        self.assertIn('ONLINE_STOP_REQUEST_EXIT_CODE', code)
        self.assertIn('if training_stop_requested():\n            return', code)


if __name__ == '__main__':
    unittest.main()
