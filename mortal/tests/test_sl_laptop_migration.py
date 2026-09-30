import unittest
from unittest.mock import patch

from scripts.migrate_sl_curriculum_probe import identity_digest, path_lookup, remap_indexes
from mortal.eval.confirmation_protocol import runtime_record


def game(root, number):
    return f'{root}/2024010100gm-00a9-0000-{number:08x}.json'


class LaptopMigrationTests(unittest.TestCase):
    def test_paths_change_but_ids_order_and_repeated_entries_do_not(self):
        old = [game('D:/data', n) for n in (3, 1, 2, 1)]
        new = [game('C:/rebuilt', n) for n in (1, 2, 3)]
        original = {'train_files': old, 'domains': {'recent': old}, 'roles': {'old': old[:1]}}
        result = remap_indexes(original, path_lookup(new))
        self.assertEqual(result['train_files'], [new[n - 1] for n in (3, 1, 2, 1)])
        self.assertEqual(identity_digest(old), identity_digest(result['train_files']))
        self.assertEqual(original['train_files'], old)

    def test_missing_game_is_not_silently_dropped(self):
        with self.assertRaisesRegex(ValueError, 'missing game'):
            remap_indexes({'train_files': [game('D:/data', 1)]}, {})

    def test_ambiguous_destination_fails(self):
        with self.assertRaisesRegex(ValueError, 'ambiguous'):
            path_lookup([game('C:/one', 1), game('C:/two', 1)])

    def test_repeated_same_path_is_unambiguous(self):
        self.assertEqual(len(path_lookup([game('C:/one', 1)] * 2)), 1)

    def test_runtime_binds_actual_threads_and_environment(self):
        with patch('mortal.eval.confirmation_protocol.sha256_file', return_value='hash'), \
             patch('mortal.eval.confirmation_protocol.native_module_file', return_value='native'), \
             patch('mortal.eval.confirmation_protocol.torch.get_num_threads', return_value=2), \
             patch('mortal.eval.confirmation_protocol.torch.get_num_interop_threads', return_value=1), \
             patch.dict('os.environ', {'OMP_NUM_THREADS': '2', 'RAYON_NUM_THREADS': '4'}):
            runtime = runtime_record('cpu')
        self.assertEqual(runtime['torch_threads'], 2)
        self.assertEqual(runtime['torch_interop_threads'], 1)
        self.assertEqual(runtime['thread_environment']['RAYON_NUM_THREADS'], '4')


if __name__ == '__main__':
    unittest.main()
