from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from mortal.core.artifacts import file_sha256, stable_json_digest
from scripts.prepare_sl_confirmation_runtime import relocated_protocol, verify_runtime_parity


class ConfirmationRuntimePreparationTests(unittest.TestCase):
    def test_source_parity_accepts_only_physical_newlines_with_honest_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            before, after = root / 'before', root / 'after'
            before.mkdir()
            after.mkdir()
            (before / 'code.py').write_bytes(b'value = 1\r\n')
            (after / 'code.py').write_bytes(b'value = 1\n')
            old = {'threads': 24, 'source_sha256': {'code.py': file_sha256(before / 'code.py')}}
            new = {'threads': 24, 'source_sha256': {'code.py': file_sha256(after / 'code.py')}}
            self.assertEqual(len(verify_runtime_parity(old, new, before, after)), 1)
            with self.assertRaises(ValueError):
                verify_runtime_parity(old, dict(new, threads=2), before, after)
            (after / 'code.py').write_bytes(b'value = 2\n')
            with self.assertRaisesRegex(ValueError, 'recorded runtime'):
                verify_runtime_parity(old, new, before, after)
            new['source_sha256']['code.py'] = file_sha256(after / 'code.py')
            with self.assertRaisesRegex(ValueError, 'source text'):
                verify_runtime_parity(old, new, before, after)

    def test_only_aa_size_and_explicit_provenance_change(self):
        original = {'aa_games_per_arm': 128, 'screen_games_per_arm': 16000,
                    'confirmation_games_per_arm': 64000, 'screen_seed_key': 12,
                    'confirmation_seed_key': 34, 'candidates': {'fixed': {'sha256': 'actor'}},
                    'inference': {'enable_amp': False}, 'primary': 'mean_pt'}
        saved = deepcopy(original)
        proof = {'mode': 'formal_inference', 'timing_mode': 'throughput', 'chunk_seeds': 64,
                 'games': 256, 'ordered_game_events_equal': True,
                 'protocol_fingerprint': stable_json_digest(original)}
        actual = relocated_protocol(original, proof, {'old_results_imported': False})
        self.assertEqual(original, saved)
        self.assertEqual(actual.pop('aa_games_per_arm'), 256)
        self.assertFalse(actual.pop('runtime_transition')['old_results_imported'])
        self.assertEqual(actual, {key: value for key, value in original.items() if key != 'aa_games_per_arm'})
        for change in ({'chunk_seeds': 128}, {'games': 128}, {'ordered_game_events_equal': False},
                       {'protocol_fingerprint': 'different'}):
            with self.assertRaises(ValueError):
                relocated_protocol(original, dict(proof, **change), {})


if __name__ == '__main__':
    unittest.main()
