"""Standard-library contract tests. These do not claim Torch/native coverage."""
import ast
import copy
import gzip
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from mortal.core.evidence_contract import fingerprint, sha256_file, validation_input_contract
from mortal.data.current_policy_manifest import (
    SOURCE_KEYS, SAMPLING_KEYS, cluster_id, load_current_policy_manifest,
    main, register_probe_blocks, validate_manifest,
)


def completed_probe(root, *, seed_start, seed_key=20261001, groups=2):
    root.mkdir()
    games = root / 'games'
    games.mkdir()
    outcomes = []
    for offset in range(groups):
        seed = seed_start + offset
        for seat in range(4):
            # Deliberately identical basenames across blocks: identity is not the filename.
            filename = games / f'{offset}_{seat}.json.gz'
            names = ['canonical'] * 4
            names[seat] = 'trainee'
            events = [{'type': 'start_game', 'names': names, 'seed': [seed, seed_key]},
                      {'type': 'start_kyoku'}, {'type': 'end_kyoku'}, {'type': 'end_game'}]
            with gzip.open(filename, 'wt', encoding='utf-8') as stream:
                stream.write('\n'.join(json.dumps(event) for event in events))
            outcomes.append({'log_path': str(filename), 'sha256': sha256_file(filename),
                             'seed': seed, 'seed_key': seed_key, 'challenger_seat': seat,
                             'challenger_rank': 1, 'kyoku_count': 1})
    (root / 'outcomes.json').write_text(json.dumps(outcomes), encoding='utf-8')
    receipt = {
        'status': 'complete', 'source_commit': 'a' * 40,
        'arguments': {'games': groups * 4, 'seed_start': seed_start, 'seed_key': seed_key,
                      'sampling_seed': seed_start + 1, 'imputation_seed': 20260905},
        'weights_and_config': {name: {'path': name + '.pth', 'sha256': fingerprint(name)}
                               for name in ('actor', 'opponent', 'config', 'warm0', 'warm40k', 'clean40k')},
        'source_hashes': {name: fingerprint(name) for name in SOURCE_KEYS},
        'native_sha256': 'a' * 64, 'native_extension_sha256': {'libriichi.so': 'b' * 64},
        'torch_version': 'fixture', 'numpy_version': 'fixture', 'probe_pts': [2, 1, 0, -3],
        **dict(zip(SAMPLING_KEYS, (1.0, 0.0, False, True, False, False))),
        'artifact_sha256': {'outcomes.json': sha256_file(root / 'outcomes.json')},
    }
    (root / 'provenance.json').write_text(json.dumps(receipt), encoding='utf-8')
    return root


def resign(payload):
    payload['fingerprint'] = fingerprint({key: value for key, value in payload.items() if key != 'fingerprint'})
    return payload


class CurrentPolicyManifestTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.train = completed_probe(self.root / 'train', seed_start=100)
        self.dev = completed_probe(self.root / 'dev', seed_start=200)

    def manifest(self):
        return register_probe_blocks(train_probe_dirs=[self.train], dev_probe_dirs=[self.dev])

    def mutate_receipt(self, root, change):
        filename = root / 'provenance.json'
        receipt = json.loads(filename.read_text())
        change(receipt)
        filename.write_text(json.dumps(receipt))

    def test_multiple_blocks_and_explicit_split_preserve_weight_receipts(self):
        extra = completed_probe(self.root / 'train2', seed_start=300)
        manifest = register_probe_blocks(train_probe_dirs=[self.train, extra], dev_probe_dirs=[self.dev])
        self.assertEqual(len(manifest['games']), 24)
        self.assertEqual(len(validate_manifest(manifest, verify_files=True)['train']), 16)
        self.assertEqual(manifest['blocks'][0]['weights_and_config']['warm40k']['sha256'], fingerprint('warm40k'))
        self.assertEqual(manifest['identity']['player_names'], ['trainee'])
        self.assertEqual(manifest['identity']['target_mode'], 'all_players')

    def test_same_basenames_do_not_merge_distinct_seed_groups(self):
        manifest = self.manifest()
        first, later = manifest['games'][0], manifest['games'][8]
        self.assertEqual(Path(first['path']).name, Path(later['path']).name)
        self.assertNotEqual(first['cluster_id'], later['cluster_id'])
        self.assertTrue(all(game['cluster_id'] == first['cluster_id'] for game in manifest['games'][:4]))

    def test_duplicate_root_and_cross_split_seed_group_rejected(self):
        with self.assertRaisesRegex(ValueError, 'registered twice'):
            register_probe_blocks(train_probe_dirs=[self.train], dev_probe_dirs=[self.train])
        overlap = completed_probe(self.root / 'overlap', seed_start=100)
        with self.assertRaisesRegex(ValueError, 'duplicate|repeated'):
            register_probe_blocks(train_probe_dirs=[self.train], dev_probe_dirs=[overlap])

    def test_incomplete_and_cross_split_groups_rejected(self):
        for change, message in (
            (lambda m: m['games'].pop(), 'incomplete'),
            (lambda m: m['games'][0].update(split='dev'), 'crosses'),
            (lambda m: m['games'][0].update(trainee_seat=1), 'repeated'),
        ):
            with self.subTest(message=message):
                manifest = self.manifest()
                change(manifest)
                with self.assertRaisesRegex(ValueError, message):
                    validate_manifest(resign(manifest), verify_files=False)

    def test_empty_train_or_dev_is_invalid(self):
        for train, dev in (([], [self.dev]), ([self.train], [])):
            with self.assertRaisesRegex(ValueError, 'non-empty'):
                register_probe_blocks(train_probe_dirs=train, dev_probe_dirs=dev)

    def test_single_dev_group_cannot_claim_cluster_inference(self):
        small = completed_probe(self.root / 'small', seed_start=500, groups=1)
        with self.assertRaisesRegex(ValueError, 'at least two'):
            register_probe_blocks(train_probe_dirs=[self.train], dev_probe_dirs=[small])

    def test_unfinished_receipt_and_source_or_actor_drift_fail(self):
        for field in ('status', 'actor', 'source'):
            with self.subTest(field=field):
                root = completed_probe(self.root / field, seed_start=500)
                def change(receipt):
                    if field == 'status':
                        receipt['status'] = 'running'
                    elif field == 'actor':
                        receipt['weights_and_config']['actor']['sha256'] = 'f' * 64
                    else:
                        receipt['source_hashes'][SOURCE_KEYS[0]] = 'f' * 64
                self.mutate_receipt(root, change)
                with self.assertRaises(ValueError):
                    register_probe_blocks(train_probe_dirs=[self.train], dev_probe_dirs=[root])

    def test_outcomes_and_log_hashes_fail_closed(self):
        manifest = self.manifest()
        Path(manifest['games'][0]['path']).write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'log hash mismatch'):
            validate_manifest(manifest, verify_files=True)
        with (self.dev / 'outcomes.json').open('a') as stream:
            stream.write(' ')
        with self.assertRaisesRegex(ValueError, 'outcomes.*mismatch'):
            self.manifest()

    def test_log_header_is_authority_for_seed_and_trainee_seat(self):
        for key, value in (('seed', 999), ('challenger_seat', 1)):
            root = completed_probe(self.root / key, seed_start=600)
            outcomes = json.loads((root / 'outcomes.json').read_text())
            if key == 'seed':
                # Keep all rows/groups consistent but make the native log disagree.
                for outcome in outcomes:
                    outcome['seed'] += 100
                self.mutate_receipt(root, lambda r: r['arguments'].update(seed_start=700))
            else:
                outcomes[0][key] = value
            (root / 'outcomes.json').write_text(json.dumps(outcomes))
            self.mutate_receipt(root, lambda r: r['artifact_sha256'].update(
                {'outcomes.json': sha256_file(root / 'outcomes.json')}))
            with self.assertRaisesRegex(ValueError, 'differs from native log'):
                register_probe_blocks(train_probe_dirs=[self.train], dev_probe_dirs=[root])

    def test_recomputed_fingerprint_cannot_forge_receipt_membership(self):
        manifest = self.manifest()
        manifest['games'][0]['block_id'] = manifest['blocks'][1]['id']
        with self.assertRaisesRegex(ValueError, 'differs from registered outcome'):
            validate_manifest(resign(manifest), verify_files=True)

    def test_load_binds_mapping_filter_receipts_and_each_log_hash_once(self):
        manifest = self.manifest()
        filename = self.root / 'manifest.json'
        filename.write_text(json.dumps(manifest))
        with patch('mortal.data.current_policy_manifest.sha256_file', wraps=sha256_file) as hasher:
            loaded = load_current_policy_manifest(filename)
        for game in manifest['games']:
            self.assertEqual(sum(str(call.args[0]) == game['path'] for call in hasher.call_args_list), 1)
        self.assertEqual(loaded['contract']['split_groups'], {'train': 2, 'dev': 2, 'test': 0})
        self.assertEqual(loaded['contract']['manifest_sha256'], sha256_file(filename))
        self.assertEqual(len(loaded['group_ids']), 16)
        self.assertEqual(len(set(loaded['group_ids'].values())), 4)

    def test_verified_validation_ledger_does_not_rehash_dev(self):
        filename = self.root / 'native.so'
        filename.write_bytes(b'native fixture')
        manifest = self.manifest()
        ledger = {g['path']: g['sha256'] for g in manifest['games']}
        files = [g['path'] for g in manifest['games'] if g['split'] == 'dev']
        original = validation_input_contract(files, settings={}, native_file=filename)
        with patch('mortal.core.evidence_contract.sha256_file', wraps=sha256_file) as hasher:
            cached = validation_input_contract(files, settings={}, native_file=filename, verified_file_sha256=ledger)
        self.assertEqual(cached, original)
        self.assertEqual(hasher.call_count, 1)
        with self.assertRaises(KeyError):
            validation_input_contract(files, settings={}, native_file=filename, verified_file_sha256={})

    def test_manifest_fingerprint_and_resume_contract_reject_drift(self):
        manifest = self.manifest()
        manifest['games'][0]['split'] = 'dev'
        with self.assertRaisesRegex(ValueError, 'fingerprint mismatch'):
            validate_manifest(manifest, verify_files=False)
        source = Path(__file__).parents[1] / 'online/pretrain_oracle_critic.py'
        tree = ast.parse(source.read_text(encoding='utf-8'))
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'validate_resume_training_contract')
        namespace = {'copy': copy}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), namespace)
        validate = namespace['validate_resume_training_contract']
        contract = {'current_policy': self.manifest()['fingerprint']}
        self.assertFalse(validate({'training_contract': contract}, contract))
        for changed in ({}, {'current_policy': 'changed'}):
            with self.assertRaisesRegex(ValueError, 'contract mismatch'):
                validate({'training_contract': contract}, changed)

    def test_cli_creates_new_ledger_without_overwriting(self):
        output = self.root / 'ledger.json'
        args = ['--train-probe-dir', str(self.train), '--dev-probe-dir', str(self.dev), '--output', str(output)]
        main(args)
        self.assertTrue(output.is_file())
        with self.assertRaises(FileExistsError):
            main(args)


if __name__ == '__main__':
    unittest.main()
