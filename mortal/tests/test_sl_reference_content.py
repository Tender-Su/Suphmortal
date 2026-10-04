"""CPU-only failures and snapshot consistency for reference content unions."""
from contextlib import closing
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from mortal.core.artifacts import file_sha256
from mortal.supervised.early_transition import TrainingContentLedger
from mortal.supervised.reference_content import reference_ledger_snapshot
from mortal.supervised.continuation import evaluation_horizons, observation_plan


def pin_digest(rows):
    value = hashlib.sha256()
    for row in sorted(rows):
        value.update(json.dumps(row, ensure_ascii=True).encode() + b'\n')
    return value.hexdigest()


class ReferenceContentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / 'source.sqlite3'
        self.reference = self.root / 'reference'
        self.reference.mkdir()
        TrainingContentLedger(self.source, 'late', create=True).verify(
            [{'file': 'shared', 'source_sha256': 'a' * 64}])
        TrainingContentLedger(self.reference / 'training_content.sqlite3', 'early', create=True).verify(
            [{'file': 'shared', 'source_sha256': 'a' * 64}, {'file': 'future', 'source_sha256': 'b' * 64}])
        self.binding = {'source_identity': 'late', 'reference_identity': 'early',
            'reference_directory': str(self.reference), 'contract_sha256': 'contract',
            'source_ledger_content_sha256': pin_digest([('shared', 'a' * 64)]),
            'reference_ledger_content_sha256': pin_digest([('shared', 'a' * 64), ('future', 'b' * 64)])}
        self.output = self.root / 'output'
        self.output.mkdir()

    def invoke(self, **overrides):
        return reference_ledger_snapshot(self.source, self.output / 'inherited.sqlite3', 'late',
            ('shared',), current_hashes={'shared': 'a' * 64},
            binding=dict(self.binding, **overrides))

    def test_union_deduplicates_and_does_not_change_sources_or_binding(self):
        paths = [self.source, self.reference / 'training_content.sqlite3']
        before = {p: file_sha256(p) for p in paths}
        contract = deepcopy(self.binding)
        result = self.invoke()
        self.assertEqual(result['pinned_games'], 2)
        self.assertEqual(result['content_sha256'], self.binding['reference_ledger_content_sha256'])
        self.assertFalse(result['reference_progress_imported'])
        self.assertEqual(self.binding, contract)
        self.assertEqual(before, {p: file_sha256(p) for p in paths})
        with closing(sqlite3.connect(self.output / 'inherited.sqlite3')) as db:
            self.assertEqual(db.execute('SELECT identity FROM metadata').fetchone(), (result['identity'],))

    def test_conflicting_path_rejected_without_published_union(self):
        with closing(sqlite3.connect(self.reference / 'training_content.sqlite3')) as db, db:
            db.execute('UPDATE games SET sha256=? WHERE path=?', ('c' * 64, 'shared'))
        self.binding['reference_ledger_content_sha256'] = pin_digest(
            [('shared', 'c' * 64), ('future', 'b' * 64)])
        with self.assertRaisesRegex(ValueError, 'pin conflict'):
            self.invoke()
        self.assertFalse((self.output / 'inherited.sqlite3').exists())
        self.assertTrue((self.output / 'inherited.sqlite3.building').exists())

    def test_wrong_identity_or_snapshot_hash_fails_closed(self):
        with self.assertRaisesRegex(ValueError, 'source ledger identity'):
            self.invoke(source_identity='different')
        with self.assertRaisesRegex(ValueError, 'snapshot digest'):
            self.invoke(reference_ledger_content_sha256='changed')
        self.assertFalse((self.output / 'inherited.sqlite3').exists())

    def test_publication_failure_leaves_no_union_and_no_receipt(self):
        with patch('mortal.supervised.reference_content.os.link', side_effect=OSError('injected failure')):
            with self.assertRaisesRegex(OSError, 'injected'):
                self.invoke()
        self.assertFalse((self.output / 'inherited.sqlite3').exists())
        self.assertFalse((self.output / 'continuation_receipt.json').exists())

    def test_backup_includes_committed_wal_and_excludes_uncommitted_transaction(self):
        path = self.reference / 'training_content.sqlite3'
        with closing(sqlite3.connect(path)) as writer:
            writer.execute('PRAGMA journal_mode=WAL')
            writer.execute('INSERT INTO games VALUES (?,?)', ('committed', 'd' * 64))
            writer.commit()
            writer.execute('INSERT INTO games VALUES (?,?)', ('uncommitted', 'e' * 64))
            self.binding['reference_ledger_content_sha256'] = pin_digest(
                [('shared', 'a' * 64), ('future', 'b' * 64), ('committed', 'd' * 64)])
            result = self.invoke()
            self.assertEqual(result['pinned_games'], 3)
            writer.rollback()
        with closing(sqlite3.connect(self.output / 'inherited.sqlite3')) as db:
            self.assertIsNone(db.execute('SELECT 1 FROM games WHERE path=?', ('uncommitted',)).fetchone())

    def test_only_original_remaining_full_observations(self):
        plan = observation_plan(start=5000, until=20000, save_updates=500, save_seconds=300,
                                trend_every=10000, full_every=10000)
        self.assertEqual(evaluation_horizons(plan), [10000, 20000])


if __name__ == '__main__':
    unittest.main()
