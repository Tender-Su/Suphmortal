import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_stage05_p1_pairwise_only as legacy_entry


class RunStage05P1PairwiseOnlyTests(unittest.TestCase):
    def test_main_accepts_legacy_args_and_shows_migration_message(self):
        with self.assertRaises(SystemExit) as exc_ctx:
            legacy_entry.main(
                [
                    '--run-name',
                    'demo_run',
                    '--dry-run',
                    '--continue-to-joint-refine',
                ]
            )

        message = str(exc_ctx.exception)
        self.assertIn('run_stage05_p1_pairwise_only.py is deprecated.', message)
        self.assertIn('python run_stage05_p1_only.py --run-name <name>', message)
        self.assertIn('Legacy --run-name was accepted and ignored here: demo_run', message)
        self.assertIn('Legacy --dry-run was accepted and ignored here.', message)
        self.assertIn('Additional legacy args were ignored: --continue-to-joint-refine', message)


if __name__ == '__main__':
    unittest.main()
