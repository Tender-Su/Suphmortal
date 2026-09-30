"""Small repository fixtures for documentation checks; no training dependencies."""
from datetime import date, timedelta
from pathlib import Path
import tempfile
import unittest

from check_docs import audit


class DocumentationChecks(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def write(self, name, text):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding='utf-8')

    def codes(self, **kwargs):
        return {item['code'] for item in audit(self.root, **kwargs)['issues']}

    def test_empty_or_wrong_root_cannot_pass(self):
        self.assertEqual({'root-index'}, self.codes())

    def test_navigation_anchors_and_fenced_examples(self):
        self.write('README.md', '# Repository\n\n[Guide](docs/guide%20one.md#重复-1)\n')
        self.write('docs/guide one.md',
                   '# Guide\n\n## 重复\n\n## 重复\n\n'
                   '[Home][root]\n\n[root]: ../README.md\n\n'
                   '```markdown\n[Example](missing.md)\n# Example title\n```\n')
        self.assertEqual(set(), self.codes())

    def test_stale_links_anchors_and_missing_reference(self):
        self.write('README.md', '# Repository\n\n[Guide](docs/guide.md#gone)\n'
                   '[Missing](docs/missing.md)\n[Ref][unknown]\n')
        self.write('docs/guide.md', '# Guide\n')
        self.assertEqual({'missing-anchor', 'missing-link', 'missing-reference'}, self.codes())

    def test_orphan_and_stale_source_command(self):
        self.write('README.md', '# Repository\n\n`mortal/retired.py`\n\n'
                   '```powershell\npython -m mortal.retired\n```\n')
        self.write('docs/orphan.md', '# Orphan\n')
        self.assertEqual({'missing-source', 'missing-module', 'unindexed'}, self.codes())

    def test_optional_artifacts_and_historical_paths(self):
        self.write('README.md', '# Repository\n\n[History](docs/archive/old.md)\n'
                   '[Evidence](logs/run/result.json)\n')
        self.write('docs/archive/old.md', '# Old\n\n> 历史归档 · 保留原始路径\n\n'
                   '`mortal/retired.py`\n\n```powershell\npython -m mortal.retired\n```\n')
        self.assertEqual(set(), self.codes())
        self.assertEqual({'missing-link'}, self.codes(check_artifacts=True))

    def test_status_date_and_page_growth(self):
        self.write('README.md', '# Repository\n\n[State](docs/status/current.md)\n')
        self.write('docs/status/current.md', '# State\n\n' + 'A' * 5100 + '\n')
        self.assertEqual({'verification-date', 'page-budget'}, self.codes())
        old = (date.today() - timedelta(days=31)).isoformat()
        self.write('docs/status/current.md', f'# State\n\n> 核验：{old} · snapshot\n')
        result = audit(self.root)
        self.assertEqual(0, result['errors'])
        self.assertEqual(1, result['warnings'])
        self.assertEqual({'stale-status'}, self.codes())

    def test_history_needs_banner_and_links_cannot_escape(self):
        self.write('README.md', '# Repository\n\n[Old](docs/archive/old.md)\n')
        self.write('docs/archive/old.md', '# Old\n\n[Outside](../../../outside.md)\n')
        self.assertEqual({'history-label', 'outside-root'}, self.codes())


if __name__ == '__main__':
    unittest.main()
