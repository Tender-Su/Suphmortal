"""Check owned Markdown without importing training code or accessing the network."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import date
import json
from pathlib import Path
import re
from urllib.parse import unquote


INLINE_LINK = re.compile(r'!?\[[^\]\n]*\]\(\s*(<[^>\n]+>|[^\s)]+)(?:\s+"[^"\n]*")?\s*\)')
REFERENCE_DEF = re.compile(r'^ {0,3}\[([^\]\n]+)\]:\s*(<[^>\n]+>|\S+)', re.MULTILINE)
REFERENCE_LINK = re.compile(r'!?\[([^\]\n]+)\]\[([^\]\n]*)\]')
INLINE_CODE = re.compile(r'(?<!`)`([^`\n]+)`(?!`)')
MODULE = re.compile(r'(?<!\S)-m\s+((?:mortal|integrations)(?:\.[A-Za-z_]\w*)+)')
SOURCE_PREFIXES = ('docs/', 'mortal/', 'libriichi/', 'scripts/', 'integrations/', 'exe-wrapper/')
GENERATED = 'docs/status/supervised-fidelity-results.md'
OPTIONAL_FILES = {'mortal/config.toml'}


def without_fences(text: str) -> str:
    """Keep line numbers while excluding fenced examples from prose checks."""
    lines, fence = [], None
    for line in text.splitlines(keepends=True):
        match = re.match(r'^\s{0,3}(`{3,}|~{3,})(.*)$', line)
        if fence:
            if match and match[1][0] == fence[0] and len(match[1]) >= len(fence) and not match[2].strip():
                fence = None
            lines.append('\n' if line.endswith('\n') else '')
        elif match:
            fence = match[1]
            lines.append('\n' if line.endswith('\n') else '')
        else:
            lines.append(line)
    return ''.join(lines)


def heading_ids(text: str) -> set[str]:
    body = without_fences(text)
    ids = set(re.findall(r'<a\s+(?:id|name)=["\']([^"\']+)["\']', body, re.I))
    counts = Counter()
    for match in re.finditer(r'^ {0,3}#{1,6}\s+(.+?)\s*#*\s*$', body, re.MULTILINE):
        title = re.sub(r'\[([^\]]+)\]\([^)]*\)', r'\1', match[1])
        title = re.sub(r'<[^>]+>', '', title).lower()
        slug = re.sub(r'[^\w\s-]', '', title).replace(' ', '-')
        suffix = f'-{counts[slug]}' if counts[slug] else ''
        counts[slug] += 1
        ids.add(slug + suffix)
    return ids


def owned_documents(root: Path) -> list[Path]:
    paths = set(root.glob('*.md'))
    paths.update((root / 'docs').rglob('*.md'))
    paths.update((root / 'integrations').rglob('*.md'))
    package_readme = root / 'mortal/README.md'
    if package_readme.is_file():
        paths.add(package_readme)
    return sorted(paths)


def is_history(name: str) -> bool:
    return name.startswith(('docs/archive/', 'docs/reflections/')) and not name.endswith('/README.md')


def is_artifact(path: Path) -> bool:
    return bool({'logs', 'checkpoints', 'target'} & set(path.parts))


def page_budget(name: str) -> tuple[int, int] | None:
    if is_history(name) or name == GENERATED:
        return None
    if name == 'docs/agent/handoff.md':
        return 45, 2500
    if name == 'CLAUDE.md':
        return 12, 1000
    if name == 'AGENTS.md':
        return 90, 6000
    if name == 'docs/agent/workflows.md':
        return 200, 9000
    if name.endswith('README.md') or name == 'RTK.md':
        return 65, 5000
    if name.startswith('docs/status/'):
        return 100, 5000
    if name.startswith('docs/agent/'):
        return 120, 6500
    return None


def audit(root: Path, *, check_artifacts: bool = False, max_age_days: int = 30) -> dict:
    root = root.resolve()
    documents = owned_documents(root)
    content = {p: p.read_text(encoding='utf-8-sig') for p in documents}
    graph = {p: set() for p in documents}
    issues, links, optional_links = [], 0, 0

    def issue(path: Path, code: str, message: str, line: int = 1, severity: str = 'error') -> None:
        issues.append(dict(file=path.relative_to(root).as_posix(), line=line,
                           severity=severity, code=code, message=message))

    if not (root / 'README.md').is_file():
        issue(root / 'README.md', 'root-index', 'Repository root needs README.md; check --root')

    def validate_link(source: Path, target: str, line: int) -> None:
        nonlocal links, optional_links
        target = target.strip('<>')
        if re.match(r'^[A-Za-z][A-Za-z0-9+.-]*:', target) or target.startswith('//'):
            return
        links += 1
        path_text, _, fragment = target.partition('#')
        path_text = unquote(path_text.split('?')[0])
        if not path_text:
            destination = source
        elif path_text.startswith('/'):
            destination = root / path_text.lstrip('/')
        else:
            destination = source.parent / path_text
        destination = destination.resolve()
        if not destination.is_relative_to(root):
            issue(source, 'outside-root', f'Local link leaves repository: {target}', line)
            return
        if is_artifact(destination.relative_to(root)):
            optional_links += 1
            if not check_artifacts:
                return
        if not destination.exists():
            issue(source, 'missing-link', f'Missing local target: {target}', line)
            return
        if destination in graph:
            graph[source].add(destination)
        if fragment and destination.suffix.lower() == '.md':
            target_body = content.get(destination)
            if target_body is None:
                target_body = destination.read_text(encoding='utf-8-sig')
            if unquote(fragment) not in heading_ids(target_body):
                issue(source, 'missing-anchor', f'Missing heading anchor: {target}', line)

    for path, text in content.items():
        name = path.relative_to(root).as_posix()
        prose = without_fences(text)
        link_prose = INLINE_CODE.sub(lambda m: ' ' * len(m[0]), prose)
        if name != GENERATED and not is_history(name):
            if not text.startswith('# ') or len(re.findall(r'^# ', prose, re.MULTILINE)) != 1:
                issue(path, 'title', 'Use exactly one top-level title at the start')
        if is_history(name) and not re.search(r'^> 历史(?:归档|复盘)', text[:800], re.MULTILINE):
            issue(path, 'history-label', 'Historical document needs a visible history banner')
        if (name.startswith('docs/status/') and name != GENERATED) or name == 'docs/agent/handoff.md':
            verified = re.search(r'^> 核验：(\d{4}-\d{2}-\d{2})', prose, re.MULTILINE)
            if not verified:
                issue(path, 'verification-date', 'Status needs an actual verification date and scope')
            else:
                try:
                    age = (date.today() - date.fromisoformat(verified[1])).days
                except ValueError:
                    issue(path, 'verification-date', 'Invalid verification date')
                else:
                    if age < 0:
                        issue(path, 'verification-date', 'Verification date is in the future')
                    elif age > max_age_days:
                        issue(path, 'stale-status', f'Last verified {age} days ago; recheck facts before updating date', severity='warning')
        budget = page_budget(name)
        if budget and (len(text.splitlines()) > budget[0] or len(text) > budget[1]):
            issue(path, 'page-budget', f'{len(text.splitlines())} lines / {len(text)} chars exceeds {budget[0]} / {budget[1]}')
        for match in INLINE_LINK.finditer(link_prose):
            validate_link(path, match[1], link_prose.count('\n', 0, match.start()) + 1)
        definitions = {m[1].casefold(): m[2] for m in REFERENCE_DEF.finditer(link_prose)}
        for match in REFERENCE_DEF.finditer(link_prose):
            validate_link(path, match[2], link_prose.count('\n', 0, match.start()) + 1)
        for match in REFERENCE_LINK.finditer(link_prose):
            key = (match[2] or match[1]).casefold()
            if key not in definitions:
                issue(path, 'missing-reference', f'Undefined reference: {key}', link_prose.count('\n', 0, match.start()) + 1)
        if is_history(name) or name == GENERATED:
            continue
        for match in INLINE_CODE.finditer(prose):
            token = match[1].replace('\\', '/')
            if not token.startswith(SOURCE_PREFIXES) or re.search(r'[\s*<>{}\[\]]', token):
                continue
            token = re.sub(r':\d+$', '', token).rstrip('/')
            if token in OPTIONAL_FILES or is_artifact(Path(token)):
                continue
            if not (root / token).exists():
                issue(path, 'missing-source', f'Missing literal source path: {token}', prose.count('\n', 0, match.start()) + 1)
        for match in MODULE.finditer(text):
            module = root.joinpath(*match[1].split('.'))
            if not module.with_suffix('.py').is_file() and not (module / '__main__.py').is_file():
                issue(path, 'missing-module', f'Missing CLI module: {match[1]}', text.count('\n', 0, match.start()) + 1)

    pending = [root / name for name in ('README.md', 'AGENTS.md', 'CLAUDE.md', 'RTK.md') if root / name in graph]
    visited = set()
    while pending:
        page = pending.pop()
        if page not in visited:
            visited.add(page)
            pending.extend(graph[page] - visited)
    for path in documents:
        if path not in visited:
            issue(path, 'unindexed', 'Document is not reachable from root navigation')
    return dict(documents=len(documents), local_links=links, optional_artifact_links=optional_links,
                errors=sum(i['severity'] == 'error' for i in issues),
                warnings=sum(i['severity'] == 'warning' for i in issues), issues=issues)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--json', action='store_true', help='Print machine-readable results')
    parser.add_argument('--check-artifacts', action='store_true', help='Also require linked local run artifacts')
    parser.add_argument('--max-age-days', type=int, default=30, help='Warn about older status snapshots')
    args = parser.parse_args()
    if args.max_age_days < 0:
        parser.error('--max-age-days must be nonnegative')
    result = audit(args.root, check_artifacts=args.check_artifacts, max_age_days=args.max_age_days)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"Docs: {result['documents']}; local links: {result['local_links']}; "
              f"optional artifact links: {result['optional_artifact_links']}; "
              f"errors: {result['errors']}; warnings: {result['warnings']}")
        for item in result['issues']:
            print(f"{item['file']}:{item['line']}: {item['severity']} [{item['code']}] {item['message']}")
    return int(result['errors'] > 0)


if __name__ == '__main__':
    raise SystemExit(main())
