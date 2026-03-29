import argparse
import gzip
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import shutil


def iter_month_dirs(src_root: Path):
    for year_dir in sorted(src_root.iterdir()):
        if not year_dir.is_dir():
            continue
        for month_dir in sorted(year_dir.iterdir()):
            if month_dir.is_dir():
                yield month_dir


def decompress_one(src_path: str, src_root: str, dst_root: str, overwrite: bool):
    src = Path(src_path)
    rel = src.relative_to(Path(src_root))
    dst = Path(dst_root) / rel.with_suffix('')
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists() and not overwrite:
        return str(src), str(dst), 'skipped'

    with gzip.open(src, 'rb') as fin, open(dst, 'wb') as fout:
        shutil.copyfileobj(fin, fout, length=1024 * 1024)
    return str(src), str(dst), 'ok'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src-root', default=r'D:\mahjong_data\dataset')
    parser.add_argument('--dst-root', default=r'D:\mahjong_data\dataset_json')
    parser.add_argument('--workers', type=int, default=12)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--month-limit', type=int, default=0)
    parser.add_argument('--start-month', default='')
    parser.add_argument('--report-every', type=int, default=1000)
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    src_root = Path(args.src_root)
    dst_root = Path(args.dst_root)
    dst_root.mkdir(parents=True, exist_ok=True)

    month_dirs = list(iter_month_dirs(src_root))
    if args.start_month:
        month_dirs = [m for m in month_dirs if m.name >= args.start_month]
    if args.month_limit > 0:
        month_dirs = month_dirs[:args.month_limit]
    if not month_dirs:
        raise FileNotFoundError(f'no month dirs found under {src_root}')

    print(f'found {len(month_dirs):,} month dirs', flush=True)

    total_done = 0
    total_skipped = 0
    total_seen = 0

    for month_index, month_dir in enumerate(month_dirs, start=1):
        files = sorted(month_dir.rglob('*.json.gz'))
        if args.limit > 0:
            remaining = max(args.limit - total_seen, 0)
            if remaining <= 0:
                break
            files = files[:remaining]
        if not files:
            continue

        total_seen += len(files)
        print(
            f'[month {month_index}/{len(month_dirs)}] start {month_dir.name} '
            f'files={len(files):,}',
            flush=True,
        )

        completed = 0
        skipped = 0
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    decompress_one,
                    str(src),
                    str(src_root),
                    str(dst_root),
                    args.overwrite,
                ): src
                for src in files
            }

            for future in as_completed(futures):
                _, _, status = future.result()
                completed += 1
                if status == 'skipped':
                    skipped += 1
                if completed % args.report_every == 0 or completed == len(files):
                    print(
                        f'[month {month_dir.name}] completed {completed:,}/{len(files):,} '
                        f'(skipped {skipped:,})',
                        flush=True,
                    )

        total_done += completed
        total_skipped += skipped
        print(
            f'[month {month_dir.name}] done files={completed:,} skipped={skipped:,} '
            f'total_done={total_done:,}',
            flush=True,
        )


if __name__ == '__main__':
    main()
