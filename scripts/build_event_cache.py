import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


def iter_month_dirs(src_root: Path, reverse: bool):
    month_dirs = []
    for year_dir in src_root.iterdir():
        if not year_dir.is_dir():
            continue
        for month_dir in year_dir.iterdir():
            if month_dir.is_dir():
                month_dirs.append(month_dir)
    month_dirs.sort(reverse=reverse)
    return month_dirs


def chunkify(items, chunk_size):
    for index in range(0, len(items), chunk_size):
        yield items[index:index + chunk_size]


def build_chunk(version: int, month_name: str, chunk_index: int, file_list, out_path: str):
    from libriichi.dataset import GameplayLoader

    loader = GameplayLoader(version=version, oracle=False, augmented=False)
    count = loader.build_event_cache_file(file_list, out_path)
    return month_name, chunk_index, count, out_path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src-root', default=r'D:\mahjong_data\dataset_json')
    parser.add_argument('--dst-root', default=r'D:\mahjong_data\dataset_event_cache_v3')
    parser.add_argument('--workers', type=int, default=14)
    parser.add_argument('--files-per-chunk', type=int, default=32)
    parser.add_argument('--month-limit', type=int, default=0)
    parser.add_argument('--start-month', default='')
    parser.add_argument('--reverse-months', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--version', type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()
    src_root = Path(args.src_root)
    dst_root = Path(args.dst_root)
    dst_root.mkdir(parents=True, exist_ok=True)

    month_dirs = iter_month_dirs(src_root, args.reverse_months)
    if args.start_month:
        month_dirs = [m for m in month_dirs if m.name >= args.start_month]
    if args.month_limit > 0:
        month_dirs = month_dirs[:args.month_limit]
    if not month_dirs:
        raise FileNotFoundError(f'no month dirs found under {src_root}')

    print(f'found {len(month_dirs):,} month dirs', flush=True)
    total_chunks = 0
    total_files = 0

    for month_index, month_dir in enumerate(month_dirs, start=1):
        files = sorted(month_dir.rglob('*.json'))
        if not files:
            continue

        month_name = month_dir.name
        month_out_dir = dst_root / month_name
        month_out_dir.mkdir(parents=True, exist_ok=True)

        tasks = []
        for chunk_index, chunk in enumerate(chunkify(files, args.files_per_chunk)):
            out_path = month_out_dir / f'chunk_{chunk_index:05d}.events.zst'
            if out_path.exists() and not args.overwrite:
                continue
            tasks.append((chunk_index, [str(path) for path in chunk], str(out_path)))

        if not tasks:
            print(f'[month {month_name}] skip all existing', flush=True)
            continue

        print(
            f'[month {month_index}/{len(month_dirs)}] start {month_name} '
            f'files={len(files):,} chunks={len(tasks):,}',
            flush=True,
        )

        completed = 0
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    build_chunk,
                    args.version,
                    month_name,
                    chunk_index,
                    file_list,
                    out_path,
                ): chunk_index
                for chunk_index, file_list, out_path in tasks
            }

            for future in as_completed(futures):
                month_name, chunk_index, file_count, out_path = future.result()
                completed += 1
                total_chunks += 1
                total_files += file_count
                if completed % 10 == 0 or completed == len(tasks):
                    print(
                        f'[month {month_name}] chunks={completed:,}/{len(tasks):,} '
                        f'total_chunks={total_chunks:,} total_files={total_files:,}',
                        flush=True,
                    )

        print(
            f'[month {month_name}] done chunks={len(tasks):,} files={len(files):,}',
            flush=True,
        )


if __name__ == '__main__':
    main()
