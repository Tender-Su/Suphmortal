import argparse
import glob
import gzip
import os
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src-root', default=r'D:\mahjong_data')
    parser.add_argument('--dst-root', default=r'D:\mahjong_data\dataset')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--year-limit', type=int, default=0)
    parser.add_argument('--report-every', type=int, default=1)
    return parser.parse_args()


def zip_members_to_extract(zip_path: str):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for info in zip_ref.infolist():
            if info.is_dir():
                continue
            filename = info.filename
            if not filename.endswith('.mjson'):
                continue
            year = filename[:4]
            month = filename[4:6]
            rel_path = Path(year) / f'{year}{month}' / filename.replace('.mjson', '.json.gz')
            yield info, rel_path


def extract_one_zip(zip_path: str, dst_root: str):
    extracted = 0
    skipped = 0
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for info, rel_path in zip_members_to_extract(zip_path):
            dst_path = Path(dst_root) / rel_path
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            if dst_path.exists():
                skipped += 1
                continue
            with zip_ref.open(info) as fin:
                first_chunk = fin.read(1024 * 1024)
                if first_chunk.startswith(b'\x1f\x8b'):
                    # Some yearly zips already store gzip-compressed payloads under the
                    # legacy `.mjson` suffix. Preserve those bytes verbatim so the output
                    # remains a valid single-layer `.json.gz`.
                    with open(dst_path, 'wb') as fout:
                        fout.write(first_chunk)
                        while True:
                            chunk = fin.read(1024 * 1024)
                            if not chunk:
                                break
                            fout.write(chunk)
                else:
                    with gzip.open(dst_path, 'wb') as fout:
                        if first_chunk:
                            fout.write(first_chunk)
                        while True:
                            chunk = fin.read(1024 * 1024)
                            if not chunk:
                                break
                            fout.write(chunk)
            extracted += 1
    return {
        'zip_path': zip_path,
        'extracted': extracted,
        'skipped': skipped,
    }


def main():
    args = parse_args()
    src_root = Path(args.src_root)
    dst_root = Path(args.dst_root)
    dst_root.mkdir(parents=True, exist_ok=True)

    zip_files = sorted(glob.glob(str(src_root / '*.zip')))
    if args.year_limit > 0:
        zip_files = zip_files[:args.year_limit]
    if not zip_files:
        raise FileNotFoundError(f'no zip files found under {src_root}')

    print(f'found {len(zip_files):,} yearly zip files', flush=True)
    total_extracted = 0
    total_skipped = 0

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(extract_one_zip, zip_path, str(dst_root)): zip_path for zip_path in zip_files}
        for index, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            total_extracted += int(result['extracted'])
            total_skipped += int(result['skipped'])
            if index % args.report_every == 0 or index == len(zip_files):
                print(
                    f'[{index}/{len(zip_files)}] {Path(result["zip_path"]).name} '
                    f'extracted={result["extracted"]:,} skipped={result["skipped"]:,} '
                    f'total_extracted={total_extracted:,} total_skipped={total_skipped:,}',
                    flush=True,
                )


if __name__ == '__main__':
    main()
