import argparse
import glob
import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import torch


os.environ.setdefault('RAYON_NUM_THREADS', '2')

from libriichi.dataset import Grp


def chunkify(items, chunk_size):
    for index in range(0, len(items), chunk_size):
        yield items[index:index + chunk_size]


def process_chunk(chunk_id, chunk_files, output_path, dtype_name):
    dtype = getattr(torch, dtype_name)
    try:
        data = Grp.load_gz_log_files(chunk_files)
        buffer = []

        for game in data:
            feature = game.take_feature()
            rank_by_player = tuple(game.take_rank_by_player())
            feature_pt = torch.as_tensor(feature, dtype=dtype)
            buffer.append((feature_pt, rank_by_player))

        random.shuffle(buffer)
        torch.save(buffer, output_path)
        return chunk_id, len(buffer), True, None
    except Exception as exc:
        return chunk_id, 0, False, str(exc)


def collect_files(patterns):
    files = []
    for pattern in patterns:
        files.extend(glob.glob(pattern, recursive=True))
    files.sort()
    return files


def build_tasks(files, split_name, out_dir, chunk_size):
    tasks = []
    for index, chunk in enumerate(chunkify(files, chunk_size)):
        output_path = out_dir / f'chunk_{index:04d}.pt'
        tasks.append((index, chunk, output_path, split_name))
    return tasks


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-glob', action='append', required=True)
    parser.add_argument('--val-glob', action='append', required=True)
    parser.add_argument('--out-dir', default=r'D:\mahjong_data\grp_pt')
    parser.add_argument('--chunk-size', type=int, default=1000)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dtype', choices=('bfloat16', 'float64'), default='bfloat16')
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    train_files = collect_files(args.train_glob)
    val_files = collect_files(args.val_glob)

    if not train_files:
        raise FileNotFoundError(f'no train files matched: {args.train_glob}')
    if not val_files:
        raise FileNotFoundError(f'no validation files matched: {args.val_glob}')

    print(f'Found {len(train_files)} train files and {len(val_files)} val files.')

    random.shuffle(train_files)
    random.shuffle(val_files)

    out_dir = Path(args.out_dir)
    train_out_dir = out_dir / 'train'
    val_out_dir = out_dir / 'val'
    train_out_dir.mkdir(parents=True, exist_ok=True)
    val_out_dir.mkdir(parents=True, exist_ok=True)

    tasks = [
        *build_tasks(train_files, 'train', train_out_dir, args.chunk_size),
        *build_tasks(val_files, 'val', val_out_dir, args.chunk_size),
    ]

    completed = 0
    total_samples = 0

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_chunk, chunk_id, chunk_files, output_path, args.dtype):
            (chunk_id, split_name)
            for chunk_id, chunk_files, output_path, split_name in tasks
        }
        for future in as_completed(futures):
            chunk_id, split_name = futures[future]
            _, num_samples, success, err = future.result()
            completed += 1
            if success:
                total_samples += num_samples
                print(
                    f'[{split_name}] completed {completed}/{len(tasks)} chunks, '
                    f'saved {num_samples} games, total {total_samples}'
                )
            else:
                print(f'[{split_name}] chunk {chunk_id} failed: {err}')


if __name__ == '__main__':
    main()
