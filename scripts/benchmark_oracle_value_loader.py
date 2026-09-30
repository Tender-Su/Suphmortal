import argparse
import json
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mortal.online.pretrain_oracle_critic import (
    build_file_splits,
    data_stream_signature,
    initial_data_progress,
    load_file_index,
    make_dataset,
    make_loader,
    oracle_pretrain_cfg,
    sanitize_sys_path_for_spawn,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Benchmark the Oracle critic training DataLoader.'
    )
    parser.add_argument('--fold-count', type=int, required=True)
    parser.add_argument('--warmup-batches', type=int, default=8)
    parser.add_argument('--batches', type=int, default=100)
    parser.add_argument('--num-workers', type=int, default=None)
    parser.add_argument('--file-batch-size', type=int, default=None)
    parser.add_argument('--prefetch-factor', type=int, default=None)
    parser.add_argument(
        '--file-index',
        default='',
        help='Benchmark this index directly instead of rebuilding train/dev/test splits.',
    )
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--label', default='')
    parser.add_argument('--output', default='')
    parser.add_argument('--validation', action='store_true')
    parser.add_argument('--exhaust', action='store_true')
    return parser.parse_args()


def main():
    sanitize_sys_path_for_spawn()
    args = parse_args()
    if args.fold_count <= 0:
        raise ValueError('--fold-count must be positive')
    if args.warmup_batches < 0 or args.batches <= 0:
        raise ValueError('warmup-batches must be non-negative and batches must be positive')

    cfg = dict(oracle_pretrain_cfg())
    if args.batch_size is not None:
        cfg['batch_size'] = args.batch_size
    fold_key = 'val_state_fold_count' if args.validation else 'state_fold_count'
    cfg[fold_key] = args.fold_count
    cfg['state_fold_backend'] = 'native_hash'
    prefix = 'val_' if args.validation else ''
    for key, value in (
        (f'{prefix}num_workers', args.num_workers),
        (f'{prefix}file_batch_size', args.file_batch_size),
        (f'{prefix}prefetch_factor', args.prefetch_factor),
    ):
        if value is not None:
            cfg[key] = value
    if args.file_index:
        selected_files = load_file_index(args.file_index)
        if selected_files is None:
            raise FileNotFoundError(f'invalid --file-index: {args.file_index}')
    else:
        train_files, dev_files, _test_files = build_file_splits(cfg)
        selected_files = dev_files if args.validation else train_files
    stream_state = initial_data_progress(data_stream_signature(cfg))
    loader = make_loader(
        make_dataset(
            selected_files,
            cfg,
            train=not args.validation,
            stream_state=stream_state,
        ),
        cfg,
        train=not args.validation,
    )
    iterator = iter(loader)
    warmup_batches = 0 if args.exhaust else args.warmup_batches
    for _ in range(warmup_batches):
        next(iterator)

    batches = 0
    samples = 0
    game_ids = set()
    started = time.perf_counter()
    selected_batches = iterator if args.exhaust else (next(iterator) for _ in range(args.batches))
    for batch in selected_batches:
        batches += 1
        samples += int(batch[0].shape[0])
        if args.validation and len(batch) > 4:
            game_ids.update(int(game_id) for game_id in batch[4].reshape(-1).tolist())
    elapsed = time.perf_counter() - started
    result = {
        'label': args.label,
        'backend': 'native_hash',
        'split': 'dev' if args.validation else 'train',
        'fold_count': args.fold_count,
        'files': len(selected_files),
        'num_workers': int(cfg.get(f'{prefix}num_workers', 0) or 0),
        'file_batch_size': int(cfg.get(f'{prefix}file_batch_size', 1) or 1),
        'batch_size': int(cfg.get('batch_size', 1) or 1),
        'warmup_batches': warmup_batches,
        'batches': batches,
        'samples': samples,
        'games': len(game_ids),
        'exhausted': bool(args.exhaust),
        'elapsed_seconds': elapsed,
        'batches_per_second': batches / elapsed,
        'samples_per_second': samples / elapsed,
    }
    encoded = json.dumps(result, sort_keys=True)
    if args.output:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = REPO_ROOT / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open('a', encoding='utf-8') as output:
            output.write(encoded + '\n')
    print(encoded)


if __name__ == '__main__':
    main()
