import prelude

import argparse
import json
import os
import secrets
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np
import torch

from config import config
from engine import MortalEngine
from libriichi.arena import OneVsThree
from model import Brain, CategoricalPolicy

SCRIPT_PATH = Path(__file__).resolve()
SCRIPT_DIR = SCRIPT_PATH.parent

DEFAULT_1V3_GPU_SEED_COUNTS = {
    'NVIDIA GeForce RTX 5070 Ti': 1024,
    'NVIDIA GeForce RTX 4060 Laptop GPU': 640,
}

DEFAULT_1V3_GPU_SHARD_COUNTS = {
    'NVIDIA GeForce RTX 5070 Ti': 4,
    'NVIDIA GeForce RTX 4060 Laptop GPU': 3,
}


def _coerce_positive_int(value, *, field_name):
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f'{field_name} must be positive, got {parsed}')
    return parsed


def _resolve_cfg_seed_count(override_cfg, *, source_name):
    if not isinstance(override_cfg, dict):
        return None
    if 'seed_count' in override_cfg:
        return _coerce_positive_int(override_cfg['seed_count'], field_name=f'{source_name}.seed_count')
    if 'games_per_iter' in override_cfg:
        games_per_iter = _coerce_positive_int(
            override_cfg['games_per_iter'],
            field_name=f'{source_name}.games_per_iter',
        )
        if games_per_iter % 4 != 0:
            raise ValueError(f'{source_name}.games_per_iter must be divisible by 4, got {games_per_iter}')
        return games_per_iter // 4
    return None


def _resolve_cfg_shard_count(override_cfg, *, source_name):
    if not isinstance(override_cfg, dict):
        return None
    if 'shard_count' not in override_cfg:
        return None
    return _coerce_positive_int(override_cfg['shard_count'], field_name=f'{source_name}.shard_count')


def resolve_gpu_name(cfg):
    challenger_device = str(cfg.get('challenger', {}).get('device', 'cpu'))
    if not challenger_device.startswith('cuda'):
        return None
    if not torch.cuda.is_available():
        return None
    return torch.cuda.get_device_name(torch.device(challenger_device))


def resolve_seed_count(cfg):
    if (env_override := os.environ.get('MORTAL_1V3_SEED_COUNT')):
        return (
            _coerce_positive_int(env_override, field_name='MORTAL_1V3_SEED_COUNT'),
            f'env MORTAL_1V3_SEED_COUNT={env_override}',
        )

    computer_name = os.environ.get('COMPUTERNAME', '')
    machine_overrides = cfg.get('machine_overrides', {})
    if computer_name:
        machine_seed_count = _resolve_cfg_seed_count(
            machine_overrides.get(computer_name),
            source_name=f'[1v3.machine_overrides.{computer_name}]',
        )
        if machine_seed_count is not None:
            return machine_seed_count, f'config machine override for {computer_name}'

    gpu_name = resolve_gpu_name(cfg)
    gpu_overrides = cfg.get('gpu_overrides', {})
    if gpu_name:
        gpu_seed_count = _resolve_cfg_seed_count(
            gpu_overrides.get(gpu_name),
            source_name=f'[1v3.gpu_overrides.{gpu_name}]',
        )
        if gpu_seed_count is not None:
            return gpu_seed_count, f'config gpu override for {gpu_name}'

        default_seed_count = DEFAULT_1V3_GPU_SEED_COUNTS.get(gpu_name)
        if default_seed_count is not None:
            return default_seed_count, f'built-in gpu default for {gpu_name}'

    seed_count = _resolve_cfg_seed_count(cfg, source_name='[1v3]')
    if seed_count is not None:
        return seed_count, 'config [1v3]'

    games_per_iter = _coerce_positive_int(cfg['games_per_iter'], field_name='[1v3].games_per_iter')
    if games_per_iter % 4 != 0:
        raise ValueError(f'[1v3].games_per_iter must be divisible by 4, got {games_per_iter}')
    return games_per_iter // 4, 'config [1v3].games_per_iter / 4'


def resolve_shard_count(cfg):
    if (env_override := os.environ.get('MORTAL_1V3_SHARD_COUNT')):
        return (
            _coerce_positive_int(env_override, field_name='MORTAL_1V3_SHARD_COUNT'),
            f'env MORTAL_1V3_SHARD_COUNT={env_override}',
        )

    computer_name = os.environ.get('COMPUTERNAME', '')
    machine_overrides = cfg.get('machine_overrides', {})
    if computer_name:
        machine_shard_count = _resolve_cfg_shard_count(
            machine_overrides.get(computer_name),
            source_name=f'[1v3.machine_overrides.{computer_name}]',
        )
        if machine_shard_count is not None:
            return machine_shard_count, f'config machine override for {computer_name}'

    gpu_name = resolve_gpu_name(cfg)
    gpu_overrides = cfg.get('gpu_overrides', {})
    if gpu_name:
        gpu_shard_count = _resolve_cfg_shard_count(
            gpu_overrides.get(gpu_name),
            source_name=f'[1v3.gpu_overrides.{gpu_name}]',
        )
        if gpu_shard_count is not None:
            return gpu_shard_count, f'config gpu override for {gpu_name}'

        default_shard_count = DEFAULT_1V3_GPU_SHARD_COUNTS.get(gpu_name)
        if default_shard_count is not None:
            return default_shard_count, f'built-in gpu default for {gpu_name}'

    shard_count = _resolve_cfg_shard_count(cfg, source_name='[1v3]')
    if shard_count is not None:
        return shard_count, 'config [1v3]'

    return 1, 'implicit default shard_count=1'


def plan_shards(total_seed_count, requested_shard_count):
    total_seed_count = _coerce_positive_int(total_seed_count, field_name='total_seed_count')
    requested_shard_count = _coerce_positive_int(requested_shard_count, field_name='requested_shard_count')
    shard_count = min(total_seed_count, requested_shard_count)
    base, remainder = divmod(total_seed_count, shard_count)
    return [base + (1 if index < remainder else 0) for index in range(shard_count)]


def load_mortal_engine(engine_cfg):
    state = torch.load(engine_cfg['state_file'], weights_only=True, map_location=torch.device('cpu'))
    saved_cfg = state['config']
    version = saved_cfg['control'].get('version', 1)
    conv_channels = saved_cfg['resnet']['conv_channels']
    num_blocks = saved_cfg['resnet']['num_blocks']
    mortal = Brain(version=version, num_blocks=num_blocks, conv_channels=conv_channels, Norm='GN').eval()
    dqn = CategoricalPolicy().eval()
    mortal.load_state_dict(state['mortal'])
    dqn.load_state_dict(state['policy_net'])
    if engine_cfg['enable_compile']:
        mortal.compile()
        dqn.compile()
    engine = MortalEngine(
        mortal,
        dqn,
        is_oracle=False,
        version=version,
        device=torch.device(engine_cfg['device']),
        enable_amp=engine_cfg['enable_amp'],
        enable_rule_based_agari_guard=engine_cfg['enable_rule_based_agari_guard'],
        name=engine_cfg['name'],
    )
    return engine


def summarize_rankings(rankings):
    rankings = np.array(rankings, dtype=np.int64)
    avg_rank = rankings @ np.arange(1, 5) / rankings.sum()
    avg_pt = rankings @ np.array([90, 45, 0, -135]) / rankings.sum()
    return rankings, float(avg_rank), float(avg_pt)


def load_eval_engines(cfg):
    engine_cham = load_mortal_engine(cfg['champion'])
    engine_chal = load_mortal_engine(cfg['challenger'])
    return {
        'engine_cham': engine_cham,
        'engine_chal': engine_chal,
    }


def run_eval_once(*, cfg, seed_start, seed_key, seed_count, log_dir, disable_progress_bar, eval_context=None):
    if eval_context is None:
        eval_context = load_eval_engines(cfg)
    engine_cham = eval_context['engine_cham']
    engine_chal = eval_context['engine_chal']
    env = OneVsThree(
        disable_progress_bar=disable_progress_bar,
        log_dir=log_dir,
    )
    rankings = env.py_vs_py(
        challenger=engine_chal,
        champion=engine_cham,
        seed_start=(seed_start, seed_key),
        seed_count=seed_count,
    )
    return rankings


def build_worker_command(*, seed_start, seed_key, seed_count, result_json, log_dir, shard_index):
    command = [
        sys.executable,
        str(SCRIPT_PATH),
        '--worker',
        '--seed-start',
        str(seed_start),
        '--seed-key',
        str(seed_key),
        '--seed-count',
        str(seed_count),
        '--result-json',
        str(result_json),
        '--shard-index',
        str(shard_index),
        '--disable-progress-bar',
    ]
    if log_dir is not None:
        command.extend(['--log-dir', str(log_dir)])
    return command


def normalize_child_env():
    env = os.environ.copy()
    config_override = env.get('MORTAL_CFG')
    if config_override:
        env['MORTAL_CFG'] = str(Path(config_override).resolve())
    return env


def build_worker_loop_command(*, shard_index, disable_progress_bar):
    command = [
        sys.executable,
        str(SCRIPT_PATH),
        '--worker-loop',
        '--shard-index',
        str(shard_index),
    ]
    if disable_progress_bar:
        command.append('--disable-progress-bar')
    return command


def read_worker_loop_message(worker_entry):
    line = worker_entry['process'].stdout.readline()
    if line == '':
        exit_code = worker_entry['process'].poll()
        log_excerpt = ''
        if worker_entry['log_path'].exists():
            log_excerpt = worker_entry['log_path'].read_text(encoding='utf-8', errors='replace')
        raise RuntimeError(
            '1v3 shard worker closed stdout unexpectedly: '
            f"shard={worker_entry['shard_index']} exit_code={exit_code} log={worker_entry['log_path']}\n{log_excerpt}"
        )
    return json.loads(line)


def run_worker_loop(args):
    cfg = config['1v3']
    eval_context = load_eval_engines(cfg)
    print(
        json.dumps({
            'kind': 'ready',
            'shard_index': args.shard_index,
        }),
        flush=True,
    )
    for raw_line in sys.stdin:
        task = json.loads(raw_line)
        if str(task.get('kind', '')) == 'stop':
            break
        if str(task.get('kind', '')) != 'run':
            continue
        try:
            rankings = run_eval_once(
                cfg=cfg,
                seed_start=int(task['seed_start']),
                seed_key=int(task['seed_key']),
                seed_count=int(task['seed_count']),
                log_dir=task.get('log_dir'),
                disable_progress_bar=args.disable_progress_bar,
                eval_context=eval_context,
            )
            rankings, avg_rank, avg_pt = summarize_rankings(rankings)
            print(
                json.dumps({
                    'kind': 'result',
                    'iter_index': int(task['iter_index']),
                    'shard_index': int(args.shard_index),
                    'seed_start': int(task['seed_start']),
                    'seed_count': int(task['seed_count']),
                    'rankings': rankings.tolist(),
                    'avg_rank': avg_rank,
                    'avg_pt': avg_pt,
                }),
                flush=True,
            )
        except BaseException as exc:
            print(
                json.dumps({
                    'kind': 'task_error',
                    'iter_index': int(task['iter_index']),
                    'shard_index': int(args.shard_index),
                    'seed_start': int(task['seed_start']),
                    'seed_count': int(task['seed_count']),
                    'error': str(exc),
                    'traceback': traceback.format_exc(),
                }),
                flush=True,
            )
            break


def stop_persistent_shard_workers(shard_worker_pool):
    if shard_worker_pool is None:
        return
    for entry in shard_worker_pool['workers']:
        try:
            if entry['process'].poll() is None and entry['process'].stdin:
                entry['process'].stdin.write(json.dumps({'kind': 'stop'}) + '\n')
                entry['process'].stdin.flush()
        except Exception:
            continue
    for entry in shard_worker_pool['workers']:
        process = entry['process']
        if process.stdin:
            try:
                process.stdin.close()
            except Exception:
                pass
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.terminate()
    for entry in shard_worker_pool['workers']:
        process = entry['process']
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        if process.stdout:
            try:
                process.stdout.close()
            except Exception:
                pass
        try:
            entry['log_handle'].close()
        except Exception:
            pass


def start_persistent_shard_workers(*, cfg, shard_count, disable_progress_bar):
    runtime_root = Path(tempfile.mkdtemp(prefix='mahjongai_1v3_worker_pool_'))
    child_env = normalize_child_env()
    workers = []
    for shard_index in range(shard_count):
        log_path = runtime_root / f'shard_{shard_index:02d}.log'
        log_handle = log_path.open('w', encoding='utf-8', newline='\n')
        process = subprocess.Popen(
            build_worker_loop_command(
                shard_index=shard_index,
                disable_progress_bar=disable_progress_bar,
            ),
            cwd=str(SCRIPT_DIR),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=log_handle,
            text=True,
            env=child_env,
        )
        workers.append({
            'shard_index': shard_index,
            'process': process,
            'log_path': log_path,
            'log_handle': log_handle,
        })

    try:
        for entry in workers:
            message = read_worker_loop_message(entry)
            if str(message.get('kind', '')) != 'ready':
                raise RuntimeError(
                    'unexpected shard worker startup message: '
                    f"shard={entry['shard_index']} message={message}"
                )
    except BaseException:
        stop_persistent_shard_workers({
            'workers': workers,
        })
        raise

    return {
        'workers': workers,
        'runtime_root': runtime_root,
    }


def run_sharded_iteration_with_workers(
    *,
    iter_index,
    seed_start,
    seed_key,
    shard_seed_counts,
    log_dir,
    shard_worker_pool,
):
    runtime_root = Path(tempfile.mkdtemp(prefix=f'mahjongai_1v3_iter_{iter_index:04d}_'))
    workers = shard_worker_pool['workers']
    if len(workers) != len(shard_seed_counts):
        raise ValueError(
            f'shard worker count mismatch: workers={len(workers)} shard_seed_counts={len(shard_seed_counts)}'
        )

    seed_cursor = seed_start
    for worker_entry, shard_seed_count in zip(workers, shard_seed_counts, strict=True):
        child_log_dir = None
        if log_dir is not None:
            child_log_dir = str(Path(log_dir) / f'iter_{iter_index:04d}' / f'shard_{worker_entry["shard_index"]:02d}')
        worker_entry['process'].stdin.write(json.dumps({
            'kind': 'run',
            'iter_index': iter_index,
            'seed_start': seed_cursor,
            'seed_key': seed_key,
            'seed_count': shard_seed_count,
            'log_dir': child_log_dir,
        }) + '\n')
        worker_entry['process'].stdin.flush()
        seed_cursor += shard_seed_count

    shard_payloads: dict[int, dict] = {}
    for worker_entry in workers:
        message = read_worker_loop_message(worker_entry)
        kind = str(message.get('kind', ''))
        if kind != 'result' and kind != 'task_error':
            raise RuntimeError(f'unexpected shard worker message: {message}')
        if int(message.get('iter_index', -1)) != iter_index:
            raise RuntimeError(f'unexpected shard worker iter index: {message}')
        shard_index = int(message['shard_index'])
        if kind == 'task_error':
            log_path = runtime_root / f'shard_{shard_index:02d}.log'
            log_path.write_text(str(message.get('traceback') or message.get('error') or ''), encoding='utf-8')
            raise RuntimeError(
                '1v3 shard failed: '
                f"iter={iter_index} shard={shard_index} seed_start={message['seed_start']} "
                f"seed_count={message['seed_count']} log={log_path}"
            )
        shard_payloads[shard_index] = {
            'rankings': list(message['rankings']),
            'avg_rank': float(message['avg_rank']),
            'avg_pt': float(message['avg_pt']),
        }
        result_path = runtime_root / f'shard_{shard_index:02d}.json'
        result_path.write_text(json.dumps(shard_payloads[shard_index], ensure_ascii=False, indent=2), encoding='utf-8')

    total_rankings = np.zeros(4, dtype=np.int64)
    for shard_index in range(len(workers)):
        total_rankings += np.array(shard_payloads[shard_index]['rankings'], dtype=np.int64)
    rankings, avg_rank, avg_pt = summarize_rankings(total_rankings)
    return rankings, avg_rank, avg_pt, runtime_root


def run_sharded_iteration(*, cfg, iter_index, seed_start, seed_key, seed_count, shard_seed_counts, log_dir):
    runtime_root = Path(tempfile.mkdtemp(prefix=f'mahjongai_1v3_iter_{iter_index:04d}_'))
    child_env = normalize_child_env()
    processes = []
    seed_cursor = seed_start

    try:
        for shard_index, shard_seed_count in enumerate(shard_seed_counts):
            result_json = runtime_root / f'shard_{shard_index:02d}.json'
            log_path = runtime_root / f'shard_{shard_index:02d}.log'
            child_log_dir = None
            if log_dir is not None:
                child_log_dir = Path(log_dir) / f'iter_{iter_index:04d}' / f'shard_{shard_index:02d}'
            command = build_worker_command(
                seed_start=seed_cursor,
                seed_key=seed_key,
                seed_count=shard_seed_count,
                result_json=result_json,
                log_dir=child_log_dir,
                shard_index=shard_index,
            )
            log_handle = log_path.open('w', encoding='utf-8', newline='\n')
            process = subprocess.Popen(
                command,
                cwd=str(SCRIPT_DIR),
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                env=child_env,
            )
            log_handle.close()
            processes.append(
                {
                    'process': process,
                    'result_json': result_json,
                    'log_path': log_path,
                    'seed_start': seed_cursor,
                    'seed_count': shard_seed_count,
                    'shard_index': shard_index,
                }
            )
            seed_cursor += shard_seed_count

        shard_payloads = []
        for entry in processes:
            return_code = entry['process'].wait()
            if return_code != 0:
                raise RuntimeError(
                    '1v3 shard failed: '
                    f"iter={iter_index} shard={entry['shard_index']} seed_start={entry['seed_start']} "
                    f"seed_count={entry['seed_count']} log={entry['log_path']}"
                )
            with entry['result_json'].open(encoding='utf-8') as f:
                shard_payloads.append(json.load(f))

        total_rankings = np.zeros(4, dtype=np.int64)
        for payload in shard_payloads:
            total_rankings += np.array(payload['rankings'], dtype=np.int64)
        rankings, avg_rank, avg_pt = summarize_rankings(total_rankings)
        return rankings, avg_rank, avg_pt, runtime_root
    except BaseException:
        for entry in processes:
            process = entry['process']
            if process.poll() is None:
                process.terminate()
        for entry in processes:
            process = entry['process']
            if process.poll() is None:
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
        raise


def run_worker(args):
    cfg = config['1v3']
    rankings = run_eval_once(
        cfg=cfg,
        seed_start=args.seed_start,
        seed_key=args.seed_key,
        seed_count=args.seed_count,
        log_dir=args.log_dir,
        disable_progress_bar=args.disable_progress_bar,
    )
    rankings, avg_rank, avg_pt = summarize_rankings(rankings)
    payload = {
        'seed_start': args.seed_start,
        'seed_key': args.seed_key,
        'seed_count': args.seed_count,
        'shard_index': args.shard_index,
        'rankings': rankings.tolist(),
        'avg_rank': avg_rank,
        'avg_pt': avg_pt,
    }
    result_path = Path(args.result_json)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')


def run_main(args):
    cfg = config['1v3']
    seeds_per_iter, seed_count_source = resolve_seed_count(cfg)
    games_per_iter = seeds_per_iter * 4
    shard_count, shard_count_source = resolve_shard_count(cfg)
    shard_seed_counts = plan_shards(seeds_per_iter, shard_count)
    effective_shard_count = len(shard_seed_counts)
    iters = _coerce_positive_int(cfg['iters'], field_name='[1v3].iters')
    log_dir = cfg.get('log_dir') or None
    disable_progress_bar = bool(cfg.get('disable_progress_bar', False))

    if (key := cfg.get('seed_key', -1)) == -1:
        key = secrets.randbits(64)

    print(f'1v3 seed_count={seeds_per_iter} ({games_per_iter} games/iter) from {seed_count_source}')
    print(f'1v3 shard_count={effective_shard_count} from {shard_count_source}; shard seeds={shard_seed_counts}')
    if log_dir is None:
        print('1v3 log_dir disabled')

    single_shard_eval_context = None
    shard_worker_pool = None
    if effective_shard_count == 1:
        single_shard_eval_context = load_eval_engines(cfg)
    else:
        shard_worker_pool = start_persistent_shard_workers(
            cfg=cfg,
            shard_count=effective_shard_count,
            disable_progress_bar=disable_progress_bar,
        )

    try:
        seed_start = 10000
        for iter_index, seed in enumerate(range(seed_start, seed_start + seeds_per_iter * iters, seeds_per_iter)):
            print('-' * 50)
            print('#', iter_index)
            if effective_shard_count == 1:
                iteration_log_dir = log_dir
                rankings = run_eval_once(
                    cfg=cfg,
                    seed_start=seed,
                    seed_key=key,
                    seed_count=seeds_per_iter,
                    log_dir=iteration_log_dir,
                    disable_progress_bar=disable_progress_bar,
                    eval_context=single_shard_eval_context,
                )
                rankings, avg_rank, avg_pt = summarize_rankings(rankings)
            else:
                rankings, avg_rank, avg_pt, runtime_root = run_sharded_iteration_with_workers(
                    iter_index=iter_index,
                    seed_start=seed,
                    seed_key=key,
                    shard_seed_counts=shard_seed_counts,
                    log_dir=log_dir,
                    shard_worker_pool=shard_worker_pool,
                )
                print(f'shard runtime: {runtime_root}')
            print(f'challenger rankings: {rankings} ({avg_rank}, {avg_pt}pt)')
    finally:
        stop_persistent_shard_workers(shard_worker_pool)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--worker-loop', action='store_true')
    parser.add_argument('--seed-start', type=int)
    parser.add_argument('--seed-key', type=int)
    parser.add_argument('--seed-count', type=int)
    parser.add_argument('--result-json')
    parser.add_argument('--log-dir', default=None)
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--disable-progress-bar', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    if args.worker_loop:
        run_worker_loop(args)
        return
    if args.worker:
        required = {
            'seed_start': args.seed_start,
            'seed_key': args.seed_key,
            'seed_count': args.seed_count,
            'result_json': args.result_json,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ValueError(f'missing worker args: {", ".join(missing)}')
        run_worker(args)
        return
    run_main(args)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
