"""Run a bounded, isolated local PPO pipeline for resource diagnostics only."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time

from profile_oracle_runtime import Sampler, digest, dump


def child(args):
    config_file = Path(args.output).resolve() / 'config.toml'
    os.environ.update(MORTAL_CFG=str(config_file), MORTAL_ORACLE_ARM='current_config',
                      MORTAL_ORACLE_ARTIFACT_SUFFIX='', PYTHONDONTWRITEBYTECODE='1')
    sys.path.insert(0, str(Path(args.source).resolve()))
    if args.windows_high_qos:
        from mortal.core.process_resources import configure_windows_high_qos
        configure_windows_high_qos(True)
    import torch
    torch.set_num_threads(args.torch_threads)
    torch.set_num_interop_threads(1)
    if args.role != 'server':
        torch.cuda.set_per_process_memory_fraction(.68 if args.role == 'trainer' else .17)
    output = Path(args.output).resolve()
    if args.role == 'trainer':
        from mortal.online import train_online as trainer
        from mortal.config import config
        from mortal.eval.oracle_experiments import apply_oracle_experiment_to_config
        apply_oracle_experiment_to_config(config)
        original_stop = trainer.online_reached_max_steps
        original_drift = trainer.policy_drift
        original_usable = trainer.behavior_version_is_usable
        original_optimizer_step = torch.optim.AdamW.step
        started = time.perf_counter()
        updates, drifts, optimizer_updates, behavior_checks = [], [], [], []
        with (output / 'trainer_trace.jsonl').open('w', encoding='utf-8') as trace:
            def reached(configuration, steps):
                row = {'event': 'accepted_step', 'step': int(steps),
                       'elapsed_s': time.perf_counter() - started}
                updates.append(row)
                trace.write(json.dumps(row) + '\n')
                trace.flush()
                return len(optimizer_updates) >= args.steps or original_stop(configuration, steps)

            def drift(*values, **keywords):
                result = original_drift(*values, **keywords)
                drifts.append({key: float(value.item()) for key, value in result.items()})
                return result

            def optimizer_step(optimizer, *values, **keywords):
                result = original_optimizer_step(optimizer, *values, **keywords)
                optimizer_updates.append(time.perf_counter() - started)
                trace.write(json.dumps({'event': 'optimizer_update', 'elapsed_s': optimizer_updates[-1]}) + '\n')
                trace.flush()
                return result

            def usable(value, history, *, published_version, max_gap):
                result = original_usable(value, history, published_version=published_version, max_gap=max_gap)
                behavior_checks.append({'version': value, 'published_version': published_version,
                                        'usable': result})
                return result

            trainer.online_reached_max_steps = reached
            trainer.policy_drift = drift
            trainer.behavior_version_is_usable = usable
            torch.optim.AdamW.step = optimizer_step
            try:
                trainer.train()
            finally:
                dump(output / 'trainer_result.json', {
                    'updates': updates, 'drifts': drifts, 'optimizer_updates': optimizer_updates,
                    'behavior_checks': behavior_checks,
                    'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(),
                    'peak_cuda_reserved_bytes': torch.cuda.max_memory_reserved(),
                })
    else:
        import importlib
        if args.role == 'client':
            from mortal.eval.player import TrainPlayer
            original_play = TrainPlayer.train_play
            def play(player, *values, **keywords):
                start = time.perf_counter()
                result = original_play(player, *values, **keywords)
                with (output / 'selfplay_trace.jsonl').open('a', encoding='utf-8') as trace:
                    trace.write(json.dumps({'wall_s': time.perf_counter() - start,
                                            'games': len(result[1]), 'time': time.time()}) + '\n')
                return result
            TrainPlayer.train_play = play
        importlib.import_module('mortal.online.' + args.role).main()


def configuration(args, output):
    import toml
    cfg = toml.loads(Path(args.template).read_text(encoding='utf-8'))
    actor, critic = str(Path(args.actor).resolve()), str(Path(args.critic).resolve())
    control = cfg['control']
    control.update(online=True, state_file=str(output / 'latest.pth'),
                   best_state_file=str(output / 'best.pth'),
                   tensorboard_dir=str(output / 'tb_log'), batch_size=192,
                   enable_compile=False, device='cuda:0')
    cfg['test_play'].update(enable=False, initial_enable=False, log_dir=str(output / 'test_play'))
    cfg['repro'].update(enabled=True, seed=20260908, strict_cuda=False, allow_cudnn_benchmark=True)
    cfg['online'].update(init_state_file=actor, stop_at_max_steps=True,
                         gae_inference_batch_size=args.infer_batch, enable_compile=False)
    cfg['online']['remote'] = {'host': '127.0.0.1', 'port': args.port,
                               'connect_max_wait_sec': 90, 'connect_retry_sec': 1}
    cfg['online']['server'].update(buffer_dir=str(output / 'buffer'), drain_dir=str(output / 'drain'),
                                   capacity=args.games, force_sequential=getattr(args, 'sequential', True),
                                   sample_reuse_rate=0, sample_reuse_threshold=0)
    cfg['value'].update(enabled=True, oracle_critic=True, oracle_critic_state_file=critic,
                        oracle_critic_arch='dual_tower', oracle_fusion_mode='residual_mlp',
                        oracle_fusion_hidden=1024, value_head_hidden=256,
                        value_loss_mode='mse', exact_zero_sum=True,
                        target_mode='all_players', reward_source='score_rank')
    cfg['oracle_experiments'].update(default_arm='current_config', suffix_artifacts=False)
    for name in ('search', 'search_distill', 'oracle_dependency_eval', 'expected_reward'):
        cfg[name]['enabled'] = False
    cfg['aux']['danger_enabled'] = False
    cfg['baseline'] = {
        'train': {'device': 'cuda:0', 'enable_compile': False, 'state_file': actor},
        'test': {'device': 'cuda:0', 'enable_compile': False, 'state_file': actor},
    }
    cfg['train_play'] = {'default': {'games': args.games, 'log_dir': str(output / 'train_play'),
                                    'repeats': 1, 'explore_rate': 1.0}}
    # The diagnostic stop is a runtime hook. The LR horizon, publication cadence,
    # logical batch, GAE grouping, PPO guardrails and objective remain fixed.
    return cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--manifest')
    parser.add_argument('--template')
    parser.add_argument('--actor')
    parser.add_argument('--critic')
    parser.add_argument('--output', required=True)
    parser.add_argument('--games', type=int, default=64)
    parser.add_argument('--infer-batch', type=int, default=512)
    parser.add_argument('--rayon', type=int, default=4)
    parser.add_argument('--torch-threads', type=int, default=1)
    parser.add_argument('--steps', type=int, default=96)
    parser.add_argument('--max-seconds', type=float, default=600)
    parser.add_argument('--min-available-gib', type=float, default=5)
    parser.add_argument('--port', type=int, default=5171)
    parser.add_argument('--sequential', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--windows-high-qos', action='store_true')
    parser.add_argument('--role', choices=('server', 'trainer', 'client'))
    args = parser.parse_args()
    if not 0 < args.steps <= 1000 or args.games < 4 or args.games % 4:
        parser.error('invalid bounded diagnostic size')
    if not 4 <= args.min_available_gib <= 8:
        parser.error('RAM reserve must be between 4 and 8 GiB')
    if args.role:
        child(args)
        return
    import psutil
    import toml
    output, source = Path(args.output).resolve(), Path(args.source).resolve()
    if output.is_relative_to(source) or source.is_relative_to(output):
        raise ValueError('diagnostic output must be separate from frozen source')
    manifest = json.loads(Path(args.manifest).read_text(encoding='utf-8'))
    if Path(manifest['source_root']).resolve() != source:
        raise ValueError('source root differs from manifest')
    for name, expected in manifest['source_sha256'].items():
        if digest(source / name) != expected:
            raise ValueError('frozen source changed: ' + name)
    if psutil.virtual_memory().available < 10 * 2**30:
        raise RuntimeError('pipeline diagnostic requires 10 GiB available RAM')
    for process in psutil.process_iter(['name', 'cmdline']):
        command = ' '.join(process.info['cmdline'] or [])
        if (process.info['name'] or '').lower() in ('r5apex.exe', 'r5apex_dx12.exe'):
            raise RuntimeError('Apex is running')
        if process.pid != os.getpid() and (process.info['name'] or '').lower() == 'python.exe':
            if ('mortal.online.pretrain_oracle_critic' in command or
                    'benchmark_oracle_training.py' in command):
                raise RuntimeError('another Oracle training workload is running')
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', args.port))
    output.mkdir(parents=True, exist_ok=False)
    cfg = configuration(args, output)
    (output / 'config.toml').write_text(toml.dumps(cfg), encoding='utf-8', newline='\n')
    dump(output / 'identity.json', {
        'arguments': vars(args), 'scientific_use': False,
        'started_at': datetime.now(timezone.utc).isoformat(),
        'actor_sha256': digest(args.actor), 'critic_sha256': digest(args.critic),
        'source_manifest_sha256': digest(args.manifest), 'script_sha256': digest(__file__),
        'config_sha256': digest(output / 'config.toml'),
        'comparability': 'Fixed seeded stochastic protocol; changing native game parallelism can change sampled events.',
    })
    environment = dict(os.environ, PYTHONPATH=str(source), RAYON_NUM_THREADS=str(args.rayon),
                       OMP_NUM_THREADS=str(args.torch_threads), MKL_NUM_THREADS=str(args.torch_threads),
                       PYTHONDONTWRITEBYTECODE='1')
    environment.pop('MORTAL_ORACLE_PAUSE_FILE', None)
    if environment.get('MORTAL_CPU_AFFINITY'):
        raise RuntimeError('pipeline diagnostic does not opt in to CPU affinity')
    children, streams, latencies = {}, [], []
    delay_stop = threading.Event()
    def delay_probe():
        while not delay_stop.is_set():
            start = time.perf_counter()
            time.sleep(.02)
            latencies.append(max(0, time.perf_counter() - start - .02))
    def launch(role):
        log = (output / (role + '.log')).open('wb')
        streams.append(log)
        command = [sys.executable, '-u', str(Path(__file__).resolve()), '--source', str(source),
                   '--output', str(output), '--role', role, '--steps', str(args.steps),
                   '--torch-threads', str(args.torch_threads)]
        if args.windows_high_qos:
            command.append('--windows-high-qos')
        children[role] = subprocess.Popen(command, cwd=source, env=environment,
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        dump(output / 'processes.json', {key: value.pid for key, value in children.items()})
    reason, trainer_code = None, None
    started = time.perf_counter()
    with Sampler(output, os.getpid()) as sampler:
        delay_thread = threading.Thread(target=delay_probe, daemon=True)
        delay_thread.start()
        try:
            launch('server')
            deadline = time.monotonic() + 40
            while 'listening on ' not in (output / 'server.log').read_text(encoding='utf-8', errors='replace'):
                if children['server'].poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError('local diagnostic server did not start')
                time.sleep(.2)
            launch('trainer')
            launch('client')
            while reason is None:
                trainer_code = children['trainer'].poll()
                if trainer_code is not None:
                    reason = 'requested_updates_completed' if trainer_code == 86 else 'trainer_failed'
                elif any(children[role].poll() is not None for role in ('server', 'client')):
                    reason = 'pipeline_role_failed'
                elif time.perf_counter() - started > args.max_seconds:
                    reason = 'time_limit'
                elif sampler.rows and sampler.rows[-1]['available_ram_bytes'] < args.min_available_gib * 2**30:
                    reason = 'RAM_reserve'
                elif sampler.rows and sampler.rows[-1].get('gpu_used_mib', 0) > 14300:
                    reason = 'VRAM_reserve'
                elif any((p.info['name'] or '').lower() in ('r5apex.exe', 'r5apex_dx12.exe')
                         for p in psutil.process_iter(['name'])):
                    reason = 'Apex_started'
                else:
                    time.sleep(.5)
        finally:
            for process in children.values():
                if process.poll() is None:
                    process.terminate()
            for process in children.values():
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            for stream in streams:
                stream.close()
            delay_stop.set()
            delay_thread.join(timeout=2)
    latencies.sort()
    report = {'reason': reason, 'trainer_returncode': trainer_code,
              'minimum_available_ram_gib_guard': args.min_available_gib,
              'wall_s': time.perf_counter() - started, 'resources': sampler.report(),
              'scheduler_delay_p95_ms': 1000 * latencies[int((len(latencies) - 1) * .95)],
              'scheduler_delay_max_ms': 1000 * max(latencies)}
    result_file = output / 'trainer_result.json'
    if result_file.exists():
        result = json.loads(result_file.read_text(encoding='utf-8'))
        updates = result['updates']
        report.update(accepted_steps=updates[-1]['step'] if updates else 0,
                      accepted_steps_per_s=updates[-1]['step'] / report['wall_s'] if updates else 0,
                      optimizer_updates=len(result['optimizer_updates']),
                      optimizer_updates_per_s=len(result['optimizer_updates']) / report['wall_s'],
                      drift_batches=len(result['drifts']),
                      rejected_drift_batches=sum(row['approx_kl'] > .02 or row['clip_fraction'] > .5
                                                for row in result['drifts']),
                      peak_cuda_allocated_bytes=result['peak_cuda_allocated_bytes'],
                      peak_cuda_reserved_bytes=result['peak_cuda_reserved_bytes'])
    dump(output / 'result.json', report)
    print(json.dumps(report), flush=True)
    if reason != 'requested_updates_completed' or report.get('optimizer_updates') != args.steps:
        raise RuntimeError('pipeline probe did not finish all requested optimizer updates')


if __name__ == '__main__':
    main()
