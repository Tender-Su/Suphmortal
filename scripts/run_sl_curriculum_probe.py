"""Prepare/run matched A/B/C observations in an immutable local runtime."""
import argparse
from copy import deepcopy
import itertools
import json
import os
from pathlib import Path
import random
import shutil
from statistics import NormalDist
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from mortal.core.artifacts import atomic_torch_save, atomic_write_json, file_sha256, stable_json_digest
from mortal.core.toml_utils import load_toml_file, write_toml_file
from mortal.supervised.auxiliary_config import effective_auxiliary_recipe
from mortal.supervised.curriculum_probe import CurriculumProbe, RECIPES, make_domains
from mortal.supervised.lr_scheduler import LinearWarmUpConstantLR


def freeze_search_parser(text):
    """Repair only the new runtime; the main file is also used by a live bot.

    Remove this adapter once the one-line call-site fix can safely be integrated
    into mortal/eval/search_runtime.py. Record both fingerprints in the manifest.
    """
    before = '_as_bool(cfg.get("hard_only", True), True)'
    after = '_as_bool(cfg.get("hard_only", True), default=True)'
    if before in text:
        if text.count(before) != 1:
            raise ValueError('unexpected search parser repair scope')
        return text.replace(before, after)
    if after not in text:
        raise ValueError('unknown search parser source; review the compatibility repair')
    return text


def read_json(file):
    return json.loads(Path(file).read_text(encoding='utf-8'))


def prepare_branch_state(source, config):
    """Explicit experimental branch: keep Adam moments/AMP/aux clock, reset local counters.

    The only scheduler change is to freeze its CURRENT LR. Historical patience,
    sampler cursor and selection metrics do not belong to the new experiment.
    """
    if effective_auxiliary_recipe(source['config']) != effective_auxiliary_recipe(config):
        raise ValueError('branch auxiliary objective differs from parent')
    for name in ('mortal', 'policy_net', 'aux_net', 'opponent_aux_net', 'danger_aux_net',
                 'optimizer', 'optimizer_param_groups', 'scaler', 'scheduler'):
        if source.get(name) is None:
            raise ValueError(f'clean parent missing {name}')
    state = {name: deepcopy(source[name]) for name in (
        'mortal', 'policy_net', 'aux_net', 'opponent_aux_net', 'danger_aux_net',
        'optimizer', 'optimizer_param_groups', 'scaler',
    )}
    lr = config['supervised']['scheduler']['peak']
    groups = state['optimizer']['param_groups']
    if any(abs(group['lr'] - lr) > 1e-12 for group in groups):
        raise ValueError('first probe must preserve current parent LR in every parameter group')
    # Obtain the real scheduler serialization using small throwaway parameters.
    dummy = torch.optim.AdamW([{'params': [torch.nn.Parameter(torch.zeros(1))]} for _ in groups], lr=1)
    scheduler = LinearWarmUpConstantLR(dummy, peak=lr, init=lr, warm_up_steps=0)
    state.update(scheduler=scheduler.state_dict(), config=deepcopy(config),
                 config_section='supervised', checkpoint_id=stable_json_digest(config),
                 run_provenance=deepcopy(config['supervised']['run_provenance']),
                 steps=0, optimizer_steps=0, skipped_optimizer_steps=0, nonfinite_batches=0,
                 auxiliary_optimizer_steps=source.get('auxiliary_optimizer_steps', source['optimizer_steps']),
                 epoch=0, epoch_complete=False, timestamp=time.time(),
                 branch_parent={'checkpoint_id': source.get('checkpoint_id'),
                                'steps': source['steps'], 'optimizer_steps': source['optimizer_steps'],
                                'initialization': 'preserve_adam_amp_aux_clock_freeze_current_lr_new_data_cursor'})
    return state


def build_config(source, arm_root, index, *, seed, device, microbatch, logical_batch, identity,
                 runtime_performance=None):
    if logical_batch % microbatch or microbatch <= 0:
        raise ValueError('logical batch must be divisible by positive microbatch size')
    config = deepcopy(source['config'])
    control, sl = config['control'], config['supervised']
    control.update(device=device, opt_step_every=logical_batch // microbatch,
                   enable_cuda_prefetch=False, enable_cudnn_benchmark=False,
                   enable_compile=False, allow_tf32=False)
    if config.get('search', {}).get('enabled', False):
        raise ValueError('matched data-only probes require search disabled in parent')
    lr = source['optimizer']['param_groups'][0]['lr']
    config['dataset'].update(num_workers=0, reserve_ratio=0, enable_augmentation=True,
                             augmented_first=False)
    sl.update(batch_size=microbatch, val_batch_size=1024, file_index=str(index), num_workers=0,
              file_batch_size=1, val_file_batch_size=1, prefetch_factor=1, val_prefetch_factor=1,
              rayon_num_threads=2, train_in_order=True, val_in_order=True,
              force_safe_training=False, log_every=128, save_every=0, max_steps=0,
              val_every_steps=0, monitor_val_batches=0, full_val_every_checks=0,
              old_regression_every_checks=0, max_epochs=1, seed=seed,
              early_stopping_patience=0, early_stopping_patience_checks=0,
              init_state_file='', candidate_portfolio_dir='', milestone_checkpoint_dir='',
              tensorboard_dir=str(arm_root / 'tensorboard'),
              convergence={'enabled': False}, adaptive_curriculum={'enabled': False},
              gradient_calibration={'enabled': False},
              scheduler={'type': 'constant', 'peak': lr, 'init': lr, 'warm_up_steps': 0},
              run_provenance={'plan_id': identity, 'probe': True})
    for key in ('state_file', 'best_state_file', 'best_loss_state_file', 'best_acc_state_file',
                'best_rank_state_file', 'best_policy_state_file', 'adaptive_best_state_file'):
        sl[key] = str(arm_root / (key + '.pth'))
    if runtime_performance is not None:
        limits = {'probe_prepare_file_batch_size': (1, 2, 4),
                  'val_file_batch_size': (1, 2, 4, 8), 'rayon_num_threads': (1, 2, 4, 8)}
        optional_limits = {'probe_prepare_workers': tuple(range(9)),
                           'val_prepare_workers': tuple(range(9)),
                           'prepare_rayon_threads': (1, 2, 4, 8)}
        if not set(limits) <= set(runtime_performance) <= set(limits) | set(optional_limits) or any(
                type(runtime_performance[name]) is not int or runtime_performance[name] not in allowed
                for name, allowed in limits.items()):
            raise ValueError('only bounded preparation settings may change in a runtime continuation')
        if any(type(runtime_performance[name]) is not int or runtime_performance[name] not in allowed
               for name, allowed in optional_limits.items() if name in runtime_performance):
            raise ValueError('only bounded ordered preparation workers may change in a runtime continuation')
        sl.update(runtime_performance)
    return config


def pick_files(files, count, seed):
    if len(files) < count:
        raise ValueError(f'not enough validation games: {len(files)} < {count}')
    values = sorted(files)
    random.Random(seed).shuffle(values)
    return sorted(values[:count])


def prepare(args):
    output = Path(args.directory).resolve()
    if output.exists():
        raise FileExistsError('new probe directory required; existing experiment remains immutable')
    anchor = Path(args.anchor).resolve()
    source = torch.load(anchor, map_location='cpu', weights_only=False)
    if source['steps'] != 2_880_000 and not args.smoke:
        raise ValueError('this approved first experiment is anchored at clean A 2.88M')
    output.mkdir(parents=True)
    frozen_anchor = output / 'parent.pth'
    shutil.copy2(anchor, frozen_anchor)
    base_index = torch.load(args.index, map_location='cpu', weights_only=True)
    all_files = sorted(set(base_index['train_files'] + base_index['val_files']))
    domains = make_domains(all_files)
    from mortal.data.split_ledger import game_identity
    by_year = {}
    for file in all_files:
        by_year.setdefault(game_identity(file)[:4], []).append(file)
    recent_count, old_count = (2, 2) if args.smoke else (512, 256)
    controller_recent = pick_files(
        [file for file in by_year['2026'] if game_identity(file)[:6] == '202601'],
        recent_count, 202609071,
    )
    controller_old = pick_files(by_year['2022'], old_count, 202609072)
    selection_recent = pick_files(by_year['2025'], 512, 202609073)
    remaining_old = sorted(set(by_year['2022']) - set(controller_old))
    selection_old = pick_files(remaining_old, 256, 202609074)
    roles = {'controller_recent': controller_recent, 'controller_old': controller_old,
             'selection_recent': selection_recent, 'selection_old': selection_old}
    for left, right in itertools.combinations(roles.values(), 2):
        if set(left) & set(right):
            raise ValueError('controller and selection overlap')
    train_files = sorted(set(itertools.chain.from_iterable(domains.values())))
    if set(train_files) & set(itertools.chain.from_iterable(roles.values())):
        raise ValueError('training leaks validation')
    # Controller inputs are open development data. Selection/test payloads stay
    # unopened by this runner; their index identities alone are recorded.
    controller_hashes = {file: file_sha256(file) for file in controller_recent + controller_old}
    indexes = output / 'indexes.pth'
    atomic_torch_save({'train_files': train_files, 'domains': domains, 'roles': roles,
                       'monitor_recent_files': controller_recent,
                       'full_recent_files': controller_recent, 'old_regression_files': controller_old}, indexes)
    runtime = output / 'source'
    # Freeze only executable source. No config/checkpoint/data directory is copied.
    for file in (ROOT / 'mortal').rglob('*.py'):
        if 'checkpoints' in file.parts or '__pycache__' in file.parts:
            continue
        target = runtime / file.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file, target)
    for name in ('scripts/run_sl_curriculum_probe.py', 'scripts/start_oracle_critic_detached.ps1',
                 'scripts/supervise_oracle_critic_around_apex.ps1'):
        target = runtime / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    shutil.copy2(args.native, runtime / 'libriichi.pyd')
    parser_file = runtime / 'mortal/eval/search_runtime.py'
    parser_source_sha256 = file_sha256(parser_file)
    parser_file.write_text(freeze_search_parser(parser_file.read_text(encoding='utf-8')), encoding='utf-8')
    source_hashes = {file.relative_to(runtime).as_posix(): file_sha256(file)
                     for file in runtime.rglob('*') if file.is_file()}
    manifest = {'format': 'matched_sl_curriculum_probe_v1', 'parent': str(frozen_anchor),
                'parent_sha256': file_sha256(frozen_anchor), 'source_root': str(runtime),
                'source_sha256': source_hashes, 'indexes': str(indexes),
                'runtime_overrides': {'mortal/eval/search_runtime.py': {
                    'main_source_sha256': parser_source_sha256,
                    'reason': 'coerce_bool default is keyword-only; main source retained for live bot',
                }},
                'indexes_sha256': file_sha256(indexes), 'recipes': RECIPES,
                'controller_source_sha256': controller_hashes,
                'domain_sizes': {name: len(files) for name, files in domains.items()},
                'split_sizes': {name: len(files) for name, files in roles.items()},
                'split_identity_sha256': {name: stable_json_digest([game_identity(f) for f in files])
                                         for name, files in roles.items()},
                'seeds': [20260907] if args.smoke else [20260907, 20260917],
                'horizons': [1, 2] if args.smoke else [1024, 4096],
                'delayed_transfer_routes': [] if args.smoke else ['AC', 'CC'],
                'microbatch': args.microbatch, 'logical_batch': 1024,
                'device': args.device, 'gpu_memory_fraction': args.gpu_memory_fraction,
                'primary': 'controller_recent.policy_loss',
                'guardrail_margins': {'action_accuracy': 0.0002, 'old_policy_loss': 0.0002},
                'meaningful_delta': 0.0002, 'familywise_alpha': 0.05,
                'planned_metric_contrasts': 84,
                'inference': 'approximate_paired_game_cluster_normal_bonferroni',
                'stage_budget': None, 'purpose': 'measure_recipe_gain_and_delayed_transfer',
                'automatic_weight_changes': False, 'automatic_publication': False,
                'selection_payloads_opened': False, 'sealed_payloads_opened': False,
                'validation_history': 'new_controller_selection_partition; parent used these calendar domains before',
                'smoke': args.smoke, 'created_at': time.time()}
    manifest['identity'] = stable_json_digest(manifest)
    atomic_write_json(output / 'manifest.json', manifest)
    spec = {'format': 'oracle_critic_apex_supervisor_spec_v1', 'repo_root': str(runtime),
            'python_executable': sys.executable, 'search_root': str(output),
            'pause_file': str(output / 'apex_pause.request'),
            'status_file': str(output / 'apex_supervisor_status.json'),
            'log_file': str(output / 'apex_supervisor.log'),
            'runner_arguments': ['-u', str(runtime / 'scripts/run_sl_curriculum_probe.py'),
                                 'run', '--directory', str(output)]}
    atomic_write_json(output / 'apex_supervisor_spec.json', spec)
    print(json.dumps({name: manifest[name] for name in ('identity', 'domain_sizes', 'split_sizes', 'horizons')}))


def verify_manifest(manifest):
    candidate = dict(manifest)
    identity = candidate.pop('identity')
    if stable_json_digest(candidate) != identity:
        raise ValueError('probe manifest changed')
    for file, expected in ((manifest['parent'], manifest['parent_sha256']),
                           (manifest['indexes'], manifest['indexes_sha256'])):
        if file_sha256(file) != expected:
            raise ValueError(f'probe input changed: {file}')
    for name, digest in manifest['source_sha256'].items():
        if file_sha256(Path(manifest['source_root']) / name) != digest:
            raise ValueError(f'frozen probe runtime changed: {name}')
    for file, digest in manifest['controller_source_sha256'].items():
        if file_sha256(file) != digest:
            raise ValueError(f'controller input changed: {file}')


def arm_spec(manifest, seed, route):
    parent = manifest['parent']
    local_seed = seed
    horizons = manifest['horizons']
    if len(route) == 2:
        parent = str(Path(manifest['parent']).parent / f'{seed}_{route[0]}' /
                     f'update_{horizons[-1]:07d}.pth')
        local_seed = seed + 100000
        horizons = [manifest['horizons'][-1]]
    return parent, local_seed, horizons


def run_arm(args):
    output = Path(args.directory).resolve()
    manifest = read_json(output / 'manifest.json')
    verify_manifest(manifest)
    arm_root = output / f'{args.seed}_{args.route}'
    arm_root.mkdir(exist_ok=True)
    parent, seed, horizons = arm_spec(manifest, args.seed, args.route)
    identity = stable_json_digest({'experiment': manifest['identity'], 'parent': file_sha256(parent),
                                  'seed': seed, 'route': args.route, 'horizons': horizons})
    source = torch.load(parent, weights_only=False, map_location='cpu')
    config = build_config(source, arm_root, manifest['indexes'], seed=seed,
                          device=manifest['device'], microbatch=manifest['microbatch'],
                          logical_batch=manifest['logical_batch'], identity=identity,
                          runtime_performance=manifest.get('runtime_performance'))
    config_path = arm_root / 'config.toml'
    if config_path.exists():
        if load_toml_file(config_path) != config:
            raise ValueError('existing arm configuration differs')
    else:
        write_toml_file(config_path, config)
    state_file = Path(config['supervised']['state_file'])
    if not state_file.exists():
        atomic_torch_save(prepare_branch_state(source, config), state_file)
    del source
    os.environ['MORTAL_CFG'] = str(config_path)
    os.environ['RAYON_NUM_THREADS'] = str(config['supervised']['rayon_num_threads'])
    os.environ['OMP_NUM_THREADS'] = '2'
    os.environ['MKL_NUM_THREADS'] = '2'
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    if manifest['device'].startswith('cuda'):
        torch.cuda.set_per_process_memory_fraction(manifest['gpu_memory_fraction'])
    torch.use_deterministic_algorithms(True)
    index = torch.load(manifest['indexes'], weights_only=True, map_location='cpu')
    probe = CurriculumProbe(config, index['domains'], recipe=args.route[-1], seed=seed,
                            output=arm_root, horizons=horizons, identity=identity,
                            eval_splits={name: index['roles'][name]
                                         for name in ('controller_recent', 'controller_old')})
    if args.until_update is not None:
        if args.until_update not in horizons:
            raise ValueError('only declared observation horizons can end a segment')
        probe.stop_at = args.until_update
    from mortal.supervised.train_supervised import train
    train(probe=probe, stage_label=f'Curriculum probe {args.route}', checkpoint_label=args.route)
    for horizon in [0, *[value for value in horizons if value <= probe.stop_at]]:
        if not (arm_root / f'update_{horizon:07d}.json').exists():
            raise RuntimeError('probe ended before all declared observations')
    receipt = {'identity': identity, 'until_update': probe.stop_at, 'completed_at': time.time()}
    atomic_write_json(arm_root / f'segment_{probe.stop_at:07d}.json', receipt)
    if probe.stop_at == horizons[-1]:
        atomic_write_json(arm_root / 'completed.json', receipt)


def comparisons(left, right, z):
    from mortal.core.adaptive_curriculum import paired_cluster_summary
    result = {}
    for split, metric, name in (
        ('controller_recent', 'policy_loss', 'policy_loss'),
        ('controller_recent', 'action_accuracy', 'action_accuracy'),
        ('controller_old', 'policy_loss', 'old_policy_loss'),
    ):
        result[name] = paired_cluster_summary(
            left['splits'][split]['_adaptive_cluster_records'][metric],
            right['splits'][split]['_adaptive_cluster_records'][metric], confidence_z=z,
        )
        if result[name]['num_games'] < 32:
            result[name]['inferentially_qualified'] = False
        else:
            result[name]['inferentially_qualified'] = True
    return result


def clear_gain(result, manifest):
    return (all(item['inferentially_qualified'] for item in result.values())
            and result['policy_loss']['ci_high'] < -manifest['meaningful_delta']
            and result['action_accuracy']['ci_low'] >= -manifest['guardrail_margins']['action_accuracy']
            and result['old_policy_loss']['ci_high'] <= manifest['guardrail_margins']['old_policy_loss'])


def summarize(output, manifest):
    z = NormalDist().inv_cdf(1 - manifest['familywise_alpha'] /
                            (2 * manifest['planned_metric_contrasts']))
    report = {'identity': manifest['identity'], 'status': 'inconclusive',
              'confidence_z': z, 'comparisons': {}, 'recommendation': None,
              'automatic_publication': False, 'selection_payloads_opened': False,
              'sealed_payloads_opened': False,
              'uncertainty': 'game sampling CI conditional on each training seed; two seeds do not estimate training variance'}
    winners = []
    for seed in manifest['seeds']:
        baselines = [read_json(output / f'{seed}_{recipe}' / 'update_0000000.json') for recipe in RECIPES]
        if len({row['learned_state_sha256'] for row in baselines}) != 1:
            raise ValueError('same-parent learned states differ between recipes')
        if any(row['splits'] != baselines[0]['splits'] for row in baselines[1:]):
            raise ValueError('same-parent baseline differs between recipes')
        for horizon in manifest['horizons']:
            observations = {recipe: read_json(output / f'{seed}_{recipe}' / f'update_{horizon:07d}.json')
                            for recipe in RECIPES}
            parent = read_json(output / f'{seed}_A' / 'update_0000000.json')
            if any(row['optimizer_updates'] != horizon or row['successful_decisions'] !=
                   horizon * manifest['logical_batch'] for row in observations.values()):
                raise ValueError('recipe observations use unequal successful update/sample counts')
            current = {}
            for left, right in itertools.combinations(RECIPES, 2):
                current[f'{left}-{right}'] = comparisons(observations[left], observations[right], z)
            for recipe in RECIPES:
                current[f'{recipe}-parent'] = comparisons(observations[recipe], parent, z)
            report['comparisons'][f'{seed}_U{horizon}'] = current
            qualified = []
            for recipe in RECIPES:
                if not clear_gain(current[f'{recipe}-parent'], manifest):
                    continue
                if all(clear_gain(comparisons(observations[recipe], observations[other], z), manifest)
                       for other in RECIPES if other != recipe):
                    qualified.append(recipe)
            winners.append(qualified[0] if len(qualified) == 1 else None)
    delayed_favors_a = False
    for seed in manifest['seeds']:
        if not manifest['delayed_transfer_routes']:
            continue
        horizon = manifest['horizons'][-1]
        ac = read_json(output / f'{seed}_AC' / f'update_{horizon:07d}.json')
        cc = read_json(output / f'{seed}_CC' / f'update_{horizon:07d}.json')
        result = comparisons(ac, cc, z)
        report['comparisons'][f'{seed}_AC-CC'] = result
        delayed_favors_a |= clear_gain(result, manifest)
    if winners and winners[0] and all(winner == winners[0] for winner in winners):
        if winners[0] != 'C' or not delayed_favors_a:
            report.update(status='consistent_controller_signal', recommendation=winners[0])
    report['delayed_transfer_favors_A'] = delayed_favors_a
    report['next_step'] = ('lock a finite confirmation candidate and evaluate selection-dev, then formal 1v3'
                           if report['recommendation'] else
                           'no automatic switch; inspect precision, horizons and LR with symmetric extensions')
    receipts = [read_json(file) for file in output.glob('*/update_*.json')]
    final_by_arm = {}
    for receipt in receipts:
        key = str(Path(receipt['checkpoint']).parent)
        if key not in final_by_arm or receipt['optimizer_updates'] > final_by_arm[key]['optimizer_updates']:
            final_by_arm[key] = receipt
    report['total_successful_updates'] = sum(row['optimizer_updates'] for row in final_by_arm.values())
    report['total_successful_decisions'] = sum(row['successful_decisions'] for row in final_by_arm.values())
    report['total_consumed_decisions'] = sum(row['exposure'].get('decisions', 0) for row in final_by_arm.values())
    attempts = [read_json(file) for file in (output / 'attempts').glob('*.json')]
    report['finished_attempt_wall_seconds'] = sum(row.get('wall_seconds', 0) for row in attempts)
    report['unfinished_attempts'] = sum(row['state'] == 'running' for row in attempts)
    if manifest.get('continuation'):
        previous = Path(manifest['continuation']['source_directory'])
        prior_attempts = [read_json(file) for file in (previous / 'attempts').glob('*.json')]
        prior_wall = sum(row.get('wall_seconds', 0) for row in prior_attempts)
        report['prior_runtime_finished_attempt_wall_seconds'] = prior_wall
        report['finished_attempt_wall_seconds'] += prior_wall
        report['unfinished_attempts'] += sum(row['state'] == 'running' for row in prior_attempts)
        report['runtime_continuation'] = manifest['continuation']
    report['cost_scope'] = ('all branches and finished attempts; decision counts are durable cursor counts; '
                            'work lost before a crash checkpoint is not exactly counted')
    atomic_write_json(output / 'comparison.json', report)
    return report


def observation_schedule(manifest):
    schedule = []
    for round_index, horizon in enumerate(manifest['horizons']):
        for seed_index, seed in enumerate(manifest['seeds']):
            order = list(RECIPES) if seed_index % 2 == 0 else list(reversed(RECIPES))
            offset = round_index % len(order)
            order = order[offset:] + order[:offset]
            schedule.extend((seed, recipe, horizon) for recipe in order)
    schedule.extend((seed, route, manifest['horizons'][-1]) for seed in manifest['seeds']
                    for route in manifest['delayed_transfer_routes'])
    return schedule


def run(args):
    output = Path(args.directory).resolve()
    manifest = read_json(output / 'manifest.json')
    verify_manifest(manifest)
    # Complete the same observation horizon for every recipe/seed before any
    # arm gets the longer horizon. Rotate order so A has no permanent priority.
    schedule = observation_schedule(manifest)
    for seed, route, horizon in schedule:
        arm_root = output / f'{seed}_{route}'
        if (arm_root / f'segment_{horizon:07d}.json').exists():
            continue
        arm_root.mkdir(exist_ok=True)
        atomic_write_json(output / 'progress.json', {'state': 'running', 'seed': seed, 'route': route,
                                                    'until_update': horizon,
                                                    'updated_at': time.time(), 'schedule': schedule})
        env = os.environ.copy()
        env.update(RAYON_NUM_THREADS=str(manifest.get('runtime_performance', {}).get('rayon_num_threads', 2)),
                   OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                   CUBLAS_WORKSPACE_CONFIG=':4096:8', PYTHONUNBUFFERED='1')
        attempts_root = output / 'attempts'
        attempts_root.mkdir(exist_ok=True)
        attempt_file = attempts_root / f'{time.time_ns()}.json'
        attempt = {'state': 'running', 'seed': seed, 'route': route, 'horizon': horizon,
                   'started_at': time.time()}
        atomic_write_json(attempt_file, attempt)
        started = time.monotonic()
        with (arm_root / 'train.log').open('a', encoding='utf-8') as stream:
            result = subprocess.run([sys.executable, '-u', str(Path(manifest['source_root']) /
                                    'scripts/run_sl_curriculum_probe.py'), 'arm', '--directory', str(output),
                                    '--seed', str(seed), '--route', route, '--until-update', str(horizon)],
                                    cwd=manifest['source_root'], env=env, stdout=stream, stderr=stream)
        attempt.update(state='finished', returncode=result.returncode, wall_seconds=time.monotonic() - started)
        atomic_write_json(attempt_file, attempt)
        if result.returncode:
            atomic_write_json(output / 'progress.json', {'state': 'paused_or_failed', 'seed': seed,
                                                        'route': route, 'returncode': result.returncode})
            raise SystemExit(result.returncode)
    report = summarize(output, manifest)
    atomic_write_json(output / 'progress.json', {'state': 'completed', 'result': report['status'],
                                                'updated_at': time.time()})
    print(json.dumps({key: report[key] for key in ('status', 'recommendation', 'total_successful_updates')}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prep = commands.add_parser('prepare')
    prep.add_argument('--directory', required=True)
    prep.add_argument('--anchor', required=True)
    prep.add_argument('--index', default=str(ROOT / 'mortal/checkpoints/file_index_supervised_json.pth'))
    prep.add_argument('--native', required=True)
    prep.add_argument('--device', default='cuda:0')
    prep.add_argument('--microbatch', type=int, default=256)
    prep.add_argument('--gpu-memory-fraction', type=float, default=0.20)
    prep.add_argument('--smoke', action='store_true')
    for name in ('run', 'arm', 'summarize'):
        child = commands.add_parser(name)
        child.add_argument('--directory', required=True)
        if name == 'arm':
            child.add_argument('--seed', type=int, required=True)
            child.add_argument('--route', choices=[*RECIPES, 'AC', 'CC'], required=True)
            child.add_argument('--until-update', type=int)
    args = parser.parse_args()
    if args.command == 'summarize':
        output = Path(args.directory).resolve()
        report = summarize(output, read_json(output / 'manifest.json'))
        print(json.dumps({'status': report['status'], 'recommendation': report['recommendation']}))
    else:
        {'prepare': prepare, 'run': run, 'arm': run_arm}[args.command](args)


if __name__ == '__main__':
    main()
