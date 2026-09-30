"""No-update, frozen categorical actor / fixed-imputed Oracle critic diagnostic.

Run as a module. The output directory must never have existed. An external
supervisor owns the compute deadline; this entry point neither resumes nor trains.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
import json
import os
from pathlib import Path
import random
import subprocess
import time

from mortal.core.artifacts import atomic_output_path, atomic_write_json, file_sha256

CRITICS = ('warm0', 'warm40k', 'clean40k')
PAIRS = (('warm40k', 'warm0'), ('clean40k', 'warm0'), ('clean40k', 'warm40k'))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'actor', 'opponent', 'warm0', 'warm40k', 'clean40k', 'output-dir'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--games', type=int, default=256, help='8 for smoke; 256 for pilot')
    parser.add_argument('--seed-start', type=int, required=True)
    parser.add_argument('--seed-key', type=int, required=True)
    parser.add_argument('--sampling-seed', type=int, required=True)
    parser.add_argument('--imputation-seed', type=int, default=20260905)
    parser.add_argument('--bootstrap-seed', type=int, default=20260905)
    parser.add_argument('--bootstrap-replicates', type=int, default=20000)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--torch-threads', type=int, default=1)
    parser.add_argument('--rayon-threads', type=int, default=4)
    args = parser.parse_args(argv)
    if args.games < 8 or args.games % 4:
        parser.error('--games must be a multiple of four, at least eight')
    if args.batch_size < 1 or args.torch_threads < 1 or args.rayon_threads < 1:
        parser.error('batch and thread counts must be positive')
    if args.bootstrap_replicates < 1000:
        parser.error('--bootstrap-replicates must be at least 1000')
    for name in ('seed_start', 'seed_key', 'sampling_seed', 'imputation_seed', 'bootstrap_seed'):
        if not 0 <= getattr(args, name) < 2**64:
            parser.error(f'{name} must be an unsigned 64-bit integer')
    if args.seed_start + args.games // 4 >= 2**64:
        parser.error('seed range exceeds u64')
    return args


def reserve_output(path):
    """Exclusive ownership before TrainPlayer's destructive log-dir handling."""
    if os.path.lexists(path):
        raise FileExistsError(f'output path already exists: {path}')
    root = Path(path).resolve()
    root.mkdir(parents=True, exist_ok=False)
    return root


def verify_complete_log(path):
    with gzip.open(path, 'rt', encoding='utf-8') as handle:
        events = [json.loads(line) for line in handle if line.strip()]
    if not events or events[0].get('type') != 'start_game' or events[-1].get('type') != 'end_game':
        raise ValueError(f'incomplete native game: {path}')
    starts = sum(e.get('type') == 'start_kyoku' for e in events)
    ends = sum(e.get('type') == 'end_kyoku' for e in events)
    if not starts or starts != ends:
        raise ValueError(f'unfinished kyoku in {path}')
    if events[0].get('names', []).count('trainee') != 1:
        raise ValueError(f'expected exactly one trainee in {path}')
    return starts


def validate_critic_contract(state, *, version, pts, role=None):
    cfg, pre = state['config'], state['oracle_critic_pretrain']
    if role is not None and state.get('steps') != {'warm0': 0, 'warm40k': 40000, 'clean40k': 40000}[role]:
        raise ValueError(f'{role} checkpoint has unexpected internal steps={state.get("steps")}')
    if any(group.get('train_mode', False) for group in state.get('optimizer', {}).get('param_groups', [])):
        raise ValueError('critic checkpoint is in schedule-free train mode, not evaluation mode')
    if cfg['control']['version'] != version or list(cfg['env']['pts']) != list(pts):
        raise ValueError('critic observation version / rank-point utility differs from actor protocol')
    if (pre['target_mode'] != 'all_players' or pre['return_mode'] != 'score_rank_mc'
            or float(pre['discount_gamma']) != 1.0):
        raise ValueError('critic must use all_players score_rank_mc gamma=1')
    if pre['critic_arch'] != 'dual_tower' or pre.get('value_loss_mode', 'mse') != 'mse':
        raise ValueError('this bounded probe supports dual_tower MSE critics only')
    return pre


def summarize_predictions(target, pred):
    import numpy as np
    pred, target = np.asarray(pred, dtype=np.float64), np.asarray(target, dtype=np.float64)
    if pred.shape != target.shape:
        raise ValueError('prediction and target shapes differ')
    error = pred - target
    if error.ndim != 2 or error.shape[1] != 4 or not np.isfinite(error).all() or len(error) == 0:
        raise ValueError('expected finite, nonempty four-head predictions and targets')
    return {'states': len(error), 'head_values': int(error.size),
            'p0_mse': float(np.square(error[:, 0]).mean()),
            'all_players_mse': float(np.square(error).mean()),
            'p0_bias': float(error[:, 0].mean()), 'all_players_bias': float(error.mean())}


def calibration(target, pred):
    import numpy as np
    edges = [-np.inf, -4, -3, -2, -1, 0, 1, 2, 3, 4, np.inf]
    result = []
    for lo, hi in zip(edges, edges[1:]):
        keep = (pred >= lo) & (pred < hi)
        result.append({'bin': f'[{lo},{hi})', 'count': int(keep.sum()),
                       'prediction_mean': float(pred[keep].mean()) if keep.any() else None,
                       'target_mean': float(target[keep].mean()) if keep.any() else None})
    return result


def distribution(values):
    import numpy as np
    values = np.asarray(values, dtype=np.float64)
    return {'count': int(values.size), 'mean': float(values.mean()), 'std': float(values.std()),
            'quantiles': dict(zip(('min', 'p01', 'p05', 'p50', 'p95', 'p99', 'max'),
                                 np.quantile(values, [0, .01, .05, .5, .95, .99, 1]).tolist()))}


def load_policy(path, torch):
    from mortal.core.model import Brain, CategoricalPolicy
    from mortal.core.checkpoint_utils import checkpoint_brain_is_oracle_structure
    state = torch.load(path, weights_only=True, map_location='cpu', mmap=True)
    cfg = state['config']
    if checkpoint_brain_is_oracle_structure(state):
        raise ValueError('frozen actor/opponent must have visible-only Brain structure')
    brain = Brain(version=cfg['control']['version'], **cfg['resnet'], Norm='GN').eval()
    policy = CategoricalPolicy().eval()
    brain.load_state_dict(state['mortal'], strict=True)
    policy.load_state_dict(state['policy_net'], strict=True)
    for model in (brain, policy):
        model.requires_grad_(False)
    return brain, policy, cfg, state.get('steps')


def run(args, root):
    # Must precede any module importing mortal.config or the native Rayon pool.
    os.environ['MORTAL_CFG'] = str(Path(args.config).resolve())
    os.environ['TRAIN_PLAY_PROFILE'] = 'frozen_critic_probe'
    os.environ['RAYON_NUM_THREADS'] = str(args.rayon_threads)
    import numpy as np
    import torch
    import libriichi
    from mortal.config import config
    from libriichi.dataset import GameplayLoader
    from mortal.core.model import OracleDualTowerBrain, ValueHead
    from mortal.data.dataloader import FileDatasetsIter
    from mortal.data.oracle_value import expand_kyoku_rewards_to_steps, discounted_returns_from_step_rewards
    from mortal.eval.engine import MortalEngine
    from mortal.eval.player import TrainPlayer
    from mortal.eval.paired_1v3 import load_games, duplicate_sets, bootstrap_interval
    from mortal.online.train_online import compute_gae_advantages_from_step_rewards

    torch.set_num_threads(args.torch_threads)
    torch.set_num_interop_threads(1)
    device = torch.device(args.device)
    native_probe = GameplayLoader(version=4, oracle=True, player_names=['trainee'])
    if not callable(getattr(native_probe, 'set_oracle_imputation_seed', None)):
        raise RuntimeError('native loader lacks fixed Oracle imputation support; no arena started')
    native_probe.set_oracle_imputation_seed(args.imputation_seed)
    del native_probe
    files = {name: Path(getattr(args, name)).resolve() for name in ('config', 'actor', 'opponent', *CRITICS)}
    hashes = {name: file_sha256(path) for name, path in files.items()}
    actor, policy, actor_cfg, actor_steps = load_policy(files['actor'], torch)
    opponent, opponent_policy, opponent_cfg, opponent_steps = load_policy(files['opponent'], torch)
    version, pts = actor_cfg['control']['version'], [2, 1, 0, -3]
    if version != 4:
        raise ValueError('expected current v4 actor observations')
    if opponent_cfg['control']['version'] != version:
        raise ValueError('opponent observation version mismatch')
    log_dir = root / 'games'
    config['control']['version'] = version
    config['control']['enable_cudnn_benchmark'] = False
    config['env']['pts'] = list(pts)
    config['repro'] = {'enabled': True, 'seed': args.sampling_seed,
                       'train_key': args.seed_key, 'train_seed_start': args.seed_start,
                       'allow_cudnn_benchmark': False}
    config.setdefault('train_play', {})['frozen_critic_probe'] = {
        'log_dir': str(log_dir), 'explore_rate': 1.0, 'games': args.games, 'repeats': 1}
    config.setdefault('baseline', {})['train'] = {
        'device': args.device, 'state_file': str(files['opponent']), 'enable_compile': False,
        'reload_each_session': False, 'pool_seed': args.sampling_seed % 2**32}
    config['search'] = {'enabled': False}
    config['oracle_guiding'] = {'actor_enabled': False}

    # Explicit frozen baseline construction bypasses checkpoint-embedded search
    # settings. The actual arena/session path remains production TrainPlayer.
    class FrozenProbePlayer(TrainPlayer):
        def _ensure_baseline_pool_ready(self, *, force=False):
            if self._baseline_pool:
                return
            engine = MortalEngine(opponent, opponent_policy, is_oracle=False, version=version,
                                  device=device, enable_amp=device.type == 'cuda',
                                  enable_rule_based_agari_guard=True, name='canonical',
                                  explore_rate=0.0, oracle_guiding_keep_prob=0.0,
                                  oracle_input_mode='zero', search_runtime_bundle=None)
            self._baseline_pool = [{'engine': engine, 'state_file': str(files['opponent']),
                                    'weight': 1.0, 'labels': ('canonical',), 'description': 'canonical'}]

    source_root = Path(__file__).resolve().parents[2]
    def git_output(*arguments):
        return subprocess.check_output(['git', *arguments], cwd=source_root, text=True).strip()
    provenance = {'schema': 1, 'status': 'running', 'started_unix': time.time(),
                  'arguments': vars(args), 'weights_and_config': {
                      name: {'path': str(path), 'sha256': hashes[name]} for name, path in files.items()},
                  'source_commit': git_output('rev-parse', 'HEAD'),
                  'source_status': git_output('status', '--porcelain'),
                  'source_hashes': {str(Path(p).relative_to(source_root)): file_sha256(p) for p in
                      [Path(__file__), source_root / 'mortal/data/dataloader.py',
                       source_root / 'mortal/online/train_online.py', source_root / 'mortal/eval/player.py',
                       source_root / 'mortal/eval/engine.py', source_root / 'libriichi/src/dataset/invisible.rs']},
                  'native_module': str(libriichi.__file__), 'native_sha256': file_sha256(libriichi.__file__),
                  'torch_version': torch.__version__, 'numpy_version': np.__version__,
                  'actor_steps': actor_steps, 'opponent_steps': opponent_steps,
                  'actor_saved_pts': actor_cfg.get('env', {}).get('pts'), 'probe_pts': pts,
                  'distribution_scope': 'this frozen sampled actor versus training-style canonical opponent; not the old formal 1v3 evaluation protocol',
                  'input_semantics': 'FIXED-IMPUTED Oracle: recorded hidden hands/drawn tiles plus deterministic completion of unobserved wall; trust_seed=false',
                  'true_oracle_reconstruction': 'deferred: native seed reconstruction consistency not established',
                  'actor_explore_rate': 1.0, 'opponent_explore_rate': 0.0,
                  'actor_agari_guard': False, 'opponent_agari_guard': True,
                  'search_enabled': False, 'actor_oracle_guiding': False,
                  'optimizer_created': False, 'actor_strength_proof': False}
    native_dir = Path(libriichi.__file__).resolve().parent
    native_extensions = sorted(set(native_dir.glob('*.pyd')) | set(native_dir.glob('*.so')))
    provenance['native_extension_sha256'] = {str(p): file_sha256(p) for p in native_extensions}
    atomic_write_json(root / 'provenance.json', provenance)
    atomic_write_json(root / 'effective_config.json', config)
    # Validate all three critics before spending any arena computation.
    critics = {}
    for name in CRITICS:
        state = torch.load(files[name], map_location='cpu', weights_only=True, mmap=True)
        pre = validate_critic_contract(state, version=version, pts=pts, role=name)
        brain = OracleDualTowerBrain(version=version, **state['config']['resnet'], Norm='GN',
                                    oracle_fusion_mode=pre.get('oracle_fusion_mode', 'linear'),
                                    oracle_fusion_hidden=pre.get('oracle_fusion_hidden', 512)).eval()
        value = ValueHead(num_players=4, hidden_size=pre.get('value_head_hidden', 256),
                          zero_sum=pre.get('exact_zero_sum', False)).eval()
        brain.load_state_dict(state['oracle_brain'], strict=True)
        value.load_state_dict(state['value_net'], strict=True)
        brain.requires_grad_(False)
        value.requires_grad_(False)
        critics[name] = (brain, value)
        provenance.setdefault('critics', {})[name] = {'steps': state.get('steps'), 'contract': pre}
        del state
    atomic_write_json(root / 'provenance.json', provenance)
    player = FrozenProbePlayer()
    player.train_seed = args.seed_start  # resolve_train_seed_start treats zero as its legacy default
    # Loading/initializing models consumes RNG. Reset immediately before arena.
    random.seed(args.sampling_seed)
    np.random.seed(args.sampling_seed % 2**32)
    torch.manual_seed(args.sampling_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.sampling_seed)
    if os.path.lexists(log_dir):
        raise FileExistsError(f'refusing destructive TrainPlayer path: {log_dir}')
    rankings, paths = player.train_play(actor, policy, device, actor_oracle_enabled=False,
                                       actor_oracle_keep_prob=0.0, actor_oracle_input_mode='zero',
                                       search_runtime_bundle=None)
    games = load_games(log_dir, 'trainee')
    groups = duplicate_sets(games)
    expected = {(seed, args.seed_key) for seed in range(args.seed_start, args.seed_start + args.games // 4)}
    if set(groups) != expected or len(games) != args.games or len(paths) != args.games:
        raise ValueError('arena output does not match requested complete four-seat seed sets')
    for game in games:
        game['kyoku_count'] = verify_complete_log(game['log_path'])
        game['sha256'] = file_sha256(game['log_path'])
    atomic_write_json(root / 'outcomes.json', games)
    # Free actor GPU storage before critic scoring. No policy calls thereafter.
    actor.cpu(); policy.cpu(); opponent.cpu(); opponent_policy.cpu()
    del player
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    for brain, value in critics.values():
        brain.to(device); value.to(device)
    targets, all_context = [], []
    predictions = {name: [] for name in CRITICS}
    advantages = {name: [] for name in CRITICS}
    clustered = defaultdict(lambda: defaultdict(list))
    identity_max = {name: 0.0 for name in CRITICS}
    game_counts = []
    with atomic_output_path(root / 'predictions.jsonl.gz') as temporary:
        with gzip.open(temporary, 'wt', encoding='utf-8', compresslevel=3) as stream:
            for game_index, game in enumerate(games):
                path = game['log_path']
                dataset = FileDatasetsIter(version=version, file_list=[path], pts=pts, oracle=True,
                                           player_names=['trainee'], value_target_mode='all_players',
                                           value_reward_source='score_rank', emit_context_meta=True,
                                           emit_opponent_state_labels=False, track_danger_labels=False,
                                           track_regret_labels=False)
                trajectories = iter(dataset.iter_game_trajectories([path], oracle_imputation_seed=args.imputation_seed))
                trajectory = next(trajectories)
                if next(trajectories, None) is not None:
                    raise ValueError('expected exactly one trainee trajectory per game')
                if game_index == 0:
                    repeated = next(dataset.iter_game_trajectories([path], oracle_imputation_seed=args.imputation_seed))
                    for field in ('obs', 'invisible_obs', 'at_kyoku', 'kyoku_value_target', 'context_meta'):
                        if not np.array_equal(trajectory[field], repeated[field]):
                            raise ValueError(f'fixed-imputation A/A mismatch: {field}')
                    del repeated
                obs, invisible = trajectory['obs'], trajectory['invisible_obs']
                rewards = expand_kyoku_rewards_to_steps(trajectory['kyoku_value_target'], trajectory['at_kyoku'])
                target = discounted_returns_from_step_rewards(rewards, 1.0)
                context = trajectory['context_meta']
                if (len(target) == 0 or target.shape != (len(obs), 4)
                        or len(trajectory['kyoku_value_target']) != game['kyoku_count']
                        or context.shape != (len(obs), 8)
                        or not np.isin(context[:, 3], [0, 1]).all()
                        or not np.isin(context[:, 4], [0, 1, 2, 3]).all()):
                    raise ValueError('invalid complete-trajectory / native context shape or range')
                if not all(np.isfinite(x).all() for x in (obs, invisible, rewards, target)):
                    raise ValueError('non-finite trajectory input or label')
                per_game = {}
                for name, (brain, value) in critics.items():
                    with torch.inference_mode():
                        pred = torch.cat([value(brain(
                            torch.as_tensor(obs[i:i + args.batch_size], device=device, dtype=torch.float32),
                            invisible_obs=torch.as_tensor(invisible[i:i + args.batch_size], device=device, dtype=torch.float32)
                        )).cpu() for i in range(0, len(obs), args.batch_size)]).numpy()
                    summary = summarize_predictions(target, pred)
                    cluster_key = (game['seed'], game['seed_key'])
                    clustered[name][cluster_key].append((len(target), summary['p0_mse'], summary['all_players_mse']))
                    per_game[name] = pred
                    predictions[name].append(pred)
                    adv1 = np.stack([compute_gae_advantages_from_step_rewards(rewards[:, h], pred[:, h], 1.0, 1.0)
                                     for h in range(4)], axis=1)
                    residual = adv1 + pred - target
                    identity_max[name] = max(identity_max[name], float(np.abs(residual).max()))
                    np.testing.assert_allclose(adv1 + pred, target, atol=2e-5, rtol=2e-5)
                    adv95 = compute_gae_advantages_from_step_rewards(rewards[:, 0], pred[:, 0], 1.0, .95)
                    if not np.isfinite(adv95).all():
                        raise ValueError('non-finite production GAE')
                    advantages[name].append(adv95)
                for i in range(len(target)):
                    stream.write(json.dumps({'game_index': game_index, 'seed': game['seed'], 'seed_key': game['seed_key'],
                                             'trainee_seat': game['challenger_seat'], 'state_index': i,
                                             'current_rank': int(context[i, 4]), 'all_last': bool(context[i, 3]),
                                             'target': target[i].tolist(),
                                             'pred': {name: per_game[name][i].tolist() for name in CRITICS}},
                                            separators=(',', ':'), allow_nan=False) + '\n')
                zero_summary = summarize_predictions(target, np.zeros_like(target))
                clustered['constant_zero'][(game['seed'], game['seed_key'])].append(
                    (len(target), zero_summary['p0_mse'], zero_summary['all_players_mse']))
                targets.append(target)
                all_context.append(context)
                game_counts.append(len(target))
                print(json.dumps({'scored_games': game_index + 1, 'total_games': len(games), 'states': sum(game_counts)}), flush=True)
    target, context = np.concatenate(targets), np.concatenate(all_context)
    result = {'games': len(games), 'seed_groups': len(groups), 'states': len(target), 'game_state_counts': game_counts,
              'rankings': rankings.tolist(), 'head_coordinate': 'relative_to_trainee_seat; p0=trainee',
              'input_semantics': provenance['input_semantics'], 'gamma': 1.0, 'metrics': {},
              'zero_baseline': summarize_predictions(target, np.zeros_like(target)),
              'finite_checks': True, 'first_game_fixed_imputation_AA': True, 'lambda1_return_identity_max_abs': identity_max,
              'paired_differences': {}, 'bootstrap_replicates': args.bootstrap_replicates,
              'bootstrap_seed': args.bootstrap_seed,
              'paired_estimand': 'equal-weight independent seed-group mean of within-group state-weighted MSE; four seats clustered together',
              'global_estimand': 'all decision states equally weighted',
              'interpretation': 'pilot diagnostic, no readiness thresholds or strength claim; clean40k versus warm0 is a candidate contrast, not its training gain',
              'selection_adjusted': False, 'training_seed_uncertainty_included': False}
    seed_mse = {}
    for name in CRITICS:
        pred = np.concatenate(predictions[name])
        result['metrics'][name] = summarize_predictions(target, pred)
        result['metrics'][name]['p0_calibration'] = calibration(target[:, 0], pred[:, 0])
        result['metrics'][name]['p0_gae_lambda095'] = distribution(np.concatenate(advantages[name]))
        result['metrics'][name]['input_strata'] = {
            f'current_rank={rank},all_last={last}': summarize_predictions(target[keep], pred[keep])
            for rank in range(4) for last in (0, 1)
            if (keep := ((context[:, 4] == rank) & (context[:, 3] == last))).any()}
        seed_mse[name] = np.asarray([
            np.average(np.asarray(clustered[name][key])[:, 1:], axis=0,
                       weights=np.asarray(clustered[name][key])[:, 0]) for key in sorted(groups)])
        result['metrics'][name]['equal_seed_group_mse'] = dict(zip(('p0', 'all_players'), seed_mse[name].mean(0).tolist()))
    seed_mse['constant_zero'] = np.asarray([
        np.average(np.asarray(clustered['constant_zero'][key])[:, 1:], axis=0,
                   weights=np.asarray(clustered['constant_zero'][key])[:, 0]) for key in sorted(groups)])
    result['zero_baseline']['equal_seed_group_mse'] = dict(zip(
        ('p0', 'all_players'), seed_mse['constant_zero'].mean(0).tolist()))
    for left, right in (*PAIRS, *((name, 'constant_zero') for name in CRITICS)):
        result['paired_differences'][f'{left}-{right}'] = {
            key: {**bootstrap_interval((seed_mse[left] - seed_mse[right])[:, i],
                                       replicates=args.bootstrap_replicates, seed=args.bootstrap_seed),
                  'seed_groups': len(groups), 'games': len(games), 'states': len(target)}
            for i, key in enumerate(('p0_mse', 'all_players_mse'))}
    if hashes != {name: file_sha256(path) for name, path in files.items()}:
        raise RuntimeError('input checkpoint or configuration changed during probe')
    result['input_files_unchanged'] = True
    atomic_write_json(root / 'metrics.json', result)
    provenance.update(status='complete', finished_unix=time.time(),
                      artifact_sha256={name: file_sha256(root / name) for name in
                                       ('outcomes.json', 'metrics.json', 'predictions.jsonl.gz', 'effective_config.json')})
    atomic_write_json(root / 'provenance.json', provenance)
    print(json.dumps({'status': 'complete', 'output_dir': str(root), 'states': len(target)}), flush=True)


def main(argv=None):
    args = parse_args(argv)
    root = reserve_output(args.output_dir)
    atomic_write_json(root / 'request.json', vars(args))
    try:
        run(args, root)
    except BaseException as exc:
        atomic_write_json(root / 'failure.json', {'status': 'failed', 'type': type(exc).__name__, 'error': str(exc), 'unix': time.time()})
        raise


if __name__ == '__main__':
    main()
