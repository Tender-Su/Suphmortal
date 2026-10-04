"""No-update, frozen categorical actor / fixed-imputed Oracle critic diagnostic.

Run as a module. The output directory must never have existed. An external
supervisor owns the compute deadline; this entry point neither resumes nor trains.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
import hashlib
import json
import os
from pathlib import Path
import random
import re
import subprocess
import time

from mortal.core.artifacts import atomic_output_path, atomic_write_json, file_sha256

CRITICS = ('warm0', 'warm40k', 'clean40k')
PAIRS = (('warm40k', 'warm0'), ('clean40k', 'warm0'), ('clean40k', 'warm40k'))


class ProbePhaseTimings:
    """Exclusive wall-clock intervals; repeated phases accumulate, never nest."""
    def __init__(self):
        self.started = self.phase_started = time.perf_counter()
        self.phase = 'preflight_input_fingerprint_model_load'
        self.finished = None
        self.seconds = {}

    def switch(self, phase):
        if self.finished is not None:
            raise RuntimeError('probe timings already finished')
        now = time.perf_counter()
        self.seconds[self.phase] = self.seconds.get(self.phase, 0.0) + (now - self.phase_started)
        self.phase, self.phase_started = phase, now

    def finish(self):
        self.switch(None)
        self.finished = self.phase_started

    def report(self):
        now = self.finished if self.finished is not None else time.perf_counter()
        return {
            'schema': 1, 'clock': 'time.perf_counter',
            'status': 'complete' if self.finished is not None else 'incomplete',
            'total_seconds': now - self.started,
            'phase_seconds': dict(self.seconds),
            'incomplete_phase': ({'name': self.phase, 'seconds': now - self.phase_started}
                                 if self.finished is None else None),
            'scope': 'before request.json write through final integrity checks and metrics/provenance publication; excludes interpreter startup, CLI parsing, output reservation, timing.json/failure.json publication and final stdout',
            'accounting': 'exclusive wall intervals, repeated phases summed; total equals phase_seconds sum plus incomplete_phase seconds when present; absent phases did not run',
            'cuda_timing': 'wall time, not kernel time; synchronize only after generated arena and critic device placement; inference includes existing blocking .cpu() results, no added per-batch synchronization',
        }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'actor', 'opponent', 'output-dir'):
        parser.add_argument('--' + name, required=True)
    for name in CRITICS:
        parser.add_argument('--' + name)
    parser.add_argument('--reference', help='One selected critic; mutually exclusive with the legacy three roles')
    parser.add_argument('--reference-sha256')
    parser.add_argument('--reference-steps', type=int)
    parser.add_argument('--reuse-rollout-dir', help='Read-only original probe rollout; never arbitrary logs')
    parser.add_argument('--candidate')
    parser.add_argument('--candidate-sha256')
    parser.add_argument('--candidate-steps', type=int)
    parser.add_argument('--shuffle-hidden', action='store_true', help='One shared fixed within-game hidden-input derangement')
    parser.add_argument('--shuffle-seed', type=int, default=20260930)
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
    if args.reference:
        if any(getattr(args, name) for name in CRITICS) or args.reuse_rollout_dir or args.shuffle_hidden:
            parser.error('--reference requires a new rollout without legacy roles or hidden shuffle')
        if not re.fullmatch('[0-9a-f]{64}', args.reference_sha256 or '') or args.reference_steps is None or args.reference_steps < 0:
            parser.error('--reference requires its SHA256 and nonnegative internal step')
    elif not all(getattr(args, name) for name in CRITICS) or args.reference_sha256 or args.reference_steps is not None:
        parser.error('provide either --reference identity or all three legacy critic roles')
    if args.candidate:
        if not args.reference or args.candidate_steps != 256 or not re.fullmatch('[0-9a-f]{64}', args.candidate_sha256 or ''):
            parser.error('paired calibration requires reference plus exact candidate256 identity')
    elif args.candidate_sha256 or args.candidate_steps is not None:
        parser.error('candidate identity requires candidate')
    if args.games < 8 or args.games % 4:
        parser.error('--games must be a multiple of four, at least eight')
    if args.batch_size < 1 or args.torch_threads < 1 or args.rayon_threads < 1:
        parser.error('batch and thread counts must be positive')
    if args.bootstrap_replicates < 1000:
        parser.error('--bootstrap-replicates must be at least 1000')
    for name in ('seed_start', 'seed_key', 'sampling_seed', 'imputation_seed', 'bootstrap_seed', 'shuffle_seed'):
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



def shuffle_hidden_indices(length, *, seed, seed_key, seat, shuffle_seed):
    """One outcome-independent full-game cycle; no fixed points or batch RNG."""
    import numpy as np
    if length < 2:
        raise ValueError('hidden shuffle requires at least two states in every game')
    key = json.dumps([seed, seed_key, seat, shuffle_seed], separators=(',', ':')).encode()
    derived_seed = int.from_bytes(hashlib.sha256(key).digest()[:16], 'little')
    rng = np.random.Generator(np.random.PCG64(derived_seed))
    order = rng.permutation(length)
    indices = np.empty(length, dtype=np.int64)
    indices[order] = np.roll(order, 1)
    return indices



def validate_reuse_model_identity(previous, current, *, repo_root=None):
    """Only bridge old probe provenance using existing local Git blobs."""
    repo_root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[2]
    old_hashes = {k.replace('\\', '/'): v for k, v in previous['source_hashes'].items()}
    new_hashes = {k.replace('\\', '/'): v for k, v in current['source_hashes'].items()}
    evidence = {}
    for path in ('mortal/core/model.py', 'mortal/core/checkpoint_utils.py'):
        if path not in new_hashes:
            raise ValueError(f'current model runtime hash missing: {path}')
        if path in old_hashes:
            if old_hashes[path] != new_hashes[path]:
                raise ValueError(f'reuse model runtime source mismatch: {path}')
            evidence[path] = 'recorded_sha256_match'
            continue
        commit = previous.get('source_commit', '')
        if not re.fullmatch(r'[0-9a-fA-F]{40}|[0-9a-fA-F]{64}', commit):
            raise ValueError('legacy reuse requires an exact recorded local source commit')
        if previous.get('source_status') != '':
            raise ValueError('legacy reuse cannot reconstruct model code from a dirty or unrecorded source status')
        git_env = {**os.environ, 'GIT_NO_LAZY_FETCH': '1', 'GIT_TERMINAL_PROMPT': '0'}
        try:
            old_blob = subprocess.check_output(['git', 'cat-file', 'blob', f'{commit}:{path}'],
                                               cwd=repo_root, stderr=subprocess.PIPE, env=git_env)
            head_blob = subprocess.check_output(['git', 'cat-file', 'blob', f'HEAD:{path}'],
                                                cwd=repo_root, stderr=subprocess.PIPE, env=git_env)
        except (OSError, subprocess.CalledProcessError) as exc:
            raise ValueError(f'legacy model identity cannot be established from local Git: {path}') from exc
        if old_blob != head_blob or hashlib.sha256(head_blob).hexdigest() != new_hashes[path]:
            raise ValueError(f'legacy model runtime blob mismatch: {path}')
        evidence[path] = {'mode': 'local_git_blob_match', 'previous_commit': commit,
                          'sha256': hashlib.sha256(head_blob).hexdigest()}
    return evidence


def validate_reuse_provenance(previous, current):
    if previous.get('schema') != 1 or previous.get('status') not in ('running', 'complete'):
        raise ValueError('unsupported original probe provenance')
    if previous.get('rollout_reuse') or previous.get('arguments', {}).get('reuse_rollout_dir'):
        raise ValueError('reuse accepts an original generated rollout, not a chain of reused outputs')
    for role in ('actor', 'opponent'):
        if previous['weights_and_config'][role]['sha256'] != current['weights_and_config'][role]['sha256']:
            raise ValueError(f'reuse {role} identity mismatch')
    for key in ('native_sha256', 'torch_version', 'numpy_version', 'actor_explore_rate',
                'opponent_explore_rate', 'actor_agari_guard', 'opponent_agari_guard',
                'search_enabled', 'actor_oracle_guiding', 'probe_pts'):
        if key not in previous or previous[key] != current[key]:
            raise ValueError(f'reuse sampling/native contract mismatch: {key}')
    def extension_identity(provenance):
        values = provenance.get('native_extension_sha256', {})
        if not values:
            raise ValueError('reuse requires recorded native extension hashes')
        return sorted(values.values())
    if extension_identity(previous) != extension_identity(current):
        raise ValueError('reuse native extension identity mismatch')
    for key in ('games', 'seed_start', 'seed_key', 'sampling_seed', 'device', 'torch_threads', 'rayon_threads'):
        if previous['arguments'].get(key) != current['arguments'][key]:
            raise ValueError(f'reuse sampling argument mismatch: {key}')
    # Probe/diagnostic code can change; the actual actor arena path cannot.
    for key in ('mortal/eval/engine.py', 'mortal/eval/player.py'):
        def source_hash(provenance):
            return {k.replace('\\', '/'): v for k, v in provenance['source_hashes'].items()}[key]
        if source_hash(previous) != source_hash(current):
            raise ValueError(f'reuse actor runtime source mismatch: {key}')


def load_verified_rollout(source, current, *, load_games, duplicate_sets):
    source = Path(source).resolve()
    provenance_path, outcomes_path = source / 'provenance.json', source / 'outcomes.json'
    previous = json.loads(provenance_path.read_text(encoding='utf-8'))
    validate_reuse_provenance(previous, current)
    model_identity = validate_reuse_model_identity(previous, current)
    outcomes_hash = file_sha256(outcomes_path)
    recorded_hash = previous.get('artifact_sha256', {}).get('outcomes.json')
    if recorded_hash is not None and recorded_hash != outcomes_hash:
        raise ValueError('reuse outcomes artifact hash mismatch')
    recorded = json.loads(outcomes_path.read_text(encoding='utf-8'))
    args = current['arguments']
    if not isinstance(recorded, list) or len(recorded) != args['games']:
        raise ValueError('reuse outcomes count mismatch')
    if (source / 'games').is_symlink():
        raise ValueError('reuse original games directory must not be a symlink')
    log_root = (source / 'games').resolve()
    by_path = {}
    for game in recorded:
        path = Path(game['log_path']).resolve()
        if not path.is_relative_to(log_root) or path in by_path:
            raise ValueError('reuse requires unique original logs under source/games')
        if file_sha256(path) != game['sha256']:
            raise ValueError(f'reuse game log hash mismatch: {path}')
        if verify_complete_log(path) != game['kyoku_count']:
            raise ValueError('reuse game completeness mismatch')
        by_path[path] = game
    fresh = load_games(log_root, 'trainee')
    if len(fresh) != args['games'] or {Path(g['log_path']).resolve() for g in fresh} != set(by_path):
        raise ValueError('reuse directory includes missing or unregistered logs')
    games = []
    for game in fresh:
        saved = by_path[Path(game['log_path']).resolve()]
        if any(game[key] != saved[key] for key in ('seed', 'seed_key', 'challenger_seat', 'challenger_rank')):
            raise ValueError('reuse recorded outcome differs from native log')
        games.append({**game, 'kyoku_count': saved['kyoku_count'], 'sha256': saved['sha256']})
    expected = {(seed, args['seed_key']) for seed in range(args['seed_start'], args['seed_start'] + args['games'] // 4)}
    if set(duplicate_sets(games)) != expected:
        raise ValueError('reuse does not contain requested complete four-seat seed groups')
    return games, {'source_dir': str(source), 'model_source_identity': model_identity, 'source_status_at_read': previous['status'],
                   'source_provenance_sha256': file_sha256(provenance_path),
                   'source_outcomes_sha256': outcomes_hash,
                   'evidence': 'full provenance, registered complete native logs and hashes, exact four-seat groups; status alone is insufficient'}

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


def run(args, root, *, timings):
    from mortal.research.frozen_probe_cache import (
        ProbeBudget, BudgetExpired, prediction_stream, temporal_fields, reference_summary, inference_slices,
    )
    candidate = getattr(args, 'candidate', None)
    from mortal.research.paired_critic_calibration import PairedBudget, input_fingerprint, paired_summary
    names = ('reference', 'candidate') if candidate else ('reference',) if args.reference else CRITICS
    budget = (PairedBudget if candidate else ProbeBudget).from_environment() if args.reference else None
    if budget:
        budget.check()
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
    files = {name: Path(getattr(args, name)).resolve() for name in ('config', 'actor', 'opponent', *names)}
    hashes = {name: file_sha256(path) for name, path in files.items()}
    if args.reference and hashes['reference'] != args.reference_sha256:
        raise ValueError('reference checkpoint SHA256 mismatch')
    if candidate and hashes['candidate'] != args.candidate_sha256:
        raise ValueError('candidate checkpoint SHA256 mismatch')
    if args.reference:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
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
                  'timing_report': 'timing.json',
                  'arguments': vars(args), 'weights_and_config': {
                      name: {'path': str(path), 'sha256': hashes[name]} for name, path in files.items()},
                  'source_commit': git_output('rev-parse', 'HEAD'),
                  'source_status': git_output('status', '--porcelain'),
                  'source_hashes': {str(Path(p).relative_to(source_root)): file_sha256(p) for p in
                      [Path(__file__), source_root / 'mortal/data/dataloader.py',
                       source_root / 'mortal/core/model.py', source_root / 'mortal/core/checkpoint_utils.py',
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
    if args.reference:
        for relative in ('mortal/research/frozen_probe_cache.py', 'mortal/data/oracle_value.py',
                         'mortal/online/pretrain_oracle_critic.py'):
            provenance['source_hashes'][relative] = file_sha256(source_root / relative)
        provenance.update(
            mode='paired_calibration256' if candidate else 'single_reference', budget=budget.manifest(),
            planned_groups=[{'seed': seed, 'seed_key': args.seed_key, 'seats': [0, 1, 2, 3]}
                            for seed in range(args.seed_start, args.seed_start + args.games // 4)],
            distribution_scope='frozen sampled actor versus the exact original baseline file',
            numerical={'critic_amp': False, 'critic_dtype': 'float32', 'actor_amp': device.type == 'cuda',
                       'opponent_amp': device.type == 'cuda', 'matmul_tf32': False, 'cudnn_tf32': False,
                       'cudnn_benchmark': False, 'physical_critic_batch': args.batch_size},
            value_unit_mapping={'head': 'four scalar MSE ValueHead outputs; not rank probabilities',
                'mapping': 'identity: scale=1, offset=0; score_rank_mc return-to-go in original rank-point units',
                'rank_points': pts, 'centering': 'training subtracts mean(pts)=0',
                'training_target': 'sum of successive kyoku rank-point changes through end_game',
                'source': ['mortal/data/oracle_value.py:score_rank_delta_rewards_by_kyoku',
                           'mortal/data/oracle_value.py:discounted_returns_from_step_rewards',
                           'mortal/online/pretrain_oracle_critic.py:output_weighted_mse'],
                'head_coordinate': 'absolute seat (trainee_seat + head) % 4; p0 is controlled actor',
                'no_new_label_scaling': True},
            temporal_contract={'observation': 'pre-action, full native trainee decision clock, no fold or shuffle',
                'reward': 'all intervening kyoku rank-point deltas until this actor next decides; final remainder once',
                'done': 'last controlled decision of a verified end_game only; end_kyoku does not terminate',
                'terminal_next_value': 0, 'truncation': 'partial games/groups excluded, never bootstrapped as complete MC',
                'gamma': 1.0, 'production_lambda': .95, 'identity_lambda': 1.0,
                'normalization': 'none', 'lambda1_tolerance': {'atol': 2e-5, 'rtol': 2e-5,
                    'source': 'existing frozen probe FP32 identity check'}})
    native_dir = Path(libriichi.__file__).resolve().parent
    native_extensions = sorted(set(native_dir.glob('*.pyd')) | set(native_dir.glob('*.so')))
    provenance['hidden_shuffle'] = {
        'enabled': args.shuffle_hidden, 'seed': args.shuffle_seed,
        'rule': 'sha256([game_seed,seed_key,trainee_seat,shuffle_seed]) first128bits little-endian -> PCG64 permutation -> one successor cycle over all game states; no fixed points',
        'scope': 'within complete trainee game; same indices for all three critics and every physical batch',
        'batch_layout': {'game_order': 'sorted native log paths', 'state_order': 'original complete temporal order',
                         'physical_batch_size': args.batch_size, 'critic_amp': False},
        'interpretation': 'visible/hidden inconsistency is OOD; dependence diagnostic, not a visible-only control or causal Oracle improvement'}
    provenance['native_extension_sha256'] = {str(p): file_sha256(p) for p in native_extensions}
    atomic_write_json(root / 'provenance.json', provenance)
    atomic_write_json(root / 'effective_config.json', config)
    # Validate all three critics before spending any arena computation.
    critics = {}
    for name in names:
        state = torch.load(files[name], map_location='cpu', weights_only=True, mmap=True)
        pre = validate_critic_contract(state, version=version, pts=pts, role=None if args.reference else name)
        expected_step = args.candidate_steps if name == 'candidate' else args.reference_steps
        if args.reference and state.get('steps') != expected_step:
            raise ValueError(f'{name} checkpoint internal step mismatch')
        if name == 'candidate':
            clock = state.get('optimizer_update_clock', {})
            if (clock.get('successes') != 256 or state.get('artifact_role') != 'calibration256_eval_weights_only'
                    or state.get('reference_sha256') != hashes['reference']):
                raise ValueError('candidate is not the fixed256 endpoint from this reference')
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
    if budget:
        budget.check()
    timings.switch('rollout_reuse_validation' if args.reuse_rollout_dir else 'arena_generation_validation')
    if args.reuse_rollout_dir:
        games, reuse = load_verified_rollout(args.reuse_rollout_dir, provenance,
                                            load_games=load_games, duplicate_sets=duplicate_sets)
        provenance['rollout_reuse'] = reuse
        groups = duplicate_sets(games)
        rankings = np.bincount([game['challenger_rank'] - 1 for game in games], minlength=4)
    else:
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
        del player
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
    if args.reference:
        games.sort(key=lambda game: (game['seed'], game['seed_key'], game['challenger_seat']))
    atomic_write_json(root / 'outcomes.json', games)
    atomic_write_json(root / 'provenance.json', provenance)
    timings.switch('scoring_setup')
    # Free actor GPU storage before critic scoring. No policy calls thereafter.
    actor.cpu(); policy.cpu(); opponent.cpu(); opponent_policy.cpu()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    for brain, value in critics.values():
        brain.to(device); value.to(device)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    targets, all_context = [], []
    predictions = {name: [] for name in names}
    shuffled_predictions = {name: [] for name in names}
    shuffle_mappings = []
    advantages = {name: [] for name in names}
    clustered = defaultdict(lambda: defaultdict(list))
    identity_max = {name: 0.0 for name in names}
    game_counts = []
    input_fingerprints = []
    with prediction_stream(root, reference=bool(args.reference)) as stream:
        try:
            for game_index, game in enumerate(games):
                if budget and budget.stopping():
                    break
                timings.switch('replay_decode_validation')
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
                if args.reference and trajectory['player_id'] != game['challenger_seat']:
                    raise ValueError('native controlled player does not match arena seat')

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
                shuffle_indices = None
                if args.shuffle_hidden:
                    shuffle_indices = shuffle_hidden_indices(len(obs), seed=game['seed'], seed_key=game['seed_key'],
                                                            seat=game['challenger_seat'], shuffle_seed=args.shuffle_seed)
                    shuffle_mappings.append({'game_index': game_index, 'states': len(obs),
                                             'indices_sha256': hashlib.sha256(shuffle_indices.astype('<i8').tobytes()).hexdigest()})
                fingerprints = input_fingerprint(trajectory, target) if candidate else None
                per_game, per_game_shuffled, raw_temporal = {}, {}, {}
                for name, (brain, value) in critics.items():
                    timings.switch('critic_inference')
                    with torch.inference_mode():
                        pred = torch.cat([value(brain(
                            torch.as_tensor(obs[i:i + args.batch_size], device=device, dtype=torch.float32),
                            invisible_obs=torch.as_tensor(invisible[i:i + args.batch_size], device=device, dtype=torch.float32)
                        )).cpu() for i in inference_slices(len(obs), args.batch_size, budget)]).numpy()
                    timings.switch('aggregation_statistics')
                    summary = summarize_predictions(target, pred)
                    cluster_key = (game['seed'], game['seed_key'])
                    clustered[name][cluster_key].append((len(target), summary['p0_mse'], summary['all_players_mse']))
                    if args.shuffle_hidden:
                        timings.switch('critic_inference')
                        with torch.inference_mode():
                            shuffled = torch.cat([value(brain(
                                torch.as_tensor(obs[i:i + args.batch_size], device=device, dtype=torch.float32),
                                invisible_obs=torch.as_tensor(invisible[shuffle_indices[i:i + args.batch_size]], device=device, dtype=torch.float32)
                            )).cpu() for i in inference_slices(len(obs), args.batch_size, budget)]).numpy()
                        timings.switch('aggregation_statistics')
                        shuffle_summary = summarize_predictions(target, shuffled)
                        clustered[name + '_shuffle'][cluster_key].append(
                            (len(target), shuffle_summary['p0_mse'], shuffle_summary['all_players_mse']))
                        shuffled_predictions[name].append(shuffled)
                        per_game_shuffled[name] = shuffled
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
                    if args.reference and name == 'reference':
                        raw_temporal = temporal_fields(trajectory, game, rewards, target, pred,
                                                       compute_gae_advantages_from_step_rewards)

                if candidate:
                    if fingerprints != input_fingerprint(trajectory, target):
                        raise ValueError('shared paired scoring inputs mutated')
                    input_fingerprints.append({'seed':game['seed'],'seed_key':game['seed_key'],
                        'seat':game['challenger_seat'],'states':len(target),'shared_C0_C1':fingerprints})
                    atomic_write_json(root / 'input_fingerprints.json', input_fingerprints)
                timings.switch('prediction_write')
                for i in range(len(target)):
                    stream.write(json.dumps({'game_index': game_index, 'seed': game['seed'], 'seed_key': game['seed_key'],
                                             'trainee_seat': game['challenger_seat'], 'state_index': i,
                                             'current_rank': int(context[i, 4]), 'all_last': bool(context[i, 3]),
                                             **({key: value[i] for key, value in raw_temporal.items()} if args.reference else {}),
                                             **({'hidden_source_index': int(shuffle_indices[i]),
                                                 'pred_shuffled': {name: per_game_shuffled[name][i].tolist() for name in names}}
                                                if args.shuffle_hidden else {}),
                                             'target': target[i].tolist(),
                                             'pred': {name: per_game[name][i].tolist() for name in names}},
                                            separators=(',', ':'), allow_nan=False) + '\n')
                if args.reference:
                    stream.finish_game(game)
                timings.switch('aggregation_statistics')
                zero_summary = summarize_predictions(target, np.zeros_like(target))
                clustered['constant_zero'][(game['seed'], game['seed_key'])].append(
                    (len(target), zero_summary['p0_mse'], zero_summary['all_players_mse']))
                targets.append(target)
                all_context.append(context)
                game_counts.append(len(target))
                print(json.dumps({'scored_games': game_index + 1, 'total_games': len(games), 'states': sum(game_counts)}), flush=True)
                timings.switch('prediction_write')  # Includes gzip close and atomic publication on the last game.
        except BudgetExpired:
            if not args.reference:
                raise
    timings.switch('aggregation_statistics')
    if args.reference:
        completed_count = stream.completed_games
        if candidate:
            result = paired_summary(args, games[:completed_count], targets[:completed_count],
                                    predictions['reference'][:completed_count], predictions['candidate'][:completed_count],
                                    all_context[:completed_count], advantages['reference'][:completed_count], stream.groups)
            result['shared_input_fingerprints_sha256'] = file_sha256(root / 'input_fingerprints.json') if input_fingerprints else None
        else:
            result = reference_summary(args, games[:completed_count], targets[:completed_count],
                                       predictions['reference'][:completed_count], all_context[:completed_count],
                                       advantages['reference'][:completed_count], stream.groups)
        timings.switch('final_integrity_write')
        if hashes != {name: file_sha256(path) for name, path in files.items()}:
            raise RuntimeError('input checkpoint or configuration changed during probe')
        result['input_files_unchanged'] = True
        result['lambda1_return_identity_max_abs'] = identity_max['reference']
        atomic_write_json(root / 'metrics.json', result)
        provenance.update(status=result['status'], finished_unix=time.time(),
                          artifact_sha256={name: file_sha256(root / name) for name in
                                           ('outcomes.json', 'metrics.json', 'effective_config.json')},
                          completed_group_artifacts=stream.groups)
        atomic_write_json(root / 'provenance.json', provenance)
        return {'status': result['status'], 'output_dir': str(root), 'games': completed_count, 'states': result['states']}
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
    for name in names:
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
    comparisons = [*PAIRS, *((name, 'constant_zero') for name in names)]
    if args.shuffle_hidden:
        result['hidden_shuffle'] = {'protocol': provenance['hidden_shuffle'], 'mapping_hashes': shuffle_mappings, 'metrics': {}}
        for name in names:
            key = name + '_shuffle'
            pred = np.concatenate(shuffled_predictions[name])
            result['hidden_shuffle']['metrics'][name] = summarize_predictions(target, pred)
            seed_mse[key] = np.asarray([
                np.average(np.asarray(clustered[key][group])[:, 1:], axis=0,
                           weights=np.asarray(clustered[key][group])[:, 0]) for group in sorted(groups)])
            result['hidden_shuffle']['metrics'][name]['equal_seed_group_mse'] = dict(zip(
                ('p0', 'all_players'), seed_mse[key].mean(0).tolist()))
            comparisons.append((key, name))
    for left, right in comparisons:
        result['paired_differences'][f'{left}-{right}'] = {
            key: {**bootstrap_interval((seed_mse[left] - seed_mse[right])[:, i],
                                       replicates=args.bootstrap_replicates, seed=args.bootstrap_seed),
                  'seed_groups': len(groups), 'games': len(games), 'states': len(target)}
            for i, key in enumerate(('p0_mse', 'all_players_mse'))}
    timings.switch('final_integrity_write')
    if hashes != {name: file_sha256(path) for name, path in files.items()}:
        raise RuntimeError('input checkpoint or configuration changed during probe')
    if args.reuse_rollout_dir:
        reuse = provenance['rollout_reuse']
        if file_sha256(Path(reuse['source_dir']) / 'outcomes.json') != reuse['source_outcomes_sha256']:
            raise RuntimeError('reused outcomes changed during scoring')
        if any(file_sha256(game['log_path']) != game['sha256'] for game in games):
            raise RuntimeError('reused game logs changed during scoring')
        result['reused_rollout_verified_unchanged'] = True
    result['input_files_unchanged'] = True
    atomic_write_json(root / 'metrics.json', result)
    provenance.update(status='complete', finished_unix=time.time(),
                      artifact_sha256={name: file_sha256(root / name) for name in
                                       ('outcomes.json', 'metrics.json', 'predictions.jsonl.gz', 'effective_config.json')})
    atomic_write_json(root / 'provenance.json', provenance)
    return {'status': 'complete', 'output_dir': str(root), 'states': len(target)}


def main(argv=None):
    args = parse_args(argv)
    if args.reuse_rollout_dir and Path(args.output_dir).resolve().is_relative_to(Path(args.reuse_rollout_dir).resolve()):
        raise ValueError('new output must be outside the read-only original rollout directory')
    root = reserve_output(args.output_dir)
    timings = ProbePhaseTimings()
    try:
        atomic_write_json(root / 'request.json', vars(args))
        completion = run(args, root, timings=timings)
        timings.finish()
        timing_report = timings.report()
        atomic_write_json(root / 'timing.json', timing_report)
        print(json.dumps({**completion, 'timing': timing_report}), flush=True)
    except BaseException as exc:
        atomic_write_json(root / 'failure.json', {'status': 'failed', 'type': type(exc).__name__,
                                                'error': str(exc), 'unix': time.time(), 'timing': timings.report()})
        raise


if __name__ == '__main__':
    main()
