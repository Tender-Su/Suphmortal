"""Engineering-only: two real critic updates on one verified, closed 8-game rollout.

This is deliberately not a training candidate or a calibration/readiness result.
It invokes production train(), replacing only server I/O and adding observers.
Run in a fresh process under the runner's independent deadline supervisor.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from functools import wraps
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from unittest.mock import patch

from mortal.core.artifacts import atomic_write_json, atomic_write_toml, file_sha256
from mortal.core.update_clock import OptimizerUpdateClock
from mortal.research.frozen_actor_critic_probe import (
    load_policy, load_verified_rollout, reserve_output, validate_critic_contract,
)

IMPUTATION_SEED = 20260905
SUCCESSFUL_UPDATES = 2
REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_FILES = (
    'mortal/research/critic_only_update_check.py',
    'mortal/research/frozen_actor_critic_probe.py',
    'mortal/online/train_online.py', 'mortal/online/critic_calibration.py',
    'mortal/core/update_clock.py', 'mortal/core/model.py',
    'mortal/core/checkpoint_utils.py', 'mortal/data/dataloader.py',
    'mortal/data/oracle_value.py', 'mortal/eval/player.py', 'mortal/eval/engine.py',
    'libriichi/src/dataset/invisible.rs',
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'actor', 'critic', 'opponent', 'reuse-rollout-dir', 'output-dir'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--engineering-lr', type=float, required=True,
                        help='Explicit positive smoke-test LR, not a research recommendation')
    for name in ('seed-start', 'seed-key', 'sampling-seed'):
        parser.add_argument('--' + name, type=int, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--torch-threads', type=int, default=1)
    parser.add_argument('--rayon-threads', type=int, default=4)
    parser.add_argument('--enable-amp', action='store_true')
    args = parser.parse_args(argv)
    if not math.isfinite(args.engineering_lr) or args.engineering_lr <= 0:
        parser.error('--engineering-lr must be finite and positive')
    if min(args.batch_size, args.torch_threads, args.rayon_threads) < 1:
        parser.error('batch and thread counts must be positive')
    for key in ('seed_start', 'seed_key', 'sampling_seed'):
        if not 0 <= getattr(args, key) < 2**64:
            parser.error(f'{key} must be an unsigned 64-bit integer')
    if args.seed_start + 2 >= 2**64:
        parser.error('seed range exceeds u64')
    return args


def reserve_check_output(output, rollout):
    output, rollout = Path(output).absolute(), Path(rollout).resolve()
    if output.resolve().is_relative_to(rollout):
        raise ValueError('output must be outside the read-only rollout')
    return reserve_output(output)


def build_config(base, actor_config, critic_state, args, root):
    """Deep-copy inputs; retain the complete checkpoint-inherited auxiliary recipe."""
    config = deepcopy(base)
    for section in ('aux', 'supervised'):
        config[section] = deepcopy(actor_config.get(section, {}))
    config['resnet'] = deepcopy(actor_config['resnet'])
    if config['resnet'] != critic_state['config']['resnet']:
        raise ValueError('production online actor/critic must share the checked resnet architecture')
    control = config.setdefault('control', {})
    control.update(version=actor_config['control']['version'], online=True,
                   state_file=str(root / 'engineering_only_latest.pth'),
                   best_state_file=str(root / 'unused_best.pth'),
                   tensorboard_dir=str(root / 'tensorboard'), device=args.device,
                   batch_size=args.batch_size, opt_step_every=1,
                   save_every=1000000, test_every=1000000,
                   submit_every=1000000, old_update_every=1000000,
                   enable_compile=False, enable_amp=args.enable_amp,
                   enable_cudnn_benchmark=False)
    config.setdefault('env', {})['pts'] = [2, 1, 0, -3]
    config.setdefault('dataset', {}).update(num_workers=0, enable_augmentation=False,
                                            augmented_first=False)
    online = config.setdefault('online', {})
    online.update(init_state_file=str(Path(args.actor).resolve()),
                  max_successful_optimizer_steps=SUCCESSFUL_UPDATES, stop_at_max_steps=False,
                  gae_inference_batch_size=args.batch_size, enable_compile=False)
    online.setdefault('importance_sampling', {}).update(
        enabled=False, vtrace_mode='disabled', drop_untracked_samples=False)
    config.setdefault('policy', {}).update(
        gae_enabled=True, gae_gamma=1.0, gae_lambda=1.0,
        actor_objective='ppo', online_action_scope='all',
        vtrace_target_rho_clip=0.0, vtrace_target_c_clip=0.0)
    pre = critic_state['oracle_critic_pretrain']
    value = config.setdefault('value', {})
    value.update(enabled=True, critic_only=True, critic_warmup_steps=0,
                 oracle_critic=True, oracle_critic_arch=pre['critic_arch'],
                 oracle_critic_state_file=str(Path(args.critic).resolve()),
                 target_mode='all_players', reward_source='score_rank',
                 exact_zero_sum=bool(pre.get('exact_zero_sum', False)))
    for name, default in (('oracle_fusion_mode', 'linear'), ('oracle_fusion_hidden', 512),
                          ('value_head_hidden', 256), ('value_loss_mode', 'mse')):
        value[name] = pre.get(name, default)
    if not math.isfinite(float(value.get('weight', 0.5))) or float(value.get('weight', 0.5)) <= 0:
        raise ValueError('value.weight must be finite and positive')
    config.setdefault('optim', {})['scheduler'] = {
        'init': args.engineering_lr, 'peak': args.engineering_lr,
        'final': args.engineering_lr, 'warm_up_steps': 0, 'max_steps': 2,
    }
    config.setdefault('test_play', {}).update(
        enable=False, initial_enable=False, log_dir=str(root / 'unused_test_games'))
    config.setdefault('baseline', {})['test'] = {
        'state_file': str(Path(args.opponent).resolve()), 'device': args.device,
        'enable_compile': False,
    }
    config.setdefault('search', {})['enabled'] = False
    config.setdefault('oracle_guiding', {}).update(actor_enabled=False, actor_source='zero')
    config['oracle_experiments'] = {'default_arm': 'current_config', 'suffix_artifacts': False}
    config.setdefault('oracle_dependency_eval', {}).update(
        enabled=False, log_dir=str(root / 'unused_dependency_games'))
    config['repro'] = {'enabled': True, 'seed': args.sampling_seed,
                       'train_key': args.seed_key, 'train_seed_start': args.seed_start,
                       'allow_cudnn_benchmark': False}
    return config


def validate_registered_roles(previous, hashes):
    """A 40k step label alone cannot identify the approved warm checkpoint."""
    for role, input_name in (('actor', 'actor'), ('opponent', 'opponent'), ('warm40k', 'critic')):
        if previous['weights_and_config'][role]['sha256'] != hashes[input_name]:
            raise ValueError(f'input does not match registered rollout role: {role}')
    if previous['arguments'].get('imputation_seed') != IMPUTATION_SEED:
        raise ValueError('rollout has a different fixed Oracle imputation seed')


class OneShotDrain:
    """Read original registered files once; never wait, replay, copy, or rename."""
    def __init__(self, directory, games):
        self.directory = Path(directory).resolve()
        self.expected = {Path(game['log_path']).resolve(): game['sha256'] for game in games}
        self.calls = 0

    def __call__(self):
        if self.calls:
            raise RuntimeError('verified closed rollout exhausted before the successful-update budget')
        self.calls += 1
        entries = list(self.directory.iterdir())
        actual = {path.resolve() for path in entries}
        if (len(entries) != len(self.expected) or actual != set(self.expected)
                or any(path.is_symlink() or not path.is_file() for path in entries)):
            raise ValueError('drain directory differs from the registered closed rollout')
        verify_hashes(self.expected)
        return str(self.directory)


def seeded_trajectories(real_method):
    @wraps(real_method)
    def wrapped(self, file_list, *, oracle_imputation_seed=None):
        if oracle_imputation_seed not in (None, IMPUTATION_SEED):
            raise ValueError('conflicting Oracle imputation seed')
        return real_method(self, file_list, oracle_imputation_seed=IMPUTATION_SEED)
    return wrapped


def observed_steps(real_step, before, after):
    @wraps(real_step)
    def wrapped(scaler, optimizer, clock):
        # Read the real train_batch loss before its graph/locals go out of scope.
        # Do not replace the loss, step, scaler, optimizer, clock, or return value.
        before(sys._getframe(1).f_locals, optimizer, clock)
        succeeded = real_step(scaler, optimizer, clock)
        after(succeeded, optimizer, clock)
        return succeeded
    return wrapped


def verify_hashes(expected):
    for path, digest in expected.items():
        if file_sha256(path) != digest:
            raise ValueError(f'read-only input changed: {path}')


def validate_saved_clock(state, observations):
    if 'optimizer_update_clock' not in state:
        raise ValueError('latest lacks the real exact optimizer clock')
    clock = OptimizerUpdateClock.from_checkpoint(state, opt_step_every=1)
    if (clock.successes != SUCCESSFUL_UPDATES or clock.legacy_attempt_offset
            or clock.inherited_progress_offset or clock.attempts != len(observations)
            or state['steps'] != clock.attempts):
        raise ValueError('latest does not prove exactly two fresh successful updates')
    if sum(row['succeeded'] for row in observations) != clock.successes:
        raise ValueError('observed real updates disagree with checkpoint successes')
    if not observations or observations[-1]['clock'] != clock.state_dict():
        raise ValueError('saved and observed optimizer clocks disagree')
    return clock.state_dict()


def assert_state_equal(actual, expected, torch, label):
    if actual.keys() != expected.keys():
        raise ValueError(f'{label} state keys differ')
    for key in actual:
        a, b = actual[key].detach().cpu(), expected[key].detach().cpu()
        if a.dtype != b.dtype or a.shape != b.shape or not torch.equal(a, b):
            raise ValueError(f'{label} state changed: {key}')


class TrainingObservation:
    def __init__(self, torch, actor_state, critic_state, fixed_inputs, output):
        self.torch, self.actor_state, self.critic_state = torch, actor_state, critic_state
        self.fixed_inputs, self.output = fixed_inputs, output
        self.models = None
        self.initial_logits = None
        self.publications, self.steps = [], []

    def actor_unchanged(self):
        for key, model in zip(('mortal', 'policy_net'), self.models):
            assert_state_equal(model.state_dict(), self.actor_state[key], self.torch, key)
            if any(module.training for module in model.modules()):
                raise ValueError(f'{key} left eval mode')
            if any(parameter.grad is not None for parameter in model.parameters()):
                raise ValueError(f'{key} received a gradient')

    def logits(self, models):
        actor, policy = models
        device = next(actor.parameters()).device
        obs, masks = (value.to(device) for value in self.fixed_inputs)
        with self.torch.inference_mode():
            return policy.logits(actor(obs), masks).detach().cpu()

    def submit(self, actor, policy, *, is_idle, runtime, aux_payload):
        if self.models is None:
            self.models = (actor, policy)
        elif self.models != (actor, policy):
            raise ValueError('publication replaced the observed live actor')
        self.actor_unchanged()
        if (runtime['actor_oracle_enabled'] or runtime['actor_oracle_keep_prob'] != 0
                or not runtime['oracle_critic_enabled']):
            raise ValueError('actual publication changed the visible actor / Oracle critic contract')
        if not self.publications:
            for key in ('oracle_brain', 'value_net'):
                assert_state_equal(aux_payload[key], self.critic_state[key], self.torch, key)
            self.initial_logits = self.logits(self.models)
        row = {'version': len(self.publications), 'is_idle': bool(is_idle),
               'runtime': deepcopy(runtime), 'input_actor_state_equal': True}
        self.publications.append(row)
        return row['version']  # Local transport acknowledgment; never an optimizer clock.

    def before_step(self, local, optimizer, clock):
        if self.models is None:
            raise ValueError('optimizer reached before the real initial policy publication')
        self.actor_unchanged()
        if local.get('policy_step_active') is not False:
            raise ValueError('production policy/auxiliary loss gate is active')
        value_loss = float(local['value_loss_val'].detach().cpu())
        total_loss = float(local['loss'].detach().cpu())
        if not math.isfinite(value_loss) or not math.isfinite(total_loss):
            raise ValueError('nonfinite actual training value/total loss')
        for key in ('aux_loss_val', 'opp_loss_val', 'danger_loss_val', 'exp_reward_loss_val',
                    'tile_eff_loss_val', 'furo_regret_loss_val', 'hand_value_regret_loss_val'):
            if float(local[key].detach().cpu()) != 0.0:
                raise ValueError(f'critic-only auxiliary objective is active: {key}')
        self.critics = (local['oracle_brain'], local['value_net'])
        self.pending = {'value_loss': value_loss, 'total_loss': total_loss,
                        'policy_step_active': False, 'actor_gradients_none': True,
                        'learning_rates': [float(group['lr']) for group in optimizer.param_groups]}
        if not any(lr > 0 and math.isfinite(lr) for lr in self.pending['learning_rates']):
            raise ValueError('real optimizer has no finite positive learning rate')

    def after_step(self, succeeded, optimizer, clock):
        self.actor_unchanged()
        gradients_finite = []
        for model in self.critics:
            grads = [p.grad for p in model.parameters() if p.grad is not None]
            finite = bool(grads) and all(bool(self.torch.isfinite(g).all()) for g in grads)
            if succeeded and not finite:
                raise ValueError('successful critic update has missing/nonfinite gradients')
            if any(not bool(self.torch.isfinite(v).all()) for v in model.state_dict().values()):
                raise ValueError('critic state became nonfinite after a real step')
            gradients_finite.append(finite)
        self.steps.append({**self.pending, 'succeeded': bool(succeeded),
                           'critic_gradients_finite': gradients_finite, 'clock': clock.state_dict()})
        atomic_write_json(self.output / 'observed_updates.json', self.steps)


def run(args, root):
    if 'mortal.config' in sys.modules:
        raise RuntimeError('run this entry point in a fresh process before importing mortal.config')
    for name in ('MORTAL_ORACLE_ARM', 'MORTAL_ORACLE_ARTIFACT_SUFFIX'):
        if os.environ.get(name):
            raise ValueError(f'unset {name}; external experiment overrides are not supported')
    os.environ['RAYON_NUM_THREADS'] = str(args.rayon_threads)
    import numpy as np
    import torch
    import libriichi
    from mortal.core.toml_utils import load_toml_file

    torch.set_num_threads(args.torch_threads)
    torch.set_num_interop_threads(1)
    files = {name: Path(getattr(args, name)).resolve() for name in ('config', 'actor', 'critic', 'opponent')}
    hashes = {name: file_sha256(path) for name, path in files.items()}
    source = Path(args.reuse_rollout_dir).resolve()
    protected = {path: hashes[name] for name, path in files.items()}
    for name in ('provenance.json', 'outcomes.json'):
        protected[source / name] = file_sha256(source / name)
    previous = json.loads((source / 'provenance.json').read_text(encoding='utf-8'))
    validate_registered_roles(previous, hashes)
    actor_state = torch.load(files['actor'], map_location='cpu', weights_only=True, mmap=True)
    critic_state = torch.load(files['critic'], map_location='cpu', weights_only=True, mmap=True)
    if actor_state.get('steps') != 50000 or actor_state['config']['control']['version'] != 4:
        raise ValueError('this engineering check requires the registered v4 C50k actor')
    pre = validate_critic_contract(critic_state, version=4, pts=[2, 1, 0, -3], role='warm40k')
    config = build_config(load_toml_file(files['config']), actor_state['config'], critic_state, args, root)
    atomic_write_toml(root / 'effective_config.toml', config)
    os.environ['MORTAL_CFG'] = str(root / 'effective_config.toml')
    from mortal.config import config as live_config
    expected_recipe = {key: deepcopy(live_config[key]) for key in ('aux', 'supervised')}
    from mortal.core import common
    from mortal.data.dataloader import FileDatasetsIter
    from mortal.eval.paired_1v3 import load_games, duplicate_sets
    from mortal.online import train_online

    native_path = Path(libriichi.__file__).resolve()
    extensions = sorted(set(native_path.parent.glob('*.pyd')) | set(native_path.parent.glob('*.so')))
    provenance = {
        'schema': 1, 'status': 'running', 'engineering_only': True, 'candidate_model': False,
        'started_unix': time.time(), 'arguments': {**vars(args), 'games': 8,
                                                  'imputation_seed': IMPUTATION_SEED},
        'weights_and_config': {name: {'path': str(path), 'sha256': hashes[name]} for name, path in files.items()},
        'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO_ROOT, text=True).strip(),
        'source_status': subprocess.check_output(['git', 'status', '--porcelain'], cwd=REPO_ROOT, text=True).strip(),
        'source_hashes': {name: file_sha256(REPO_ROOT / name) for name in SOURCE_FILES},
        'native_sha256': file_sha256(native_path),
        'native_extension_sha256': {str(path): file_sha256(path) for path in extensions},
        'torch_version': torch.__version__, 'numpy_version': np.__version__,
        'actor_explore_rate': 1.0, 'opponent_explore_rate': 0.0,
        'actor_agari_guard': False, 'opponent_agari_guard': True,
        'search_enabled': False, 'actor_oracle_guiding': False, 'probe_pts': [2, 1, 0, -3],
        'critic_contract': pre, 'engineering_lr_not_research_recommendation': args.engineering_lr,
        'closed_replay_contract': 'exact frozen registered actor; no PV filename, importance sampling and V-trace disabled',
        'input_semantics': 'FIXED-IMPUTED Oracle, trust_seed=false; not true hidden-wall reconstruction',
        'aux_recipe_source': 'complete actor checkpoint aux and supervised sections; production inactive policy gate',
    }
    games, reuse = load_verified_rollout(source, provenance, load_games=load_games, duplicate_sets=duplicate_sets)
    provenance['rollout_reuse'] = reuse
    protected.update({Path(game['log_path']).resolve(): game['sha256'] for game in games})
    atomic_write_json(root / 'provenance.json', provenance)
    drain = OneShotDrain(source / 'games', games)
    observer = None
    try:
        # Strict preflight uses the actual production model constructors/loaders.
        actor, policy, _, _ = load_policy(files['actor'], torch)
        del actor, policy
        train_online.validate_oracle_critic_init_checkpoint(critic_state, live_config)
        critic, head = train_online.build_online_value_models(live_config, device=torch.device('cpu'))
        critic.load_state_dict(critic_state['oracle_brain'], strict=True)
        head.load_state_dict(critic_state['value_net'], strict=True)
        del critic, head
        paths = sorted(str(Path(game['log_path']).resolve()) for game in games)
        dataset = FileDatasetsIter(version=4, file_list=paths[:1], pts=[2, 1, 0, -3],
                                   oracle=True, player_names=['trainee'],
                                   value_target_mode='all_players', value_reward_source='score_rank')
        trajectory = next(dataset.iter_game_trajectories(paths[:1], oracle_imputation_seed=IMPUTATION_SEED))
        fixed = (torch.from_numpy(trajectory['obs'][:2].copy()).float(),
                 torch.from_numpy(trajectory['masks'][:2].copy()).bool())
        del trajectory, dataset
        observer = TrainingObservation(torch, actor_state, critic_state, fixed, root)
        # No model, loss, optimizer, TestPlayer, GAE, scaler, saver, or clock replacement.
        with patch.object(common, 'drain', drain), patch.object(common, 'submit_param', observer.submit), \
                patch.object(FileDatasetsIter, 'iter_game_trajectories',
                             seeded_trajectories(FileDatasetsIter.iter_game_trajectories)), \
                patch.object(train_online, 'observed_scaler_step', observed_steps(
                    train_online.observed_scaler_step, observer.before_step, observer.after_step)):
            try:
                train_online.train()
            except SystemExit as exc:
                if exc.code != train_online.ONLINE_MAX_STEPS_EXIT_CODE:
                    raise
            else:
                raise RuntimeError('production trainer returned without its saved-budget exit')
        latest_path = Path(live_config['control']['state_file'])
        latest = torch.load(latest_path, map_location='cpu', weights_only=True, mmap=True)
        clock = validate_saved_clock(latest, observer.steps)
        observer.actor_unchanged()
        for key in ('mortal', 'policy_net'):
            assert_state_equal(latest[key], actor_state[key], torch, 'saved ' + key)
        changed = {}
        for key in ('oracle_brain', 'value_net'):
            if any(not bool(torch.isfinite(value).all()) for value in latest[key].values()):
                raise ValueError(f'nonfinite saved critic: {key}')
            changed[key] = sum(not torch.equal(value, critic_state[key][name]) for name, value in latest[key].items())
        if not all(changed.values()):
            raise ValueError('both the Oracle brain and value head must actually change')
        for section in ('aux', 'supervised'):
            if (live_config[section] != expected_recipe[section]
                    or latest['config'][section] != expected_recipe[section]):
                raise ValueError(f'complete inherited recipe changed: {section}')
        reloaded_critic, reloaded_value = train_online.build_online_value_models(
            latest['config'], device=torch.device('cpu'))
        for key, model in (('oracle_brain', reloaded_critic), ('value_net', reloaded_value)):
            model.load_state_dict(latest[key], strict=True)
            assert_state_equal(model.state_dict(), latest[key], torch, 'reloaded ' + key)
        del reloaded_critic, reloaded_value
        reloaded_actor, reloaded_policy, _, _ = load_policy(latest_path, torch)
        device = next(observer.models[0].parameters()).device
        reloaded = (reloaded_actor.to(device), reloaded_policy.to(device))
        for key, model in zip(('mortal', 'policy_net'), reloaded):
            assert_state_equal(model.state_dict(), actor_state[key], torch, 'reloaded ' + key)
        if not torch.equal(observer.initial_logits, observer.logits(reloaded)):
            raise ValueError('strictly reloaded actor logits changed on fixed real observations')
        result = {'status': 'passed', 'engineering_only': True, 'candidate_model': False,
                  'trainer_exit_code': 86, 'drain_calls': drain.calls,
                  'optimizer_update_clock': clock, 'critic_changed_tensors': changed,
                  'actor_state_buffers_logits_equal': True, 'actual_value_losses_finite': True,
                  'observed_updates': observer.steps, 'publications': observer.publications,
                  'latest_sha256': file_sha256(latest_path),
                  'interpretation': 'two-update wiring check only; no critic readiness, model strength, or research LR claim'}
        atomic_write_json(root / 'effective_config_after_train.json', live_config)
    finally:
        verify_hashes(protected)
        atomic_write_json(root / 'input_immutability.json', {'unchanged': True,
                          'sha256': {str(path): digest for path, digest in protected.items()}})
    atomic_write_json(root / 'result.json', result)
    provenance.update(status='complete', finished_unix=time.time(),
                      result_sha256=file_sha256(root / 'result.json'))
    atomic_write_json(root / 'provenance.json', provenance)
    return result


def main(argv=None):
    args = parse_args(argv)
    root = reserve_check_output(args.output_dir, args.reuse_rollout_dir)
    atomic_write_json(root / 'request.json', vars(args))
    try:
        run(args, root)
    except BaseException as exc:
        atomic_write_json(root / 'failure.json', {'status': 'failed', 'type': type(exc).__name__,
                                                'error': str(exc), 'unix': time.time()})
        raise


if __name__ == '__main__':
    main()
