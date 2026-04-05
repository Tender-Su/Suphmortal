import argparse
import json
import os
import subprocess
import sys
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

import run_stage05_ab as s05
import run_stage05_formal as stage05_formal


REPO_ROOT = Path(__file__).resolve().parent.parent
STAGE1_AB_ROOT = REPO_ROOT / 'logs' / 'stage1_ab'
STAGE1_AB_ROOT.mkdir(parents=True, exist_ok=True)

STAGE1_AB_DEFAULTS = {
    'batch_size': 1024,
    'num_workers': 4,
    'file_batch_size': 10,
    'val_file_batch_size': 8,
    'prefetch_factor': 3,
    'val_prefetch_factor': 5,
    'force_safe_training': False,
    'log_every': 1000,
    'save_every': 4000,
    'val_every_steps': 4000,
    'monitor_val_batches': 128,
    'full_val_every_checks': 1,
    'old_regression_every_checks': 1,
    'max_epochs': 99,
    'max_steps': 12000,
    'eval_files': {
        'full_recent': 128,
        'old_regression': 64,
    },
    'seed': 20260319,
}


@dataclass(frozen=True)
class DropoutProfile:
    arm_name: str
    description: str
    schedule: str
    decay_ratio: float
    continue_lr_scale: float = 0.1


PROFILE_ARMS = {
    'linear_075': DropoutProfile(
        arm_name='linear_075',
        description='linear 1->0 over 75% budget, then normal continuation',
        schedule='linear',
        decay_ratio=0.75,
    ),
    'cosine_075': DropoutProfile(
        arm_name='cosine_075',
        description='cosine 1->0 over 75% budget, then normal continuation',
        schedule='cosine',
        decay_ratio=0.75,
    ),
    'linear_050': DropoutProfile(
        arm_name='linear_050',
        description='linear 1->0 over 50% budget, then longer normal continuation',
        schedule='linear',
        decay_ratio=0.50,
    ),
    'cosine_050': DropoutProfile(
        arm_name='cosine_050',
        description='cosine 1->0 over 50% budget, then longer normal continuation',
        schedule='cosine',
        decay_ratio=0.50,
    ),
    'linear_100': DropoutProfile(
        arm_name='linear_100',
        description='linear 1->0 over full budget, no continuation control',
        schedule='linear',
        decay_ratio=1.0,
    ),
    'cosine_100': DropoutProfile(
        arm_name='cosine_100',
        description='cosine 1->0 over full budget, no continuation control',
        schedule='cosine',
        decay_ratio=1.0,
    ),
}

DEFAULT_PROFILE_ARMS = (
    'linear_075',
    'cosine_075',
    'linear_050',
    'cosine_050',
)

CONTROL_PROFILE_ARMS = (
    'linear_100',
    'cosine_100',
)


def ensure_stage1_section(base_cfg: dict[str, Any]) -> dict[str, Any]:
    cfg = deepcopy(base_cfg)
    stage1_cfg = deepcopy(cfg.get('stage1', cfg.get('supervised', {})))
    if not stage1_cfg:
        raise RuntimeError('config.toml is missing both [stage1] and [supervised]')

    supervised_cfg = cfg.get('supervised', {})
    if not isinstance(supervised_cfg, dict):
        supervised_cfg = {}
    oracle_cfg = cfg.get('oracle', {})
    if not isinstance(oracle_cfg, dict):
        oracle_cfg = {}
    optim_scheduler_cfg = cfg.get('optim', {}).get('scheduler', {})
    if not isinstance(optim_scheduler_cfg, dict):
        optim_scheduler_cfg = {}

    fallback_init_state_file = (
        oracle_cfg.get('init_state_file')
        or supervised_cfg.get('best_loss_state_file')
        or supervised_cfg.get('best_state_file')
        or supervised_cfg.get('init_state_file', '')
    )
    if fallback_init_state_file and not stage1_cfg.get('init_state_file'):
        stage1_cfg['init_state_file'] = fallback_init_state_file

    stage1_cfg.setdefault('publish_stage2_handoff', False)
    stage1_cfg.setdefault('enable_oracle', True)
    stage1_cfg.setdefault('validation_use_oracle', False)
    stage1_cfg.setdefault('save_every', STAGE1_AB_DEFAULTS['save_every'])
    stage1_cfg.setdefault('val_every_steps', STAGE1_AB_DEFAULTS['val_every_steps'])
    stage1_cfg.setdefault('monitor_val_batches', STAGE1_AB_DEFAULTS['monitor_val_batches'])
    stage1_cfg.setdefault('full_val_every_checks', STAGE1_AB_DEFAULTS['full_val_every_checks'])
    stage1_cfg.setdefault('old_regression_every_checks', STAGE1_AB_DEFAULTS['old_regression_every_checks'])
    stage1_cfg.setdefault('max_steps', STAGE1_AB_DEFAULTS['max_steps'])
    stage1_cfg.setdefault('max_epochs', STAGE1_AB_DEFAULTS['max_epochs'])
    stage1_cfg.setdefault(
        'lr',
        supervised_cfg.get('lr', optim_scheduler_cfg.get('peak', 3e-4)),
    )

    rank_aux_cfg = deepcopy(stage1_cfg.get('rank_aux', supervised_cfg.get('rank_aux', {})))
    stage1_cfg['rank_aux'] = rank_aux_cfg

    aux_cfg = deepcopy(cfg.get('aux', {}))
    aux_cfg.update(deepcopy(stage1_cfg.get('aux', {})))
    if aux_cfg.get('danger_weight', 0.0):
        aux_cfg.setdefault('danger_enabled', True)
    stage1_cfg['aux'] = aux_cfg

    scheduler_cfg = deepcopy(stage1_cfg.get('scheduler', supervised_cfg.get('scheduler', {})))
    if not scheduler_cfg:
        scheduler_cfg = deepcopy(optim_scheduler_cfg)
    scheduler_cfg.setdefault('type', 'cosine')
    scheduler_cfg.setdefault('warm_up_steps', 0)
    scheduler_cfg.setdefault('init', 1e-8)
    scheduler_cfg.setdefault('max_steps', int(stage1_cfg['max_steps']))
    if scheduler_cfg.get('type') == 'cosine':
        scheduler_cfg.setdefault('final', optim_scheduler_cfg.get('final', 1e-5))
    else:
        scheduler_cfg.setdefault('min_lr', 1e-6)
    stage1_cfg['scheduler'] = scheduler_cfg

    oracle_dropout_cfg = deepcopy(stage1_cfg.get('oracle_dropout', {}))
    default_max_steps = int(stage1_cfg.get('max_steps', STAGE1_AB_DEFAULTS['max_steps']) or STAGE1_AB_DEFAULTS['max_steps'])
    oracle_dropout_cfg.setdefault('enabled', True)
    oracle_dropout_cfg.setdefault('schedule', 'linear')
    oracle_dropout_cfg.setdefault('gamma_start', 1.0)
    oracle_dropout_cfg.setdefault('gamma_end', 0.0)
    oracle_dropout_cfg.setdefault('hold_steps', 0)
    oracle_dropout_cfg.setdefault('decay_steps', max(1, int(round(default_max_steps * 0.75))))
    stage1_cfg['oracle_dropout'] = oracle_dropout_cfg

    cfg['stage1'] = stage1_cfg
    return cfg


def stage1_recipe_snapshot(base_cfg: dict[str, Any]) -> dict[str, Any]:
    stage1_cfg = base_cfg.get('stage1', {})
    return {
        'rank_aux': deepcopy(stage1_cfg.get('rank_aux', {})),
        'aux': deepcopy(stage1_cfg.get('aux', {})),
    }


def stage1_recipe_problems(base_cfg: dict[str, Any]) -> list[str]:
    stage1_cfg = base_cfg.get('stage1', {})
    aux_cfg = stage1_cfg.get('aux', {})
    rank_aux_cfg = stage1_cfg.get('rank_aux', {})
    problems = []

    rank_base = float(rank_aux_cfg.get('base_weight', aux_cfg.get('next_rank_weight', 0.0)) or 0.0)
    if rank_base <= 0:
        problems.append('rank aux base weight <= 0')

    opp_weight = float(aux_cfg.get('opponent_state_weight', 0.0) or 0.0)
    if opp_weight <= 0:
        problems.append('opponent_state_weight <= 0')

    danger_weight = float(aux_cfg.get('danger_weight', 0.0) or 0.0)
    danger_enabled = bool(aux_cfg.get('danger_enabled', False) or danger_weight > 0)
    if not danger_enabled or danger_weight <= 0:
        problems.append('danger aux disabled or danger_weight <= 0')

    return problems


def ensure_stage1_recipe_fixed(base_cfg: dict[str, Any]) -> None:
    problems = stage1_recipe_problems(base_cfg)
    if problems:
        raise RuntimeError(
            'Stage 1 fixed full-aux recipe is invalid: ' + '; '.join(problems)
        )


def resolve_stage1_peak_lr(base_cfg: dict[str, Any]) -> float:
    stage1_cfg = base_cfg.get('stage1', {})
    optim_scheduler_cfg = base_cfg.get('optim', {}).get('scheduler', {})
    return float(stage1_cfg.get('lr', optim_scheduler_cfg.get('peak', 3e-4)))


def build_scaled_scheduler(
    base_cfg: dict[str, Any],
    *,
    max_steps: int,
    lr_scale: float,
    warm_up_steps_override: int | None = None,
) -> dict[str, Any]:
    scheduler_cfg = deepcopy(base_cfg.get('stage1', {}).get('scheduler', {}))
    optim_scheduler_cfg = base_cfg.get('optim', {}).get('scheduler', {})
    if not scheduler_cfg:
        scheduler_cfg = deepcopy(optim_scheduler_cfg)
    scheduler_cfg.setdefault('type', 'cosine')
    scheduler_cfg.setdefault('init', 1e-8)
    scheduler_cfg.setdefault('warm_up_steps', 0)
    scheduler_cfg['max_steps'] = int(max_steps)
    warm_up_steps = int(
        warm_up_steps_override
        if warm_up_steps_override is not None
        else scheduler_cfg.get('warm_up_steps', 0) or 0
    )
    scheduler_cfg['warm_up_steps'] = min(max(warm_up_steps, 0), int(max_steps))
    scheduler_cfg['init'] = float(scheduler_cfg.get('init', 1e-8)) * lr_scale
    if scheduler_cfg.get('type') == 'cosine':
        scheduler_cfg['final'] = float(
            scheduler_cfg.get('final', optim_scheduler_cfg.get('final', 1e-5))
        ) * lr_scale
    else:
        scheduler_cfg['min_lr'] = float(scheduler_cfg.get('min_lr', 1e-6)) * lr_scale
    return scheduler_cfg


def stage1_checkpoint_paths(exp_dir: Path) -> dict[str, Path]:
    ckpt_dir = exp_dir / 'checkpoints'
    tb_dir = exp_dir / 'tb'
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tb_dir.mkdir(parents=True, exist_ok=True)
    return {
        'state_file': ckpt_dir / 'stage1_latest_oracle.pth',
        'best_state_file': ckpt_dir / 'stage1_best_loss_oracle.pth',
        'best_loss_state_file': ckpt_dir / 'stage1_best_loss_oracle.pth',
        'best_acc_state_file': ckpt_dir / 'stage1_best_acc_oracle.pth',
        'best_rank_state_file': ckpt_dir / 'stage1_best_rank_oracle.pth',
        'best_loss_normal_state_file': ckpt_dir / 'stage1_best_loss_normal.pth',
        'best_acc_normal_state_file': ckpt_dir / 'stage1_best_acc_normal.pth',
        'best_rank_normal_state_file': ckpt_dir / 'stage1_best_rank_normal.pth',
        'tensorboard_dir': tb_dir,
        'file_index': exp_dir / 'file_index.pth',
    }


def stage1_arm_exp_dir(
    *,
    ab_name: str,
    profile_arm_name: str | None = None,
    phase_name: str | None = None,
    recipe_arm_name: str | None = None,
    gamma_arm: str | None = None,
    isolate_gamma_artifacts: bool = False,
) -> Path:
    if profile_arm_name is not None:
        exp_dir = STAGE1_AB_ROOT / ab_name / profile_arm_name
        if phase_name:
            exp_dir = exp_dir / phase_name
        return exp_dir
    if recipe_arm_name is not None:
        exp_dir = STAGE1_AB_ROOT / ab_name / recipe_arm_name
        if isolate_gamma_artifacts:
            exp_dir = exp_dir / str(gamma_arm)
        return exp_dir
    raise TypeError('missing profile_arm_name')


def run_stage1_training(cfg_path: Path, log_path: Path) -> None:
    env = os.environ.copy()
    env['MORTAL_CFG'] = str(cfg_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    attempt = 0
    while True:
        attempt += 1
        attempt_log_start = log_path.stat().st_size if log_path.exists() else 0
        mode = 'a' if log_path.exists() else 'w'
        with log_path.open(mode, encoding='utf-8', newline='\n') as f:
            f.write(f'\n=== train_stage1_refine attempt {attempt} @ {time.strftime("%Y-%m-%d %H:%M:%S")} ===\n')
            f.flush()
            proc = subprocess.run(
                [sys.executable, 'train_stage1_refine.py'],
                cwd=s05.MORTAL_DIR,
                env=env,
                stdout=f,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if proc.returncode == 0:
            return
        marker = s05.transient_training_failure_marker(log_path, start_offset=attempt_log_start)
        if marker is None:
            raise RuntimeError(f'train_stage1_refine.py failed, see {log_path}')
        sleep_seconds = min(30, 5 + attempt)
        with log_path.open('a', encoding='utf-8', newline='\n') as f:
            f.write(
                f'=== transient training failure detected: {marker}; '
                f'retrying with same config after {sleep_seconds}s ===\n'
            )
        time.sleep(sleep_seconds)


def exclude_stage1_eval_files(train_files: list[str], eval_splits: dict[str, list[str]]) -> list[str]:
    excluded_files = {
        file_path
        for split_files in eval_splits.values()
        for file_path in split_files
    }
    return [file_path for file_path in train_files if file_path not in excluded_files]


def resolve_stage1_splits(base_cfg: dict[str, Any], seed: int) -> tuple[list[str], dict[str, list[str]], dict[str, list[str]]]:
    grouped = s05.group_files_by_month(s05.load_all_files())
    eval_splits = s05.build_eval_splits(grouped, seed, STAGE1_AB_DEFAULTS['eval_files'])

    if s05.BASE_INDEX_PATH.exists():
        base_index = torch.load(s05.BASE_INDEX_PATH, weights_only=True)
        train_files = list(base_index.get('train_files', []))
    else:
        train_files = []

    if not train_files:
        train_files = s05.select_range(grouped, *s05.WINDOWS['broad_all'])

    train_files = exclude_stage1_eval_files(train_files, eval_splits)
    if not train_files:
        raise RuntimeError('Stage 1 train split is empty after excluding evaluation files')

    max_train_files = int(base_cfg.get('stage1', {}).get('max_train_files', 0) or 0)
    if max_train_files > 0:
        train_files = s05.sample_files(train_files, max_train_files, seed + 41)

    return train_files, eval_splits, grouped


def disabled_oracle_dropout() -> dict[str, Any]:
    return {
        'enabled': False,
        'schedule': 'none',
        'gamma_start': 1.0,
        'gamma_end': 1.0,
        'hold_steps': 0,
        'decay_steps': 0,
    }


def split_profile_steps(total_steps: int, profile: DropoutProfile) -> tuple[int, int]:
    decay_steps = min(total_steps, max(1, int(round(total_steps * profile.decay_ratio))))
    continuation_steps = max(total_steps - decay_steps, 0)
    return decay_steps, continuation_steps


def build_phase_overrides(
    base_cfg: dict[str, Any],
    *,
    ckpts: dict[str, Path],
    seed: int,
    max_steps: int,
    init_state_file: str,
    enable_oracle: bool,
    oracle_dropout: dict[str, Any],
    lr_scale: float,
    warm_up_steps_override: int | None = None,
) -> dict[str, Any]:
    stage1_cfg = base_cfg.get('stage1', {})
    stage1_overrides: dict[str, Any] = {
        'state_file': str(ckpts['state_file']),
        'best_state_file': str(ckpts['best_state_file']),
        'best_loss_state_file': str(ckpts['best_loss_state_file']),
        'best_acc_state_file': str(ckpts['best_acc_state_file']),
        'best_rank_state_file': str(ckpts['best_rank_state_file']),
        'best_loss_normal_state_file': str(ckpts['best_loss_normal_state_file']),
        'best_acc_normal_state_file': str(ckpts['best_acc_normal_state_file']),
        'best_rank_normal_state_file': str(ckpts['best_rank_normal_state_file']),
        'publish_stage2_handoff': False,
        'tensorboard_dir': str(ckpts['tensorboard_dir']),
        'file_index': str(ckpts['file_index']),
        'init_state_file': init_state_file,
        'seed': seed,
        'batch_size': STAGE1_AB_DEFAULTS['batch_size'],
        'num_workers': STAGE1_AB_DEFAULTS['num_workers'],
        'file_batch_size': STAGE1_AB_DEFAULTS['file_batch_size'],
        'val_file_batch_size': STAGE1_AB_DEFAULTS['val_file_batch_size'],
        'prefetch_factor': STAGE1_AB_DEFAULTS['prefetch_factor'],
        'val_prefetch_factor': STAGE1_AB_DEFAULTS['val_prefetch_factor'],
        'force_safe_training': STAGE1_AB_DEFAULTS['force_safe_training'],
        'log_every': STAGE1_AB_DEFAULTS['log_every'],
        'save_every': STAGE1_AB_DEFAULTS['save_every'],
        'val_every_steps': STAGE1_AB_DEFAULTS['val_every_steps'],
        'monitor_val_batches': STAGE1_AB_DEFAULTS['monitor_val_batches'],
        'full_val_every_checks': STAGE1_AB_DEFAULTS['full_val_every_checks'],
        'old_regression_every_checks': STAGE1_AB_DEFAULTS['old_regression_every_checks'],
        'max_epochs': int(stage1_cfg.get('max_epochs', STAGE1_AB_DEFAULTS['max_epochs']) or STAGE1_AB_DEFAULTS['max_epochs']),
        'max_steps': int(max_steps),
        'enable_oracle': enable_oracle,
        'validation_use_oracle': False,
        'lr': resolve_stage1_peak_lr(base_cfg) * lr_scale,
        'scheduler': build_scaled_scheduler(
            base_cfg,
            max_steps=max_steps,
            lr_scale=lr_scale,
            warm_up_steps_override=warm_up_steps_override,
        ),
        'oracle_dropout': deepcopy(oracle_dropout),
        'aux': deepcopy(stage1_cfg.get('aux', {})),
        'rank_aux': deepcopy(stage1_cfg.get('rank_aux', {})),
    }
    return {'stage1': stage1_overrides}


def load_phase_summaries(ckpts: dict[str, Path]) -> dict[str, Any]:
    return {
        'best_loss': s05.load_state_summary(ckpts['best_loss_state_file']),
        'best_acc': s05.load_state_summary(ckpts['best_acc_state_file']),
        'best_rank': s05.load_state_summary(ckpts['best_rank_state_file']),
        'latest': s05.load_state_summary(ckpts['state_file']),
    }


def run_training_phase(
    base_cfg: dict[str, Any],
    *,
    ab_name: str,
    profile: DropoutProfile,
    phase_name: str,
    phase_description: str,
    phase_dir: Path,
    train_files: list[str],
    eval_splits: dict[str, list[str]],
    cfg_overrides: dict[str, Any],
) -> dict[str, Any]:
    ckpts = stage1_checkpoint_paths(phase_dir)
    s05.write_index(
        ckpts['file_index'],
        train_files=train_files,
        monitor_recent_files=eval_splits['monitor_recent_files'],
        full_recent_files=eval_splits['full_recent_files'],
        old_regression_files=eval_splits['old_regression_files'],
        meta={
            'ab_name': ab_name,
            'arm_name': profile.arm_name,
            'description': profile.description,
            'phase_name': phase_name,
            'phase_description': phase_description,
            'train_files': len(train_files),
        },
    )

    cfg = s05.merge_dict(base_cfg, cfg_overrides)
    cfg_path = phase_dir / 'config.toml'
    log_path = phase_dir / 'train.log'
    s05.write_toml(cfg_path, cfg)
    run_stage1_training(cfg_path, log_path)

    return {
        'phase_name': phase_name,
        'description': phase_description,
        'final': load_phase_summaries(ckpts),
        'paths': {
            name: str(path) for name, path in ckpts.items()
        } | {
            'config_path': str(cfg_path),
            'log_path': str(log_path),
        },
    }


def run_profile_arm(
    base_cfg: dict[str, Any],
    *,
    ab_name: str,
    profile: DropoutProfile,
    train_files: list[str],
    eval_splits: dict[str, list[str]],
    init_state_file: str,
    seed: int,
    step_scale: float,
) -> dict[str, Any]:
    base_max_steps = int(base_cfg['stage1'].get('max_steps', STAGE1_AB_DEFAULTS['max_steps']) or STAGE1_AB_DEFAULTS['max_steps'])
    total_steps = max(1, int(round(base_max_steps * step_scale)))
    decay_steps, continuation_steps = split_profile_steps(total_steps, profile)

    transition_dir = stage1_arm_exp_dir(
        ab_name=ab_name,
        profile_arm_name=profile.arm_name,
        phase_name='oracle_transition',
    )
    transition_ckpts = stage1_checkpoint_paths(transition_dir)
    transition_result = run_training_phase(
        base_cfg,
        ab_name=ab_name,
        profile=profile,
        phase_name='oracle_transition',
        phase_description=f'{profile.schedule} oracle-dropout transition',
        phase_dir=transition_dir,
        train_files=train_files,
        eval_splits=eval_splits,
        cfg_overrides=build_phase_overrides(
            base_cfg,
            ckpts=transition_ckpts,
            seed=seed,
            max_steps=decay_steps,
            init_state_file=init_state_file,
            enable_oracle=True,
            oracle_dropout={
                'enabled': True,
                'schedule': profile.schedule,
                'gamma_start': 1.0,
                'gamma_end': 0.0,
                'hold_steps': 0,
                'decay_steps': decay_steps,
            },
            lr_scale=1.0,
        ),
    )

    phases = {
        'oracle_transition': transition_result,
    }
    final_result = transition_result

    if continuation_steps > 0:
        continuation_init_state = transition_result['paths']['state_file']
        if not Path(continuation_init_state).exists():
            raise RuntimeError(
                'missing Stage 1 normal continuation init checkpoint: '
                f'{continuation_init_state}'
            )
        continuation_dir = stage1_arm_exp_dir(
            ab_name=ab_name,
            profile_arm_name=profile.arm_name,
            phase_name='normal_continue',
        )
        continuation_ckpts = stage1_checkpoint_paths(continuation_dir)
        continuation_result = run_training_phase(
            base_cfg,
            ab_name=ab_name,
            profile=profile,
            phase_name='normal_continue',
            phase_description='normal continuation after gamma=0',
            phase_dir=continuation_dir,
            train_files=train_files,
            eval_splits=eval_splits,
            cfg_overrides=build_phase_overrides(
                base_cfg,
                ckpts=continuation_ckpts,
                seed=seed + 1,
                max_steps=continuation_steps,
                init_state_file=continuation_init_state,
                enable_oracle=False,
                oracle_dropout=disabled_oracle_dropout(),
                lr_scale=profile.continue_lr_scale,
                warm_up_steps_override=0,
            ),
        )
        phases['normal_continue'] = continuation_result
        final_result = continuation_result

    return {
        'arm_name': profile.arm_name,
        'description': profile.description,
        'schedule': profile.schedule,
        'decay_ratio': profile.decay_ratio,
        'continuation_ratio': continuation_steps / total_steps,
        'continue_lr_scale': profile.continue_lr_scale,
        'phase_steps': {
            'total_steps': total_steps,
            'oracle_transition_steps': decay_steps,
            'normal_continue_steps': continuation_steps,
        },
        'final': final_result['final'],
        'phases': phases,
        'paths': final_result['paths'],
        'score': s05.score_summary(final_result['final']['best_loss']),
    }


def rank_results(results: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    winner_name, _ = s05.select_winner_by_policy(results)
    best_loss = min(s05.full_recent_loss(result['final']['best_loss']) for result in results.values())
    ranking = []
    for name, result in results.items():
        summary = result['final']['best_loss']
        full_metrics = summary.get('last_full_recent_metrics') or {}
        old_metrics = summary.get('last_old_regression_metrics') or {}
        recent_loss = s05.full_recent_loss(summary)
        ranking.append(
            {
                'arm_name': name,
                'description': result.get('description'),
                'schedule': result.get('schedule'),
                'decay_ratio': result.get('decay_ratio'),
                'continuation_ratio': result.get('continuation_ratio'),
                'eligible': recent_loss <= best_loss + s05.LOSS_EPSILON,
                'winner': name == winner_name,
                'full_recent_loss': recent_loss,
                'action_priority': list(s05.action_priority(summary)),
                'action_quality_score': s05.action_quality_score(full_metrics or summary),
                'scenario_quality_score': s05.scenario_quality_score(full_metrics or summary),
                'rank_acc': float(full_metrics.get('rank_acc', summary.get('best_full_recent_rank_acc', 0.0))),
                'old_regression_loss': float(old_metrics.get('loss', float('inf'))),
                'paths': result.get('paths', {}),
            }
        )
    ranking.sort(
        key=lambda item: (
            not item['eligible'],
            item['full_recent_loss'],
            tuple(-float(v) for v in item['action_priority']),
        )
    )
    return ranking


def save_results(ab_name: str, payload: dict[str, Any]) -> Path:
    out_path = STAGE1_AB_ROOT / ab_name / 'summary.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', encoding='utf-8', newline='\n') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return out_path


def normalize_profile_arm_names(profile_arm_names: list[str] | None) -> list[str]:
    if not profile_arm_names:
        return list(DEFAULT_PROFILE_ARMS)
    normalized = []
    for arm_name in profile_arm_names:
        if arm_name not in PROFILE_ARMS:
            raise ValueError(f'unknown Stage 1 profile arm: {arm_name}')
        if arm_name not in normalized:
            normalized.append(arm_name)
    return normalized


def run_profile_ab(
    base_cfg: dict[str, Any],
    *,
    ab_name: str,
    seed: int,
    step_scale: float,
    profile_arm_names: list[str],
    init_state_file: str,
) -> dict[str, Any]:
    selected_profiles = normalize_profile_arm_names(profile_arm_names)
    train_files, eval_splits, _ = resolve_stage1_splits(base_cfg, seed)
    results = {}
    for arm_name in selected_profiles:
        profile = PROFILE_ARMS[arm_name]
        results[arm_name] = run_profile_arm(
            base_cfg,
            ab_name=ab_name,
            profile=profile,
            train_files=train_files,
            eval_splits=eval_splits,
            init_state_file=init_state_file,
            seed=seed,
            step_scale=step_scale,
        )
    winner_name, selection = s05.select_winner_by_policy(results)
    payload = {
        'mode': 'profile',
        'seed': seed,
        'step_scale': step_scale,
        'profile_arms': selected_profiles,
        'init_state_file': init_state_file,
        'train_files': len(train_files),
        'eval_split_counts': {
            key: len(value) for key, value in eval_splits.items()
        },
        'recipe': stage1_recipe_snapshot(base_cfg),
        'winner': winner_name,
        'winner_metrics': results[winner_name]['final']['best_loss'],
        'selection': selection,
        'ranking': rank_results(results),
        'results': results,
    }
    payload['summary_path'] = str(save_results(ab_name, payload))
    return payload


def list_arms_payload(base_cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = {
        'profile_arms': {
            name: {
                'description': arm.description,
                'schedule': arm.schedule,
                'decay_ratio': arm.decay_ratio,
                'continue_lr_scale': arm.continue_lr_scale,
                'continuation_ratio': 1.0 - arm.decay_ratio,
            }
            for name, arm in PROFILE_ARMS.items()
        },
        'default_shortlist': list(DEFAULT_PROFILE_ARMS),
        'control_profiles': list(CONTROL_PROFILE_ARMS),
    }
    if base_cfg is not None:
        payload['recipe'] = stage1_recipe_snapshot(base_cfg)
        payload['recipe_problems'] = stage1_recipe_problems(base_cfg)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--ab-name')
    parser.add_argument('--seed', type=int, default=STAGE1_AB_DEFAULTS['seed'])
    parser.add_argument('--step-scale', type=float, default=1.0)
    parser.add_argument('--profile-arm', action='append', dest='profile_arms')
    parser.add_argument('--include-controls', action='store_true')
    parser.add_argument('--init-state')
    parser.add_argument('--list-arms', action='store_true')
    args = parser.parse_args()

    base_cfg = ensure_stage1_section(s05.build_base_config())

    if args.list_arms:
        print(json.dumps(list_arms_payload(base_cfg), ensure_ascii=False, indent=2))
        return

    ensure_stage1_recipe_fixed(base_cfg)

    init_state_file = args.init_state or base_cfg['stage1'].get('init_state_file', '')
    if not init_state_file:
        raise RuntimeError('missing Stage 1 init checkpoint; pass --init-state or set stage1.init_state_file')
    stage05_formal.ensure_stage1_canonical_handoff_ready(init_state_file)
    if not Path(init_state_file).exists():
        raise RuntimeError(f'Stage 1 init checkpoint does not exist: {init_state_file}')

    profile_arm_names = normalize_profile_arm_names(args.profile_arms)
    if args.include_controls:
        profile_arm_names = normalize_profile_arm_names(
            profile_arm_names + list(CONTROL_PROFILE_ARMS)
        )

    ab_name = args.ab_name or 'stage1_profile_screen'
    payload = run_profile_ab(
        base_cfg,
        ab_name=ab_name,
        seed=args.seed,
        step_scale=args.step_scale,
        profile_arm_names=profile_arm_names,
        init_state_file=init_state_file,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
