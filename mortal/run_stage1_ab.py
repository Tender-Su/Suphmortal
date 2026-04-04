from __future__ import annotations

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

STAGE1_RECIPE_DEFAULT_OPP_WEIGHT = 0.064
STAGE1_RECIPE_DEFAULT_DANGER_WEIGHT = 0.144


@dataclass(frozen=True)
class RecipeArm:
    arm_name: str
    description: str
    use_oracle: bool
    enable_rank: bool
    enable_opp: bool
    enable_danger: bool


RECIPE_ARMS = {
    'S1-A': RecipeArm(
        arm_name='S1-A',
        description='visible-only CE',
        use_oracle=False,
        enable_rank=False,
        enable_opp=False,
        enable_danger=False,
    ),
    'S1-B': RecipeArm(
        arm_name='S1-B',
        description='oracle-dropout CE + rank aux',
        use_oracle=True,
        enable_rank=True,
        enable_opp=False,
        enable_danger=False,
    ),
    'S1-C': RecipeArm(
        arm_name='S1-C',
        description='oracle-dropout CE + rank aux + opponent_state aux',
        use_oracle=True,
        enable_rank=True,
        enable_opp=True,
        enable_danger=False,
    ),
    'S1-D': RecipeArm(
        arm_name='S1-D',
        description='oracle-dropout CE + rank aux + opponent_state aux + danger aux',
        use_oracle=True,
        enable_rank=True,
        enable_opp=True,
        enable_danger=True,
    ),
}


GAMMA_PROFILES = {
    'G0': {
        'label': 'no dropout',
        'enabled': False,
        'schedule': 'none',
    },
    'G1': {
        'label': 'linear 1->0',
        'enabled': True,
        'schedule': 'linear',
    },
    'G2': {
        'label': 'cosine-like 1->0',
        'enabled': True,
        'schedule': 'cosine',
    },
}


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
    fallback_init_state_file = (
        oracle_cfg.get('init_state_file')
        or supervised_cfg.get('best_loss_state_file')
        or supervised_cfg.get('best_state_file')
        or supervised_cfg.get('init_state_file', '')
    )
    if fallback_init_state_file and not stage1_cfg.get('init_state_file'):
        stage1_cfg['init_state_file'] = fallback_init_state_file

    stage1_cfg.setdefault('enable_oracle', True)
    stage1_cfg.setdefault('validation_use_oracle', False)
    stage1_cfg.setdefault('save_every', STAGE1_AB_DEFAULTS['save_every'])
    stage1_cfg.setdefault('val_every_steps', STAGE1_AB_DEFAULTS['val_every_steps'])
    stage1_cfg.setdefault('monitor_val_batches', STAGE1_AB_DEFAULTS['monitor_val_batches'])
    stage1_cfg.setdefault('full_val_every_checks', STAGE1_AB_DEFAULTS['full_val_every_checks'])
    stage1_cfg.setdefault('old_regression_every_checks', STAGE1_AB_DEFAULTS['old_regression_every_checks'])
    stage1_cfg.setdefault('max_steps', STAGE1_AB_DEFAULTS['max_steps'])
    stage1_cfg.setdefault('max_epochs', STAGE1_AB_DEFAULTS['max_epochs'])

    rank_aux_cfg = deepcopy(stage1_cfg.get('rank_aux', supervised_cfg.get('rank_aux', {})))
    stage1_cfg['rank_aux'] = rank_aux_cfg

    aux_cfg = deepcopy(cfg.get('aux', {}))
    aux_cfg.update(deepcopy(stage1_cfg.get('aux', {})))
    stage1_cfg['aux'] = aux_cfg

    oracle_dropout_cfg = deepcopy(stage1_cfg.get('oracle_dropout', {}))
    oracle_dropout_cfg.setdefault('enabled', True)
    oracle_dropout_cfg.setdefault('schedule', 'linear')
    oracle_dropout_cfg.setdefault('gamma_start', 1.0)
    oracle_dropout_cfg.setdefault('gamma_end', 0.0)
    oracle_dropout_cfg.setdefault('hold_steps', 0)
    stage1_cfg['oracle_dropout'] = oracle_dropout_cfg

    cfg['stage1'] = stage1_cfg
    return cfg


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
    recipe_arm_name: str,
    gamma_arm: str,
    isolate_gamma_artifacts: bool,
) -> Path:
    exp_dir = STAGE1_AB_ROOT / ab_name / recipe_arm_name
    if isolate_gamma_artifacts:
        exp_dir = exp_dir / gamma_arm
    return exp_dir


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


def recipe_aux_weight(base_cfg: dict[str, Any], key: str, *, fallback: float) -> float:
    stage1_aux = base_cfg.get('stage1', {}).get('aux', {})
    weight = float(stage1_aux.get(key, 0.0) or 0.0)
    if weight > 0:
        return weight
    return fallback


def apply_recipe_feature_overrides(
    base_cfg: dict[str, Any],
    recipe_arm: RecipeArm,
    *,
    aux_overrides: dict[str, Any],
    rank_aux_overrides: dict[str, Any],
) -> None:
    if not recipe_arm.enable_rank:
        rank_aux_overrides['base_weight'] = 0.0
        aux_overrides['next_rank_weight'] = 0.0

    if recipe_arm.enable_opp:
        aux_overrides['opponent_state_weight'] = recipe_aux_weight(
            base_cfg,
            'opponent_state_weight',
            fallback=STAGE1_RECIPE_DEFAULT_OPP_WEIGHT,
        )
    else:
        aux_overrides['opponent_state_weight'] = 0.0

    if recipe_arm.enable_danger:
        aux_overrides['danger_enabled'] = True
        aux_overrides['danger_weight'] = recipe_aux_weight(
            base_cfg,
            'danger_weight',
            fallback=STAGE1_RECIPE_DEFAULT_DANGER_WEIGHT,
        )
    else:
        aux_overrides['danger_enabled'] = False
        aux_overrides['danger_weight'] = 0.0


def effective_recipe_sections(base_cfg: dict[str, Any], recipe_arm: RecipeArm) -> tuple[dict[str, Any], dict[str, Any]]:
    stage1_cfg = base_cfg.get('stage1', {})
    aux_cfg = deepcopy(stage1_cfg.get('aux', {}))
    rank_aux_cfg = deepcopy(stage1_cfg.get('rank_aux', {}))
    apply_recipe_feature_overrides(
        base_cfg,
        recipe_arm,
        aux_overrides=aux_cfg,
        rank_aux_overrides=rank_aux_cfg,
    )
    return aux_cfg, rank_aux_cfg


def gamma_overrides(gamma_arm: str, max_steps: int) -> dict[str, Any]:
    profile = GAMMA_PROFILES[gamma_arm]
    if not profile['enabled']:
        return {
            'enabled': False,
            'schedule': 'none',
            'gamma_start': 1.0,
            'gamma_end': 1.0,
            'hold_steps': 0,
            'decay_steps': 0,
        }
    return {
        'enabled': True,
        'schedule': profile['schedule'],
        'gamma_start': 1.0,
        'gamma_end': 0.0,
        'hold_steps': 0,
        'decay_steps': max_steps,
    }


def build_recipe_overrides(
    base_cfg: dict[str, Any],
    recipe_arm: RecipeArm,
    *,
    gamma_arm: str,
    max_steps: int,
    init_state_file: str,
    ckpts: dict[str, Path],
    seed: int,
) -> dict[str, Any]:
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
        'max_epochs': STAGE1_AB_DEFAULTS['max_epochs'],
        'max_steps': max_steps,
        'enable_oracle': recipe_arm.use_oracle,
        'validation_use_oracle': False,
        'oracle_dropout': gamma_overrides(gamma_arm, max_steps),
        'aux': {},
        'rank_aux': {},
    }

    aux_overrides = stage1_overrides['aux']
    rank_aux_overrides = stage1_overrides['rank_aux']

    apply_recipe_feature_overrides(
        base_cfg,
        recipe_arm,
        aux_overrides=aux_overrides,
        rank_aux_overrides=rank_aux_overrides,
    )

    if not recipe_arm.use_oracle:
        stage1_overrides['validation_use_oracle'] = False
        stage1_overrides['oracle_dropout'] = gamma_overrides('G0', max_steps)

    return {'stage1': stage1_overrides}


def ineffective_reason(base_cfg: dict[str, Any], recipe_arm: RecipeArm) -> str | None:
    stage1_aux, stage1_rank = effective_recipe_sections(base_cfg, recipe_arm)

    if recipe_arm.enable_rank:
        rank_base = float(stage1_rank.get('base_weight', stage1_aux.get('next_rank_weight', 0.0)) or 0.0)
        if rank_base <= 0:
            return 'rank aux base weight <= 0'
    if recipe_arm.enable_opp:
        opp_weight = float(stage1_aux.get('opponent_state_weight', 0.0) or 0.0)
        if opp_weight <= 0:
            return 'opponent_state_weight <= 0'
    if recipe_arm.enable_danger:
        danger_weight = float(stage1_aux.get('danger_weight', 0.0) or 0.0)
        danger_enabled = bool(stage1_aux.get('danger_enabled', False) or danger_weight > 0)
        if not danger_enabled or danger_weight <= 0:
            return 'danger aux disabled or danger_weight <= 0'
    return None


def run_recipe_arm(
    base_cfg: dict[str, Any],
    *,
    ab_name: str,
    recipe_arm: RecipeArm,
    gamma_arm: str,
    isolate_gamma_artifacts: bool = False,
    train_files: list[str],
    eval_splits: dict[str, list[str]],
    init_state_file: str,
    seed: int,
    step_scale: float,
) -> dict[str, Any]:
    exp_dir = stage1_arm_exp_dir(
        ab_name=ab_name,
        recipe_arm_name=recipe_arm.arm_name,
        gamma_arm=gamma_arm,
        isolate_gamma_artifacts=isolate_gamma_artifacts,
    )
    ckpts = stage1_checkpoint_paths(exp_dir)
    base_max_steps = int(base_cfg['stage1'].get('max_steps', STAGE1_AB_DEFAULTS['max_steps']) or STAGE1_AB_DEFAULTS['max_steps'])
    max_steps = max(1, int(round(base_max_steps * step_scale)))

    s05.write_index(
        ckpts['file_index'],
        train_files=train_files,
        monitor_recent_files=eval_splits['monitor_recent_files'],
        full_recent_files=eval_splits['full_recent_files'],
        old_regression_files=eval_splits['old_regression_files'],
        meta={
            'ab_name': ab_name,
            'arm_name': recipe_arm.arm_name,
            'description': recipe_arm.description,
            'gamma_arm': gamma_arm,
            'train_files': len(train_files),
        },
    )

    cfg = s05.merge_dict(
        base_cfg,
        build_recipe_overrides(
            base_cfg,
            recipe_arm,
            gamma_arm=gamma_arm,
            max_steps=max_steps,
            init_state_file=init_state_file,
            ckpts=ckpts,
            seed=seed,
        ),
    )
    cfg_path = exp_dir / 'config.toml'
    log_path = exp_dir / 'train.log'
    s05.write_toml(cfg_path, cfg)
    run_stage1_training(cfg_path, log_path)

    final_best_loss = s05.load_state_summary(ckpts['best_loss_state_file'])
    final_best_acc = s05.load_state_summary(ckpts['best_acc_state_file'])
    final_best_rank = s05.load_state_summary(ckpts['best_rank_state_file'])
    latest = s05.load_state_summary(ckpts['state_file'])

    return {
        'arm_name': recipe_arm.arm_name,
        'description': recipe_arm.description,
        'gamma_arm': gamma_arm,
        'ineffective_reason': ineffective_reason(base_cfg, recipe_arm),
        'use_oracle': recipe_arm.use_oracle,
        'enable_rank': recipe_arm.enable_rank,
        'enable_opp': recipe_arm.enable_opp,
        'enable_danger': recipe_arm.enable_danger,
        'final': {
            'best_loss': final_best_loss,
            'best_acc': final_best_acc,
            'best_rank': final_best_rank,
            'latest': latest,
        },
        'paths': {
            name: str(path) for name, path in ckpts.items()
        } | {
            'config_path': str(cfg_path),
            'log_path': str(log_path),
        },
        'score': s05.score_summary(final_best_loss),
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
                'gamma_arm': result.get('gamma_arm'),
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


def collect_effectiveness_warnings(base_cfg: dict[str, Any], recipe_arm_names: list[str]) -> list[str]:
    warnings = []
    for arm_name in recipe_arm_names:
        recipe_arm = RECIPE_ARMS[arm_name]
        reason = ineffective_reason(base_cfg, recipe_arm)
        if reason is not None:
            warnings.append(f'{arm_name}: {reason}')
    return warnings


def run_recipe_ab(
    base_cfg: dict[str, Any],
    *,
    ab_name: str,
    seed: int,
    step_scale: float,
    gamma_arm: str,
    init_state_file: str,
) -> dict[str, Any]:
    train_files, eval_splits, _ = resolve_stage1_splits(base_cfg, seed)
    results = {}
    for recipe_arm in RECIPE_ARMS.values():
        results[recipe_arm.arm_name] = run_recipe_arm(
            base_cfg,
            ab_name=ab_name,
            recipe_arm=recipe_arm,
            gamma_arm=gamma_arm,
            isolate_gamma_artifacts=False,
            train_files=train_files,
            eval_splits=eval_splits,
            init_state_file=init_state_file,
            seed=seed,
            step_scale=step_scale,
        )
    winner_name, selection = s05.select_winner_by_policy(results)
    payload = {
        'mode': 'recipe',
        'seed': seed,
        'step_scale': step_scale,
        'gamma_arm': gamma_arm,
        'init_state_file': init_state_file,
        'warnings': collect_effectiveness_warnings(base_cfg, list(RECIPE_ARMS)),
        'train_files': len(train_files),
        'eval_split_counts': {
            key: len(value) for key, value in eval_splits.items()
        },
        'winner': winner_name,
        'winner_metrics': results[winner_name]['final']['best_loss'],
        'selection': selection,
        'ranking': rank_results(results),
        'results': results,
    }
    payload['summary_path'] = str(save_results(ab_name, payload))
    return payload


def run_gamma_ab(
    base_cfg: dict[str, Any],
    *,
    ab_name: str,
    seed: int,
    step_scale: float,
    recipe_arm_name: str,
    init_state_file: str,
) -> dict[str, Any]:
    if recipe_arm_name not in RECIPE_ARMS:
        raise ValueError(f'unknown recipe arm: {recipe_arm_name}')
    recipe_arm = RECIPE_ARMS[recipe_arm_name]
    train_files, eval_splits, _ = resolve_stage1_splits(base_cfg, seed)
    results = {}
    for gamma_arm in GAMMA_PROFILES:
        arm_name = f'{recipe_arm_name}_{gamma_arm}'
        result = run_recipe_arm(
            base_cfg,
            ab_name=ab_name,
            recipe_arm=recipe_arm,
            gamma_arm=gamma_arm,
            isolate_gamma_artifacts=True,
            train_files=train_files,
            eval_splits=eval_splits,
            init_state_file=init_state_file,
            seed=seed,
            step_scale=step_scale,
        )
        result['arm_name'] = arm_name
        results[arm_name] = result
    winner_name, selection = s05.select_winner_by_policy(results)
    payload = {
        'mode': 'gamma',
        'seed': seed,
        'step_scale': step_scale,
        'recipe_arm': recipe_arm_name,
        'init_state_file': init_state_file,
        'warnings': collect_effectiveness_warnings(base_cfg, [recipe_arm_name]),
        'train_files': len(train_files),
        'eval_split_counts': {
            key: len(value) for key, value in eval_splits.items()
        },
        'winner': winner_name,
        'winner_metrics': results[winner_name]['final']['best_loss'],
        'selection': selection,
        'ranking': rank_results(results),
        'results': results,
    }
    payload['summary_path'] = str(save_results(ab_name, payload))
    return payload


def run_block_c(
    base_cfg: dict[str, Any],
    *,
    ab_name: str,
    seed: int,
    step_scale: float,
    recipe_gamma_arm: str,
    gamma_recipe_arm: str,
    init_state_file: str,
) -> dict[str, Any]:
    recipe_ab_name = f'{ab_name}_recipe'
    recipe_payload = run_recipe_ab(
        base_cfg,
        ab_name=recipe_ab_name,
        seed=seed,
        step_scale=step_scale,
        gamma_arm=recipe_gamma_arm,
        init_state_file=init_state_file,
    )
    gamma_recipe = recipe_payload['winner']
    if gamma_recipe == 'S1-A':
        gamma_recipe = gamma_recipe_arm
    gamma_ab_name = f'{ab_name}_gamma'
    gamma_payload = run_gamma_ab(
        base_cfg,
        ab_name=gamma_ab_name,
        seed=seed + 1000,
        step_scale=step_scale,
        recipe_arm_name=gamma_recipe,
        init_state_file=init_state_file,
    )
    payload = {
        'mode': 'block_c',
        'seed': seed,
        'step_scale': step_scale,
        'init_state_file': init_state_file,
        'recipe_gamma_arm': recipe_gamma_arm,
        'gamma_recipe_arm': gamma_recipe,
        'recipe': recipe_payload,
        'gamma': gamma_payload,
    }
    payload['summary_path'] = str(save_results(ab_name, payload))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('recipe', 'gamma', 'block_c'), default='block_c')
    parser.add_argument('--ab-name')
    parser.add_argument('--seed', type=int, default=STAGE1_AB_DEFAULTS['seed'])
    parser.add_argument('--step-scale', type=float, default=1.0)
    parser.add_argument('--gamma-arm', choices=sorted(GAMMA_PROFILES), default='G1')
    parser.add_argument('--recipe-arm', choices=sorted(RECIPE_ARMS), default='S1-B')
    parser.add_argument('--init-state')
    parser.add_argument('--list-arms', action='store_true')
    args = parser.parse_args()

    if args.list_arms:
        print(json.dumps(
            {
                'recipe_arms': {
                    name: {
                        'description': arm.description,
                        'use_oracle': arm.use_oracle,
                        'enable_rank': arm.enable_rank,
                        'enable_opp': arm.enable_opp,
                        'enable_danger': arm.enable_danger,
                    }
                    for name, arm in RECIPE_ARMS.items()
                },
                'gamma_profiles': GAMMA_PROFILES,
            },
            ensure_ascii=False,
            indent=2,
        ))
        return

    base_cfg = ensure_stage1_section(s05.build_base_config())
    init_state_file = args.init_state or base_cfg['stage1'].get('init_state_file', '')
    if not init_state_file:
        raise RuntimeError('missing Stage 1 init checkpoint; pass --init-state or set stage1.init_state_file')
    stage05_formal.ensure_stage1_canonical_handoff_ready(init_state_file)
    if not Path(init_state_file).exists():
        raise RuntimeError(f'Stage 1 init checkpoint does not exist: {init_state_file}')

    if args.mode == 'recipe':
        ab_name = args.ab_name or f'stage1_recipe_{args.gamma_arm.lower()}'
        payload = run_recipe_ab(
            base_cfg,
            ab_name=ab_name,
            seed=args.seed,
            step_scale=args.step_scale,
            gamma_arm=args.gamma_arm,
            init_state_file=init_state_file,
        )
    elif args.mode == 'gamma':
        ab_name = args.ab_name or f'stage1_gamma_{args.recipe_arm.lower()}'
        payload = run_gamma_ab(
            base_cfg,
            ab_name=ab_name,
            seed=args.seed,
            step_scale=args.step_scale,
            recipe_arm_name=args.recipe_arm,
            init_state_file=init_state_file,
        )
    else:
        ab_name = args.ab_name or 'stage1_block_c'
        payload = run_block_c(
            base_cfg,
            ab_name=ab_name,
            seed=args.seed,
            step_scale=args.step_scale,
            recipe_gamma_arm=args.gamma_arm,
            gamma_recipe_arm=args.recipe_arm,
            init_state_file=init_state_file,
        )

    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
