from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mortal.core.adaptive_curriculum import (
    AdaptiveCurriculumConfig,
    inherit_adaptive_curriculum_baseline,
)
from mortal.core.toml_utils import load_toml_file, write_toml_file
from mortal.online.pretrain_oracle_critic import adaptive_curriculum_contract
from mortal.research.oracle_critic_curriculum import (
    checkpoint_training_state_hashes,
    migrate_checkpoint_for_phase,
    split_summary,
)


EXTERNAL_PAUSE_EXIT_CODE = 75
CASE_NAME = 'sf_lr200_wd000'
PHASES = ('phase_a', 'phase_b', 'phase_c')
GATE_EVERY_STEPS = 50_000
REQUIRED_FUTILE_GATES = 2
MEANINGFUL_PRIMARY_DELTA = 2e-4
FINAL_LR_LEVELS = (2e-4, 1e-4, 5e-5, 2.5e-5, 1e-5)
DEFAULT_HARD_MAX_STEPS = 5_000_000
DEFAULT_RUN_ROOT = (
    REPO_ROOT
    / 'logs/oracle_critic_formal/'
    's70_broad_to_recent_strong24m12m_adaptive_sf200_wd0_20260901_r1'
)
DEFAULT_CACHE_ROOT = (
    REPO_ROOT
    / 'logs/oracle_event_cache/'
    's70_broad_to_recent_strong24m12m_20260901_r1'
)
DEFAULT_SOURCE_CACHE_ROOT = (
    REPO_ROOT
    / 'logs/oracle_event_cache/s70_temporal_dev202512_test202601_chunk16_v1'
)
BASE_CONFIG = (
    REPO_ROOT
    / 'logs/oracle_critic_search/'
    's70_temporal_scalar_hand_wd003_formal_20260818_r1/'
    'visible_transfer_hand_aligned_wd003/config.toml'
)
RUNTIME_OVERLAY = REPO_ROOT / 'logs/runtime_overlays/libriichi_native_fold_capacity_v2'
BASE_CASE_FILE = REPO_ROOT / 'scripts/oracle_critic_curriculum_case_v1.json'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Run the Oracle critic A/B/C data curriculum with paired, '
            'evidence-triggered transitions and final LR reductions.'
        )
    )
    parser.add_argument('--run-root', default=str(DEFAULT_RUN_ROOT))
    parser.add_argument('--cache-root', default=str(DEFAULT_CACHE_ROOT))
    parser.add_argument('--source-cache-root', default=str(DEFAULT_SOURCE_CACHE_ROOT))
    parser.add_argument('--python-exe', default=sys.executable)
    parser.add_argument('--hard-max-steps', type=int, default=DEFAULT_HARD_MAX_STEPS)
    parser.add_argument('--validate-only', action='store_true')
    parser.add_argument('--emit-supervisor-spec', action='store_true')
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    candidate = Path(value)
    return candidate.resolve() if candidate.is_absolute() else (REPO_ROOT / candidate).resolve()


def file_sha256(source: Path) -> str:
    digest = hashlib.sha256()
    with source.open('rb') as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(destination: Path, payload: Any) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + '.tmp')
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )
    temporary.replace(destination)


def atomic_torch_save(destination: Path, payload: Any) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + '.tmp')
    if temporary.exists():
        temporary.unlink()
    torch.save(payload, temporary)
    temporary.replace(destination)


def pause_file() -> Path | None:
    value = os.environ.get('MORTAL_ORACLE_PAUSE_FILE', '').strip()
    return Path(value) if value else None


def pause_requested() -> bool:
    marker = pause_file()
    return marker is not None and marker.is_file()


def child_environment() -> dict[str, str]:
    if not (RUNTIME_OVERLAY / 'libriichi/__init__.py').is_file():
        raise FileNotFoundError(f'libriichi runtime overlay is incomplete: {RUNTIME_OVERLAY}')
    environment = os.environ.copy()
    overlay_text = str(RUNTIME_OVERLAY.resolve())
    existing = [
        item for item in environment.get('PYTHONPATH', '').split(os.pathsep) if item
    ]
    environment['PYTHONPATH'] = os.pathsep.join(
        [overlay_text, *[item for item in existing if item != overlay_text]]
    )
    return environment


def run_child(arguments: list[str], *, python_exe: Path) -> int:
    if pause_requested():
        return EXTERNAL_PAUSE_EXIT_CODE
    completed = subprocess.run(
        [str(python_exe), '-u', *arguments],
        cwd=REPO_ROOT,
        env=child_environment(),
        check=False,
    )
    return int(completed.returncode)


def run_restartable_evaluation(arguments: list[str], *, python_exe: Path) -> int:
    if pause_requested():
        return EXTERNAL_PAUSE_EXIT_CODE
    process = subprocess.Popen(
        [str(python_exe), '-u', *arguments],
        cwd=REPO_ROOT,
        env=child_environment(),
    )
    while process.poll() is None:
        if pause_requested():
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            return EXTERNAL_PAUSE_EXIT_CODE
        time.sleep(1)
    return int(process.returncode)


def adaptive_mapping(phase: str) -> dict[str, Any]:
    final_phase = phase == PHASES[-1]
    return {
        'enabled': True,
        'phase_name': phase,
        'final_phase': final_phase,
        'gate_every_steps': GATE_EVERY_STEPS,
        'required_futile_gates': REQUIRED_FUTILE_GATES,
        'confidence_z': 1.96,
        'primary_noninferiority_margin': MEANINGFUL_PRIMARY_DELTA,
        'primary': {
            'name': 'primary_loss',
            'direction': 'lower',
            'meaningful_delta': MEANINGFUL_PRIMARY_DELTA,
        },
        'guardrails': [
            {
                'name': 'all_players_loss',
                'direction': 'lower',
                'meaningful_delta': MEANINGFUL_PRIMARY_DELTA,
            },
            {
                'name': 'p0_mae',
                'direction': 'lower',
                'meaningful_delta': 1e-4,
            },
            *[
                {
                    'name': name,
                    'direction': 'lower',
                    'meaningful_delta': 5e-4,
                }
                for name in (
                    'exact_zero_loss',
                    'nonzero_loss',
                    'abs_ge_2_loss',
                    'abs_ge_4_loss',
                )
            ],
        ],
        'lr_levels': list(FINAL_LR_LEVELS) if final_phase else [],
    }


def phase_case_payload(phase: str) -> list[dict[str, Any]]:
    payload = json.loads(BASE_CASE_FILE.read_text(encoding='utf-8'))
    if not isinstance(payload, list) or len(payload) != 1:
        raise ValueError(f'expected one base curriculum case in {BASE_CASE_FILE}')
    case = copy.deepcopy(payload[0])
    case['pretrain']['convergence'] = {'enabled': False}
    case['pretrain']['adaptive_curriculum'] = adaptive_mapping(phase)
    return [case]


def write_phase_case(run_root: Path, phase: str) -> Path:
    destination = run_root / 'phase_specs' / f'{phase}_case.json'
    payload = phase_case_payload(phase)
    if destination.is_file():
        saved = json.loads(destination.read_text(encoding='utf-8'))
        if saved != payload:
            raise RuntimeError(f'adaptive phase case changed; use a new run root: {phase}')
    else:
        atomic_write_json(destination, payload)
    return destination


def curriculum_design(
    *,
    run_root: Path,
    cache_root: Path,
    source_cache_root: Path,
    hard_max_steps: int,
) -> dict[str, Any]:
    return {
        'format': 'oracle_critic_adaptive_curriculum_v1',
        'objective': 'strongest standalone Oracle critic before actor integration',
        'initialization': {
            'type': 'sl_policy_checkpoint',
            'source': str(
                (
                    REPO_ROOT
                    / 'logs/sl_fidelity/'
                    'sl_anchor_longabc_s70_20260609_r1_1v3_compare/'
                    'best_action_score.pth'
                ).resolve()
            ),
            'note': 'fresh Oracle/value path; no ad-hoc long critic checkpoint',
        },
        'audited_recipe': {
            'optimizer': 'Schedule-Free AdamW',
            'lr': 2e-4,
            'weight_decay': 0.0,
            'warmup_steps': 2000,
        },
        'curriculum': {
            'phases': list(PHASES),
            'phase_a': {'weights': {'recent_24m': 0.60, 'mid': 0.25, 'early': 0.15}},
            'phase_b': {'weights': {'recent_24m': 0.90, 'replay': 0.10}},
            'phase_c': {'weights': {'recent_12m': 0.98, 'replay': 0.02}},
            'monitor_every_steps': 10_000,
            'paired_gate_every_steps': GATE_EVERY_STEPS,
            'transition_rule': (
                'two consecutive same-monitor paired gates whose optimistic '
                '95% interval cannot contain a meaningful primary gain, with '
                'no significant noninferior guardrail compensation'
            ),
            'fixed_phase_lengths': None,
        },
        'final_phase_lr': {
            'levels': list(FINAL_LR_LEVELS),
            'reduction_rule': 'the same two-gate futility rule at each level',
            'stop_rule': 'the same rule at the final level',
        },
        'hard_max_steps': int(hard_max_steps),
        'selection': {
            'monitor_remainder': 0,
            'formal_dev_remainders': [1, 2, 3],
            'historical_regression_guard': True,
            'human_sealed_test': 'closed',
            'actor_replay_sid0_sid1': 'consumed_do_not_reuse',
        },
        'run_root': str(run_root),
        'cache_root': str(cache_root),
        'source_cache_root': str(source_cache_root),
        'source_fingerprints': {
            'orchestrator_sha256': file_sha256(Path(__file__).resolve()),
            'adaptive_controller_sha256': file_sha256(
                REPO_ROOT / 'mortal/core/adaptive_curriculum.py'
            ),
            'oracle_trainer_sha256': file_sha256(
                REPO_ROOT / 'mortal/online/pretrain_oracle_critic.py'
            ),
            'search_runner_sha256': file_sha256(
                REPO_ROOT / 'scripts/run_oracle_critic_search.py'
            ),
            'base_case_sha256': file_sha256(BASE_CASE_FILE),
        },
    }


def write_runtime_files(
    *,
    run_root: Path,
    cache_root: Path,
    source_cache_root: Path,
    python_exe: Path,
    hard_max_steps: int,
) -> tuple[Path, Path]:
    run_root.mkdir(parents=True, exist_ok=True)
    design_path = run_root / 'curriculum_design.json'
    design = curriculum_design(
        run_root=run_root,
        cache_root=cache_root,
        source_cache_root=source_cache_root,
        hard_max_steps=hard_max_steps,
    )
    if design_path.is_file():
        if json.loads(design_path.read_text(encoding='utf-8')) != design:
            raise RuntimeError('adaptive curriculum design changed; use a new run root')
    else:
        atomic_write_json(design_path, design)

    spec_path = run_root / 'apex_supervisor_spec.json'
    spec = {
        'format': 'oracle_critic_apex_supervisor_spec_v1',
        'repo_root': str(REPO_ROOT.resolve()),
        'search_root': str(run_root.resolve()),
        'python_executable': str(python_exe.resolve()),
        'pause_file': str((run_root / 'apex_pause.request').resolve()),
        'status_file': str((run_root / 'apex_supervisor_status.json').resolve()),
        'log_file': str((run_root / 'apex_supervisor.log').resolve()),
        'runner_arguments': [
            '-u',
            'scripts/run_oracle_critic_adaptive_curriculum.py',
            '--run-root',
            str(run_root.resolve()),
            '--cache-root',
            str(cache_root.resolve()),
            '--source-cache-root',
            str(source_cache_root.resolve()),
            '--python-exe',
            str(python_exe.resolve()),
            '--hard-max-steps',
            str(int(hard_max_steps)),
        ],
    }
    if spec_path.is_file():
        if json.loads(spec_path.read_text(encoding='utf-8')) != spec:
            raise RuntimeError('Apex supervisor spec changed; use a new run root')
    else:
        atomic_write_json(spec_path, spec)
    return design_path, spec_path


def validate_static_inputs(
    *, source_cache_root: Path, python_exe: Path, hard_max_steps: int
) -> None:
    required = (
        python_exe,
        BASE_CONFIG,
        BASE_CASE_FILE,
        source_cache_root / 'manifest.json',
        REPO_ROOT / 'scripts/build_oracle_critic_curriculum_cache.py',
        REPO_ROOT / 'scripts/run_oracle_critic_search.py',
        REPO_ROOT / 'scripts/evaluate_oracle_critic_checkpoints.py',
        REPO_ROOT / 'scripts/supervise_oracle_critic_around_apex.ps1',
    )
    missing = [str(item) for item in required if not item.is_file()]
    if missing:
        raise FileNotFoundError(f'adaptive curriculum inputs are missing: {missing}')
    if hard_max_steps <= GATE_EVERY_STEPS * REQUIRED_FUTILE_GATES:
        raise ValueError('hard max is too small for one adaptive decision window')


def cache_indexes(cache_root: Path) -> dict[str, Path]:
    return {
        'phase_a': cache_root / 'phase_a_train_index.pth',
        'phase_b': cache_root / 'phase_b_train_index.pth',
        'phase_c': cache_root / 'phase_c_train_index.pth',
        'old_regression': cache_root / 'old_regression_eval_index.pth',
        'dev': cache_root / 'dev_index.pth',
        'test': cache_root / 'test_index.pth',
    }


def ensure_cache(
    *, cache_root: Path, source_cache_root: Path, python_exe: Path
) -> int:
    manifest = cache_root / 'manifest.json'
    if manifest.is_file():
        payload = json.loads(manifest.read_text(encoding='utf-8'))
        if bool(payload.get('complete', False)):
            return 0
    return run_child(
        [
            'scripts/build_oracle_critic_curriculum_cache.py',
            '--source-cache-root',
            str(source_cache_root),
            '--base-config',
            str(BASE_CONFIG),
            '--runtime-overlay',
            str(RUNTIME_OVERLAY),
            '--output-dir',
            str(cache_root),
            '--files-per-chunk',
            '16',
            '--old-regression-eval-chunks',
            '64',
            '--seed',
            '20260416',
        ],
        python_exe=python_exe,
    )


def phase_case_dir(run_root: Path, phase: str) -> Path:
    return run_root / 'phases' / phase / CASE_NAME


def phase_config(run_root: Path, phase: str) -> Path:
    return phase_case_dir(run_root, phase) / 'config.toml'


def search_arguments(
    *,
    run_root: Path,
    phase: str,
    case_file: Path,
    hard_max_steps: int,
    indexes: dict[str, Path],
    prepare_only: bool,
) -> list[str]:
    arguments = [
        'scripts/run_oracle_critic_search.py',
        '--base-config',
        str(BASE_CONFIG),
        '--output-root',
        str(run_root / 'phases'),
        '--search-name',
        phase,
        '--runtime-overlay',
        str(RUNTIME_OVERLAY),
        '--case-file',
        str(case_file),
        '--case',
        CASE_NAME,
        '--stage-steps',
        str(int(hard_max_steps)),
        '--scheduler-horizon-steps',
        str(int(hard_max_steps)),
        '--val-every-steps',
        '10000',
        '--val-batches',
        '256',
        '--test-batches',
        '1024',
        '--num-workers',
        '2',
        '--file-batch-size',
        '6',
        '--prefetch-factor',
        '2',
        '--val-num-workers',
        '0',
        '--val-file-batch-size',
        '8',
        '--val-prefetch-factor',
        '5',
        '--dependency-val-every-steps',
        '0',
        '--save-every',
        '10000',
        '--seed',
        '20260416',
        '--split-seed',
        '20260416',
        '--val-game-id-modulus',
        '5',
        '--val-game-id-remainder',
        '0',
        '--selection-game-id-modulus',
        '5',
        '--selection-game-id-remainder',
        '1',
        '--selection-game-id-remainder',
        '2',
        '--selection-game-id-remainder',
        '3',
        '--selection-max-batches',
        '0',
        '--selection-state-fold-count',
        '128',
        '--train-file-index',
        str(indexes[phase]),
        '--dev-file-index',
        str(indexes['dev']),
        '--test-file-index',
        str(indexes['test']),
        '--stop-on-error',
    ]
    if prepare_only:
        arguments.append('--prepare-only')
    return arguments


def load_index(source: Path) -> list[str]:
    payload = torch.load(source, weights_only=False, map_location='cpu')
    if isinstance(payload, dict):
        payload = payload.get('file_list')
    if not isinstance(payload, (list, tuple)) or not payload:
        raise ValueError(f'invalid or empty file index: {source}')
    return [str(filename) for filename in payload]


def destination_splits(config_payload: dict[str, Any]) -> dict[str, Any]:
    pretrain = config_payload['oracle_critic_pretrain']
    return split_summary(
        load_index(resolve_path(pretrain['train_file_index'])),
        load_index(resolve_path(pretrain['dev_file_index'])),
        load_index(resolve_path(pretrain['test_file_index'])),
        seed=int(pretrain.get('split_seed', pretrain.get('seed', 20260416))),
    )


def migrate_phase_best(
    *,
    source: Path,
    source_phase: str,
    destination_phase: str,
    destination_case_dir: Path,
) -> Path:
    destination = destination_case_dir / 'checkpoints/latest.pth'
    adaptive_best_destination = (
        destination_case_dir / 'checkpoints/adaptive_best.pth'
    )
    if destination.is_file():
        if not adaptive_best_destination.is_file():
            raise RuntimeError(
                'migrated phase latest exists without its inherited adaptive best: '
                f'{destination}'
            )
        return destination
    if pause_requested():
        raise InterruptedError('Apex pause requested before phase migration')
    source_state = torch.load(source, weights_only=False, map_location='cpu')
    destination_config_payload = load_toml_file(
        destination_case_dir / 'config.toml'
    )
    splits = destination_splits(destination_config_payload)
    adaptive_config = AdaptiveCurriculumConfig.from_mapping(
        destination_config_payload['oracle_critic_pretrain']['adaptive_curriculum']
    )
    destination_contract = copy.deepcopy(source_state['training_contract'])
    destination_contract['adaptive_curriculum'] = adaptive_curriculum_contract(
        adaptive_config
    )
    before = checkpoint_training_state_hashes(source_state)
    migrated = migrate_checkpoint_for_phase(
        source_state,
        destination_config=destination_config_payload,
        destination_file_splits=splits,
        source_phase=source_phase,
        destination_phase=destination_phase,
        source_checkpoint=str(source.resolve()),
        source_checkpoint_sha256=file_sha256(source),
        reset_data_cursor=True,
        destination_training_contract=destination_contract,
        destination_adaptive_curriculum_state=(
            inherit_adaptive_curriculum_baseline(
                adaptive_config,
                source_state.get('adaptive_curriculum_state') or {},
            )
        ),
        destination_adaptive_state_action=(
            'inherit_phase_best_reset_gate_counters'
        ),
    )
    atomic_torch_save(adaptive_best_destination, migrated)
    atomic_torch_save(destination, migrated)
    reloaded = torch.load(destination, weights_only=False, map_location='cpu')
    if before != checkpoint_training_state_hashes(reloaded):
        raise RuntimeError('adaptive phase migration changed protected training state')
    atomic_write_json(
        destination_case_dir / 'phase_transition_manifest.json',
        {
            'format': 'oracle_critic_adaptive_phase_transition_v1',
            'created_at_utc': datetime.now(timezone.utc).isoformat(),
            'source_phase': source_phase,
            'destination_phase': destination_phase,
            'source_checkpoint': str(source.resolve()),
            'source_checkpoint_sha256': file_sha256(source),
            'destination_checkpoint': str(destination.resolve()),
            'destination_checkpoint_sha256': file_sha256(destination),
            'destination_adaptive_best': str(
                adaptive_best_destination.resolve()
            ),
            'destination_adaptive_best_sha256': file_sha256(
                adaptive_best_destination
            ),
            'source_steps': int(source_state['steps']),
            'data_cursor_action': 'reset_for_changed_train_split',
            'protected_training_state_sha256': before,
            'adaptive_state_action': 'inherit_phase_best_reset_gate_counters',
        },
    )
    return destination


def completed_phase_state(run_root: Path, phase: str) -> dict[str, Any] | None:
    latest = phase_case_dir(run_root, phase) / 'checkpoints/latest.pth'
    if not latest.is_file():
        return None
    state = torch.load(latest, weights_only=False, map_location='cpu')
    adaptive = state.get('adaptive_curriculum_state') or {}
    if not bool(adaptive.get('completed', False)):
        return None
    if adaptive.get('last_action') == 'inconclusive':
        raise RuntimeError(
            'adaptive evidence budget exhausted without a decision; retain phase-best '
            'and expand fixed validation before restarting (do not advance curriculum)'
        )
    if adaptive.get('phase_name') != phase:
        raise RuntimeError(f'completed checkpoint has wrong adaptive phase: {latest}')
    return state


def run_phase(
    *,
    run_root: Path,
    phase: str,
    indexes: dict[str, Path],
    python_exe: Path,
    hard_max_steps: int,
    source_phase: str | None = None,
    source_checkpoint: Path | None = None,
) -> int:
    case_file = write_phase_case(run_root, phase)
    prepare = search_arguments(
        run_root=run_root,
        phase=phase,
        case_file=case_file,
        hard_max_steps=hard_max_steps,
        indexes=indexes,
        prepare_only=True,
    )
    return_code = run_child(prepare, python_exe=python_exe)
    if return_code != 0:
        return return_code
    if source_checkpoint is not None:
        try:
            migrate_phase_best(
                source=source_checkpoint,
                source_phase=str(source_phase),
                destination_phase=phase,
                destination_case_dir=phase_case_dir(run_root, phase),
            )
        except InterruptedError:
            return EXTERNAL_PAUSE_EXIT_CODE
    return_code = run_child(
        search_arguments(
            run_root=run_root,
            phase=phase,
            case_file=case_file,
            hard_max_steps=hard_max_steps,
            indexes=indexes,
            prepare_only=False,
        ),
        python_exe=python_exe,
    )
    if return_code != 0:
        return return_code
    state = completed_phase_state(run_root, phase)
    if state is None:
        raise RuntimeError(
            f'{phase} reached its runner exit without an adaptive completion state'
        )
    return 0


def phase_best_checkpoint(run_root: Path, phase: str) -> Path:
    checkpoint = phase_case_dir(run_root, phase) / 'checkpoints/adaptive_best.pth'
    if not checkpoint.is_file():
        raise FileNotFoundError(f'adaptive phase-best checkpoint is missing: {checkpoint}')
    return checkpoint


def candidate_checkpoints(run_root: Path) -> list[tuple[str, Path]]:
    candidates: list[tuple[str, Path]] = []
    seen = set()
    for phase in PHASES:
        checkpoint_dir = phase_case_dir(run_root, phase) / 'checkpoints'
        filenames = [('adaptive_best', 'adaptive_best.pth')]
        if phase == PHASES[-1]:
            filenames.extend(
                (
                    ('latest', 'latest.pth'),
                    ('best_primary', 'best_primary.pth'),
                    ('best_dev', 'best_dev.pth'),
                )
            )
        for role, filename in filenames:
            checkpoint = checkpoint_dir / filename
            if not checkpoint.is_file():
                continue
            digest = file_sha256(checkpoint)
            if digest in seen:
                continue
            seen.add(digest)
            candidates.append((f'{phase}_{role}', checkpoint))
    if not candidates:
        raise FileNotFoundError('adaptive curriculum produced no candidate checkpoints')
    return candidates


def evaluation_arguments(
    *,
    config: Path,
    output: Path,
    checkpoints: list[tuple[str, Path]],
    split_override_reason: str = '',
) -> list[str]:
    arguments = [
        'scripts/evaluate_oracle_critic_checkpoints.py',
        '--config',
        str(config),
        '--split',
        'dev',
        '--input-mode',
        'true',
        '--eval-state-fold-count',
        '128',
        '--game-id-modulus',
        '5',
        '--game-id-remainder',
        '1',
        '--game-id-remainder',
        '2',
        '--game-id-remainder',
        '3',
        '--max-batches',
        '0',
        '--output',
        str(output),
        '--no-print-result',
    ]
    for name, checkpoint in checkpoints:
        arguments.extend(['--checkpoint', f'{name}={checkpoint}'])
    if split_override_reason:
        arguments.extend(['--eval-split-override-reason', split_override_reason])
    return arguments


def run_final_selection(
    *, run_root: Path, indexes: dict[str, Path], python_exe: Path
) -> int:
    selection_dir = run_root / 'final_candidate_selection'
    selection_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = candidate_checkpoints(run_root)
    config = phase_config(run_root, PHASES[-1])
    paired_output = selection_dir / 'paired_dev_remainders123.json'
    if not paired_output.is_file():
        return_code = run_restartable_evaluation(
            evaluation_arguments(
                config=config,
                output=paired_output,
                checkpoints=checkpoints,
            ),
            python_exe=python_exe,
        )
        if return_code != 0:
            return return_code

    historical_config = selection_dir / 'historical_regression_config.toml'
    if not historical_config.is_file():
        payload = load_toml_file(config)
        payload['oracle_critic_pretrain']['dev_file_index'] = str(
            indexes['old_regression'].resolve()
        )
        payload['oracle_critic_pretrain']['max_val_files'] = 0
        write_toml_file(historical_config, payload)
    historical_output = selection_dir / 'paired_historical_regression.json'
    if not historical_output.is_file():
        return run_restartable_evaluation(
            evaluation_arguments(
                config=historical_config,
                output=historical_output,
                checkpoints=checkpoints,
                split_override_reason=(
                    'predeclared historical regression guard for final adaptive '
                    'Oracle critic selection'
                ),
            ),
            python_exe=python_exe,
        )
    return 0


def write_completion(run_root: Path) -> None:
    latest = phase_case_dir(run_root, PHASES[-1]) / 'checkpoints/latest.pth'
    state = torch.load(latest, weights_only=False, map_location='cpu')
    adaptive = state.get('adaptive_curriculum_state') or {}
    atomic_write_json(
        run_root / 'completion.json',
        {
            'format': 'oracle_critic_adaptive_curriculum_completion_v1',
            'completed_at_utc': datetime.now(timezone.utc).isoformat(),
            'latest_checkpoint': str(latest.resolve()),
            'latest_checkpoint_sha256': file_sha256(latest),
            'steps': int(state['steps']),
            'adaptive_curriculum_state': adaptive,
            'candidate_checkpoints': [
                {
                    'name': name,
                    'path': str(checkpoint.resolve()),
                    'sha256': file_sha256(checkpoint),
                }
                for name, checkpoint in candidate_checkpoints(run_root)
            ],
            'candidate_selection': str(
                (run_root / 'final_candidate_selection').resolve()
            ),
            'human_sealed_test': 'closed',
            'actor_replay_sid0_sid1': 'not_reused',
        },
    )


def main() -> int:
    args = parse_args()
    run_root = resolve_path(args.run_root)
    cache_root = resolve_path(args.cache_root)
    source_cache_root = resolve_path(args.source_cache_root)
    python_exe = resolve_path(args.python_exe)
    validate_static_inputs(
        source_cache_root=source_cache_root,
        python_exe=python_exe,
        hard_max_steps=args.hard_max_steps,
    )
    design_path, spec_path = write_runtime_files(
        run_root=run_root,
        cache_root=cache_root,
        source_cache_root=source_cache_root,
        python_exe=python_exe,
        hard_max_steps=args.hard_max_steps,
    )
    if args.validate_only or args.emit_supervisor_spec:
        print(json.dumps({
            'validated': True,
            'design': str(design_path),
            'supervisor_spec': str(spec_path),
        }, sort_keys=True))
        return 0
    if pause_requested():
        return EXTERNAL_PAUSE_EXIT_CODE

    return_code = ensure_cache(
        cache_root=cache_root,
        source_cache_root=source_cache_root,
        python_exe=python_exe,
    )
    if return_code != 0:
        return return_code
    indexes = cache_indexes(cache_root)
    missing = [str(item) for item in indexes.values() if not item.is_file()]
    if missing:
        raise FileNotFoundError(f'adaptive curriculum indexes are missing: {missing}')

    source_phase = None
    source_checkpoint = None
    for phase in PHASES:
        print(f'[{phase}] dynamic paired curriculum phase', flush=True)
        return_code = run_phase(
            run_root=run_root,
            phase=phase,
            indexes=indexes,
            python_exe=python_exe,
            hard_max_steps=args.hard_max_steps,
            source_phase=source_phase,
            source_checkpoint=source_checkpoint,
        )
        if return_code != 0:
            return return_code
        source_phase = phase
        source_checkpoint = phase_best_checkpoint(run_root, phase)

    print('[final_selection] paired dev remainders 1/2/3 and historical guard', flush=True)
    return_code = run_final_selection(
        run_root=run_root,
        indexes=indexes,
        python_exe=python_exe,
    )
    if return_code != 0:
        return return_code
    write_completion(run_root)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
