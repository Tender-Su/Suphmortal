"""Narrow contracts for extending one SL phase without resetting its state."""
from contextlib import closing
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import random
import sqlite3

from mortal.core.artifacts import stable_json_digest
from mortal.supervised.early_transition import validate_parent


STATE_PATHS = ('state_file', 'best_state_file', 'best_loss_state_file', 'best_acc_state_file',
               'best_rank_state_file', 'best_policy_state_file', 'adaptive_best_state_file')
RELOCATION_FIELDS = (*STATE_PATHS, 'file_index', 'tensorboard_dir', 'probe_training_content_ledger',
                     'run_provenance')
SAVED_PHASE_FIELDS = frozenset('''checkpoint_id run_provenance mortal policy_net aux_net opponent_aux_net
    danger_aux_net optimizer optimizer_param_groups scheduler scaler steps optimizer_steps auxiliary_optimizer_steps
    skipped_optimizer_steps nonfinite_batches epoch epoch_complete timestamp best_val_loss best_val_action_acc
    best_val_action_score best_val_rank_acc best_full_recent_loss best_full_recent_policy_loss
    best_full_recent_action_acc best_full_recent_action_score best_full_recent_rank_acc patience_val_loss
    patience_counter num_lr_reductions validation_checks full_validation_checks old_regression_checks
    last_monitor_recent_metrics last_full_recent_metrics last_old_regression_metrics convergence_state
    adaptive_curriculum_state config_section stage_label checkpoint_label config curriculum_probe'''.split())


def microbatch_migration(state, target, reason):
    """Only a declared 256x4 <-> 512x2 numerical branch, never gradient equivalence."""
    if target is None:
        if reason:
            raise ValueError('microbatch change reason requires an explicit target microbatch')
        return None
    config = state['config']
    sl, control, probe = config['supervised'], config['control'], state['curriculum_probe']
    source, accumulation = sl['batch_size'], control['opt_step_every']
    if (type(target) is not int or type(source) is not int or type(accumulation) is not int
            or {source, target} != {256, 512} or source * accumulation != 1024 or not reason.strip()):
        raise ValueError('explicit reason and 256<->512 migration with unchanged logical batch1024 required')
    if set(state) - SAVED_PHASE_FIELDS or set(probe) - {
            'identity', 'dataset', 'observed', 'rng', 'elapsed_seconds', 'continuous_observation'}:
        raise ValueError('unknown saved state/counter; inspect its units before microbatch migration')
    if any(sl.get(key, {}).get('enabled', False) for key in ('convergence', 'adaptive_curriculum', 'gradient_calibration')):
        raise ValueError('microbatch migration requires disabled historical controllers/calibration')
    if any(state.get(key) is not None for key in ('convergence_state', 'adaptive_curriculum_state')):
        raise ValueError('controller state has unknown microstep history; migration refused')
    if (any(type(sl.get(key)) is not int or sl[key] != 0 for key in ('max_steps', 'val_every_steps', 'save_every'))
            or sl.get('num_workers') != 0 or control.get('enable_cuda_prefetch') is not False):
        raise ValueError('microbatch migration requires existing consumed-cursor probe clocks')
    steps, updates, skipped = (state[key] for key in ('steps', 'optimizer_steps', 'skipped_optimizer_steps'))
    data = probe['dataset']
    if (set(data) != {'sampler', 'current', 'offset', 'consumed', 'files'}
            or set(data['sampler']) != {'seed', 'recipe', 'positions', 'rng', 'draws'}
            or ('continuous_observation' in probe and set(probe['continuous_observation']) != {'done', 'last_saved_update'})):
        raise ValueError('unknown cursor/observation counter; inspect its units before microbatch migration')
    if (any(type(n) is not int or n < 0 for n in (steps, updates, skipped)) or updates == 0
            or steps != (updates + skipped) * accumulation or state['epoch_complete']
            or type(state['nonfinite_batches']) is not int or state['nonfinite_batches'] != 0 or not data['consumed']
            or any(type(n) is not int or n < 0 for n in data['consumed'].values())):
        raise ValueError('microbatch migration requires a complete optimizer boundary and known counters')
    decisions = sum(data['consumed'].values())
    log_every = sl.get('log_every')
    if (decisions != steps * source or decisions % target or type(log_every) is not int
            or log_every <= 0 or log_every * source % target):
        raise ValueError('consumed decisions/log cadence cannot be exactly converted to target microbatches')
    return {'mode': 'declared_microbatch_numerical_branch', 'reason': reason,
            'source_checkpoint_id': state['checkpoint_id'], 'logical_batch': 1024,
            'source_microbatch': source, 'target_microbatch': target,
            'source_accumulation': accumulation, 'target_accumulation': 1024 // target,
            'source_microsteps': steps, 'target_microsteps': decisions // target,
            'consumed_decisions': decisions, 'optimizer_updates': updates, 'skipped_optimizer_updates': skipped,
            'source_log_every': log_every, 'target_log_every': log_every * source // target,
            'numerical_equivalence': 'not_bitwise_or_gradient_equivalent',
            'historical_statistics': 'prior metrics/observations/elapsed time retain their original numerical protocol; '
                                     'unsaved running log window restarts as in ordinary resume'}


def apply_microbatch_config(config, migration):
    if migration is not None:
        config['supervised'].update(batch_size=migration['target_microbatch'], log_every=migration['target_log_every'])
        config['control']['opt_step_every'] = migration['target_accumulation']
    return config


def observation_plan(*, start, until, save_updates, save_seconds, trend_every, full_every):
    values = (start, until, save_updates, trend_every, full_every)
    if any(type(n) is not int or n <= 0 for n in values) or until <= start:
        raise ValueError('continuation needs a positive saved update and explicit larger phase endpoint/cadences')
    if not math.isfinite(save_seconds) or save_seconds <= 0:
        raise ValueError('latest-save wall-time cadence must be finite and positive')
    return dict(start_update=start, until_update=until, save_updates=save_updates,
                save_seconds=float(save_seconds), trend_every=trend_every, full_every=full_every)


def evaluation_kind(plan, update):
    if update <= plan['start_update']:
        return None  # The inherited point is not a new U0 or baseline.
    if update == plan['until_update'] or update % plan['full_every'] == 0:
        return 'full'
    if update % plan['trend_every'] == 0:
        return 'trend'
    return None


def evaluation_horizons(plan):
    start, end = plan['start_update'], plan['until_update']
    points = {end}
    for key in ('trend_every', 'full_every'):
        interval = plan[key]
        points.update(range((start // interval + 1) * interval, end + 1, interval))
    return sorted(points)


def trend_splits(full, *, recent_games, old_games, seed):
    result = {}
    for index, (name, count) in enumerate((('controller_recent', recent_games), ('controller_old', old_games))):
        files = sorted(full[name])
        if type(count) is not int or not 0 < count <= len(files) or len(set(files)) != len(files):
            raise ValueError('trend validation must be a nonempty fixed subset of each full role')
        random.Random(seed + index).shuffle(files)
        result[name] = sorted(files[:count])
    return result


def validate_saved_phase(state, domains, recipes):
    """No inferred cursor, RNG, auxiliary clock, AMP history, or skipped updates."""
    validate_parent(state)
    required = ('auxiliary_optimizer_steps', 'skipped_optimizer_steps', 'nonfinite_batches',
                'epoch', 'epoch_complete', 'timestamp', 'config_section', 'curriculum_probe', 'run_provenance')
    if any(key not in state for key in required):
        raise ValueError('missing exact continuation state; use a separately declared non-exact branch')
    config, provenance = state['config'], state['run_provenance']
    if config['supervised'].get('val_batch_size', config['supervised']['batch_size']) != 1024:
        raise ValueError('this continuation lane requires the saved validation batch1024/Brain256 protocol')
    if (state['config_section'] != 'supervised' or type(state['epoch']) is not int or state['epoch'] < 0
            or not math.isfinite(state['timestamp'])):
        raise ValueError('checkpoint trainer section/epoch/timestamp is not resumable')
    if not config['supervised'].get('probe_training_content_ledger'):
        raise ValueError('checkpoint lacks its source content ledger; exact input history is unavailable')
    if config['control'].get('enable_amp') and not {'scale', 'growth_factor', 'backoff_factor',
            'growth_interval', '_growth_tracker'} <= state['scaler'].keys():
        raise ValueError('enabled AMP checkpoint lacks complete GradScaler state')
    phase = provenance.get('phase')
    if phase not in ('B', 'C') or provenance.get('branch_mode') != 'preserve_adam_declared_phase_lr':
        raise ValueError('only an explicit existing B/C phase is supported; no implicit phase transition')
    if config['supervised'].get('run_provenance') != provenance:
        raise ValueError('checkpoint config and persisted provenance differ')
    probe = state['curriculum_probe']
    if not isinstance(probe, dict) or probe.get('identity') != provenance.get('plan_id'):
        raise ValueError('checkpoint probe identity differs from its declared phase')
    if not {'dataset', 'rng', 'observed', 'elapsed_seconds'} <= probe.keys():
        raise ValueError('checkpoint lacks complete consumed-data/RNG observation state')
    rng = probe['rng']
    if not isinstance(rng, dict) or not {'python', 'numpy', 'torch', 'cuda'} <= rng.keys():
        raise ValueError('checkpoint lacks complete model RNG state')
    if any(rng[key] is None for key in ('python', 'numpy', 'torch', 'cuda')):
        raise ValueError('checkpoint RNG cannot be reconstructed')
    if config['control']['device'].startswith('cuda') and not rng['cuda']:
        raise ValueError('CUDA checkpoint lacks CUDA RNG state')
    updates, steps, skipped = state['optimizer_steps'], state['steps'], state['skipped_optimizer_steps']
    accumulation = config['control']['opt_step_every']
    if (type(accumulation) is not int or accumulation <= 0 or updates <= 0
            or type(skipped) is not int or skipped < 0 or state['epoch_complete']
            or steps != (updates + skipped) * accumulation or state['nonfinite_batches'] != 0):
        raise ValueError('checkpoint is not a verified complete optimizer boundary')
    data = probe['dataset']
    if not isinstance(data, dict) or not {'sampler', 'current', 'offset', 'consumed', 'files'} <= data.keys():
        raise ValueError('checkpoint lacks its real consumed-data cursor')
    if (not data['consumed'] or any(type(n) is not int or n < 0 for n in data['consumed'].values())
            or sum(data['consumed'].values()) != steps * config['supervised']['batch_size']
            or not set(data['consumed']) <= data['files'].keys()):
        raise ValueError('consumed decisions do not match complete microbatches')
    sampler = data['sampler']
    if (not {'seed', 'recipe', 'positions', 'rng', 'draws'} <= sampler.keys()
            or sampler['seed'] != config['supervised']['seed'] or sampler['recipe'] != recipes[phase]
            or set(sampler['positions']) != set(recipes[phase])):
        raise ValueError('saved sampler seed/recipe/positions do not match the same phase')
    for name, (cycle, position) in sampler['positions'].items():
        if type(cycle) is not int or cycle < 0 or type(position) is not int or not 0 <= position <= len(domains[name]):
            raise ValueError('sampler cycle/position cannot be reconstructed')
    if type(data['offset']) is not int or data['offset'] < 0 or sampler['draws'] < 4:
        raise ValueError('invalid consumed block offset or draw count')
    if data['current'] is not None:
        draws = data['current'].get('draws', [])
        counts = {row['file']: row['available_decisions_per_draw'] for row in data['files'].values()}
        if (len(draws) != 4 or any(not {'file', 'domain', 'cycle', 'draw', 'source_sha256'} <= row.keys()
                for row in draws) or [row['draw'] for row in draws] != list(range(sampler['draws'] - 4, sampler['draws']))
                or any(row['file'] not in domains[row['domain']] or row['file'] not in counts for row in draws)
                or data['offset'] > sum(counts[row['file']] for row in draws)):
            raise ValueError('incomplete current block; cannot fabricate next sample position')
    elif data['offset'] != 0:
        raise ValueError('nonzero offset without a current block')
    scheduler = state['scheduler']
    if (scheduler.get('last_epoch') != updates or scheduler.get('_last_lr') !=
            [group['lr'] for group in state['optimizer']['param_groups']]):
        raise ValueError('saved scheduler clock/LR does not match the next optimizer update')
    return {'phase': phase, 'arm': provenance.get('arm', 'mainline'), 'optimizer_updates': updates,
            'microsteps': steps, 'consumed_decisions': sum(data['consumed'].values()),
            'next_update_lrs': list(scheduler['_last_lr']),
            'auxiliary_optimizer_steps': state['auxiliary_optimizer_steps'],
            'elapsed_seconds': probe['elapsed_seconds']}


def relocate_config(config, output, identity, parent, source_commit, source_identity, *, runtime_sha256,
                    migration=None):
    result = deepcopy(config)
    sl = result['supervised']
    for key in STATE_PATHS:
        sl[key] = str(Path(output) / (key + '.pth'))
    sl.update(file_index=str(Path(output) / 'indexes.pth'), tensorboard_dir=str(Path(output) / 'tensorboard'),
              probe_training_content_ledger=str(Path(output) / 'training_content.sqlite3'))
    provenance = deepcopy(sl['run_provenance'])
    old_runtime = provenance.get('source_runtime_sha256', '')
    provenance.update(plan_id=identity, experiment_id=identity, source_git_commit=source_commit,
                      source_runtime_sha256=runtime_sha256,
                      parent_checkpoint_id=parent['checkpoint_id'],
                      continuation={'source_experiment_id': source_identity, 'parent': deepcopy(parent),
                                    'source_runtime_sha256': old_runtime,
                                    'mode': 'same_phase_state_preserving_extension',
                                    'numerical_equivalence': 'not_a_cross_runtime_bitwise_proof'})
    provenance['parent_chain'] = [*provenance.get('parent_chain', []), deepcopy(parent)]
    if migration is not None:
        provenance['continuation'].update(mode=migration['mode'], numerical_equivalence=migration['numerical_equivalence'])
        provenance['microbatch_migrations'] = [*provenance.get('microbatch_migrations', []), deepcopy(migration)]
    sl['run_provenance'] = provenance
    return apply_microbatch_config(result, migration)


def training_semantics(config):
    result = deepcopy(config)
    for key in RELOCATION_FIELDS:
        result['supervised'].pop(key, None)
    return result


def rebind_saved_state(state, config, identity, *, migration=None):
    if migration is not None and migration != microbatch_migration(
            state, migration['target_microbatch'], migration['reason']):
        raise ValueError('microbatch migration differs from verified source counters')
    expected = apply_microbatch_config(deepcopy(state['config']), migration)
    if training_semantics(expected) != training_semantics(config):
        raise ValueError('continuation cannot change training settings, LR, warmup, seed, or batch size')
    result = deepcopy(state)
    result['config'] = deepcopy(config)
    result['run_provenance'] = deepcopy(config['supervised']['run_provenance'])
    result['curriculum_probe']['identity'] = identity
    result['checkpoint_id'] = stable_json_digest({'source_checkpoint_id': state['checkpoint_id'], 'identity': identity})
    if migration is not None:
        result['steps'] = migration['target_microsteps']
    return result


def ledger_snapshot(source, destination, source_identity, consumed_files, *, current_hashes=None):
    """SQLite backup is a consistent read even while another arm appends pins."""
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError('new inherited content snapshot required')
    with closing(sqlite3.connect(Path(source).resolve().as_uri() + '?mode=ro', uri=True)) as src:
        with closing(sqlite3.connect(destination)) as dst:
            src.backup(dst)
            if dst.execute('SELECT identity FROM metadata').fetchall() != [(source_identity,)]:
                raise ValueError('source ledger identity does not match the source experiment')
            for filename in consumed_files:
                if dst.execute('SELECT sha256 FROM games WHERE path=?', (filename,)).fetchone() is None:
                    raise ValueError('source ledger lacks a consumed game; exact input history is unavailable')
            for filename, expected in (current_hashes or {}).items():
                if dst.execute('SELECT sha256 FROM games WHERE path=?', (filename,)).fetchone() != (expected,):
                    raise ValueError('source ledger differs from the saved current-block fingerprint')
            digest, count = hashlib.sha256(), 0
            for row in dst.execute('SELECT path, sha256 FROM games ORDER BY path'):
                digest.update(json.dumps(row, ensure_ascii=True).encode() + b'\n')
                count += 1
    return {'pinned_games': count, 'content_sha256': digest.hexdigest(),
            'scope': 'all source pins at snapshot, possibly including later/other-arm pins; untouched corpus is unpinned'}


def validate_inherited_pins(inherited, live, identity, consumed_files=()):
    """A mutable working ledger may grow, but cannot forget inherited history."""
    with closing(sqlite3.connect(Path(live).resolve().as_uri() + '?mode=ro', uri=True)) as db:
        if db.execute('SELECT identity FROM metadata').fetchall() != [(identity,)]:
            raise ValueError('working content ledger has a different identity')
        db.execute('ATTACH DATABASE ? AS inherited', (Path(inherited).resolve().as_uri() + '?mode=ro',))
        mismatch = db.execute('SELECT old.path FROM inherited.games AS old LEFT JOIN main.games AS new '
                              'ON old.path=new.path WHERE new.sha256 IS NULL OR old.sha256!=new.sha256 LIMIT 1').fetchone()
        if mismatch:
            raise ValueError('working ledger lost or changed an inherited content pin: ' + mismatch[0])
        for filename in consumed_files:
            if db.execute('SELECT 1 FROM main.games WHERE path=?', (filename,)).fetchone() is None:
                raise ValueError('working ledger lost a checkpoint-consumed content pin: ' + filename)
