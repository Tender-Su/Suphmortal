"""Explicit early/late A→B→C branches, independent of the legacy data-only probe."""
from copy import deepcopy
from contextlib import closing
import math
from pathlib import Path
import sqlite3
import time

from mortal.core.artifacts import stable_json_digest
from mortal.supervised.auxiliary_config import effective_auxiliary_recipe


ARMS = ('early', 'late')
PHASES = ('B', 'C')
PARENTS = {'early': (1_366_000, 1_365_487), 'late': (2_880_000, 2_878_912)}
PARENT_LRS = {'early': 4.91502392e-5, 'late': 5e-6}
LEARNED_KEYS = ('mortal', 'policy_net', 'aux_net', 'opponent_aux_net',
                'danger_aux_net', 'optimizer', 'optimizer_param_groups', 'scaler')


class TrainingContentLedger:
    """Pin bytes at first consumption, shared by all arms/cycles and resumes.

    Never requires an up-front corpus scan. Commits precede consumption; replaying
    work after a crash only checks already pinned inputs. Missing ledgers fail.
    """
    def __init__(self, path, identity, *, create=False):
        self.path, self.identity = Path(path).resolve(), identity
        if create:
            if self.path.exists():
                raise FileExistsError('new content ledger required')
            with closing(sqlite3.connect(self.path)) as db, db:
                db.execute('CREATE TABLE metadata (identity TEXT NOT NULL)')
                db.execute('INSERT INTO metadata VALUES (?)', (identity,))
                db.execute('CREATE TABLE games (path TEXT PRIMARY KEY, sha256 TEXT NOT NULL)')
        self.verify([])

    def verify(self, draws):
        with closing(sqlite3.connect(self.path.as_uri() + '?mode=rw', uri=True)) as db, db:
            if db.execute('SELECT identity FROM metadata').fetchall() != [(self.identity,)]:
                raise ValueError('training content ledger belongs to another experiment')
            for draw in draws:
                filename, digest = str(draw['file']), draw['source_sha256']
                previous = db.execute('SELECT sha256 FROM games WHERE path=?', (filename,)).fetchone()
                if previous is not None and previous[0] != digest:
                    raise ValueError(f'training content changed across arms/cycles: {filename}')
                db.execute('INSERT OR IGNORE INTO games VALUES (?, ?)', (filename, digest))


def validate_corrected_recipe(config):
    recipe = effective_auxiliary_recipe(config)
    required = ((recipe['rank_aux'], 'base_weight', 0.001548),
                (recipe['rank_aux'], 'max_weight', 0.00516),
                (recipe['aux'], 'opponent_state_weight', 0.00135),
                (recipe['aux'], 'danger_weight', 0.00804))
    if any(not math.isclose(float(section.get(key, -1)), value, rel_tol=0, abs_tol=1e-12)
           for section, key, value in required) or recipe['aux'].get('danger_enabled') is not True:
        raise ValueError('early transition requires the inherited corrected auxiliary recipe')
    return recipe


def phase_spec(*, updates, observations, seed, peak, init, warmup):
    if type(updates) is not int or updates <= 0:
        raise ValueError('phase budget must be positive successful optimizer updates')
    points = list(observations)
    if (not points or any(type(n) is not int or n <= 0 for n in points)
            or points != sorted(set(points)) or points[-1] != updates):
        raise ValueError('observation points must increase and end at the explicit phase budget')
    if (not all(math.isfinite(value) for value in (peak, init)) or not peak >= init > 0
            or type(warmup) is not int or not 0 <= warmup <= updates
            or (warmup == 0 and peak != init)):
        raise ValueError('declare finite positive init/peak LR and a warmup within the phase budget')
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('phase seed must be an unsigned 32-bit integer')
    return {'updates': updates, 'observations': points, 'seed': seed,
            'scheduler': {'type': 'constant', 'peak': float(peak), 'init': float(init),
                          'warm_up_steps': warmup}}


def validate_parent(source, *, arm=None):
    validate_corrected_recipe(source['config'])
    if not source.get('checkpoint_id') or source.get('scheduler') is None:
        raise ValueError('parent needs checkpoint identity and scheduler')
    for name in LEARNED_KEYS:
        if source.get(name) is None:
            raise ValueError(f'parent missing learned state: {name}')
    groups = source['optimizer']['param_groups']
    names = source['optimizer_param_groups']
    state = source['optimizer']['state']
    if (not groups or len(groups) != len(names)
            or any(len(group['params']) != len(mapping) for group, mapping in zip(groups, names))):
        raise ValueError('parent Adam parameter group mapping is incomplete')
    ids = [param for group in groups for param in group['params']]
    mapped = [name for group in names for name in group]
    if len(set(ids)) != len(ids) or len(set(mapped)) != len(mapped) or set(ids) != set(state):
        raise ValueError('parent Adam state does not cover the exact parameter mapping')
    if any(not {'step', 'exp_avg', 'exp_avg_sq'} <= set(value) for value in state.values()):
        raise ValueError('parent Adam moments are incomplete')
    if any(not {'lr', 'betas', 'eps', 'weight_decay'} <= set(group) for group in groups):
        raise ValueError('parent Adam hyperparameters are incomplete')
    for name in ('steps', 'optimizer_steps'):
        if type(source[name]) is not int or source[name] < 0:
            raise ValueError(f'invalid parent {name}')
    auxiliary_steps = source.get('auxiliary_optimizer_steps', source['optimizer_steps'])
    if type(auxiliary_steps) is not int or auxiliary_steps < source['optimizer_steps']:
        raise ValueError('parent cumulative auxiliary clock is invalid')
    if arm is not None and ((source['steps'], source['optimizer_steps']) != PARENTS[arm]
                            or len(state) != 414 or any(not math.isclose(
                                group['lr'], PARENT_LRS[arm], rel_tol=0, abs_tol=1e-12) for group in groups)):
        raise ValueError(f'{arm} A parent differs from the declared checkpoint inventory')


def parent_record(source, *, path, sha256, phase):
    return {'phase': phase, 'checkpoint': str(path), 'sha256': sha256,
            'checkpoint_id': source['checkpoint_id'], 'microsteps': source['steps'],
            'optimizer_updates': source['optimizer_steps'],
            'auxiliary_optimizer_steps': source.get('auxiliary_optimizer_steps', source['optimizer_steps']),
            'source_lrs': [float(group['lr']) for group in source['optimizer']['param_groups']]}


def validate_optimizer_mapping(saved, current):
    if tuple(map(tuple, saved)) != tuple(map(tuple, current)):
        raise ValueError('declared phase branch requires the exact Adam parameter mapping; no remap')


def prepare_transition_state(source, config):
    """Keep every learned tensor and Adam clock; reset only declared phase state.

    Unlike prepare_branch_state in the old probe, the explicit phase LR may differ
    from the parent. initial_lr is scheduler bookkeeping, not an Adam moment.
    """
    import torch
    from mortal.supervised.lr_scheduler import LinearWarmUpConstantLR

    validate_parent(source)
    if effective_auxiliary_recipe(source['config']) != validate_corrected_recipe(config):
        raise ValueError('branch auxiliary objective differs from parent')
    sl = config['supervised']
    provenance = sl['run_provenance']
    if provenance.get('branch_mode') != 'preserve_adam_declared_phase_lr':
        raise ValueError('declared phase LR requires explicit branch mode')
    scheduler_cfg = sl['scheduler']
    peak = sl.get('lr', config['optim']['scheduler']['peak'])
    if scheduler_cfg['type'] != 'constant' or peak != scheduler_cfg['peak']:
        raise ValueError('effective trainer LR and declared scheduler must match')
    state = {name: deepcopy(source[name]) for name in LEARNED_KEYS}
    groups = state['optimizer']['param_groups']
    dummy = torch.optim.AdamW([{'params': [torch.nn.Parameter(torch.zeros(1))]} for _ in groups], lr=1)
    scheduler = LinearWarmUpConstantLR(dummy, peak=peak, init=scheduler_cfg['init'],
                                      warm_up_steps=scheduler_cfg['warm_up_steps'])
    for group, new in zip(groups, dummy.param_groups):
        group.update(lr=new['lr'], initial_lr=new['initial_lr'])
    state.update(scheduler=scheduler.state_dict(), config=deepcopy(config),
                 config_section='supervised', checkpoint_id=stable_json_digest(provenance),
                 run_provenance=deepcopy(provenance), steps=0, optimizer_steps=0,
                 skipped_optimizer_steps=0, nonfinite_batches=0,
                 auxiliary_optimizer_steps=source.get('auxiliary_optimizer_steps', source['optimizer_steps']),
                 epoch=0, epoch_complete=False, timestamp=time.time())
    # Metrics/controllers/cursor/RNG are intentionally absent. The trainer uses
    # fresh defaults; the probe initializes RNG after model construction.
    return state


def observation_schedule(phases):
    """Finish both arms at each horizon before extending either; B precedes C."""
    return [(arm, phase, horizon)
            for phase in PHASES
            for index, horizon in enumerate(phases[phase]['observations'])
            for arm in (ARMS if index % 2 == 0 else ARMS[::-1])]
