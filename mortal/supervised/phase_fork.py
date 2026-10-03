"""One declared B-to-C branch; existing same-phase and balanced gates stay intact."""
from copy import deepcopy
import json
from pathlib import Path
import time

from mortal.core.artifacts import atomic_torch_save, atomic_write_json, file_sha256, stable_json_digest
from mortal.supervised.continuation import STATE_PATHS
from mortal.supervised.curriculum_probe import CurriculumProbe, learned_state_digest


def fork_config(source, output, identity, parent, commit, runtime_digest, spec):
    provenance = source['supervised']['run_provenance']
    chain = provenance.get('parent_chain', [])
    if (provenance.get('phase') != 'B' or not chain or chain[0].get('phase') != 'A'
            or any(row.get('phase') != 'B' for row in chain[1:]) or parent['phase'] != 'B'):
        raise ValueError('single-parent C fork needs the complete A/B continuation chain')
    if len({row['checkpoint_id'] for row in [*chain, parent]}) != len(chain) + 1:
        raise ValueError('duplicate checkpoint in the parent chain')
    config = deepcopy(source)
    sl = config['supervised']
    for key in STATE_PATHS:
        sl[key] = str(output / (key + '.pth'))
    sl.update(file_index=str(output / 'indexes.pth'), tensorboard_dir=str(output / 'tensorboard'),
              probe_training_content_ledger=str(output / 'training_content.sqlite3'), seed=spec['seed'],
              lr=spec['scheduler']['peak'], scheduler=deepcopy(spec['scheduler']))
    config['optim']['scheduler'] = deepcopy(spec['scheduler'])
    branch = deepcopy(provenance)
    if 'continuation' in branch:
        branch['inherited_continuation'] = branch.pop('continuation')
    branch.update(plan_id=identity, experiment_id=identity, phase='C', source_git_commit=commit,
                  source_runtime_sha256=runtime_digest, parent_checkpoint_id=parent['checkpoint_id'],
                  parent_chain=[*deepcopy(chain), deepcopy(parent)], source_lrs=parent['source_lrs'],
                  phase_scheduler=deepcopy(spec['scheduler']),
                  initialization='experimental_branch_not_exact_parent_resume',
                  reset=['microsteps', 'optimizer_updates', 'scheduler_clock', 'metrics',
                         'controllers', 'data_cursor', 'rng'],
                  preserve=['all_learned_heads', 'adam_moments_and_steps', 'parameter_mapping',
                            'adam_betas_eps_weight_decay', 'amp', 'cumulative_auxiliary_clock'],
                  phase_fork={'source_experiment_id': provenance['experiment_id'],
                              'mode': 'explicit_single_parent_B_to_C', 'automatic_extension': False})
    sl['run_provenance'] = branch
    return config


def validate_baseline(row, source, roles_identity):
    if (row.get('identity') != source['run_provenance']['plan_id']
            or row.get('optimizer_updates') != source['optimizer_steps']
            or row.get('kind') != 'full' or row.get('split_identity') != roles_identity
            or row.get('learned_state_sha256') != learned_state_digest(source)):
        raise ValueError('U0 reuse needs the exact parent learned state and full fixed panel')


class PhaseForkProbe(CurriculumProbe):
    """Use existing full observations with periodic durable cursor saves."""

    def __init__(self, *args, baseline=None, save_updates=500, save_seconds=300, **kwargs):
        super().__init__(*args, reset_branch_rng=True, **kwargs)
        self.baseline = baseline
        self.save_updates, self.save_seconds = save_updates, save_seconds
        self.last_saved_update = 0
        self.last_saved_time = time.monotonic()

    def restore(self, state):
        # Parent creates no C cursor. Once C has saved one, always restore it,
        # including a U0 snapshot, instead of reseeding on each invocation.
        super().restore(state)
        self.last_saved_update = state['optimizer_steps']
        self.last_saved_time = time.monotonic()

    def observe(self, optimizer_steps, evaluate, build_state, save_latest, epoch):
        if optimizer_steps in self.observed:
            return
        result_path = self.output / f'update_{optimizer_steps:07d}.json'
        checkpoint = self.output / f'update_{optimizer_steps:07d}.pth'
        if result_path.exists():
            # As in ContinuousPhaseProbe, JSON commits the completed observation.
            # A crash before latest was saved must not re-evaluate or replace it.
            if not checkpoint.is_file():
                raise FileExistsError('unpaired observation result; inspect before resuming')
            result = json.loads(result_path.read_text(encoding='utf-8'))
            required = {'identity', 'optimizer_updates', 'recipe', 'seed', 'splits', 'exposure',
                        'elapsed_seconds', 'evaluation_seconds', 'checkpoint', 'checkpoint_sha256',
                        'learned_state_sha256', 'successful_decisions', 'skipped_optimizer_steps'}
            if not isinstance(result, dict) or not required.issubset(result):
                raise ValueError('existing phase-fork observation is incomplete')
            if ((result['identity'], result['optimizer_updates'], result['recipe'], result['seed']) !=
                    (self.identity, optimizer_steps, self.recipe, self.seed)
                    or Path(result['checkpoint']).resolve() != checkpoint.resolve()
                    or result.get('kind', 'full') != 'full'
                    or result.get('split_identity', stable_json_digest(self.eval_splits)) !=
                    stable_json_digest(self.eval_splits)):
                raise ValueError('existing phase-fork observation has different provenance')
            if (not isinstance(result['splits'], dict) or set(result['splits']) != set(self.eval_splits)
                    or any(not isinstance(metrics, dict) or not metrics for metrics in result['splits'].values())):
                raise ValueError('existing phase-fork observation has incomplete panels')
            if file_sha256(checkpoint) != result['checkpoint_sha256']:
                raise ValueError('archived phase-fork checkpoint changed')
            if learned_state_digest(build_state(epoch, epoch_complete=False)) != result['learned_state_sha256']:
                raise ValueError('existing phase-fork observation has different learned state')
            self.observed.append(optimizer_steps)
            self.last_saved_update = optimizer_steps
            self.last_saved_time = time.monotonic()
            save_latest(epoch, epoch_complete=False, reason='recover_completed_observation')
            return
        if checkpoint.exists():
            # Preserve partial evidence. No implicit rerun or archive overwrite.
            raise FileExistsError('existing unacknowledged observation; inspect before resuming')
        save_latest(epoch, epoch_complete=False, reason='before_phase_fork_observation')
        if optimizer_steps == 0 and self.baseline is not None:
            self.observed.append(0)
            state = build_state(epoch, epoch_complete=False)
            atomic_torch_save(state, checkpoint)
            result = deepcopy(self.baseline)
            origin = {key: result.get(key) for key in
                      ('identity', 'optimizer_updates', 'checkpoint', 'checkpoint_sha256',
                       'learned_state_sha256', 'evaluation_seconds', 'seed', 'recipe')}
            result.update(identity=self.identity, optimizer_updates=0, recipe=self.recipe, seed=self.seed,
                          exposure={}, successful_decisions=0, skipped_optimizer_steps=0,
                          checkpoint=str(checkpoint), checkpoint_sha256=file_sha256(checkpoint),
                          learned_state_sha256=learned_state_digest(state), evaluation_seconds=0.0,
                          elapsed_seconds=self.elapsed_before + time.monotonic() - self.started,
                          reused_parent_observation=origin)
            atomic_write_json(result_path, result)
            save_latest(epoch, epoch_complete=False, reason='phase_fork_reused_U0')
        else:
            super().observe(optimizer_steps, evaluate, build_state, save_latest, epoch)
        self.last_saved_update = optimizer_steps
        self.last_saved_time = time.monotonic()

    def after_update(self, optimizer_steps, evaluate, build_state, save_latest, epoch):
        if optimizer_steps in self.horizons:
            self.observe(optimizer_steps, evaluate, build_state, save_latest, epoch)
        elif (optimizer_steps - self.last_saved_update >= self.save_updates
              or time.monotonic() - self.last_saved_time >= self.save_seconds):
            save_latest(epoch, epoch_complete=False, reason='phase_fork_cursor')
            self.last_saved_update = optimizer_steps
            self.last_saved_time = time.monotonic()
        return optimizer_steps >= self.stop_at
