"""Independent save/trend/full clocks on the existing consumed-cursor SL hook."""
import json
from pathlib import Path
import time

from mortal.core.artifacts import atomic_torch_save, atomic_write_json, file_sha256, stable_json_digest
from mortal.supervised.continuation import evaluation_horizons, evaluation_kind
from mortal.supervised.curriculum_probe import CurriculumProbe, capture_rng, learned_state_digest, restore_rng


class ContinuousPhaseProbe(CurriculumProbe):
    def __init__(self, config, domains, *, recipe, seed, output, identity, plan,
                 full_splits, trend_splits, seal_at=None):
        super().__init__(config, domains, recipe=recipe, seed=seed, output=output,
                         identity=identity, horizons=evaluation_horizons(plan), eval_splits=full_splits)
        self.plan, self.trend_splits, self.seal_at = plan, trend_splits, seal_at
        if seal_at is not None:
            self.horizons = sorted(set([*self.horizons, seal_at]))
            self.stop_at = seal_at
        self.done = {'trend': [], 'full': []}
        self.last_saved_update = plan['start_update']
        self.last_saved_time = time.monotonic()

    def restore(self, state):
        super().restore(state)
        saved = state['curriculum_probe'].get('continuous_observation')
        if saved is not None:
            self.done = {name: list(saved['done'][name]) for name in ('trend', 'full')}
        else:
            self.done['full'] = list(self.observed)
        # A loaded checkpoint itself is a completed latest-save boundary.
        self.last_saved_update = state['optimizer_steps']
        self.last_saved_time = time.monotonic()

    def state_dict(self):
        result = super().state_dict()
        result['continuous_observation'] = {'done': {key: list(values) for key, values in self.done.items()},
                                            'last_saved_update': self.last_saved_update}
        return result

    def save(self, update, save_latest, epoch, reason):
        self.last_saved_update = update
        self.last_saved_time = time.monotonic()
        save_latest(epoch, epoch_complete=False, reason=reason)

    def observe(self, optimizer_steps, evaluate, build_state, save_latest, epoch):
        kind = 'full' if optimizer_steps == self.seal_at else evaluation_kind(self.plan, optimizer_steps)
        if kind is None or optimizer_steps in self.done[kind]:
            return
        directory = self.output / kind
        result_path = directory / f'update_{optimizer_steps:09d}.json'
        splits = self.eval_splits if kind == 'full' else self.trend_splits
        split_identity = stable_json_digest(splits)
        if result_path.exists():
            # Crash after durable result but before latest: do not re-evaluate or
            # overwrite a completed observation. Evaluation restored training RNG.
            result = json.loads(result_path.read_text(encoding='utf-8'))
            if (result['identity'], result['optimizer_updates'], result['kind'], result['split_identity']) != (
                    self.identity, optimizer_steps, kind, split_identity):
                raise ValueError('existing continuation observation has different provenance')
            if kind == 'full' and file_sha256(result['checkpoint']) != result['checkpoint_sha256']:
                raise ValueError('archived full-validation checkpoint changed')
            self.done[kind].append(optimizer_steps)
            if kind == 'full' and optimizer_steps not in self.observed:
                self.observed.append(optimizer_steps)
            self.save(optimizer_steps, save_latest, epoch, 'recover_completed_observation')
            return
        self.save(optimizer_steps, save_latest, epoch, 'before_continuation_validation')
        before = capture_rng()
        started = time.monotonic()
        result = {'identity': self.identity, 'optimizer_updates': optimizer_steps, 'kind': kind,
                  'recipe': self.recipe, 'seed': self.seed, 'split_identity': split_identity,
                  'successful_decisions': optimizer_steps * self.config['supervised']['batch_size']
                                          * self.config['control']['opt_step_every'],
                  'splits': {}, 'games': {name: len(files) for name, files in splits.items()}}
        try:
            for name, files in splits.items():
                metrics, batches = evaluate(files, optimizer_steps, desc=f'{kind} {name} U{optimizer_steps}',
                    scalar_prefix=f'continuation/{kind}/{name}', collect_cluster_records=(kind == 'full'))
                if metrics is None or not batches:
                    raise ValueError('empty continuation evaluation: ' + name)
                result['splits'][name] = metrics
        finally:
            restore_rng(before)
        self.done[kind].append(optimizer_steps)
        if kind == 'full':
            self.observed.append(optimizer_steps)
            state = build_state(epoch, epoch_complete=False)
            checkpoint = directory / f'update_{optimizer_steps:09d}.pth'
            # An incomplete earlier archive is recoverable only by the caller
            # explicitly keeping its existing file; never replace it silently.
            if checkpoint.exists():
                raise FileExistsError('unpaired full archive exists; inspect it before retrying')
            atomic_torch_save(state, checkpoint)
            result.update(checkpoint=str(checkpoint), checkpoint_sha256=file_sha256(checkpoint),
                          learned_state_sha256=learned_state_digest(state))
        result.update(evaluation_seconds=time.monotonic() - started,
                      elapsed_seconds=self.elapsed_before + time.monotonic() - self.started,
                      exposure=self.dataset.exposure() if self.dataset else {})
        atomic_write_json(result_path, result)
        self.save(optimizer_steps, save_latest, epoch, 'after_continuation_validation')

    def after_update(self, optimizer_steps, evaluate, build_state, save_latest, epoch):
        kind = evaluation_kind(self.plan, optimizer_steps)
        if kind is not None:
            self.observe(optimizer_steps, evaluate, build_state, save_latest, epoch)
        elif (optimizer_steps - self.last_saved_update >= self.plan['save_updates']
              or time.monotonic() - self.last_saved_time >= self.plan['save_seconds']):
            self.save(optimizer_steps, save_latest, epoch, 'continuation_latest')
        return optimizer_steps >= self.stop_at
