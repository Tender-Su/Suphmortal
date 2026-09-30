"""Matched curriculum observations, with an explicit consumed-data cursor.

This is an experimental runner contract, not an automatic production selector.
Mixture weights are probabilities of drawing a game, not decision proportions.
"""
from collections import Counter
from copy import deepcopy
import hashlib
import logging
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from mortal.core.artifacts import atomic_torch_save, atomic_write_json, file_sha256


RECIPES = {
    'A': {'recent': 0.60, 'mid': 0.25, 'early': 0.15},
    'B': {'recent': 0.90, 'replay': 0.10},
    'C': {'latest': 0.98, 'replay': 0.02},
}


def derived_seed(seed, *parts):
    text = ':'.join(map(str, (seed, *parts)))
    return int.from_bytes(hashlib.blake2b(text.encode(), digest_size=8).digest(), 'little')


def make_domains(files):
    """Use original game identity for deduplication and calendar assignment."""
    from mortal.data.split_ledger import game_identity
    unique = {}
    for filename in files:
        identity = game_identity(filename)
        if identity in unique and unique[identity] != filename:
            raise ValueError(f'ambiguous duplicate source game: {identity}')
        unique[identity] = filename
    domains = {name: [] for name in ('early', 'mid', 'recent', 'latest', 'replay')}
    for identity, filename in sorted(unique.items()):
        year = int(identity[:4])
        if 2009 <= year <= 2020:
            domains['early'].append(filename)
            domains['replay'].append(filename)
        elif year == 2021:
            domains['mid'].append(filename)
            domains['replay'].append(filename)
        elif year in (2023, 2024):
            domains['recent'].append(filename)
            if year == 2024:
                domains['latest'].append(filename)
    if any(not values for values in domains.values()):
        raise ValueError('all curriculum domains must be nonempty')
    return domains


class RotatingGameSampler:
    """Common random numbers for domain choice; no replacement within a domain.

    Only cycle/position/RNG state is saved. Permutations reconstruct from seed.
    No fixed small pool is selected. Every eligible game has an opportunity.
    """
    def __init__(self, domains, recipe, seed):
        self.domains = domains
        self.recipe = dict(recipe)
        if not recipe or abs(sum(recipe.values()) - 1.0) > 1e-10:
            raise ValueError('mixture weights must sum to one')
        if any(weight <= 0 or not domains.get(name) for name, weight in recipe.items()):
            raise ValueError('positive weights and nonempty domains required')
        self.seed = int(seed)
        self.rng = random.Random(derived_seed(seed, 'domain_choice'))
        self.positions = {name: [0, 0] for name in recipe}
        self.orders = {}
        self.draws = 0

    def draw(self):
        value = self.rng.random()
        chosen = next(reversed(self.recipe))
        for name, weight in self.recipe.items():
            value -= weight
            if value < 0:
                chosen = name
                break
        cycle, position = self.positions[chosen]
        files = self.domains[chosen]
        if position == len(files):
            cycle, position = cycle + 1, 0
        cached = self.orders.get(chosen)
        if cached is None or cached[0] != cycle:
            order = np.random.default_rng(derived_seed(self.seed, chosen, cycle)).permutation(len(files))
            self.orders[chosen] = cycle, order
        filename = files[int(self.orders[chosen][1][position])]
        self.positions[chosen] = [cycle, position + 1]
        draw = self.draws
        self.draws += 1
        return {'file': filename, 'domain': chosen, 'cycle': cycle, 'draw': draw}

    def state_dict(self):
        return {'seed': self.seed, 'recipe': self.recipe, 'positions': deepcopy(self.positions),
                'rng': self.rng.getstate(), 'draws': self.draws}

    def load_state_dict(self, state):
        if state['seed'] != self.seed or state['recipe'] != self.recipe:
            raise ValueError('sampler seed/recipe changed during resume')
        self.positions = deepcopy(state['positions'])
        self.rng.setstate(state['rng'])
        self.draws = state['draws']
        self.orders.clear()


class RotatingGameDataset(IterableDataset):
    """Single-process dataset; cursor advances BEFORE yield, without prefetch.

    Four draws include both views, shuffled together to mix games in updates.
    Resume reparses at most this block, then skips already consumed rows.
    """
    def __init__(self, domains, recipe, seed, loader_kwargs, *, sample_loader=None,
                 prepare_file_batch_size=1):
        super().__init__()
        self.sampler = RotatingGameSampler(domains, recipe, seed)
        self.loader_kwargs = dict(loader_kwargs)
        self.sample_loader = sample_loader
        if prepare_file_batch_size not in (1, 2, 4):
            raise ValueError('probe preparation must stay within its four-draw block')
        self.prepare_file_batch_size = prepare_file_batch_size
        self.current = None
        self.offset = 0
        self.consumed = Counter()
        self.files = {}

    def load_samples(self, filename):
        if self.sample_loader is not None:
            return self.sample_loader(filename)
        from mortal.data.dataloader import SupervisedFileDatasetsIter
        dataset = SupervisedFileDatasetsIter(
            file_list=[filename], file_batch_size=1, reserve_ratio=0,
            shuffle_files=False, num_epochs=1, emit_game_id=True,
            **self.loader_kwargs,
        )
        return list(dataset)

    def load_sample_block(self, draws):
        if self.prepare_file_batch_size == 1 or self.sample_loader is not None:
            return [self.load_samples(draw['file']) for draw in draws]
        from mortal.data.dataloader import SupervisedFileDatasetsIter, stable_source_game_id

        filenames = list(dict.fromkeys(draw['file'] for draw in draws))
        by_game = {stable_source_game_id(filename): [] for filename in filenames}
        if len(by_game) != len(filenames):
            raise ValueError('source game ID collision in preparation block')
        dataset = SupervisedFileDatasetsIter(
            file_list=filenames, file_batch_size=self.prepare_file_batch_size, reserve_ratio=0,
            shuffle_files=False, num_epochs=1, emit_game_id=True, **self.loader_kwargs,
        )
        # Bulk iteration groups augmentation by view. Regroup by game before the
        # existing row permutation, retaining original-then-augmented row order.
        for sample in dataset:
            by_game[int(sample[-1])].append(sample)
        return [by_game[stable_source_game_id(draw['file'])] for draw in draws]

    def __iter__(self):
        if get_worker_info() is not None:
            raise RuntimeError('probe cursor requires num_workers=0')
        from mortal.data.dataloader import stable_source_game_id
        while True:
            if self.current is None:
                self.current = {'draws': [self.sampler.draw() for _ in range(4)]}
                self.offset = 0
            samples = []
            for draw in self.current['draws']:
                filename = draw['file']
                if self.sample_loader is None:
                    digest = file_sha256(filename)
                    if draw.get('source_sha256', digest) != digest:
                        raise ValueError('current probe game changed since checkpoint')
                    draw['source_sha256'] = digest
            prepared = self.load_sample_block(self.current['draws'])
            for draw, game_samples in zip(self.current['draws'], prepared):
                filename = draw['file']
                game_id = stable_source_game_id(filename)
                if not game_samples:
                    raise ValueError(f'empty game in probe input: {filename}')
                self.files[game_id] = {'file': filename, 'available_decisions_per_draw': len(game_samples)}
                samples.extend((game_id, sample) for sample in game_samples)
                del game_samples
            del prepared
            order = np.random.default_rng(derived_seed(
                self.sampler.seed, 'rows', self.current['draws'][0]['draw'],
            )).permutation(len(samples))
            if not 0 <= self.offset <= len(samples):
                raise ValueError('probe sample cursor exceeds reconstructed game')
            while self.offset < len(samples):
                game_id, sample = samples[int(order[self.offset])]
                self.offset += 1
                self.consumed[game_id] += 1
                yield sample
            self.current = None
            self.offset = 0
            del samples

    def state_dict(self):
        return {'sampler': self.sampler.state_dict(), 'current': deepcopy(self.current),
                'offset': self.offset, 'consumed': dict(self.consumed), 'files': deepcopy(self.files)}

    def load_state_dict(self, state):
        self.sampler.load_state_dict(state['sampler'])
        self.current = deepcopy(state['current'])
        self.offset = state['offset']
        self.consumed = Counter(state['consumed'])
        self.files = deepcopy(state['files'])

    def exposure(self):
        from mortal.data.split_ledger import game_identity
        years = {}
        for game_id, count in self.consumed.items():
            row = self.files[game_id]
            year = game_identity(row['file'])[:4]
            totals = years.setdefault(year, {'decisions': 0, 'unique_games': 0,
                                            'decisions_beyond_one_augmented_draw': 0})
            totals['decisions'] += count
            totals['unique_games'] += 1
            totals['decisions_beyond_one_augmented_draw'] += max(
                0, count - row['available_decisions_per_draw'],
            )
        return {'by_year': years, 'decisions': sum(self.consumed.values()),
                'unique_games': len(self.consumed), 'draws': self.sampler.draws,
                'includes_skipped_update_inputs': True,
                'augmentation': 'both_views_per_draw',
                'shuffled_games_per_block': 4,
                'sampling_unit': 'game', 'sampler_positions': deepcopy(self.sampler.positions)}


def capture_rng():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []}


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda']:
        torch.cuda.set_rng_state_all([value.cpu() for value in state['cuda']])


def learned_state_digest(state):
    """Hash learned state independent of paths, UUIDs and serialization storage ids."""
    digest = hashlib.sha256()

    def add(value):
        if torch.is_tensor(value):
            add(('tensor', str(value.dtype), tuple(value.shape)))
            digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(value, dict):
            digest.update(b'{')
            for key in sorted(value, key=repr):
                add(key)
                add(value[key])
            digest.update(b'}')
        elif isinstance(value, (list, tuple)):
            digest.update(b'[')
            for item in value:
                add(item)
            digest.update(b']')
        else:
            data = repr(value).encode()
            digest.update(len(data).to_bytes(8, 'little'))
            digest.update(data)

    for name in ('mortal', 'policy_net', 'aux_net', 'opponent_aux_net', 'danger_aux_net',
                 'optimizer', 'optimizer_param_groups', 'scheduler', 'scaler',
                 'auxiliary_optimizer_steps'):
        add(name)
        add(state[name])
    return digest.hexdigest()


class CurriculumProbe:
    def __init__(self, config, domains, *, recipe, seed, output, horizons, eval_splits, identity):
        self.config = config
        self.domains = domains
        self.recipe = recipe
        self.seed = seed
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.horizons = list(horizons)
        if not self.horizons or self.horizons != sorted(set(self.horizons)) or self.horizons[0] <= 0:
            raise ValueError('probe horizons must be increasing positive optimizer updates')
        self.stop_at = self.horizons[-1]
        control, sl = config['control'], config['supervised']
        if sl['num_workers'] != 0 or control['enable_cuda_prefetch']:
            raise ValueError('exact probe cursor requires no workers or CUDA prefetch')
        if sl['max_steps'] or sl['val_every_steps'] or sl['save_every']:
            raise ValueError('probe hooks own observation and stop points')
        if any(sl.get(name, {}).get('enabled', False) for name in ('adaptive_curriculum', 'convergence')):
            raise ValueError('historical curriculum controllers must be disabled for matched probes')
        self.eval_splits = eval_splits
        self.identity = identity
        self.dataset = None
        self.pending_dataset = None
        self.observed = []
        self.started = time.monotonic()
        self.elapsed_before = 0.0

    def restore(self, state):
        saved = state.get('curriculum_probe')
        if saved is None:
            if state['optimizer_steps'] != 0:
                raise ValueError('a nonzero probe resume needs its consumed-data cursor')
            return
        if saved['identity'] != self.identity:
            raise ValueError('probe contract changed during resume')
        self.pending_dataset = saved['dataset']
        self.observed = list(saved['observed'])
        self.elapsed_before = saved['elapsed_seconds']
        restore_rng(saved['rng'])

    def build_dataset(self, loader_kwargs):
        if self.dataset is None:
            sl = self.config['supervised']
            dataset_class, preparation = RotatingGameDataset, {}
            if sl.get('probe_prepare_workers', 0):
                from mortal.supervised.ordered_preparation import OrderedRotatingGameDataset
                dataset_class = OrderedRotatingGameDataset
                preparation = {'prepare_workers': sl['probe_prepare_workers'],
                               'prepare_rayon_threads': sl.get('prepare_rayon_threads', 4)}
            self.dataset = dataset_class(
                self.domains, RECIPES[self.recipe], self.seed, loader_kwargs,
                prepare_file_batch_size=sl.get('probe_prepare_file_batch_size', 1), **preparation,
            )
            if self.pending_dataset is not None:
                self.dataset.load_state_dict(self.pending_dataset)
        return self.dataset

    def build_validation_dataset(self, dataset):
        sl = self.config['supervised']
        if not sl.get('val_prepare_workers', 0):
            return dataset
        from mortal.supervised.ordered_preparation import OrderedValidationDataset
        return OrderedValidationDataset(dataset, prepare_workers=sl['val_prepare_workers'],
            prepare_file_batch_size=sl.get('val_file_batch_size', 4),
            prepare_rayon_threads=sl.get('prepare_rayon_threads', 4))

    def state_dict(self):
        return {'identity': self.identity,
                'dataset': self.dataset.state_dict() if self.dataset else self.pending_dataset,
                'observed': list(self.observed), 'rng': capture_rng(),
                'elapsed_seconds': self.elapsed_before + time.monotonic() - self.started}

    def observe(self, optimizer_steps, evaluate, build_state, save_latest, epoch):
        if optimizer_steps in self.observed:
            return
        saved_rng = capture_rng()
        evaluation_started = time.monotonic()
        result = {'identity': self.identity, 'optimizer_updates': optimizer_steps,
                  'recipe': self.recipe, 'seed': self.seed, 'splits': {},
                  'exposure': self.dataset.exposure() if self.dataset else {},
                  'elapsed_seconds': self.elapsed_before + time.monotonic() - self.started}
        try:
            for name, files in self.eval_splits.items():
                metrics, batches = evaluate(
                    files, optimizer_steps, desc=f'PROBE {name} U{optimizer_steps}',
                    scalar_prefix=f'probe/{name}', collect_cluster_records=True,
                )
                if metrics is None or not batches:
                    raise ValueError(f'empty probe evaluation: {name}')
                result['splits'][name] = metrics
        finally:
            restore_rng(saved_rng)
        result['evaluation_seconds'] = time.monotonic() - evaluation_started
        self.observed.append(optimizer_steps)
        state = build_state(epoch, epoch_complete=False)
        checkpoint = self.output / f'update_{optimizer_steps:07d}.pth'
        atomic_torch_save(state, checkpoint)
        result.update(checkpoint=str(checkpoint), checkpoint_sha256=file_sha256(checkpoint),
                      learned_state_sha256=learned_state_digest(state),
                      successful_decisions=optimizer_steps * self.config['supervised']['batch_size']
                      * self.config['control']['opt_step_every'],
                      skipped_optimizer_steps=state['skipped_optimizer_steps'])
        if torch.cuda.is_initialized():
            result['peak_cuda_allocated_bytes'] = torch.cuda.max_memory_allocated()
        result['elapsed_seconds'] = self.elapsed_before + time.monotonic() - self.started
        atomic_write_json(self.output / f'update_{optimizer_steps:07d}.json', result)
        save_latest(epoch, epoch_complete=False, reason='probe_observation')
        logging.info('probe observation complete: recipe=%s updates=%s', self.recipe, optimizer_steps)

    def after_update(self, optimizer_steps, evaluate, build_state, save_latest, epoch):
        if optimizer_steps in self.horizons:
            self.observe(optimizer_steps, evaluate, build_state, save_latest, epoch)
        elif optimizer_steps > 0 and optimizer_steps % 64 == 0:
            save_latest(epoch, epoch_complete=False, reason='probe_cursor')
        return optimizer_steps >= self.stop_at
