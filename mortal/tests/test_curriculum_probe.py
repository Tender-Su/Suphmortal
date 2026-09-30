from copy import deepcopy
import itertools
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import torch
from torch.utils.data import DataLoader

from mortal.supervised.curriculum_probe import (
    RECIPES, CurriculumProbe, RotatingGameDataset, RotatingGameSampler, make_domains,
    learned_state_digest,
)
from scripts.run_sl_curriculum_probe import (
    build_config, prepare_branch_state, comparisons, freeze_search_parser, observation_schedule,
)


def game(year, number):
    return f'D:/data/{year}/{year}010100gm-00a9-0000-{number:08x}.json'


def fake_rows(filename):
    return [(filename, i) for i in range(7)]


class CurriculumSamplerTests(unittest.TestCase):
    def setUp(self):
        self.domains = make_domains([game(year, i) for year in (2012, 2021, 2023, 2024)
                                     for i in range(6)])

    def test_time_windows_exclude_guards_and_latest_from_replay(self):
        files = [game(year, 1) for year in range(2009, 2027)]
        domains = make_domains(files)
        self.assertEqual(domains['recent'], [game(2023, 1), game(2024, 1)])
        self.assertEqual(domains['latest'], [game(2024, 1)])
        self.assertEqual(domains['mid'], [game(2021, 1)])
        self.assertNotIn(game(2022, 1), list(itertools.chain.from_iterable(domains.values())))

    def test_entire_domain_is_visited_before_repetition(self):
        sampler = RotatingGameSampler(self.domains, {'recent': 1.0}, 41)
        first = [sampler.draw()['file'] for _ in range(12)]
        second = [sampler.draw()['file'] for _ in range(12)]
        self.assertEqual(set(first), set(self.domains['recent']))
        self.assertEqual(set(second), set(first))
        self.assertNotEqual(first, second)

    def test_sampler_reconstructs_exactly_across_cycles(self):
        sampler = RotatingGameSampler(self.domains, RECIPES['A'], 41)
        for _ in range(101):
            sampler.draw()
        state = sampler.state_dict()
        expected = [sampler.draw() for _ in range(100)]
        restored = RotatingGameSampler(self.domains, RECIPES['A'], 41)
        restored.load_state_dict(state)
        self.assertEqual(expected, [restored.draw() for _ in range(100)])

    def test_recipe_is_checked_before_resume(self):
        sampler = RotatingGameSampler(self.domains, RECIPES['A'], 41)
        with self.assertRaisesRegex(ValueError, 'recipe'):
            RotatingGameSampler(self.domains, RECIPES['B'], 41).load_state_dict(sampler.state_dict())

    def test_consumed_cursor_reconstructs_mid_game_and_block_boundary(self):
        for consumed in (1, 28, 29, 117):
            dataset = RotatingGameDataset(self.domains, RECIPES['B'], 42, {}, sample_loader=fake_rows)
            stream = iter(dataset)
            for _ in range(consumed):
                next(stream)
            saved = dataset.state_dict()
            expected = [next(stream) for _ in range(70)]
            restored = RotatingGameDataset(self.domains, RECIPES['B'], 42, {}, sample_loader=fake_rows)
            restored.load_state_dict(saved)
            self.assertEqual(expected, list(itertools.islice(iter(restored), 70)))
            self.assertEqual(dataset.state_dict(), restored.state_dict())
            self.assertEqual(restored.exposure()['decisions'], consumed + 70)
            self.assertEqual(sum(row['decisions'] for row in restored.exposure()['by_year'].values()), consumed + 70)

    def test_dataloader_does_not_count_prefetched_samples(self):
        dataset = RotatingGameDataset(self.domains, RECIPES['C'], 42, {}, sample_loader=fake_rows)
        loader = iter(DataLoader(dataset, batch_size=8, num_workers=0,
                                 generator=torch.Generator().manual_seed(9)))
        next(loader)
        self.assertEqual(dataset.exposure()['decisions'], 8)
        saved = dataset.state_dict()
        expected = next(loader)
        restored = RotatingGameDataset(self.domains, RECIPES['C'], 42, {}, sample_loader=fake_rows)
        restored.load_state_dict(saved)
        actual = next(iter(DataLoader(restored, batch_size=8, num_workers=0,
                                      generator=torch.Generator().manual_seed(9))))
        self.assertEqual(expected[0], actual[0])
        torch.testing.assert_close(expected[1], actual[1], rtol=0, atol=0)

    def test_bulk_preparation_preserves_views_duplicates_and_resume_cursor(self):
        from mortal.data.dataloader import stable_source_game_id

        def loader(**kwargs):
            views = (0, 1) if kwargs['enable_augmentation'] else (0,)
            if kwargs['augmented_first']:
                views = views[::-1]
            for view in views:
                for filename in kwargs['file_list']:
                    for row in range(7):
                        yield (filename, view, row, stable_source_game_id(filename))

        for unique_games in (1, 3, 8):
            domains = {'recent': [game(2024, i) for i in range(unique_games)]}
            for first in (False, True):
                kwargs = {'version': 4, 'enable_augmentation': True, 'augmented_first': first}

                def build(size):
                    return RotatingGameDataset(domains, {'recent': 1.0}, 41, kwargs,
                                               prepare_file_batch_size=size)

                with patch('mortal.data.dataloader.SupervisedFileDatasetsIter', side_effect=loader), \
                     patch('mortal.supervised.curriculum_probe.file_sha256', return_value='fixed'):
                    for size in (2, 4):
                        baseline, bulk = build(1), build(size)
                        left, right = iter(baseline), iter(bulk)
                        self.assertEqual(list(itertools.islice(left, 61)), list(itertools.islice(right, 61)))
                        self.assertEqual(baseline.state_dict(), bulk.state_dict())
                        resumed = build(size)
                        resumed.load_state_dict(baseline.state_dict())
                        self.assertEqual(list(itertools.islice(left, 73)),
                                         list(itertools.islice(iter(resumed), 73)))
                        self.assertEqual(baseline.state_dict(), resumed.state_dict())
                        self.assertEqual(baseline.exposure(), resumed.exposure())

    def test_preparation_cannot_extend_draw_block(self):
        with self.assertRaisesRegex(ValueError, 'four-draw'):
            RotatingGameDataset(self.domains, RECIPES['A'], 1, {}, prepare_file_batch_size=8)


class CurriculumContractTests(unittest.TestCase):
    def test_all_recipes_get_short_horizon_before_any_longer_horizon(self):
        manifest = {'seeds': [1, 2], 'horizons': [10, 40], 'delayed_transfer_routes': ['AC', 'CC']}
        schedule = observation_schedule(manifest)
        self.assertEqual({row[2] for row in schedule[:6]}, {10})
        self.assertEqual({row[2] for row in schedule[6:12]}, {40})
        self.assertEqual({(seed, recipe) for seed, recipe, _ in schedule[:6]},
                         {(seed, recipe) for seed in (1, 2) for recipe in 'ABC'})
        self.assertNotEqual(schedule[0][1], schedule[6][1])

    def test_frozen_search_parser_preserves_default_without_mutating_live_module(self):
        text = '_as_bool(cfg.get("hard_only", True), True)'
        fixed = freeze_search_parser(text)
        self.assertEqual(fixed, '_as_bool(cfg.get("hard_only", True), default=True)')
        self.assertEqual(freeze_search_parser(fixed), fixed)
        from mortal.core.config_utils import coerce_bool
        for value, expected in ((None, True), ('off', False), ('on', True)):
            self.assertEqual(eval(fixed, {'_as_bool': coerce_bool, 'cfg': {'hard_only': value}}), expected)

    def source(self):
        config = {'aux': {'next_rank_weight': 0.001, 'danger_weight': 0.008},
                  'supervised': {'rank_aux': {'base_weight': 0.001}},
                  'control': {}, 'dataset': {}}
        return {'config': config, 'steps': 100, 'optimizer_steps': 99,
                'optimizer': {'param_groups': [{'lr': 5e-6, 'params': [0]}],
                              'state': {0: {'step': torch.tensor(99), 'exp_avg': torch.tensor([0.7])}}},
                'optimizer_param_groups': [('mortal.a',)], 'scaler': {'scale': 128}, 'scheduler': {},
                **{key: {'weight': torch.tensor([1.0])} for key in
                   ('mortal', 'policy_net', 'aux_net', 'opponent_aux_net', 'danger_aux_net')}}

    def test_parent_branch_preserves_moments_heads_scaler_and_aux_clock(self):
        source = self.source()
        config = build_config(source, Path('new'), Path('index'), seed=1, device='cpu',
                              microbatch=64, logical_batch=1024, identity='fixture')
        branch = prepare_branch_state(source, config)
        self.assertEqual(branch['optimizer_steps'], 0)
        self.assertEqual(branch['auxiliary_optimizer_steps'], 99)
        self.assertEqual(branch['scaler'], source['scaler'])
        self.assertEqual(branch['optimizer'], source['optimizer'])
        self.assertEqual(branch['scheduler']['_last_lr'], [5e-6])
        before = learned_state_digest(branch)
        unchanged = deepcopy(branch)
        unchanged['config']['supervised']['state_file'] = 'different-output-path'
        self.assertEqual(before, learned_state_digest(unchanged))
        branch['mortal']['weight'].add_(1)
        self.assertNotEqual(before, learned_state_digest(branch))
        self.assertEqual(source['mortal']['weight'].item(), 1)
        config['aux']['danger_weight'] = 0
        with self.assertRaisesRegex(ValueError, 'auxiliary objective'):
            prepare_branch_state(source, config)

    def test_lr_rewarm_is_rejected_in_data_only_probe(self):
        source = self.source()
        config = build_config(source, Path('new'), Path('index'), seed=1, device='cpu',
                              microbatch=64, logical_batch=1024, identity='fixture')
        config['supervised']['scheduler']['peak'] = 1e-5
        with self.assertRaisesRegex(ValueError, 'preserve current parent LR'):
            prepare_branch_state(source, config)

    def test_runtime_profile_only_changes_bounded_preparation(self):
        source = self.source()
        kwargs = dict(seed=1, device='cpu', microbatch=64, logical_batch=1024, identity='fixture')
        profile = {'probe_prepare_file_batch_size': 4, 'val_file_batch_size': 4, 'rayon_num_threads': 4}
        before = build_config(source, Path('new'), Path('index'), **kwargs)
        after = build_config(source, Path('new'), Path('index'), **kwargs, runtime_performance=profile)
        for name in profile:
            self.assertEqual(after['supervised'][name], 4)
            if name in before['supervised']:
                after['supervised'][name] = before['supervised'][name]
            else:
                after['supervised'].pop(name)
        self.assertEqual(before, after)
        for invalid in ({**profile, 'lr': 1e-4}, {**profile, 'probe_prepare_file_batch_size': 8},
                        {**profile, 'rayon_num_threads': True}):
            with self.assertRaisesRegex(ValueError, 'bounded preparation'):
                build_config(source, Path('new'), Path('index'), **kwargs, runtime_performance=invalid)
        ordered = {**profile, 'probe_prepare_workers': 4, 'val_prepare_workers': 2,
                   'prepare_rayon_threads': 4}
        actual = build_config(source, Path('new'), Path('index'), **kwargs, runtime_performance=ordered)
        self.assertEqual(actual['supervised']['num_workers'], 0)
        self.assertEqual(actual['supervised']['probe_prepare_workers'], 4)
        self.assertEqual(actual['control']['opt_step_every'], 16)
        for name, value in (('probe_prepare_workers', 9), ('val_prepare_workers', True),
                            ('prepare_rayon_threads', 0)):
            with self.assertRaisesRegex(ValueError, 'bounded ordered preparation'):
                build_config(source, Path('new'), Path('index'), **kwargs,
                             runtime_performance={**ordered, name: value})

    def test_observations_count_successful_updates_and_do_not_extend_A(self):
        source = self.source()
        config = build_config(source, Path('new'), Path('index'), seed=1, device='cpu',
                              microbatch=64, logical_batch=1024, identity='fixture')
        with tempfile.TemporaryDirectory() as directory:
            probe = CurriculumProbe(config, {}, recipe='A', seed=1, output=directory,
                                     horizons=[10, 20], eval_splits={}, identity='fixture')
            probe.observe = Mock()
            self.assertFalse(probe.after_update(9, None, None, Mock(), 0))
            probe.observe.assert_not_called()
            self.assertFalse(probe.after_update(10, None, None, Mock(), 0))
            self.assertTrue(probe.after_update(20, None, None, Mock(), 0))
            self.assertEqual(probe.observe.call_count, 2)

    def test_one_game_cannot_qualify_and_mismatched_decisions_fail(self):
        row = {'splits': {name: {'_adaptive_cluster_records': {
            metric: [[1, 0.1, 10]] for metric in ('policy_loss', 'action_accuracy')}}
            for name in ('controller_recent', 'controller_old')}}
        result = comparisons(row, row, 3.4)
        self.assertFalse(result['policy_loss']['inferentially_qualified'])
        changed = deepcopy(row)
        changed['splits']['controller_recent']['_adaptive_cluster_records']['policy_loss'][0][2] = 11
        with self.assertRaisesRegex(ValueError, 'sample counts'):
            comparisons(changed, row, 3.4)


if __name__ == '__main__':
    unittest.main()
