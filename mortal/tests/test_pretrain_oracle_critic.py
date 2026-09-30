import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from libriichi.consts import obs_shape

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mortal.core.model import Brain, HLGaussValueHead, OracleDualTowerBrain, ValueHead
from mortal.online.pretrain_oracle_critic import (
    ORACLE_EXTERNAL_PAUSE_ENV,
    ORACLE_EXTERNAL_PAUSE_EXIT_CODE,
    binned_calibration_metrics,
    advance_data_progress_cycle,
    batch_metrics,
    build_file_splits,
    configure_trainable_parameters,
    external_pause_due,
    external_pause_requested,
    initial_data_progress,
    first_conv_module,
    finalize_metrics,
    grad_scaler_step_succeeded,
    in_training_eval_config,
    init_dual_tower_from_visible_state,
    maybe_init_from_checkpoint,
    normalize_critic_arch,
    normalize_eval_input_modes,
    normalize_game_id_subset,
    normalize_oracle_fusion_mode,
    normalize_oracle_tower_init,
    normalize_target_output_weights,
    normalize_train_scope,
    normalize_value_loss_mode,
    output_weighted_mse,
    optimizer_param_groups,
    resolve_convergence_config,
    resolve_external_pause_file,
    save_checkpoint,
    shutdown_data_loader_iterator,
    summarize_oracle_dependency,
    summarize_regression_slices,
    target_output_weights_at_step,
    training_contract,
    transform_eval_invisible_obs,
    update_data_progress,
    validate_convergence_schedule,
    validate_resume_data_stream,
    validate_resume_training_contract,
)


def small_oracle_brain():
    return Brain(
        version=4,
        is_oracle=True,
        conv_channels=32,
        num_blocks=4,
        Norm='GN',
    )


class OracleCriticPretrainScopeTests(unittest.TestCase):
    def test_external_pause_request_resolves_from_environment(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            pause_file = Path(tmp_dir) / 'pause.request'
            pause_file.touch()
            with patch.dict(
                os.environ,
                {ORACLE_EXTERNAL_PAUSE_ENV: str(pause_file)},
            ):
                resolved = resolve_external_pause_file(
                    {'external_pause_file': 'ignored.request'}
                )

            self.assertEqual(str(pause_file), resolved)
            self.assertTrue(external_pause_requested(resolved))
            self.assertTrue(external_pause_due(resolved, 99, 100))
            self.assertFalse(external_pause_due(resolved, 100, 100))
            self.assertEqual(75, ORACLE_EXTERNAL_PAUSE_EXIT_CODE)

    def test_in_training_eval_defaults_to_single_process(self):
        self.assertEqual(
            0,
            in_training_eval_config({'val_num_workers': 4})['val_num_workers'],
        )

    def test_shutdown_data_loader_iterator_clears_persistent_reference(self):
        class FakeIterator:
            def __init__(self):
                self.shutdown = False

            def _shutdown_workers(self):
                self.shutdown = True

        iterator = FakeIterator()
        loader = type('FakeLoader', (), {'_iterator': iterator})()

        self.assertTrue(shutdown_data_loader_iterator(loader, iterator))
        self.assertTrue(iterator.shutdown)
        self.assertIsNone(loader._iterator)
        self.assertEqual(
            1,
            in_training_eval_config({
                'val_num_workers': 4,
                'in_training_val_num_workers': 1,
            })['val_num_workers'],
        )

    def test_training_contract_tracks_explicit_amp_scaler_settings(self):
        scheduler = {
            'init': 1e-8,
            'peak': 2.5e-5,
            'final': 1e-5,
            'warm_up_steps': 2000,
            'max_steps': 2_500_000,
        }
        legacy = training_contract({}, scheduler)
        explicit = training_contract(
            {
                'amp_init_scale': 2048.0,
                'amp_growth_interval': 1_000_000,
            },
            scheduler,
        )

        self.assertNotIn('amp', legacy)
        self.assertEqual(
            {'init_scale': 2048.0, 'growth_interval': 1_000_000},
            explicit['amp'],
        )
        self.assertTrue(legacy['release_train_loader_for_eval'])

    def test_training_contract_records_film_architecture(self):
        contract = training_contract(
            {
                'critic_arch': 'dual_tower',
                'oracle_fusion_mode': 'film',
                'oracle_fusion_hidden': 384,
            },
            {},
        )

        self.assertEqual('film', contract['oracle_fusion_mode'])
        self.assertEqual(384, contract['oracle_fusion_hidden'])

    def test_training_contract_records_only_explicit_output_weights(self):
        legacy = training_contract({}, {})
        explicit_none = training_contract({'target_output_weights': None}, {})
        weighted = training_contract(
            {'target_output_weights': [2.0, 1.0, 1.0, 1.0]},
            {},
        )

        self.assertNotIn('target_output_weights', legacy)
        self.assertNotIn('target_output_weights', explicit_none)
        self.assertEqual([1.6, 0.8, 0.8, 0.8], weighted['target_output_weights'])

    def test_training_contract_records_value_head_and_weight_schedule(self):
        contract = training_contract(
            {
                'value_loss_mode': 'hl_gauss',
                'value_head_hidden': 1024,
                'value_num_bins': 100,
                'value_target_min': -6.0,
                'value_target_max': 6.0,
                'value_sigma_to_bin_ratio': 2.0,
                'value_padding_sigma': 3.0,
                'target_output_weights_initial': [1.0, 1.0, 1.0, 1.0],
                'target_output_weights': [4.0, 1.0, 1.0, 1.0],
                'target_output_weight_ramp_start_steps': 15_000,
                'target_output_weight_ramp_end_steps': 20_000,
            },
            {},
        )

        self.assertEqual('hl_gauss', contract['value_loss_mode'])
        self.assertEqual(1024, contract['value_head_hidden'])
        self.assertEqual(100, contract['value_distribution']['num_bins'])
        self.assertEqual(
            {
                'initial': [1.0, 1.0, 1.0, 1.0],
                'ramp_start_steps': 15_000,
                'ramp_end_steps': 20_000,
            },
            contract['target_output_weight_schedule'],
        )

    def test_training_contract_preserves_zero_lr_scales(self):
        contract = training_contract(
            {
                'encoder_lr_scale': 0.0,
                'visible_lr_scale': 0.0,
                'oracle_lr_scale': 0.0,
                'fusion_lr_scale': 0.0,
                'value_lr_scale': 0.0,
            },
            {},
        )

        for key in (
            'encoder_lr_scale',
            'visible_lr_scale',
            'oracle_lr_scale',
            'fusion_lr_scale',
            'value_lr_scale',
        ):
            self.assertEqual(0.0, contract[key])

    def test_training_contract_records_global_recipe_semantics(self):
        contract = training_contract(
            {
                'optimizer': {'type': 'adamw'},
                'val_game_id_modulus': 5,
                'val_game_id_remainders': [0],
            },
            {
                'type': 'wsd',
                'init': 1e-8,
                'peak': 5e-5,
                'final': 1e-6,
                'warm_up_steps': 2000,
                'stable_steps': 800000,
                'decay_steps': 200000,
                'decay_style': 'linear',
            },
        )

        self.assertEqual('adamw', contract['optimizer']['type'])
        self.assertEqual('wsd', contract['scheduler']['type'])
        self.assertEqual(
            {'game_id_modulus': 5, 'game_id_remainders': [0]},
            contract['validation_subset'],
        )

    def test_game_id_subset_is_normalized_and_validated(self):
        self.assertEqual((5, (0, 2)), normalize_game_id_subset(5, [2, 0, 2]))
        self.assertEqual((3, (0, 1, 2)), normalize_game_id_subset(3, []))
        with self.assertRaisesRegex(ValueError, 'remainders'):
            normalize_game_id_subset(5, [5])

    def test_convergence_contract_is_normalized_and_allows_post_core_tail(self):
        convergence = resolve_convergence_config({
            'convergence': {
                'enabled': True,
                'core_optimizer_steps': 100,
                'tail_lr_levels': [1e-3, 5e-4],
                'metric': 'primary_loss',
            }
        })
        scheduler = {
            'init': 1e-8,
            'peak': 1e-2,
            'final': 1e-3,
            'warm_up_steps': 10,
            'max_steps': 100,
        }

        validate_convergence_schedule(
            convergence,
            scheduler_cfg=scheduler,
            scheduler_horizon_steps=100,
            max_steps=200,
        )
        contract = training_contract({}, scheduler, convergence)

        self.assertEqual('primary_loss', contract['convergence']['metric'])
        self.assertEqual([1e-3, 5e-4], contract['convergence']['tail_lr_levels'])

    def test_convergence_contract_only_migrates_matching_pre_core_checkpoint(self):
        convergence = resolve_convergence_config({
            'convergence': {
                'enabled': True,
                'core_optimizer_steps': 100,
                'tail_lr_levels': [1e-3, 5e-4],
                'metric': 'primary_loss',
            }
        })
        scheduler = {
            'init': 1e-8,
            'peak': 1e-2,
            'final': 1e-3,
            'warm_up_steps': 10,
            'max_steps': 100,
        }
        expected = training_contract({}, scheduler, convergence)
        legacy = dict(expected)
        legacy.pop('convergence')

        self.assertTrue(validate_resume_training_contract(
            {'steps': 99, 'training_contract': legacy},
            expected,
            convergence_config=convergence,
        ))
        with self.assertRaisesRegex(ValueError, 'contract mismatch'):
            validate_resume_training_contract(
                {'steps': 100, 'training_contract': legacy},
                expected,
                convergence_config=convergence,
            )

    def test_atomic_checkpoint_preserves_previous_file_on_save_failure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            destination = Path(tmpdir) / 'latest.pth'
            destination.write_bytes(b'previous')
            with patch(
                'mortal.online.pretrain_oracle_critic.torch.save',
                side_effect=RuntimeError('injected failure'),
            ):
                with self.assertRaisesRegex(RuntimeError, 'injected failure'):
                    save_checkpoint(str(destination), {'steps': 1})

            self.assertEqual(b'previous', destination.read_bytes())
            self.assertEqual([], list(destination.parent.glob('*.tmp')))

    def test_normalize_train_scope_accepts_tail_aliases(self):
        self.assertEqual('oracle_tail', normalize_train_scope('tail'))
        self.assertEqual('oracle_tail', normalize_train_scope('oracle_input_tail_value'))

    def test_normalize_critic_arch_accepts_dual_tower_alias(self):
        self.assertEqual('dual_tower', normalize_critic_arch('dual'))
        self.assertEqual('single_tower', normalize_critic_arch('bridge'))

    def test_normalize_oracle_tower_init_accepts_explicit_modes(self):
        self.assertEqual('visible_transfer', normalize_oracle_tower_init('transfer'))
        self.assertEqual('random', normalize_oracle_tower_init('native_random'))

    def test_normalize_oracle_fusion_mode_accepts_conditioning_alias(self):
        self.assertEqual('linear', normalize_oracle_fusion_mode('late_linear'))
        self.assertEqual('film', normalize_oracle_fusion_mode('conditioned'))
        self.assertEqual('residual_mlp', normalize_oracle_fusion_mode('residual'))

    def test_target_output_weights_are_mean_normalized_and_validated(self):
        self.assertEqual(
            (1.6, 0.8, 0.8, 0.8),
            normalize_target_output_weights([2.0, 1.0, 1.0, 1.0]),
        )
        with self.assertRaisesRegex(ValueError, 'must contain 4 values'):
            normalize_target_output_weights([1.0, 1.0])
        with self.assertRaisesRegex(ValueError, 'finite and non-negative'):
            normalize_target_output_weights([1.0, -1.0, 1.0, 1.0])
        with self.assertRaisesRegex(ValueError, 'at least one positive'):
            normalize_target_output_weights([0.0, 0.0, 0.0, 0.0])

    def test_target_output_weight_schedule_interpolates_normalized_endpoints(self):
        cfg = {
            'target_output_weights_initial': [1.0, 1.0, 1.0, 1.0],
            'target_output_weights': [4.0, 1.0, 1.0, 1.0],
            'target_output_weight_ramp_start_steps': 15_000,
            'target_output_weight_ramp_end_steps': 20_000,
        }
        final = normalize_target_output_weights([4.0, 1.0, 1.0, 1.0])

        self.assertEqual((1.0, 1.0, 1.0, 1.0), target_output_weights_at_step(cfg, 14_999))
        self.assertEqual((1.0, 1.0, 1.0, 1.0), target_output_weights_at_step(cfg, 15_000))
        midpoint = target_output_weights_at_step(cfg, 17_500)
        self.assertEqual(tuple((1.0 + value) * 0.5 for value in final), midpoint)
        self.assertEqual(final, target_output_weights_at_step(cfg, 20_000))

    def test_target_output_weight_schedule_validates_bounds(self):
        with self.assertRaisesRegex(ValueError, 'require both'):
            target_output_weights_at_step(
                {'target_output_weights_initial': [1.0] * 4},
                0,
            )
        with self.assertRaisesRegex(ValueError, 'start_steps <= end_steps'):
            target_output_weights_at_step(
                {
                    'target_output_weights_initial': [1.0] * 4,
                    'target_output_weights': [2.0, 1.0, 1.0, 1.0],
                    'target_output_weight_ramp_start_steps': 10,
                    'target_output_weight_ramp_end_steps': 9,
                },
                0,
            )

    def test_output_weighted_mse_keeps_output_mean_scale(self):
        pred = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        target = torch.zeros_like(pred)
        weights = normalize_target_output_weights([2.0, 1.0, 1.0, 1.0])

        actual = output_weighted_mse(pred, target, weights)

        self.assertAlmostEqual(
            (1.6 + 3.2 + 7.2 + 12.8) / 4.0,
            actual.item(),
            places=6,
        )

    def test_bridge_scope_trains_only_oracle_input_slice_and_value_head(self):
        brain = small_oracle_brain()
        value_net = ValueHead(num_players=4)

        info = configure_trainable_parameters(
            brain,
            value_net,
            scope='oracle_input_value',
            version=4,
        )

        first_conv = first_conv_module(brain)
        visible_channels = obs_shape(4)[0]
        self.assertTrue(first_conv.weight.requires_grad)
        self.assertFalse(brain.encoder.net[1].res_unit[2].weight.requires_grad)
        self.assertIsNotNone(info['first_conv_mask'])
        self.assertEqual(0, int(info['first_conv_mask'][:, :visible_channels, :].sum().item()))
        self.assertGreater(int(info['first_conv_mask'][:, visible_channels:, :].sum().item()), 0)
        self.assertTrue(all(param.requires_grad for param in value_net.parameters()))

    def test_oracle_tail_scope_unfreezes_tail_blocks_and_encoder_output(self):
        brain = small_oracle_brain()
        value_net = ValueHead(num_players=4)

        info = configure_trainable_parameters(
            brain,
            value_net,
            scope='oracle_tail',
            version=4,
            tail_blocks=2,
        )

        self.assertEqual('oracle_tail', info['scope'])
        self.assertEqual(2, info['tail_blocks'])
        self.assertFalse(brain.encoder.net[1].res_unit[2].weight.requires_grad)
        self.assertFalse(brain.encoder.net[2].res_unit[2].weight.requires_grad)
        self.assertTrue(brain.encoder.net[3].res_unit[2].weight.requires_grad)
        self.assertTrue(brain.encoder.net[4].res_unit[2].weight.requires_grad)
        self.assertTrue(brain.encoder.net[-1].weight.requires_grad)
        self.assertLess(
            info['trainable_params'],
            sum(param.numel() for param in brain.parameters()) + sum(param.numel() for param in value_net.parameters()),
        )

    def test_tail_optimizer_uses_scaled_lr_for_unfrozen_encoder_tail(self):
        brain = small_oracle_brain()
        value_net = ValueHead(num_players=4)
        configure_trainable_parameters(
            brain,
            value_net,
            scope='oracle_tail',
            version=4,
            tail_blocks=2,
        )

        groups = optimizer_param_groups(
            brain,
            value_net,
            scope='oracle_tail',
            encoder_lr_scale=0.2,
        )

        lrs = [group.get('lr', 1.0) for group in groups]
        self.assertIn(0.2, lrs)
        self.assertIn(1.0, lrs)
        grouped = {id(param) for group in groups for param in group['params']}
        trainable = {id(param) for param in (*brain.parameters(), *value_net.parameters()) if param.requires_grad}
        self.assertEqual(trainable, grouped)

    def test_dual_tower_brain_keeps_oracle_critic_forward_contract(self):
        brain = OracleDualTowerBrain(
            version=4,
            conv_channels=32,
            num_blocks=2,
            Norm='GN',
        )
        obs = torch.randn(3, obs_shape(4)[0], 34)
        invisible_obs = torch.randn(3, 217, 34)

        phi = brain(obs, invisible_obs=invisible_obs)

        self.assertEqual((3, 1024), tuple(phi.shape))

    def test_dual_tower_optimizer_can_scale_visible_oracle_and_fusion_separately(self):
        brain = OracleDualTowerBrain(
            version=4,
            conv_channels=32,
            num_blocks=2,
            Norm='GN',
        )
        value_net = ValueHead(num_players=4)

        groups = optimizer_param_groups(
            brain,
            value_net,
            scope='all',
            visible_lr_scale=0.25,
            oracle_lr_scale=1.0,
            fusion_lr_scale=0.5,
            value_lr_scale=0.75,
            weight_decay=0.03,
        )

        by_name = {group['name']: group for group in groups}
        self.assertEqual(0.25, by_name['visible_encoder_decay']['lr'])
        self.assertEqual(1.0, by_name['oracle_encoder_decay']['lr'])
        self.assertEqual(0.5, by_name['fusion_decay']['lr'])
        self.assertEqual(0.75, by_name['value_decay']['lr'])
        self.assertEqual(0.03, by_name['fusion_decay']['weight_decay'])
        grouped = [id(param) for group in groups for param in group['params']]
        self.assertEqual(len(grouped), len(set(grouped)))
        self.assertEqual(
            {id(param) for param in (*brain.parameters(), *value_net.parameters())},
            set(grouped),
        )

    def test_three_way_file_split_is_disjoint_and_deterministic(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            index_file = Path(tmpdir) / 'files.pth'
            torch.save({'file_list': [f'{idx}.json' for idx in range(100)]}, index_file)
            cfg = {
                'file_index': str(index_file),
                'seed': 7,
                'val_ratio': 0.1,
                'test_ratio': 0.2,
                'min_val_files': 0,
                'min_test_files': 0,
            }

            first = build_file_splits(cfg)
            second = build_file_splits(cfg)

        self.assertEqual(first, second)
        train_files, dev_files, test_files = first
        self.assertEqual((70, 10, 20), tuple(map(len, first)))
        self.assertFalse(set(train_files) & set(dev_files))
        self.assertFalse(set(train_files) & set(test_files))
        self.assertFalse(set(dev_files) & set(test_files))

    def test_explicit_file_indexes_preserve_declared_splits(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            indexes = {}
            expected = {
                'train': ['cache-0.events.zst', 'cache-1.events.zst'],
                'dev': ['dev-0.json', 'dev-1.json'],
                'test': ['test-0.json'],
            }
            for split, file_list in expected.items():
                index_file = root / f'{split}.pth'
                torch.save({'file_list': file_list}, index_file)
                indexes[f'{split}_file_index'] = str(index_file)

            actual = build_file_splits(indexes)

        self.assertEqual(
            (expected['train'], expected['dev'], expected['test']),
            actual,
        )

    def test_explicit_file_indexes_require_train_and_dev(self):
        with self.assertRaisesRegex(ValueError, 'require train_file_index'):
            build_file_splits({'train_file_index': 'train.pth'})

    def test_data_progress_tracks_safe_worker_cursors_and_rejects_signature_drift(self):
        signature = {'batch_size': 8, 'num_workers': 2}
        progress = initial_data_progress(signature)

        update_data_progress(
            progress,
            torch.tensor([
                [0, 0, 12],
                [0, 0, 18],
                [1, 0, 6],
            ]),
        )

        self.assertEqual([0, 12], progress['resume_cursors'][0])
        self.assertEqual([0, 6], progress['resume_cursors'][1])
        self.assertEqual(3, progress['samples_consumed'])
        self.assertEqual(progress, validate_resume_data_stream({'data_progress': progress}, signature))
        with self.assertRaisesRegex(ValueError, 'data stream mismatch'):
            validate_resume_data_stream(
                {'data_progress': progress},
                {'batch_size': 16, 'num_workers': 2},
            )

    def test_data_progress_cycle_advance_clears_resume_cursors(self):
        progress = initial_data_progress({'state_fold_count': 4})
        progress['resume_cursors'] = {0: [0, 12], 1: [0, 8]}

        returned = advance_data_progress_cycle(progress)

        self.assertIs(progress, returned)
        self.assertEqual(1, progress['cycle'])
        self.assertEqual({}, progress['resume_cursors'])

    def test_dual_tower_requires_all_scope(self):
        brain = OracleDualTowerBrain(
            version=4,
            conv_channels=32,
            num_blocks=2,
            Norm='GN',
        )
        value_net = ValueHead(num_players=4)

        with self.assertRaisesRegex(ValueError, "requires train_scope='all'"):
            configure_trainable_parameters(
                brain,
                value_net,
                scope='oracle_tail',
                version=4,
                tail_blocks=1,
            )

    def test_dual_tower_can_initialize_from_visible_brain_state(self):
        visible = Brain(
            version=4,
            is_oracle=False,
            conv_channels=32,
            num_blocks=2,
            Norm='GN',
        )
        dual = OracleDualTowerBrain(
            version=4,
            conv_channels=32,
            num_blocks=2,
            Norm='GN',
        )

        info = init_dual_tower_from_visible_state(dual, visible.state_dict())

        self.assertIsInstance(info, dict)
        self.assertTrue(torch.equal(
            dual.visible_encoder.state_dict()['net.1.res_unit.2.weight'],
            visible.encoder.state_dict()['net.1.res_unit.2.weight'],
        ))
        self.assertTrue(torch.equal(
            dual.oracle_encoder.state_dict()['net.1.res_unit.2.weight'],
            visible.encoder.state_dict()['net.1.res_unit.2.weight'],
        ))
        self.assertEqual(
            (32, 217, 3),
            tuple(dual.oracle_encoder.state_dict()['net.0.weight'].shape),
        )
        self.assertEqual('visible_channel_mean_scaled', info['oracle_encoder']['first_conv_init'])

    def test_dual_tower_can_initialize_oracle_slice_from_single_tower_state(self):
        single = Brain(
            version=4,
            is_oracle=True,
            conv_channels=32,
            num_blocks=2,
            Norm='GN',
        )
        dual = OracleDualTowerBrain(
            version=4,
            conv_channels=32,
            num_blocks=2,
            Norm='GN',
        )

        info = init_dual_tower_from_visible_state(dual, single.state_dict())

        visible_channels = obs_shape(4)[0]
        self.assertEqual('single_oracle_slice', info['oracle_encoder']['first_conv_init'])
        self.assertTrue(torch.equal(
            dual.oracle_encoder.state_dict()['net.0.weight'],
            single.encoder.state_dict()['net.0.weight'][:, visible_channels:, :],
        ))

    def test_dual_tower_fusion_can_start_as_visible_identity(self):
        dual = OracleDualTowerBrain(
            version=4,
            conv_channels=32,
            num_blocks=2,
            Norm='GN',
            oracle_fusion_init=0.0,
        )

        weight = dual.fusion.weight.detach()
        self.assertTrue(torch.equal(weight[:, :1024], torch.eye(1024)))
        self.assertEqual(0, torch.count_nonzero(weight[:, 1024:]).item())

    def test_dual_tower_film_starts_as_exact_visible_identity(self):
        dual = OracleDualTowerBrain(
            version=4,
            conv_channels=32,
            num_blocks=2,
            Norm='GN',
            oracle_fusion_mode='film',
            oracle_fusion_hidden=64,
        )
        dual.eval()
        obs = torch.randn(3, obs_shape(4)[0], 34)
        invisible_obs = torch.randn(3, 217, 34)

        with torch.no_grad():
            expected = dual.actv(dual.visible_encoder(obs))
            actual = dual(obs, invisible_obs=invisible_obs)

        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
        self.assertEqual(0, torch.count_nonzero(dual.fusion[-1].weight).item())
        self.assertEqual(0, torch.count_nonzero(dual.fusion[-1].bias).item())

    def test_dual_tower_film_zero_output_layer_receives_first_step_gradient(self):
        dual = OracleDualTowerBrain(
            version=4,
            conv_channels=32,
            num_blocks=2,
            Norm='GN',
            oracle_fusion_mode='film',
            oracle_fusion_hidden=64,
        )
        obs = torch.randn(2, obs_shape(4)[0], 34)
        invisible_obs = torch.randn(2, 217, 34)

        dual(obs, invisible_obs=invisible_obs).square().mean().backward()

        self.assertGreater(dual.fusion[-1].weight.grad.abs().sum().item(), 0.0)

    def test_dual_tower_residual_mlp_starts_as_exact_visible_identity(self):
        dual = OracleDualTowerBrain(
            version=4,
            conv_channels=32,
            num_blocks=2,
            Norm='GN',
            oracle_fusion_mode='residual_mlp',
            oracle_fusion_hidden=64,
        )
        dual.eval()
        obs = torch.randn(3, obs_shape(4)[0], 34)
        invisible_obs = torch.randn(3, 217, 34)

        with torch.no_grad():
            expected = dual.actv(dual.visible_encoder(obs))
            actual = dual(obs, invisible_obs=invisible_obs)

        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
        self.assertEqual(0, torch.count_nonzero(dual.fusion[-1].weight).item())
        self.assertEqual(0, torch.count_nonzero(dual.fusion[-1].bias).item())

    def test_dual_tower_residual_mlp_output_layer_receives_first_step_gradient(self):
        dual = OracleDualTowerBrain(
            version=4,
            conv_channels=32,
            num_blocks=2,
            Norm='GN',
            oracle_fusion_mode='residual_mlp',
            oracle_fusion_hidden=64,
        )
        obs = torch.randn(2, obs_shape(4)[0], 34)
        invisible_obs = torch.randn(2, 217, 34)

        dual(obs, invisible_obs=invisible_obs).square().mean().backward()

        self.assertGreater(dual.fusion[-1].weight.grad.abs().sum().item(), 0.0)

    def test_dual_tower_hand_aligned_init_maps_each_opponent_hand(self):
        visible = Brain(
            version=4,
            is_oracle=False,
            conv_channels=32,
            num_blocks=2,
            Norm='GN',
        )
        dual = OracleDualTowerBrain(
            version=4,
            conv_channels=32,
            num_blocks=2,
            Norm='GN',
        )

        info = init_dual_tower_from_visible_state(
            dual,
            visible.state_dict(),
            oracle_first_conv_init='hand_aligned',
            oracle_hand_init_scale=0.25,
        )

        source = visible.encoder.state_dict()['net.0.weight'][:, :7, :] * 0.25
        target = dual.oracle_encoder.state_dict()['net.0.weight']
        for start in (0, 17, 34):
            self.assertTrue(torch.equal(source, target[:, start:start + 7, :]))
        self.assertEqual(0, torch.count_nonzero(target[:, 7:17, :]).item())
        self.assertEqual('opponent_hand_aligned', info['oracle_encoder']['first_conv_init'])

    def test_dual_tower_random_init_preserves_native_first_conv(self):
        visible = Brain(
            version=4,
            is_oracle=False,
            conv_channels=32,
            num_blocks=2,
            Norm='GN',
        )
        dual = OracleDualTowerBrain(
            version=4,
            conv_channels=32,
            num_blocks=2,
            Norm='GN',
        )
        before = dual.oracle_encoder.state_dict()['net.0.weight'].clone()

        info = init_dual_tower_from_visible_state(
            dual,
            visible.state_dict(),
            oracle_first_conv_init='random',
        )

        self.assertTrue(torch.equal(before, dual.oracle_encoder.state_dict()['net.0.weight']))
        self.assertEqual('native_random', info['oracle_encoder']['first_conv_init'])

    def test_dual_tower_random_tower_preserves_all_native_oracle_weights(self):
        visible = Brain(
            version=4,
            is_oracle=False,
            conv_channels=32,
            num_blocks=2,
            Norm='GN',
        )
        dual = OracleDualTowerBrain(
            version=4,
            conv_channels=32,
            num_blocks=2,
            Norm='GN',
        )
        before = {
            key: value.clone()
            for key, value in dual.oracle_encoder.state_dict().items()
        }

        info = init_dual_tower_from_visible_state(
            dual,
            visible.state_dict(),
            oracle_tower_init='random',
            oracle_first_conv_init='random',
        )

        after = dual.oracle_encoder.state_dict()
        self.assertTrue(all(torch.equal(before[key], after[key]) for key in before))
        self.assertEqual('random', info['oracle_encoder']['tower_init'])
        self.assertEqual('native_random', info['oracle_encoder']['first_conv_init'])

    def test_value_head_exact_zero_sum_projection(self):
        value_net = ValueHead(num_players=4, zero_sum=True)
        pred = value_net(torch.randn(7, 1024))

        self.assertTrue(torch.allclose(pred.sum(dim=-1), torch.zeros(7), atol=1e-6))

    def test_value_head_hidden_size_is_configurable(self):
        value_net = ValueHead(num_players=4, hidden_size=64)

        self.assertEqual(64, value_net.hidden_size)
        self.assertEqual((64, 1024), tuple(value_net.net[0].weight.shape))
        self.assertEqual((4, 64), tuple(value_net.net[-1].weight.shape))

    def test_hl_gauss_support_matches_paper_recipe(self):
        value_net = HLGaussValueHead(num_players=4)

        self.assertAlmostEqual(12.0 / 88.0, value_net.bin_width)
        self.assertAlmostEqual(24.0 / 88.0, value_net.sigma)
        self.assertAlmostEqual(-6.0 - 72.0 / 88.0, value_net.support_min)
        self.assertAlmostEqual(6.0 + 72.0 / 88.0, value_net.support_max)

    def test_hl_gauss_targets_are_normalized_and_nearly_unbiased(self):
        value_net = HLGaussValueHead(num_players=4)
        target = torch.tensor([
            [-6.0, -4.0, 0.0, 6.0],
            [-3.0, -1.0, 1.0, 3.0],
        ])

        probs = value_net.target_probs(target)
        encoded = (probs * value_net.bin_centers).sum(dim=-1)

        torch.testing.assert_close(
            probs.sum(dim=-1),
            torch.ones_like(target),
            rtol=1e-6,
            atol=1e-6,
        )
        self.assertLess((encoded - target).abs().max().item(), 0.0013)

    def test_hl_gauss_forward_is_zero_sum_and_cross_entropy_trains(self):
        value_net = HLGaussValueHead(
            num_players=4,
            hidden_size=32,
            zero_sum=True,
        )
        phi = torch.randn(3, 1024)
        target = torch.tensor([
            [-6.0, -2.0, 2.0, 6.0],
            [-3.0, -1.0, 1.0, 3.0],
            [0.0, 0.0, 0.0, 0.0],
        ])

        logits = value_net.logits(phi)
        pred = value_net.values_from_logits(logits)
        loss = value_net.cross_entropy(logits, target, [1.6, 0.8, 0.8, 0.8])
        loss.backward()

        self.assertEqual((3, 4, 100), tuple(logits.shape))
        torch.testing.assert_close(
            pred.sum(dim=-1),
            torch.zeros(3),
            rtol=0.0,
            atol=1e-6,
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(value_net.net[-1].weight.grad.abs().sum().item(), 0.0)

    def test_value_loss_mode_aliases_are_normalized(self):
        self.assertEqual('mse', normalize_value_loss_mode('scalar_mse'))
        self.assertEqual('hl_gauss', normalize_value_loss_mode('HL-Gauss'))
        with self.assertRaisesRegex(ValueError, 'unsupported value_loss_mode'):
            normalize_value_loss_mode('quantile')

    def test_single_tower_can_start_with_zero_oracle_input_weights(self):
        visible = Brain(
            version=4,
            is_oracle=False,
            conv_channels=32,
            num_blocks=2,
            Norm='GN',
        )
        oracle = Brain(
            version=4,
            is_oracle=True,
            conv_channels=32,
            num_blocks=2,
            Norm='GN',
        )
        value_net = ValueHead(num_players=4)
        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = Path(tmp_dir) / 'visible_only.pth'
            torch.save({'mortal': visible.state_dict()}, checkpoint_path)

            maybe_init_from_checkpoint(
                oracle,
                value_net,
                str(checkpoint_path),
                torch.device('cpu'),
                oracle_input_init_scale=0.0,
            )

        visible_channels = obs_shape(4)[0]
        first_conv = oracle.encoder.net[0].weight.detach()
        self.assertTrue(torch.equal(
            first_conv[:, :visible_channels],
            visible.encoder.net[0].weight.detach(),
        ))
        self.assertEqual(0, torch.count_nonzero(first_conv[:, visible_channels:]).item())

    def test_strict_oracle_init_rejects_visible_only_checkpoint(self):
        visible = Brain(
            version=4,
            is_oracle=False,
            conv_channels=32,
            num_blocks=2,
            Norm='GN',
        )
        oracle = Brain(
            version=4,
            is_oracle=True,
            conv_channels=32,
            num_blocks=2,
            Norm='GN',
        )
        value_net = ValueHead(num_players=4)
        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = Path(tmp_dir) / 'visible_only.pth'
            torch.save({'mortal': visible.state_dict()}, checkpoint_path)

            with self.assertRaisesRegex(ValueError, 'must contain oracle_brain'):
                maybe_init_from_checkpoint(
                    oracle,
                    value_net,
                    str(checkpoint_path),
                    torch.device('cpu'),
                    strict_oracle_checkpoint=True,
                )

    def test_strict_oracle_init_rejects_wrong_oracle_shape(self):
        visible = Brain(
            version=4,
            is_oracle=False,
            conv_channels=32,
            num_blocks=2,
            Norm='GN',
        )
        oracle = Brain(
            version=4,
            is_oracle=True,
            conv_channels=32,
            num_blocks=2,
            Norm='GN',
        )
        value_net = ValueHead(num_players=4)
        with tempfile.TemporaryDirectory() as tmp_dir:
            checkpoint_path = Path(tmp_dir) / 'wrong_shape.pth'
            torch.save({'oracle_brain': visible.state_dict()}, checkpoint_path)

            with self.assertRaisesRegex(RuntimeError, 'does not match'):
                maybe_init_from_checkpoint(
                    oracle,
                    value_net,
                    str(checkpoint_path),
                    torch.device('cpu'),
                    strict_oracle_checkpoint=True,
                )

    def test_batch_metrics_tracks_objective_and_teacher_loss_separately(self):
        pred = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        target = torch.tensor([[0.0, 2.0, 2.0, 6.0]])
        teacher = torch.tensor([[1.0, 1.0, 4.0, 4.0]])
        target_mse = torch.nn.functional.mse_loss(pred, target)
        teacher_mse = torch.nn.functional.mse_loss(pred, teacher)
        zero_sum_loss = pred.sum(dim=-1).square().mean()
        objective = target_mse + 0.5 * teacher_mse + 0.01 * zero_sum_loss

        metrics = finalize_metrics([
            batch_metrics(
                pred,
                target,
                objective_loss=objective,
                target_mse=target_mse,
                teacher_mse=teacher_mse,
                zero_sum_loss=zero_sum_loss,
            )
        ])

        self.assertAlmostEqual(target_mse.item(), metrics['loss'])
        self.assertAlmostEqual(objective.item(), metrics['objective_loss'])
        self.assertAlmostEqual(target_mse.item(), metrics['target_mse'])
        self.assertAlmostEqual(teacher_mse.item(), metrics['teacher_mse'])
        self.assertAlmostEqual(zero_sum_loss.item(), metrics['zero_sum_loss'])
        self.assertIn('relative_player_0', metrics['outputs'])
        self.assertAlmostEqual(target.square().mean().item(), metrics['zero_baseline_loss'])

    def test_regression_slices_cover_zero_nonzero_and_large_targets(self):
        pred = torch.tensor([1.0, 2.0, 0.0, 1.0, -2.0])
        target = torch.tensor([0.0, 1.0, 2.0, 4.0, -4.0])

        slices = summarize_regression_slices(pred, target)

        self.assertEqual(1, slices['exact_zero']['count'])
        self.assertEqual(4, slices['nonzero']['count'])
        self.assertEqual(3, slices['abs_ge_2']['count'])
        self.assertEqual(2, slices['abs_ge_4']['count'])
        self.assertAlmostEqual(1.0, slices['exact_zero']['loss'])

    def test_finalize_metrics_attaches_primary_tail_slices(self):
        pred = torch.zeros((2, 4), dtype=torch.float32)
        target = torch.tensor([
            [0.0, 1.0, 2.0, 4.0],
            [-4.0, -2.0, -1.0, 0.0],
        ])

        metrics = finalize_metrics([batch_metrics(pred, target)])
        primary_slices = metrics['outputs']['relative_player_0']['slices']

        self.assertEqual(1, primary_slices['exact_zero']['count'])
        self.assertEqual(1, primary_slices['abs_ge_4']['count'])

    def test_eval_metrics_report_game_cluster_uncertainty(self):
        pred = torch.zeros((4, 4), dtype=torch.float32)
        target = torch.tensor([
            [1.0] * 4,
            [3.0] * 4,
            [2.0] * 4,
            [4.0] * 4,
        ])

        metrics = finalize_metrics([
            batch_metrics(pred, target, game_id=torch.tensor([10, 10, 20, 20]))
        ])

        self.assertEqual(2, metrics['num_games'])
        self.assertEqual(4, metrics['num_samples'])
        self.assertEqual(16, metrics['num_values'])
        self.assertAlmostEqual(7.5, metrics['loss'])
        self.assertAlmostEqual(7.5, metrics['game_balanced_loss'])
        self.assertAlmostEqual(2.5, metrics['loss_cluster_se'])
        self.assertAlmostEqual(2.5, metrics['game_balanced_loss_se'])

    def test_eval_metrics_report_zero_sum_preserving_calibration(self):
        pred = torch.tensor([[1.0, -1.0, 2.0, -2.0]])
        target = 2.0 * pred

        metrics = finalize_metrics([batch_metrics(pred, target)])

        self.assertAlmostEqual(2.0, metrics['calibration_scale'])
        self.assertAlmostEqual(0.0, metrics['calibrated_loss'])
        self.assertAlmostEqual(
            2.0,
            metrics['outputs']['relative_player_0']['calibration_scale'],
        )
        self.assertAlmostEqual(
            0.0,
            metrics['outputs']['relative_player_0']['calibrated_loss'],
        )

    def test_binned_calibration_reports_local_mean_error(self):
        pred = torch.tensor([-1.0, -0.5, 0.5, 1.0])

        perfect = binned_calibration_metrics(pred, pred, num_bins=2)
        shifted = binned_calibration_metrics(pred, pred + 1.0, num_bins=2)

        self.assertEqual(2, perfect['binned_calibration_bins'])
        self.assertAlmostEqual(0.0, perfect['binned_calibration_mae'])
        self.assertAlmostEqual(0.0, perfect['binned_calibration_rmse'])
        self.assertAlmostEqual(1.0, shifted['binned_calibration_mae'])
        self.assertAlmostEqual(1.0, shifted['binned_calibration_rmse'])
        self.assertAlmostEqual(1.0, shifted['binned_calibration_max_abs'])

    def test_eval_input_modes_are_deduplicated_and_validated(self):
        self.assertEqual(
            ('true', 'zero', 'shuffled'),
            normalize_eval_input_modes('true,zero shuffled true'),
        )
        with self.assertRaisesRegex(ValueError, 'unsupported Oracle eval input mode'):
            normalize_eval_input_modes(('fake',))

    def test_shuffled_eval_input_uses_distant_batch_rows(self):
        invisible_obs = torch.arange(8, dtype=torch.float32).reshape(4, 2, 1)

        shuffled = transform_eval_invisible_obs(invisible_obs, 'shuffled')

        self.assertTrue(torch.equal(shuffled, torch.roll(invisible_obs, shifts=2, dims=0)))
        self.assertTrue(torch.equal(
            transform_eval_invisible_obs(invisible_obs, 'zero'),
            torch.zeros_like(invisible_obs),
        ))

    def test_grad_scaler_step_only_succeeds_when_scale_does_not_drop(self):
        self.assertTrue(grad_scaler_step_succeeded(65536.0, 65536.0))
        self.assertTrue(grad_scaler_step_succeeded(65536.0, 131072.0))
        self.assertFalse(grad_scaler_step_succeeded(65536.0, 32768.0))

    def test_oracle_dependency_summary_uses_shuffled_as_primary_counterfactual(self):
        summary = summarize_oracle_dependency({
            'true': {'loss': 3.0, 'corr': 0.6, 'explained_variance': 0.3},
            'shuffled': {'loss': 4.0, 'corr': 0.4, 'explained_variance': 0.2},
            'zero': {'loss': 5.0, 'corr': 0.3, 'explained_variance': 0.1},
        })

        self.assertAlmostEqual(
            0.25,
            summary['true_vs_shuffled']['relative_loss_improvement'],
        )
        self.assertAlmostEqual(0.2, summary['true_vs_shuffled']['corr_gain'])
        self.assertAlmostEqual(2.0, summary['true_vs_zero']['loss_improvement'])


if __name__ == '__main__':
    unittest.main()
