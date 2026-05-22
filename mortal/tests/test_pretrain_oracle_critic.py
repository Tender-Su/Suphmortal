import sys
import tempfile
import unittest
from pathlib import Path

import torch
from libriichi.consts import obs_shape

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mortal.core.model import Brain, OracleDualTowerBrain, ValueHead
from mortal.online.pretrain_oracle_critic import (
    batch_metrics,
    configure_trainable_parameters,
    first_conv_module,
    finalize_metrics,
    init_dual_tower_from_visible_state,
    maybe_init_from_checkpoint,
    normalize_critic_arch,
    normalize_train_scope,
    optimizer_param_groups,
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
    def test_normalize_train_scope_accepts_tail_aliases(self):
        self.assertEqual('oracle_tail', normalize_train_scope('tail'))
        self.assertEqual('oracle_tail', normalize_train_scope('oracle_input_tail_value'))

    def test_normalize_critic_arch_accepts_dual_tower_alias(self):
        self.assertEqual('dual_tower', normalize_critic_arch('dual'))
        self.assertEqual('single_tower', normalize_critic_arch('bridge'))

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


if __name__ == '__main__':
    unittest.main()
