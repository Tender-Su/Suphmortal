import copy
import json
import tempfile
import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path

import torch

from mortal.core.evidence_contract import require_finalist_decision, sha256_file, native_module_file
from mortal.data.split_ledger import build_ledger, require_disjoint
from mortal.eval.confirmation_protocol import actor_fingerprint, prepare_protocol, run_arm
from mortal.online.train_online import build_online_value_models, validate_oracle_critic_init_checkpoint


class EvidenceProtocolTests(unittest.TestCase):
    def test_wheel_provenance_hashes_binary_instead_of_python_wrapper(self):
        from types import SimpleNamespace
        wrapper = SimpleNamespace(__name__='testnative', __file__='wheel/__init__.py')
        binary = SimpleNamespace(__name__='testnative.core', __file__='wheel/core.pyd')
        with patch.dict('sys.modules', {'testnative': wrapper, 'testnative.core': binary}):
            self.assertEqual(Path('wheel/core.pyd').resolve(), native_module_file(wrapper))

    def test_completed_chunks_resume_and_source_change_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            record = {'path': 'fixed-model', 'sha256': 'fixed', 'actor_sha256': 'a'}
            protocol = prepare_protocol(record, record, {'x': {'actor_sha256': 'b'}})
            protocol['aa_games_per_arm'] = 4
            def emit_chunk(**kwargs):
                Path(kwargs['log_dir']).mkdir(parents=True)
            with patch('mortal.eval.confirmation_protocol.verify_checkpoint'), \
                 patch('mortal.eval.confirmation_protocol.runtime_record', return_value={'source': 'fixed'}), \
                 patch('mortal.eval.confirmation_protocol.load_games', return_value=[{}] * 4), \
                 patch('mortal.eval.confirmation_protocol.event_hashes', return_value=['event'] * 4), \
                 patch('mortal.eval.one_vs_three.load_mortal_engine', return_value=MagicMock()), \
                 patch('mortal.eval.one_vs_three.run_eval_once', side_effect=emit_chunk) as run:
                run_arm(protocol, directory, stage='aa', name='reference', chunk_seeds=1)
                (Path(directory) / 'aa/reference/result.json').unlink()
                run_arm(protocol, directory, stage='aa', name='reference', chunk_seeds=1)
                self.assertEqual(1, run.call_count)
                with patch('mortal.eval.confirmation_protocol.runtime_record', return_value={'source': 'changed'}):
                    with self.assertRaisesRegex(ValueError, 'manifest changed'):
                        run_arm(protocol, directory, stage='aa', name='reference', chunk_seeds=1)

    def test_actor_dedup_ignores_critic_but_detects_policy_change(self):
        state = {'mortal': {'w': torch.ones(2)}, 'policy_net': {'w': torch.ones(2)},
                 'value_net': {'w': torch.ones(2)}, 'config': {'control': {'version': 4}, 'resnet': {}}}
        original = actor_fingerprint(state)
        state['value_net']['w'].add_(1)
        self.assertEqual(original, actor_fingerprint(state))
        state['policy_net']['w'].add_(1)
        self.assertNotEqual(original, actor_fingerprint(state))

    def test_duplicate_candidates_and_reused_confirmation_seed_rejected(self):
        reference, candidate = {'actor_sha256': 'a'}, {'actor_sha256': 'b'}
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            prepare_protocol(reference, reference, {'x': candidate, 'y': candidate})
        with self.assertRaisesRegex(ValueError, 'independent'):
            prepare_protocol(reference, reference, {'x': candidate}, screen_seed=1, confirmation_seed=1)

    def test_sealed_test_rejects_missing_or_changed_finalist(self):
        with self.assertRaisesRegex(ValueError, 'sealed test'):
            require_finalist_decision('', [])
        with tempfile.TemporaryDirectory() as directory:
            model, decision = Path(directory) / 'model', Path(directory) / 'decision.json'
            model.write_bytes(b'fixed')
            decision.write_text(json.dumps({'decision': 'selected', 'validation_input_fingerprint': 'fixed-input',
                                            'allowed_checkpoint_sha256': [sha256_file(model)]}))
            require_finalist_decision(decision, [model])
            model.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'not frozen'):
                require_finalist_decision(decision, [model])

    def test_split_ledger_matches_games_across_paths_and_compression(self):
        with tempfile.TemporaryDirectory() as directory:
            train, test = Path(directory) / 'train.pth', Path(directory) / 'test.pth'
            game = '2025120100gm-00a9-0000-1703bbc3'
            torch.save({'file_list': [f'D:/data/{game}.json']}, train)
            torch.save({'file_list': [f'C:/cache/{game}.json.gz']}, test)
            ledger = build_ledger({'sl:train': train, 'oracle:test': test})
            with self.assertRaisesRegex(ValueError, 'overlaps'):
                require_disjoint(ledger, 'oracle:test', exposed_roles=['sl:train'])

    def test_residual_critic_constructor_and_loading_contract(self):
        cfg = {'control': {'version': 4}, 'resnet': {'conv_channels': 32, 'num_blocks': 1},
               'env': {'pts': [2, 1, 0, -3]}, 'policy': {'gae_gamma': 1.0},
               'value': {'enabled': True, 'oracle_critic': True, 'oracle_critic_arch': 'dual_tower',
                         'oracle_fusion_mode': 'residual_mlp', 'oracle_fusion_hidden': 64,
                         'value_head_hidden': 64, 'target_mode': 'all_players', 'reward_source': 'score_rank'}}
        oracle, head = build_online_value_models(cfg, device='cpu')
        pre = {'critic_arch': 'dual_tower', 'oracle_fusion_mode': 'residual_mlp', 'oracle_fusion_hidden': 64,
               'value_head_hidden': 64, 'target_mode': 'all_players', 'return_mode': 'score_rank_mc', 'discount_gamma': 1.0}
        state = {'oracle_brain': oracle.state_dict(), 'value_net': head.state_dict(),
                 'config': cfg, 'oracle_critic_pretrain': pre}
        validate_oracle_critic_init_checkpoint(state, cfg)
        wrong = copy.deepcopy(cfg)
        wrong['value']['oracle_fusion_mode'] = 'linear'
        with self.assertRaisesRegex(ValueError, 'fusion_mode mismatch'):
            validate_oracle_critic_init_checkpoint(state, wrong)


if __name__ == '__main__':
    unittest.main()
