"""Prepare new-reward runs without rewriting old configs or checkpoint contracts."""
import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from mortal.core.evidence_contract import sha256_file
from mortal.core.toml_utils import load_toml_file, write_toml_file
from mortal.eval.confirmation_protocol import write_new
from mortal.online.policy_objective import ACTOR_OBJECTIVE_VERSION


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-config', required=True)
    parser.add_argument('--critic-anchor', required=True)
    parser.add_argument('--actor-anchor', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('use an empty output directory; frozen profiles are not overwritten')
    output.mkdir(parents=True, exist_ok=True)
    base = load_toml_file(args.base_config)
    cfg = deepcopy(base)
    cfg['env']['pts'] = [2.0, 1.0, 0.0, -3.0]
    cfg['policy'].update({
        'gae_gamma': 1.0, 'actor_objective': 'ppo',
        'actor_objective_contract': ACTOR_OBJECTIVE_VERSION,
        'importance_rho_clip': 0.0, 'importance_c_clip': 0.0,
        'vtrace_target_rho_clip': 0.0, 'vtrace_target_c_clip': 0.0,
        'vtrace_rho_clip': 0.0, 'vtrace_c_clip': 0.0,
        'dual_clip': 0.0, 'logit_thres': 0.0,
        'target_kl': 0.02, 'max_clip_fraction': 0.5, 'max_behavior_version_gap': 1,
    })
    cfg.setdefault('online', {}).setdefault('importance_sampling', {}).update({
        'enabled': True, 'drop_untracked_samples': True,
        'max_policy_versions': 2, 'vtrace_mode': 'disabled',
    })
    pretrain = cfg['oracle_critic_pretrain']
    pretrain.update({
        'run_name': output.name, 'init_state_file': str(Path(args.critic_anchor).resolve()),
        'strict_init_checkpoint': True, 'discount_gamma': 1.0,
        'state_fold_backend': 'native_hash', 'state_fold_count': 64,
        'val_state_fold_count': 32, 'val_oracle_imputation_seed': 20260905,
        'batch_size': 640, 'num_workers': 2, 'val_num_workers': 0,
        'file_batch_size': 2, 'val_file_batch_size': 2,
        'prefetch_factor': 1, 'val_prefetch_factor': 1,
        'val_batches': 0, 'val_every_steps': 10000, 'max_steps': 40000,
        'save_every': 1000, 'final_test_enabled': False,
    })
    for key, name in [('state_file', 'latest.pth'), ('best_state_file', 'best_dev.pth'),
                      ('best_primary_state_file', 'best_primary.pth'), ('adaptive_best_state_file', 'adaptive_best.pth')]:
        pretrain[key] = str(output / 'critic' / 'checkpoints' / name)
    pretrain['metrics_file'] = str(output / 'critic' / 'metrics.jsonl')
    pretrain['tensorboard_dir'] = str(output / 'critic' / 'tb_log')
    gate = pretrain['adaptive_curriculum']
    gate.update({'phase_name': 'formal_pt_calibration', 'gate_every_steps': 10000,
                 'max_unresolved_gates': 4, 'min_paired_games': 32})
    for guard in gate['guardrails']:
        guard['noninferiority_margin'] = 0.0
    write_toml_file(output / 'critic_config.toml', cfg)

    # A separate near-on-policy reference. Launch only through qualification gates.
    ppo = deepcopy(cfg)
    ppo['control'].update({'state_file': str(output / 'ppo_reference' / 'latest.pth'),
                          'best_state_file': str(output / 'ppo_reference' / 'best.pth'),
                          'tensorboard_dir': str(output / 'ppo_reference' / 'tb_log')})
    ppo['online']['init_state_file'] = str(Path(args.actor_anchor).resolve())
    ppo.setdefault('value', {}).update({'enabled': True, 'oracle_critic': False,
                                       'target_mode': 'all_players', 'reward_source': 'score_rank'})
    ppo['policy']['gae_lambda'] = 0.95
    ppo['control']['max_steps'] = 500
    ppo['control']['save_every'] = 100
    ppo['online'].update({'stop_at_max_steps': True})
    ppo['optim']['scheduler'].update({'max_steps': 500, 'warm_up_steps': 50,
                                     'peak': 1e-5, 'final': 1e-5})
    ppo['online'].setdefault('remote', {}).update({'host': '127.0.0.1', 'port': 5151})
    ppo['online'].setdefault('server', {}).update({
        'buffer_dir': str(output / 'ppo_reference' / 'buffer'),
        'drain_dir': str(output / 'ppo_reference' / 'drain'),
        'sample_reuse_rate': 0, 'sample_reuse_threshold': 0,
    })
    for section in ('test_play', '1v3', 'oracle_dependency'):
        if section in ppo:
            ppo[section]['log_dir'] = str(output / 'ppo_reference' / section)
    if 'train_play' in ppo:
        ppo['train_play'].setdefault('default', {})['log_dir'] = str(output / 'ppo_reference' / 'train_play')
    ppo.setdefault('oracle_guiding', {})['actor_enabled'] = False
    for key in list(ppo.setdefault('aux', {})):
        if key.endswith('_weight'):
            ppo['aux'][key] = 0.0
    ppo.setdefault('expected_reward', {})['enabled'] = False
    write_toml_file(output / 'ppo_reference_config.toml', ppo)
    oracle_ppo = deepcopy(ppo)
    oracle_ppo['value'].update({
        'oracle_critic': True, 'oracle_critic_arch': pretrain['critic_arch'],
        'oracle_fusion_mode': pretrain.get('oracle_fusion_mode', 'linear'),
        'oracle_fusion_hidden': int(pretrain.get('oracle_fusion_hidden', 512)),
        'value_head_hidden': int(pretrain.get('value_head_hidden', 256)),
        'value_loss_mode': pretrain.get('value_loss_mode', 'mse'),
        'exact_zero_sum': bool(pretrain.get('exact_zero_sum', False)),
        'oracle_critic_state_file': str(output / 'qualified_critic.pth'),
    })
    for key in ('state_file', 'best_state_file', 'tensorboard_dir'):
        oracle_ppo['control'][key] = oracle_ppo['control'][key].replace('ppo_reference', 'ppo_oracle')
    oracle_ppo['online']['remote']['port'] = 5152
    for key in ('buffer_dir', 'drain_dir'):
        oracle_ppo['online']['server'][key] = oracle_ppo['online']['server'][key].replace('ppo_reference', 'ppo_oracle')
    for section in ('test_play', '1v3', 'oracle_dependency'):
        if section in oracle_ppo:
            oracle_ppo[section]['log_dir'] = oracle_ppo[section]['log_dir'].replace('ppo_reference', 'ppo_oracle')
    if 'train_play' in oracle_ppo:
        oracle_ppo['train_play']['default']['log_dir'] = oracle_ppo['train_play']['default']['log_dir'].replace('ppo_reference', 'ppo_oracle')
    write_toml_file(output / 'ppo_oracle_config.toml', oracle_ppo)

    # Small real-data development checks, explicitly excluded from qualification.
    pre = base['oracle_critic_pretrain']
    dev = torch.load(pre['dev_file_index'], weights_only=True)['file_list']
    dev = [dev[index] for index in np.linspace(0, len(dev) - 1, min(32, len(dev)), dtype=int)]
    dev_index = output / 'smoke_dev_index.pth'
    torch.save({'file_list': dev}, dev_index)
    legacy_eval = deepcopy(base)
    legacy_eval['oracle_critic_pretrain'].update({
        'dev_file_index': str(dev_index), 'test_file_index': '', 'max_val_files': 0,
        'val_state_fold_count': 16, 'val_file_batch_size': 1, 'val_num_workers': 0,
        'batch_size': 16, 'val_game_id_modulus': 1, 'val_game_id_remainders': [],
        'val_batches': 0, 'eval_enable_amp': False,
    })
    write_toml_file(output / 'legacy_rebase_eval.toml', legacy_eval)
    smoke = deepcopy(cfg)
    smoke_pre = smoke['oracle_critic_pretrain']
    # Use disjoint human training data; no sealed test payload is opened.
    train_file = next(Path(dev[0]).parents[1].glob('202511/*.json'))
    train_index = output / 'smoke_train_index.pth'
    torch.save({'file_list': [str(train_file)]}, train_index)
    smoke_pre.update({'train_file_index': str(train_index), 'dev_file_index': str(dev_index),
                      'test_file_index': '', 'batch_size': 2, 'state_fold_count': 16,
                      'num_workers': 0, 'val_num_workers': 0, 'val_batches': 2,
                      'val_state_fold_count': 16, 'val_game_id_modulus': 1,
                      'val_game_id_remainders': [], 'max_steps': 2, 'val_every_steps': 1,
                      'save_every': 1, 'log_every': 1, 'device': 'cpu', 'enable_amp': False})
    smoke_pre['adaptive_curriculum'].update({'gate_every_steps': 1, 'min_paired_games': 2})
    # Sparse smoke batches need not contain every rare guard; production retains all guards.
    smoke_pre['adaptive_curriculum']['guardrails'] = []
    for key in ('state_file', 'best_state_file', 'best_primary_state_file', 'adaptive_best_state_file',
                'metrics_file', 'tensorboard_dir'):
        smoke_pre[key] = smoke_pre[key].replace(str(output / 'critic'), str(output / 'smoke_critic'))
    write_toml_file(output / 'cpu_smoke_config.toml', smoke)
    write_new(output / 'preparation.json', {
        'base_config_sha256': sha256_file(args.base_config),
        'critic_anchor_sha256': sha256_file(args.critic_anchor),
        'actor_anchor_sha256': sha256_file(args.actor_anchor),
        'initialization': 'weights only; new optimizer, scheduler, scaler, cursor and validation baseline',
        'old_reward_rank_points': base['env']['pts'], 'new_reward_rank_points': cfg['env']['pts'],
        'new_gamma': 1.0, 'sealed_test_opened': False, 'training_seed_replicates': [20260416, 20260417, 20260418],
        'qualification_imputation_seeds': [20260905, 20260906, 20260907],
        'rl_gates_optimizer_steps': [500, 1500, 3000, 20000, 40000],
        'ppo_launch_state': 'requires actor finalist and fixed-distribution critic qualification',
        'smoke_evidence_is_not_qualification': True,
    })


if __name__ == '__main__':
    main()
