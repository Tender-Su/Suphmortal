def online_resume_model_signature(config):
    if not isinstance(config, dict):
        return None

    control_cfg = config.get('control', {})
    resnet_cfg = config.get('resnet', {})
    aux_cfg = config.get('aux', {})
    value_cfg = config.get('value', {})
    exp_reward_cfg = config.get('expected_reward', {})
    if control_cfg and not isinstance(control_cfg, dict):
        return None
    if resnet_cfg and not isinstance(resnet_cfg, dict):
        return None
    if aux_cfg and not isinstance(aux_cfg, dict):
        return None

    # `control.online` changes runtime behavior, not the model/optimizer layout
    # expected by train_online checkpoints, so it must not block exact resume.
    return {
        'version': control_cfg.get('version'),
        'resnet': dict(resnet_cfg),
        'aux_enabled': float(aux_cfg.get('next_rank_weight', 0.0) or 0.0) > 0.0,
        'opp_enabled': float(aux_cfg.get('opponent_state_weight', 0.0) or 0.0) > 0.0,
        'danger_enabled': bool(aux_cfg.get('danger_enabled', False)) or float(aux_cfg.get('danger_weight', 0.0) or 0.0) > 0.0,
        'value_enabled': bool(value_cfg.get('enabled', False) if isinstance(value_cfg, dict) else False),
        'oracle_critic': bool(value_cfg.get('oracle_critic', True) if isinstance(value_cfg, dict) and value_cfg.get('enabled', False) else False),
        'tile_eff_enabled': float(aux_cfg.get('tile_efficiency_weight', 0.0) or 0.0) > 0.0,
        'furo_regret_enabled': float(aux_cfg.get('furo_regret_weight', 0.0) or 0.0) > 0.0,
        'hand_value_regret_enabled': float(aux_cfg.get('hand_value_regret_weight', 0.0) or 0.0) > 0.0,
        'exp_reward_enabled': bool(exp_reward_cfg.get('enabled', False) if isinstance(exp_reward_cfg, dict) else False),
    }


def optimizer_state_matches_current_layout(saved_optimizer_state, optimizer):
    if not isinstance(saved_optimizer_state, dict):
        return False

    saved_param_groups = saved_optimizer_state.get('param_groups')
    if not isinstance(saved_param_groups, list):
        return False
    if len(saved_param_groups) != len(optimizer.param_groups):
        return False

    for saved_group, current_group in zip(saved_param_groups, optimizer.param_groups):
        if not isinstance(saved_group, dict):
            return False
        saved_params = saved_group.get('params')
        current_params = current_group.get('params')
        if not isinstance(saved_params, list):
            return False
        if len(saved_params) != len(current_params):
            return False

    return True


def checkpoint_supports_online_resume(state, *, current_config, optimizer):
    if not isinstance(state, dict):
        return False
    if state.get('resume_supported') is False:
        return False
    saved_config = state.get('config', {})
    if saved_config and not isinstance(saved_config, dict):
        return False
    saved_control = saved_config.get('control', {})
    if saved_control and not isinstance(saved_control, dict):
        return False
    required_keys = ('optimizer', 'scheduler', 'scaler', 'best_perf', 'steps')
    if not all(key in state for key in required_keys):
        return False

    saved_signature = online_resume_model_signature(saved_config)
    current_signature = online_resume_model_signature(current_config)
    if saved_signature is None or current_signature is None:
        return False
    if saved_signature != current_signature:
        return False

    return optimizer_state_matches_current_layout(state['optimizer'], optimizer)


def resolve_online_init_state_file(config):
    if not isinstance(config, dict):
        return ''

    online_cfg = config.get('online', {})
    if isinstance(online_cfg, dict):
        init_state_file = str(online_cfg.get('init_state_file', '') or '').strip()
        if init_state_file:
            return init_state_file

    supervised_cfg = config.get('supervised', {})
    if not isinstance(supervised_cfg, dict):
        return ''

    return str(
        supervised_cfg.get('best_loss_state_file', '')
        or supervised_cfg.get('best_state_file', '')
        or ''
    ).strip()


def ensure_online_init_state_file_ready(init_state_file):
    if not init_state_file:
        return

    from os import path
    import run_sl_formal as sl_formal

    sl_formal.ensure_supervised_canonical_handoff_ready(init_state_file)
    if not path.exists(init_state_file):
        raise FileNotFoundError(f'online.init_state_file does not exist: {init_state_file}')


def train():
    import prelude
    import logging
    import sys
    import os
    import gc
    import gzip
    import json
    import shutil
    import random
    import torch
    import math
    from os import path
    from glob import glob
    from datetime import datetime
    from itertools import chain
    from torch import optim, nn
    from torch.amp import GradScaler
    from torch.nn.utils import clip_grad_norm_
    from torch.utils.data import DataLoader
    from torch.distributions import Categorical
    from torch.utils.tensorboard import SummaryWriter
    from common import submit_param, parameter_count, drain, filtered_trimmed_lines, tqdm
    from player import TestPlayer
    from dataloader import FileDatasetsIter, worker_init_fn
    from lr_scheduler import LinearWarmUpCosineAnnealingLR
    from model import Brain, CategoricalPolicy, AuxNet
    from libriichi.consts import obs_shape
    from config import config
    from checkpoint_utils import load_brain_state_with_input_bridge
    from copy import deepcopy
    from multiprocessing import Manager

    # --- Value Head / Oracle Critic imports ---
    value_cfg = config.get('value', {})
    value_enabled = value_cfg.get('enabled', False)
    value_weight = value_cfg.get('weight', 0.5) if value_enabled else 0.0
    value_num_players = value_cfg.get('num_players', 4) if value_enabled else 4
    zero_sum_weight = value_cfg.get('zero_sum_weight', 0.01) if value_enabled else 0.0
    oracle_critic = value_cfg.get('oracle_critic', True) if value_enabled else False
    oracle_dropout_start = value_cfg.get('oracle_dropout_start', 0.0) if value_enabled else 0.0
    oracle_dropout_steps = value_cfg.get('oracle_dropout_steps', 0) if value_enabled else 0

    # --- Local Regret Heads config ---
    tile_eff_weight = config.get('aux', {}).get('tile_efficiency_weight', 0.0)
    furo_regret_weight = config.get('aux', {}).get('furo_regret_weight', 0.0)
    hand_value_regret_weight = config.get('aux', {}).get('hand_value_regret_weight', 0.0)
    online_regret_enabled = (tile_eff_weight > 0 or furo_regret_weight > 0 or hand_value_regret_weight > 0)

    # --- Expected Reward Network config ---
    exp_reward_cfg = config.get('expected_reward', {})
    exp_reward_enabled = exp_reward_cfg.get('enabled', False)
    exp_reward_weight = exp_reward_cfg.get('weight', 0.1) if exp_reward_enabled else 0.0
    exp_reward_warmup = exp_reward_cfg.get('warmup_steps', 10000) if exp_reward_enabled else 0

    version = config['control']['version']

    online = config['control']['online']
    batch_size = config['control']['batch_size']
    opt_step_every = config['control']['opt_step_every']
    save_every = config['control']['save_every']
    test_every = config['control']['test_every']
    submit_every = config['control']['submit_every']
    old_update_every= config['control']['old_update_every']
    test_games = config['test_play']['games']
    assert save_every % opt_step_every == 0
    assert test_every % save_every == 0

    device = torch.device(config['control']['device'])
    torch.backends.cudnn.benchmark = config['control']['enable_cudnn_benchmark']
    enable_amp = config['control']['enable_amp']
    enable_compile = config['control']['enable_compile']

    pts = config['env']['pts']
    file_batch_size = config['dataset']['file_batch_size']
    reserve_ratio = config['dataset']['reserve_ratio']
    num_workers = config['dataset']['num_workers']
    prefetch_factor = config['dataset'].get('prefetch_factor', 2)
    num_epochs = config['dataset']['num_epochs']
    enable_augmentation = config['dataset']['enable_augmentation']
    augmented_first = config['dataset']['augmented_first']
    eps = config['optim']['eps']
    betas = config['optim']['betas']
    weight_decay = config['optim']['weight_decay']
    max_grad_norm = config['optim']['max_grad_norm']

    entropy_weight = config['policy']['entropy_weight']
    entropy_target = config['policy'].get('entropy_target', 0)
    entropy_adjust_rate = config['policy'].get('entropy_adjust_rate', 1e-4)
    clip_ratio = config['policy']['clip_ratio']
    dual_clip = config['policy']['dual_clip']
    logit_thres = config['policy'].get('logit_thres', 0.0)
    vtrace_rho_clip = config['policy'].get('vtrace_rho_clip', 0.0)
    vtrace_c_clip = config['policy'].get('vtrace_c_clip', 0.0)
    next_rank_weight = config.get('aux', {}).get('next_rank_weight', 0.0)
    # --- Opponent State + Danger aux heads for online ---
    aux_cfg = config.get('aux', {})
    online_opponent_weight = aux_cfg.get('opponent_state_weight', 0.0)
    online_danger_weight = aux_cfg.get('danger_weight', 0.0)
    online_danger_enabled = bool(aux_cfg.get('danger_enabled', False)) or online_danger_weight > 0
    online_opp_enabled = online_opponent_weight > 0
    opponent_shanten_weight = aux_cfg.get('opponent_shanten_weight', 0.85)
    opponent_tenpai_weight = aux_cfg.get('opponent_tenpai_weight', 1.15)
    danger_mix_weights = [
        aux_cfg.get('danger_any_weight', 0.09),
        aux_cfg.get('danger_value_weight', 0.82),
        aux_cfg.get('danger_player_weight', 0.09),
    ]
    danger_value_cap = aux_cfg.get('danger_value_cap', 96000.0)

    dynamic_entropy_weight = entropy_weight
    log_entropy_alpha = math.log(max(entropy_weight, 1e-8))

    mortal = Brain(version=version, **config['resnet'], Norm="GN").to(device)
    policy_net = CategoricalPolicy().to(device)
    aux_net = AuxNet(dims=(4,)).to(device) if next_rank_weight > 0 else None

    # --- Opponent State + Danger aux heads ---
    if online_opp_enabled:
        from model import OpponentStateAuxNet
        opponent_aux_net = OpponentStateAuxNet().to(device)
    else:
        opponent_aux_net = None

    if online_danger_enabled:
        from model import DangerAuxNet
        danger_aux_net = DangerAuxNet().to(device)
    else:
        danger_aux_net = None

    # --- Oracle Critic + Value Head ---
    if value_enabled:
        from model import ValueHead
        oracle_brain = Brain(version=version, is_oracle=True, **config['resnet'], Norm="GN").to(device) if oracle_critic else None
        value_net = ValueHead(num_players=value_num_players).to(device)
    else:
        oracle_brain = None
        value_net = None

    # --- Local Regret Heads ---
    if tile_eff_weight > 0:
        from model import TileEfficiencyRegretHead
        tile_eff_net = TileEfficiencyRegretHead().to(device)
    else:
        tile_eff_net = None

    if furo_regret_weight > 0:
        from model import FuroRegretHead
        furo_regret_net = FuroRegretHead().to(device)
    else:
        furo_regret_net = None

    if hand_value_regret_weight > 0:
        from model import HandValueRegretHead
        hand_value_regret_net = HandValueRegretHead().to(device)
    else:
        hand_value_regret_net = None

    # --- Expected Reward Network ---
    if exp_reward_enabled:
        from model import ExpectedRewardNet
        exp_reward_net = ExpectedRewardNet(num_players=value_num_players).to(device)
        grp_label_smoothing = config.get('grp', {}).get('label_smoothing', 0.0)
        if grp_label_smoothing > 0:
            logging.info(
                'note: both ExpectedRewardNet and label_smoothing (%.2f) are active; '
                'both reduce terminal reward variance — ExpectedRewardNet subsumes '
                'label_smoothing once trained; consider setting label_smoothing=0 later',
                grp_label_smoothing,
            )
    else:
        exp_reward_net = None

    all_models_list = [mortal, policy_net]
    if aux_net is not None:
        all_models_list.append(aux_net)
    if opponent_aux_net is not None:
        all_models_list.append(opponent_aux_net)
    if danger_aux_net is not None:
        all_models_list.append(danger_aux_net)
    if oracle_brain is not None:
        all_models_list.append(oracle_brain)
    if value_net is not None:
        all_models_list.append(value_net)
    if tile_eff_net is not None:
        all_models_list.append(tile_eff_net)
    if furo_regret_net is not None:
        all_models_list.append(furo_regret_net)
    if hand_value_regret_net is not None:
        all_models_list.append(hand_value_regret_net)
    if exp_reward_net is not None:
        all_models_list.append(exp_reward_net)
    all_models = tuple(all_models_list)
    if enable_compile:
        for m in all_models:
            m.compile()

    Old_mortal = deepcopy(mortal)
    Old_policy_net = deepcopy(policy_net)

    logging.info(f'version: {version}')
    logging.info(f'obs shape: {obs_shape(version)}')
    logging.info(f'mortal params: {parameter_count(mortal):,}')
    logging.info(f'policy params: {parameter_count(policy_net):,}')
    if aux_net is not None:
        logging.info(f'aux params: {parameter_count(aux_net):,}')
    if opponent_aux_net is not None:
        logging.info(f'opponent_aux params: {parameter_count(opponent_aux_net):,}')
    if danger_aux_net is not None:
        logging.info(f'danger_aux params: {parameter_count(danger_aux_net):,}')
    if oracle_brain is not None:
        logging.info(f'oracle_brain params: {parameter_count(oracle_brain):,}')
    if value_net is not None:
        logging.info(f'value_net params: {parameter_count(value_net):,}')
    if tile_eff_net is not None:
        logging.info(f'tile_eff_net params: {parameter_count(tile_eff_net):,}')
    if furo_regret_net is not None:
        logging.info(f'furo_regret_net params: {parameter_count(furo_regret_net):,}')
    if exp_reward_net is not None:
        logging.info(f'exp_reward_net params: {parameter_count(exp_reward_net):,}')

    decay_params = []
    no_decay_params = []
    models_for_optim = [mortal, policy_net]
    if aux_net is not None:
        models_for_optim.append(aux_net)
    if opponent_aux_net is not None:
        models_for_optim.append(opponent_aux_net)
    if danger_aux_net is not None:
        models_for_optim.append(danger_aux_net)
    if oracle_brain is not None:
        models_for_optim.append(oracle_brain)
    if value_net is not None:
        models_for_optim.append(value_net)
    if tile_eff_net is not None:
        models_for_optim.append(tile_eff_net)
    if furo_regret_net is not None:
        models_for_optim.append(furo_regret_net)
    if hand_value_regret_net is not None:
        models_for_optim.append(hand_value_regret_net)
    if exp_reward_net is not None:
        models_for_optim.append(exp_reward_net)
    for model in models_for_optim:
        params_dict = {}
        to_decay = set()
        for mod_name, mod in model.named_modules():
            for name, param in mod.named_parameters(prefix=mod_name, recurse=False):
                params_dict[name] = param
                if isinstance(mod, (nn.Linear, nn.Conv1d)) and name.endswith('weight'):
                    to_decay.add(name)
        decay_params.extend(params_dict[name] for name in sorted(to_decay))
        no_decay_params.extend(params_dict[name] for name in sorted(params_dict.keys() - to_decay))
    param_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params},
    ]
    optimizer = optim.AdamW(param_groups, lr=1, weight_decay=0, betas=betas, eps=eps)
    scheduler = LinearWarmUpCosineAnnealingLR(optimizer, **config['optim']['scheduler'])
    scaler = GradScaler(device.type, enabled=enable_amp)
    test_player = TestPlayer()
    best_perf = {
        'avg_rank': 4.,
        'avg_pt': -135.,
    }

    steps = 0
    state_file = config['control']['state_file']
    init_state_file = resolve_online_init_state_file(config)
    best_state_file = config['control']['best_state_file']
    manager = Manager()
    shared_stats = {
        'count': manager.Value('i', 0),
        'mean': manager.Value('d', 0.0),
        'M2': manager.Value('d', 0.0),
        'lock': manager.Lock()
            }
    if path.exists(state_file):
        state = torch.load(state_file, weights_only=False, map_location=device)
        timestamp = datetime.fromtimestamp(state['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
        logging.info(f'loaded: {timestamp}')
        mortal.load_state_dict(state['mortal'])
        Old_mortal.load_state_dict(state['mortal'])
        policy_net.load_state_dict(state['policy_net'])
        Old_policy_net.load_state_dict(state['policy_net'])
        if aux_net is not None and 'aux_net' in state:
            aux_net.load_state_dict(state['aux_net'])
        if opponent_aux_net is not None and 'opponent_aux_net' in state:
            opponent_aux_net.load_state_dict(state['opponent_aux_net'])
        if danger_aux_net is not None and 'danger_aux_net' in state:
            danger_aux_net.load_state_dict(state['danger_aux_net'])
        if oracle_brain is not None and 'oracle_brain' in state:
            oracle_brain.load_state_dict(state['oracle_brain'])
        if value_net is not None and 'value_net' in state:
            value_net.load_state_dict(state['value_net'])
        if tile_eff_net is not None and 'tile_eff_net' in state:
            tile_eff_net.load_state_dict(state['tile_eff_net'])
        if furo_regret_net is not None and 'furo_regret_net' in state:
            furo_regret_net.load_state_dict(state['furo_regret_net'])
        if hand_value_regret_net is not None and 'hand_value_regret_net' in state:
            hand_value_regret_net.load_state_dict(state['hand_value_regret_net'])
        if exp_reward_net is not None and 'exp_reward_net' in state:
            exp_reward_net.load_state_dict(state['exp_reward_net'])
        if checkpoint_supports_online_resume(state, current_config=config, optimizer=optimizer):
            optimizer.load_state_dict(state['optimizer'])
            scheduler.load_state_dict(state['scheduler'])
            if 'shared_stats' in state and state['shared_stats'] is not None:
                shared_stats['count'].value = state['shared_stats']['count']
                shared_stats['mean'].value = state['shared_stats']['mean']
                shared_stats['M2'].value = float(state['shared_stats']['variance'] * state['shared_stats']['count'])
            scaler.load_state_dict(state['scaler'])
            best_perf = state['best_perf']
            steps = state['steps']
            if 'dynamic_entropy_weight' in state:
                dynamic_entropy_weight = state['dynamic_entropy_weight']
            if 'log_entropy_alpha' in state:
                log_entropy_alpha = state['log_entropy_alpha']
            else:
                log_entropy_alpha = math.log(max(dynamic_entropy_weight, 1e-8))
            logging.info('resumed optimizer/scheduler state from checkpoint')
        else:
            logging.info(
                'initialized training from checkpoint weights only; '
                'optimizer/scheduler/scaler/best_perf were reset'
            )
    elif init_state_file:
        ensure_online_init_state_file_ready(init_state_file)
        state = torch.load(init_state_file, weights_only=False, map_location=device)
        bridge_info = load_brain_state_with_input_bridge(mortal, state['mortal'])
        Old_mortal.load_state_dict(mortal.state_dict())
        policy_net.load_state_dict(state['policy_net'])
        Old_policy_net.load_state_dict(state['policy_net'])
        if aux_net is not None and state.get('aux_net') is not None:
            aux_net.load_state_dict(state['aux_net'])
        if oracle_brain is not None:
            # Initialize oracle brain from supervised brain weights via bridge
            load_brain_state_with_input_bridge(oracle_brain, state['mortal'])
        timestamp = datetime.fromtimestamp(state['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
        logging.info(
            'initialized online weights from supervised checkpoint: %s (%s); '
            'brain bridge loaded=%s skipped=%s',
            init_state_file,
            timestamp,
            len(bridge_info['loaded_keys']),
            len(bridge_info['skipped_keys']),
        )

    optimizer.zero_grad(set_to_none=True)

    if device.type == 'cuda':
        logging.info(f'device: {device} ({torch.cuda.get_device_name(device)})')
    else:
        logging.info(f'device: {device}')

    if online:
        submit_param(mortal, policy_net, is_idle=True)
        logging.info('param has been submitted')

    writer = SummaryWriter(config['control']['tensorboard_dir'])
    stats = {
        'important_ratio': 0,
        'ratio_var': 0,
        'ratio_max': 0,
        'entropy': 0,
        'loss': 0,
        'aux_loss': 0,
        'opp_loss': 0,
        'danger_loss': 0,
        'value_loss': 0,
        'exp_reward_loss': 0,
        'tile_eff_loss': 0,
        'furo_regret_loss': 0,
        'hand_value_regret_loss': 0,
        'stats': 0,
    }
    idx = 0

    def train_epoch():
        nonlocal steps
        nonlocal idx
        nonlocal stats
        nonlocal Old_mortal
        nonlocal Old_policy_net
        nonlocal dynamic_entropy_weight
        nonlocal log_entropy_alpha
        if online:
            player_names = ['trainee']
            dirname = drain()
            file_list = list(map(lambda p: path.join(dirname, p), os.listdir(dirname)))
        else:
            player_names_set = set()
            for filename in config['dataset']['player_names_files']:
                with open(filename) as f:
                    player_names_set.update(filtered_trimmed_lines(f))
            player_names = list(player_names_set)
            logging.info(f'loaded {len(player_names):,} players')

            file_index = config['dataset']['file_index']
            if path.exists(file_index):
                index = torch.load(file_index, weights_only=True)
                file_list = index['file_list']
            else:
                logging.info('building file index...')
                file_list = []
                for pat in config['dataset']['globs']:
                    file_list.extend(glob(pat, recursive=True))
                if len(player_names_set) > 0:
                    filtered = []
                    for filename in tqdm(file_list, unit='file'):
                        with gzip.open(filename, 'rt') as f:
                            start = json.loads(next(f))
                            if not set(start['names']).isdisjoint(player_names_set):
                                filtered.append(filename)
                    file_list = filtered
                file_list.sort(reverse=True)
                torch.save({'file_list': file_list}, file_index)
        logging.info(f'file list size: {len(file_list):,}')

        before_next_test_play = (test_every - steps % test_every) % test_every
        logging.info(f'total steps: {steps:,} (~{before_next_test_play:,})')

        if num_workers > 1:
            random.shuffle(file_list)
        file_data = FileDatasetsIter(
            version = version,
            file_list = file_list,
            pts = pts,
            shared_stats=shared_stats,
            oracle = oracle_critic,
            file_batch_size = file_batch_size,
            reserve_ratio = reserve_ratio,
            player_names = player_names,
            num_epochs = num_epochs,
            enable_augmentation = enable_augmentation,
            augmented_first = augmented_first,
            emit_opponent_state_labels = online_opp_enabled,
            track_danger_labels = online_danger_enabled,
            track_regret_labels = online_regret_enabled,
        )
        data_loader_kwargs = {
            'dataset': file_data,
            'batch_size': batch_size,
            'drop_last': False,
            'num_workers': num_workers,
            'pin_memory': True,
            'worker_init_fn': worker_init_fn,
        }
        if num_workers > 0:
            data_loader_kwargs['persistent_workers'] = True
            data_loader_kwargs['prefetch_factor'] = prefetch_factor
        data_loader = iter(DataLoader(**data_loader_kwargs))

        remaining_obs = []
        remaining_invisible_obs = []
        remaining_actions = []
        remaining_masks = []
        remaining_advantage = []
        remaining_player_rank = []
        remaining_extra = []  # opp/danger labels (variable-length tail)
        remaining_bs = 0
        pb = tqdm(total=save_every, desc='TRAIN', initial=steps % save_every)

        def train_batch(obs, actions, masks, advantage, player_rank,
                        invisible_obs=None, opp_shanten=None, opp_tenpai=None,
                        danger_valid=None, danger_any=None,
                        danger_value=None, danger_player_mask=None,
                        tile_eff_valid=None, tile_eff_delta=None,
                        furo_valid=None, furo_label=None,
                        hand_value_valid=None, hand_value_points=None):
            nonlocal steps
            nonlocal idx
            nonlocal pb
            nonlocal Old_mortal
            nonlocal Old_policy_net
            nonlocal dynamic_entropy_weight
            nonlocal log_entropy_alpha

            obs = obs.to(dtype=torch.float32, device=device)
            actions = actions.to(dtype=torch.int64, device=device)
            masks = masks.to(dtype=torch.bool, device=device)
            advantage = advantage.to(dtype=torch.float32, device=device)
            player_rank = player_rank.to(dtype=torch.int64, device=device)
            assert masks[range(batch_size), actions].all()

            with torch.no_grad():
                with torch.autocast(device.type, enabled=enable_amp):
                    old_logits = Old_policy_net.logits(Old_mortal(obs), masks)
                    if logit_thres > 0:
                        old_logits = old_logits.clamp(-logit_thres, logit_thres).masked_fill(~masks, -torch.inf)
                    old_dist = Categorical(logits=old_logits)
                    old_log_prob = old_dist.log_prob(actions)

            with torch.autocast(device.type, enabled=enable_amp):
                phi = mortal(obs)
                logits = policy_net.logits(phi, masks)
                if logit_thres > 0:
                    logits = logits.clamp(-logit_thres, logit_thres).masked_fill(~masks, -torch.inf)
                dist = Categorical(logits=logits)
                new_log_prob = dist.log_prob(actions)
                ratio = (new_log_prob - old_log_prob).exp()

                # V-trace IS truncation: clip ratio before advantage weighting
                # to reduce variance from off-policy samples.
                if vtrace_rho_clip > 0:
                    rho = ratio.clamp(max=vtrace_rho_clip)
                else:
                    rho = ratio

                loss1 = rho * advantage
                loss2 = torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * advantage
                min_loss = torch.min(loss1, loss2)

                clip_loss = torch.where(
                advantage < 0,
                torch.max(min_loss , dual_clip * advantage),
                min_loss
                )
                entropy = dist.entropy().view(-1, 1)
                entropy_loss = entropy * dynamic_entropy_weight

                loss = -(clip_loss + entropy_loss).mean()

                # AuxNet auxiliary loss (rank prediction)
                # Auto-reduce weight when ValueHead is active: ValueHead already
                # encodes rank information via its 4-player value output.
                aux_loss_val = torch.tensor(0.0, device=device)
                effective_rank_weight = next_rank_weight
                if aux_net is not None:
                    if value_enabled:
                        effective_rank_weight = min(next_rank_weight, 0.05)
                    rank_logits = aux_net(phi.detach())[0]
                    aux_loss_val = nn.functional.cross_entropy(rank_logits, player_rank)
                    loss = loss + effective_rank_weight * aux_loss_val

                # Opponent State auxiliary loss
                opp_loss_val = torch.tensor(0.0, device=device)
                if opponent_aux_net is not None and opp_shanten is not None:
                    opp_shanten_dev = opp_shanten.to(dtype=torch.int64, device=device)
                    opp_tenpai_dev = opp_tenpai.to(dtype=torch.int64, device=device)
                    shanten_logits, tenpai_logits = opponent_aux_net(phi.detach())
                    shanten_losses = []
                    tenpai_losses = []
                    for i in range(3):
                        shanten_losses.append(
                            nn.functional.cross_entropy(shanten_logits[i], opp_shanten_dev[:, i], reduction='mean')
                        )
                        tenpai_losses.append(
                            nn.functional.cross_entropy(tenpai_logits[i], opp_tenpai_dev[:, i], reduction='mean')
                        )
                    shanten_loss = torch.stack(shanten_losses).mean()
                    tenpai_loss = torch.stack(tenpai_losses).mean()
                    opp_loss_val = opponent_shanten_weight * shanten_loss + opponent_tenpai_weight * tenpai_loss
                    loss = loss + online_opponent_weight * opp_loss_val

                # Danger auxiliary loss
                danger_loss_val = torch.tensor(0.0, device=device)
                if danger_aux_net is not None and danger_valid is not None:
                    danger_valid_dev = danger_valid.to(dtype=torch.bool, device=device)
                    danger_any_dev = danger_any.to(dtype=torch.float32, device=device)
                    danger_value_dev = danger_value.to(dtype=torch.float32, device=device)
                    danger_player_dev = danger_player_mask.to(dtype=torch.float32, device=device)
                    any_logits, value_pred, player_logits = danger_aux_net(phi.detach())
                    # Eligible mask: valid steps AND legal discards (first 37 actions)
                    eligible = danger_valid_dev.unsqueeze(-1) & masks[:, :37]
                    eligible_count = eligible.sum().clamp(min=1.0)
                    # any loss: BCE on eligible tiles
                    any_loss = nn.functional.binary_cross_entropy_with_logits(
                        any_logits, danger_any_dev, reduction='none'
                    )
                    any_loss = (any_loss * eligible).sum() / eligible_count
                    # value loss: smooth L1 on positive-danger tiles
                    import math as _math
                    _dvc_log = _math.log1p(danger_value_cap)
                    value_positive = eligible & (danger_any_dev > 0.5)
                    positive_target = torch.log1p(danger_value_dev.clamp(min=0, max=danger_value_cap)) / _dvc_log
                    value_loss = nn.functional.smooth_l1_loss(
                        value_pred.sigmoid(), positive_target, reduction='none'
                    )
                    value_pos_count = value_positive.sum().clamp(min=1.0)
                    value_loss = (value_loss * value_positive).sum() / value_pos_count
                    # player loss: BCE on eligible tiles
                    eligible_player = eligible.unsqueeze(-1).expand_as(player_logits)
                    eligible_player_count = eligible_player.sum().clamp(min=1.0)
                    player_loss = nn.functional.binary_cross_entropy_with_logits(
                        player_logits, danger_player_dev, reduction='none'
                    )
                    player_loss = (player_loss * eligible_player).sum() / eligible_player_count
                    danger_loss_val = (
                        danger_mix_weights[0] * any_loss
                        + danger_mix_weights[1] * value_loss
                        + danger_mix_weights[2] * player_loss
                    )
                    loss = loss + online_danger_weight * danger_loss_val

                # Suphx-style oracle dropout schedule: linear decay
                if oracle_dropout_steps > 0 and oracle_dropout_start > 0:
                    oracle_drop = oracle_dropout_start * max(0.0, 1.0 - steps / oracle_dropout_steps)
                else:
                    oracle_drop = 0.0

                # Compute oracle features once, reuse for ValueHead + ExpectedRewardNet
                oracle_phi_cached = None
                if oracle_critic and oracle_brain is not None and invisible_obs is not None:
                    invisible_obs_dev = invisible_obs.to(dtype=torch.float32, device=device)
                    oracle_phi_cached = oracle_brain(obs, invisible_obs=invisible_obs_dev, oracle_dropout=oracle_drop)

                # Oracle Critic / Value Head loss (RVR-style)
                value_loss_val = torch.tensor(0.0, device=device)
                if value_enabled and value_net is not None:
                    if oracle_phi_cached is not None:
                        value_pred = value_net(oracle_phi_cached)
                    else:
                        value_pred = value_net(phi.detach())
                    # Train the trainee's value prediction against kyoku-level advantage
                    trainee_value = value_pred[:, 0]
                    value_loss_val = nn.functional.mse_loss(trainee_value, advantage)
                    # RVR zero-sum constraint: sum of all 4 player values should be 0
                    if zero_sum_weight > 0:
                        value_sum = value_pred.sum(dim=-1)
                        value_loss_val = value_loss_val + zero_sum_weight * value_sum.square().mean()
                    loss = loss + value_weight * value_loss_val

                # Local Regret Heads
                tile_eff_loss_val = torch.tensor(0.0, device=device)
                if tile_eff_net is not None and tile_eff_valid is not None:
                    tile_eff_pred = tile_eff_net(phi.detach())
                    te_valid = tile_eff_valid.to(device=device)
                    if te_valid.any():
                        te_target = tile_eff_delta.to(dtype=torch.float32, device=device)
                        tile_eff_loss_val = nn.functional.smooth_l1_loss(
                            tile_eff_pred[te_valid], te_target[te_valid])
                        loss = loss + tile_eff_weight * tile_eff_loss_val

                furo_regret_loss_val = torch.tensor(0.0, device=device)
                if furo_regret_net is not None and furo_valid is not None:
                    furo_pred = furo_regret_net(phi.detach())
                    fr_valid = furo_valid.to(device=device)
                    if fr_valid.any():
                        fr_target = furo_label.to(dtype=torch.float32, device=device)
                        # Furo label is [called, shanten_before, shanten_after]
                        # FuroRegretHead outputs 2-dim [call_regret, pass_regret]
                        # Map: call label -> target = [shanten_before - shanten_after, 0]
                        #       pass label -> target = [0, shanten_after - shanten_before]
                        called = fr_target[fr_valid, 0]  # 1.0 or 0.0
                        sh_before = fr_target[fr_valid, 1]
                        sh_after = fr_target[fr_valid, 2]
                        sh_delta = sh_before - sh_after  # positive = shanten improved
                        furo_target = torch.zeros_like(furo_pred[fr_valid])
                        furo_target[:, 0] = called * sh_delta  # call regret
                        furo_target[:, 1] = (1.0 - called) * (-sh_delta)  # pass regret
                        furo_regret_loss_val = nn.functional.smooth_l1_loss(
                            furo_pred[fr_valid], furo_target)
                        loss = loss + furo_regret_weight * furo_regret_loss_val

                hand_value_regret_loss_val = torch.tensor(0.0, device=device)
                if hand_value_regret_net is not None and hand_value_valid is not None:
                    hv_pred = hand_value_regret_net(phi.detach())
                    hv_valid = hand_value_valid.to(device=device)
                    if hv_valid.any():
                        hv_target = hand_value_points.to(dtype=torch.float32, device=device)
                        # Normalize to [0, 1] range for training stability
                        hv_target_norm = hv_target[hv_valid] / 32000.0
                        hv_pred_norm = hv_pred[hv_valid].sigmoid()  # output in [0, 1]
                        hand_value_regret_loss_val = nn.functional.smooth_l1_loss(
                            hv_pred_norm, hv_target_norm)
                        loss = loss + hand_value_regret_weight * hand_value_regret_loss_val

                # Expected Reward Network (reuses cached oracle features)
                exp_reward_loss_val = torch.tensor(0.0, device=device)
                if exp_reward_net is not None and steps >= exp_reward_warmup:
                    if oracle_phi_cached is not None:
                        oracle_phi_for_reward = oracle_phi_cached.detach()
                    else:
                        oracle_phi_for_reward = phi.detach()
                    reward_pred = exp_reward_net(oracle_phi_for_reward)
                    trainee_reward = reward_pred[:, 0]
                    exp_reward_loss_val = nn.functional.smooth_l1_loss(trainee_reward, advantage)
                    loss = loss + exp_reward_weight * exp_reward_loss_val

            scaler.scale(loss / opt_step_every).backward()

            # Multiplicative (log-space) entropy adjustment
            if entropy_target > 0:
                log_entropy_alpha += entropy_adjust_rate * (entropy_target - entropy.mean().item())
                log_entropy_alpha = max(math.log(1e-4), min(math.log(1e-2), log_entropy_alpha))
                dynamic_entropy_weight = math.exp(log_entropy_alpha)

            with torch.inference_mode():
                stats['important_ratio'] += ratio.mean()
                stats['ratio_var'] += ratio.var()
                stats['ratio_max'] += ratio.max()
                stats['entropy'] += entropy.mean()
                stats['loss'] += loss
                stats['aux_loss'] += aux_loss_val
                stats['opp_loss'] += opp_loss_val
                stats['danger_loss'] += danger_loss_val
                stats['value_loss'] += value_loss_val
                stats['exp_reward_loss'] += exp_reward_loss_val
                stats['tile_eff_loss'] += tile_eff_loss_val
                stats['furo_regret_loss'] += furo_regret_loss_val
                stats['hand_value_regret_loss'] += hand_value_regret_loss_val

            steps += 1
            idx += 1
            if idx % opt_step_every == 0:
                if max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    params = chain.from_iterable(g['params'] for g in optimizer.param_groups)
                    clip_grad_norm_(params, max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            pb.update(1)

            if online and steps % submit_every == 0:
                submit_param(mortal, policy_net, is_idle=False)
                logging.info('param has been submitted')

            if steps % save_every == 0:
                pb.close()

                writer.add_scalar('important_ratio/ratio', stats['important_ratio'] / save_every, steps)
                writer.add_scalar('important_ratio/variance', stats['ratio_var'] / save_every, steps)
                writer.add_scalar('important_ratio/max', stats['ratio_max'] / save_every, steps)
                writer.add_scalar('entropy/entropy', stats['entropy'] / save_every, steps)
                writer.add_scalar('entropy/dynamic_weight', dynamic_entropy_weight, steps)
                writer.add_scalar('loss', stats['loss'] / save_every, steps)
                if aux_net is not None:
                    writer.add_scalar('aux_loss', stats['aux_loss'] / save_every, steps)
                if online_opp_enabled:
                    writer.add_scalar('opp_loss', stats['opp_loss'] / save_every, steps)
                if online_danger_enabled:
                    writer.add_scalar('danger_loss', stats['danger_loss'] / save_every, steps)
                if value_enabled:
                    writer.add_scalar('value_loss', stats['value_loss'] / save_every, steps)
                if exp_reward_enabled:
                    writer.add_scalar('exp_reward_loss', stats['exp_reward_loss'] / save_every, steps)
                if tile_eff_weight > 0:
                    writer.add_scalar('tile_eff_loss', stats['tile_eff_loss'] / save_every, steps)
                if furo_regret_weight > 0:
                    writer.add_scalar('furo_regret_loss', stats['furo_regret_loss'] / save_every, steps)
                if hand_value_regret_weight > 0:
                    writer.add_scalar('hand_value_regret_loss', stats['hand_value_regret_loss'] / save_every, steps)
                if not online:
                    pass
                writer.flush()

                for k in stats:
                    stats[k] = 0
                idx = 0

                before_next_test_play = (test_every - steps % test_every) % test_every
                logging.info(f'total steps: {steps:,} (~{before_next_test_play:,})')
                stats_dict = save_shared_stats(shared_stats, steps, writer)
                state = {
                    'mortal': mortal.state_dict(),
                    'policy_net': policy_net.state_dict(),
                    'aux_net': aux_net.state_dict() if aux_net is not None else None,
                    'opponent_aux_net': opponent_aux_net.state_dict() if opponent_aux_net is not None else None,
                    'danger_aux_net': danger_aux_net.state_dict() if danger_aux_net is not None else None,
                    'oracle_brain': oracle_brain.state_dict() if oracle_brain is not None else None,
                    'value_net': value_net.state_dict() if value_net is not None else None,
                    'tile_eff_net': tile_eff_net.state_dict() if tile_eff_net is not None else None,
                    'furo_regret_net': furo_regret_net.state_dict() if furo_regret_net is not None else None,
                    'hand_value_regret_net': hand_value_regret_net.state_dict() if hand_value_regret_net is not None else None,
                    'exp_reward_net': exp_reward_net.state_dict() if exp_reward_net is not None else None,
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'scaler': scaler.state_dict(),
                    'steps': steps,
                    'timestamp': datetime.now().timestamp(),
                    'best_perf': best_perf,
                    'config': config,
                    'shared_stats': stats_dict,
                    'dynamic_entropy_weight': dynamic_entropy_weight,
                    'log_entropy_alpha': log_entropy_alpha,
                }
                torch.save(state, state_file)

                if online and steps % submit_every != 0:
                    submit_param(mortal, policy_net, is_idle=False)
                    logging.info('param has been submitted')

                if steps % old_update_every == 0:
                    Old_mortal = deepcopy(mortal)
                    Old_policy_net = deepcopy(policy_net)
    
                if steps % test_every == 0:
                    stat = test_player.test_play(test_games // 4, mortal, policy_net, device)
                    mortal.train()
                    policy_net.train()
                    if aux_net is not None:
                        aux_net.train()
                    if opponent_aux_net is not None:
                        opponent_aux_net.train()
                    if danger_aux_net is not None:
                        danger_aux_net.train()
                    if oracle_brain is not None:
                        oracle_brain.train()
                    if value_net is not None:
                        value_net.train()
                    
                    
                    avg_pt = stat.avg_pt([90, 45, 0, -135]) # for display only, never used in training
                    better = avg_pt >= best_perf['avg_pt'] and stat.avg_rank <= best_perf['avg_rank']
                    if better:
                        past_best = best_perf.copy()
                        best_perf['avg_pt'] = avg_pt
                        best_perf['avg_rank'] = stat.avg_rank

                    logging.info(f'avg rank: {stat.avg_rank:.6}')
                    logging.info(f'avg pt: {avg_pt:.6}')
                    writer.add_scalar('test_play/avg_ranking', stat.avg_rank, steps)
                    writer.add_scalar('test_play/avg_pt', avg_pt, steps)
                    writer.add_scalars('test_play/ranking', {
                        '1st': stat.rank_1_rate,
                        '2nd': stat.rank_2_rate,
                        '3rd': stat.rank_3_rate,
                        '4th': stat.rank_4_rate,
                    }, steps)
                    writer.add_scalars('test_play/behavior', {
                        'agari': stat.agari_rate,
                        'houjuu': stat.houjuu_rate,
                        'fuuro': stat.fuuro_rate,
                        'riichi': stat.riichi_rate,
                    }, steps)
                    writer.add_scalars('test_play/agari_point', {
                        'overall': stat.avg_point_per_agari,
                        'riichi': stat.avg_point_per_riichi_agari,
                        'fuuro': stat.avg_point_per_fuuro_agari,
                        'dama': stat.avg_point_per_dama_agari,
                    }, steps)
                    writer.add_scalar('test_play/houjuu_point', stat.avg_point_per_houjuu, steps)
                    writer.add_scalar('test_play/point_per_round', stat.avg_point_per_round, steps)
                    writer.add_scalars('test_play/key_step', {
                        'agari_jun': stat.avg_agari_jun,
                        'houjuu_jun': stat.avg_houjuu_jun,
                        'riichi_jun': stat.avg_riichi_jun,
                    }, steps)
                    writer.add_scalars('test_play/riichi', {
                        'agari_after_riichi': stat.agari_rate_after_riichi,
                        'houjuu_after_riichi': stat.houjuu_rate_after_riichi,
                        'chasing_riichi': stat.chasing_riichi_rate,
                        'riichi_chased': stat.riichi_chased_rate,
                    }, steps)
                    writer.add_scalar('test_play/riichi_point', stat.avg_riichi_point, steps)
                    writer.add_scalars('test_play/fuuro', {
                        'agari_after_fuuro': stat.agari_rate_after_fuuro,
                        'houjuu_after_fuuro': stat.houjuu_rate_after_fuuro,
                    }, steps)
                    writer.add_scalar('test_play/fuuro_num', stat.avg_fuuro_num, steps)
                    writer.add_scalar('test_play/fuuro_point', stat.avg_fuuro_point, steps)
                    writer.flush()

                    if better:
                        torch.save(state, state_file)
                        logging.info(
                            'a new record has been made, '
                            f'pt: {past_best["avg_pt"]:.4} -> {best_perf["avg_pt"]:.4}, '
                            f'rank: {past_best["avg_rank"]:.4} -> {best_perf["avg_rank"]:.4}, '
                            f'saving to {best_state_file}'
                        )
                        shutil.copy(state_file, best_state_file)
                    if online:
                        # BUG: This is a bug with unknown reason. When training
                        # in online mode, the process will get stuck here. This
                        # is the reason why `main` spawns a sub process to train
                        # in online mode instead of going for training directly.
                        sys.exit(0)
                pb = tqdm(total=save_every, desc='TRAIN')

        def _unpack_batch(batch):
            """Unpack variable-length data tuple into named fields."""
            it = iter(batch)
            obs = next(it)
            invisible_obs_batch = next(it) if oracle_critic else None
            actions = next(it)
            masks = next(it)
            advantage = next(it)
            player_rank = next(it)
            opp_shanten = next(it) if online_opp_enabled else None
            opp_tenpai = next(it) if online_opp_enabled else None
            danger_valid = next(it) if online_danger_enabled else None
            danger_any = next(it) if online_danger_enabled else None
            danger_value_b = next(it) if online_danger_enabled else None
            danger_player = next(it) if online_danger_enabled else None
            tile_eff_valid = next(it) if online_regret_enabled else None
            tile_eff_delta = next(it) if online_regret_enabled else None
            furo_valid = next(it) if online_regret_enabled else None
            furo_label = next(it) if online_regret_enabled else None
            hand_value_valid = next(it) if online_regret_enabled else None
            hand_value_points = next(it) if online_regret_enabled else None
            return {
                'obs': obs, 'invisible_obs': invisible_obs_batch,
                'actions': actions, 'masks': masks,
                'advantage': advantage, 'player_rank': player_rank,
                'opp_shanten': opp_shanten, 'opp_tenpai': opp_tenpai,
                'danger_valid': danger_valid, 'danger_any': danger_any,
                'danger_value': danger_value_b, 'danger_player_mask': danger_player,
                'tile_eff_valid': tile_eff_valid, 'tile_eff_delta': tile_eff_delta,
                'furo_valid': furo_valid, 'furo_label': furo_label,
                'hand_value_valid': hand_value_valid, 'hand_value_points': hand_value_points,
            }

        def _call_train_batch(fields, start=None, end=None):
            """Call train_batch from a fields dict, optionally slicing."""
            if start is not None:
                s = {k: (v[start:end] if v is not None else None) for k, v in fields.items()}
            else:
                s = fields
            train_batch(
                s['obs'], s['actions'], s['masks'], s['advantage'], s['player_rank'],
                invisible_obs=s['invisible_obs'],
                opp_shanten=s['opp_shanten'], opp_tenpai=s['opp_tenpai'],
                danger_valid=s['danger_valid'], danger_any=s['danger_any'],
                danger_value=s['danger_value'], danger_player_mask=s['danger_player_mask'],
                tile_eff_valid=s['tile_eff_valid'], tile_eff_delta=s['tile_eff_delta'],
                furo_valid=s['furo_valid'], furo_label=s['furo_label'],
                hand_value_valid=s['hand_value_valid'], hand_value_points=s['hand_value_points'],
            )

        remaining_fields = []  # list of field dicts

        for batch in data_loader:
            fields = _unpack_batch(batch)
            bs = fields['obs'].shape[0]
            if bs != batch_size:
                remaining_fields.append(fields)
                remaining_bs += bs
                continue
            _call_train_batch(fields)

        if remaining_bs >= batch_size and remaining_fields:
            # Concatenate all remaining fields
            cat_fields = {}
            for key in remaining_fields[0]:
                tensors = [f[key] for f in remaining_fields if f[key] is not None]
                cat_fields[key] = torch.cat(tensors, dim=0) if tensors else None

            start = 0
            end = batch_size
            while end <= remaining_bs:
                _call_train_batch(cat_fields, start, end)
                start = end
                end += batch_size
        pb.close()

        if online:
            submit_param(mortal, policy_net, is_idle=True)
            logging.info('param has been submitted')

    def save_shared_stats(shared_stats, steps, writer):
  
        count = shared_stats['count'].value
        mean = shared_stats['mean'].value
        m2 = shared_stats['M2'].value

        if count == 0:
            return {'count': 0, 'mean': 0, 'variance': 0}
        
        if count > 0:
       
            variance = m2 / count if count > 1 else 0
            std_dev = math.sqrt(variance)
            writer.add_scalar('stats/count', count, steps)
            writer.add_scalar('stats/mean', mean, steps)
            writer.add_scalar('stats/std_dev', std_dev, steps)
            return {
            'count': count,
            'mean': mean,
            'variance': variance,
            'std_dev': std_dev
            }
       
  

    while True:
        train_epoch()
        gc.collect()
        # torch.cuda.empty_cache()
        # torch.cuda.synchronize()
        if not online:
            # only run one epoch for offline for easier control
            break
    

def main():
    import os
    import sys
    import time
    from subprocess import Popen
    from config import config

    # do not set this env manually
    is_sub_proc_key = 'MORTAL_IS_SUB_PROC'
    online = config['control']['online']
    if not online or os.environ.get(is_sub_proc_key, '0') == '1':
        train()
        return

    cmd = (sys.executable, __file__)
    env = {
        is_sub_proc_key: '1',
        **os.environ.copy(),
    }
    while True:
        child = Popen(
            cmd,
            stdin = sys.stdin,
            stdout = sys.stdout,
            stderr = sys.stderr,
            env = env,
        )
        if (code := child.wait()) != 0:
            sys.exit(code)
        time.sleep(3)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
