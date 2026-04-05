import atexit
import json
import logging
import os
import time
import traceback
import torch
import numpy as np
from torch.distributions import Normal, Categorical
from libriichi.consts import oracle_obs_shape
from typing import *


def coerce_batch_inputs(obs, masks, invisible_obs):
    obs = np.ascontiguousarray(
        obs if isinstance(obs, np.ndarray) else np.stack(obs, axis=0),
        dtype=np.float32,
    )
    masks = np.ascontiguousarray(
        masks if isinstance(masks, np.ndarray) else np.stack(masks, axis=0),
        dtype=np.bool_,
    )
    if invisible_obs is not None:
        invisible_obs = np.ascontiguousarray(
            invisible_obs if isinstance(invisible_obs, np.ndarray) else np.stack(invisible_obs, axis=0),
            dtype=np.float32,
        )
    return obs, masks, invisible_obs


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    value = value.strip().lower()
    if value in {'1', 'true', 'yes', 'on'}:
        return True
    if value in {'0', 'false', 'no', 'off'}:
        return False
    return default

class MortalEngine:
    def __init__(
        self,
        brain,
        dqn,
        is_oracle,
        version,
        device = None,
        stochastic_latent = False,
        enable_amp = False,
        enable_quick_eval = True,
        enable_rule_based_agari_guard = False,
        enable_metadata = True,
        name = 'NoName',
        explore_rate = 0,
    ):
        self.engine_type = 'mortal'
        self.device = device or torch.device('cpu')
        assert isinstance(self.device, torch.device)
        self.brain = brain.to(self.device).eval()
        self.dqn = dqn.to(self.device).eval()   # 
        self.is_oracle = is_oracle
        self.version = version
        self.stochastic_latent = stochastic_latent

        self.enable_amp = enable_amp
        self.enable_quick_eval = enable_quick_eval
        self.enable_rule_based_agari_guard = enable_rule_based_agari_guard
        self.enable_metadata = enable_metadata
        self.name = name
        self.explore_rate = explore_rate
        self.profile_enabled = _env_flag('MORTAL_ENGINE_PROFILE', False)
        self._profile_stats = {
            'calls': 0,
            'batch_items': 0,
            'max_batch_size': 0,
            'prepare_elapsed': 0.0,
            'forward_elapsed': 0.0,
            'select_elapsed': 0.0,
        }
        if self.profile_enabled:
            atexit.register(self._log_profile)

    def react_batch(self, obs, masks, invisible_obs):
        try:
            with (
                torch.autocast(self.device.type, enabled=self.enable_amp),
                torch.inference_mode(),
            ):
                return self._react_batch(obs, masks, invisible_obs)
        except Exception as ex:
            raise Exception(f'{ex}\n{traceback.format_exc()}')

    def react_batch_action_only(self, obs, masks, invisible_obs):
        try:
            with (
                torch.autocast(self.device.type, enabled=self.enable_amp),
                torch.inference_mode(),
            ):
                actions, alt_actions, _, _, _ = self._react_batch_impl(obs, masks, invisible_obs)
                return actions.tolist(), alt_actions.tolist()
        except Exception as ex:
            raise Exception(f'{ex}\n{traceback.format_exc()}')

    def _react_batch(self, obs, masks, invisible_obs):
        actions, _, q_out, masks, is_greedy = self._react_batch_impl(obs, masks, invisible_obs)
        return actions.tolist(), q_out.tolist(), masks.tolist(), is_greedy.tolist()

    def _prepare_batch_tensors(self, obs, masks, invisible_obs):
        if not (
            isinstance(obs, torch.Tensor)
            and isinstance(masks, torch.Tensor)
            and (invisible_obs is None or isinstance(invisible_obs, torch.Tensor))
        ):
            obs, masks, invisible_obs = coerce_batch_inputs(obs, masks, invisible_obs)

        def _to_tensor(value, *, dtype):
            if value is None:
                return None
            tensor = value if isinstance(value, torch.Tensor) else torch.as_tensor(value)
            if tensor.dtype != dtype:
                tensor = tensor.to(dtype=dtype)
            if not tensor.is_contiguous():
                tensor = tensor.contiguous()
            if tensor.device != self.device:
                tensor = tensor.to(
                    device=self.device,
                    non_blocking=self.device.type == 'cuda' and tensor.device.type == 'cpu',
                )
            return tensor

        obs = _to_tensor(obs, dtype=torch.float32)
        masks = _to_tensor(masks, dtype=torch.bool)
        invisible_obs = _to_tensor(invisible_obs, dtype=torch.float32)
        return obs, masks, invisible_obs

    def _react_batch_impl(self, obs, masks, invisible_obs):
        prepare_started = time.perf_counter()
        obs, masks, invisible_obs = self._prepare_batch_tensors(obs, masks, invisible_obs)
        prepare_elapsed = time.perf_counter() - prepare_started
        batch_size = obs.shape[0]

        forward_started = time.perf_counter()
        match self.version:
            case 1:
                mu, logsig = self.brain(obs, invisible_obs)
                if self.stochastic_latent:
                    latent = Normal(mu, logsig.exp() + 1e-6).sample()
                else:
                    latent = mu
                q_out = self.dqn(latent, masks)
            case 2 | 3 | 4:
                # Handle oracle Brain: if Brain is oracle but no invisible_obs,
                # fill with zeros so inference works without oracle data
                if self.brain.is_oracle:
                    if invisible_obs is None:
                        oracle_ch = oracle_obs_shape(self.version)[0]
                        invisible_obs = torch.zeros(
                            batch_size, oracle_ch, obs.shape[2],
                            dtype=obs.dtype, device=self.device,
                        )
                    phi = self.brain(obs, invisible_obs)
                else:
                    phi = self.brain(obs)
                q_out = self.dqn(phi, masks)
        forward_elapsed = time.perf_counter() - forward_started

        select_started = time.perf_counter()
        if self.explore_rate > 0:
            is_greedy = torch.full((batch_size,), 1-self.explore_rate, device=self.device).bernoulli().to(torch.bool)
            actions = torch.where(is_greedy, q_out.argmax(-1), Categorical(probs=q_out).sample())
        else:
            is_greedy = torch.ones(batch_size, dtype=torch.bool, device=self.device)
            actions = q_out.argmax(-1)

        q_out_alt = q_out.clone()
        q_out_alt[:, 43] = float('-inf')
        alt_actions = q_out_alt.argmax(-1)
        select_elapsed = time.perf_counter() - select_started

        if self.profile_enabled:
            stats = self._profile_stats
            stats['calls'] += 1
            stats['batch_items'] += int(batch_size)
            stats['max_batch_size'] = max(stats['max_batch_size'], int(batch_size))
            stats['prepare_elapsed'] += prepare_elapsed
            stats['forward_elapsed'] += forward_elapsed
            stats['select_elapsed'] += select_elapsed

        return actions, alt_actions, q_out, masks, is_greedy

    def _log_profile(self):
        stats = self._profile_stats
        calls = int(stats['calls'])
        if calls <= 0:
            return
        avg_batch_size = stats['batch_items'] / calls
        logging.info(
            'mortal engine profile: name=%s device=%s calls=%d avg_batch_size=%.1f max_batch_size=%d prepare=%.3fs forward=%.3fs select=%.3fs',
            self.name,
            self.device,
            calls,
            avg_batch_size,
            int(stats['max_batch_size']),
            stats['prepare_elapsed'],
            stats['forward_elapsed'],
            stats['select_elapsed'],
        )

class ExampleMjaiLogEngine:
    def __init__(self, name: str):
        self.engine_type = 'mjai-log'
        self.name = name
        self.player_ids = None

    def set_player_ids(self, player_ids: List[int]):
        self.player_ids = player_ids

    def react_batch(self, game_states):
        res = []
        for game_state in game_states:
            game_idx = game_state.game_index
            state = game_state.state
            events_json = game_state.events_json

            events = json.loads(events_json)
            assert events[0]['type'] == 'start_kyoku'

            player_id = self.player_ids[game_idx]
            cans = state.last_cans
            if cans.can_discard:
                tile = state.last_self_tsumo()
                res.append(json.dumps({
                    'type': 'dahai',
                    'actor': player_id,
                    'pai': tile,
                    'tsumogiri': True,
                }))
            else:
                res.append('{"type":"none"}')
        return res

    # They will be executed at specific events. They can be no-op but must be
    # defined.
    def start_game(self, game_idx: int):
        pass
    def end_kyoku(self, game_idx: int):
        pass
    def end_game(self, game_idx: int, scores: List[int]):
        pass
