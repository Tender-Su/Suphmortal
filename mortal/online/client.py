import mortal.core.prelude

import logging
import socket
import torch
import numpy as np
import time
import gc
from os import path
from mortal.core.model import Brain, CategoricalPolicy
from mortal.eval.player import TrainPlayer
from mortal.core.common import send_msg, recv_msg
from mortal.config import config
from mortal.core.checkpoint_utils import load_brain_state_with_input_bridge
from mortal.eval.oracle_experiments import apply_oracle_experiment_to_config, normalize_oracle_input_mode
from mortal.eval.search_runtime import build_search_runtime_bundle_from_state_file
from mortal.core.repro import apply_reproducibility


def remote_socket(remote, *, timeout_sec: float, retry_sec: float, max_wait_sec: float):
    deadline = None if max_wait_sec <= 0 else time.monotonic() + max_wait_sec
    attempts = 0
    while True:
        attempts += 1
        conn = socket.socket()
        try:
            if timeout_sec > 0:
                conn.settimeout(timeout_sec)
            conn.connect(remote)
            conn.settimeout(None)
            if attempts > 1:
                logging.info('connected to online server after %s attempts', attempts)
            return conn
        except OSError:
            conn.close()
            if deadline is not None and time.monotonic() >= deadline:
                raise
            if attempts == 1 or attempts % 10 == 0:
                logging.warning(
                    'online server connect failed; retrying (attempt=%s remote=%s)',
                    attempts,
                    remote,
                    exc_info=True,
                )
            time.sleep(max(retry_sec, 0.01))


def resolve_search_runtime_source_file():
    control_state = str(config.get('control', {}).get('state_file', '') or '').strip()
    if control_state and path.exists(control_state):
        return control_state
    online_state = str(config.get('online', {}).get('init_state_file', '') or '').strip()
    if online_state and path.exists(online_state):
        return online_state
    return ''

def main():
    apply_oracle_experiment_to_config(config)
    repro_runtime = apply_reproducibility(config, process_name='client')
    remote = (config['online']['remote']['host'], config['online']['remote']['port'])
    remote_cfg = config['online'].get('remote', {})
    connect_timeout_sec = float(remote_cfg.get('connect_timeout_sec', 10.0) or 0.0)
    connect_retry_sec = float(remote_cfg.get('connect_retry_sec', 3.0) or 0.0)
    connect_max_wait_sec = float(remote_cfg.get('connect_max_wait_sec', 300.0) or 0.0)
    device = torch.device(config['control']['device'])
    version = config['control']['version']
    num_blocks = config['resnet']['num_blocks']
    conv_channels = config['resnet']['conv_channels']
    oracle_guiding_cfg = config.get('oracle_guiding', {})
    actor_oracle_enabled = bool(
        oracle_guiding_cfg.get('actor_enabled', False)
        if isinstance(oracle_guiding_cfg, dict)
        else False
    )

    mortal = Brain(
        version=version,
        num_blocks=num_blocks,
        conv_channels=conv_channels,
        is_oracle=actor_oracle_enabled,
        Norm="GN",
    ).to(device).eval()
    dqn = CategoricalPolicy().to(device).eval()
    if config['online']['enable_compile']:
        mortal.compile()
        dqn.compile()

    train_player = TrainPlayer()
    if repro_runtime.enabled:
        logging.info(
            'repro mode active: process=%s base_seed=%s process_seed=%s train_key=%s train_seed_start=%s cudnn_benchmark=%s strict_cuda=%s',
            repro_runtime.process_name,
            repro_runtime.base_seed,
            repro_runtime.process_seed,
            repro_runtime.train_key,
            repro_runtime.train_seed_start,
            repro_runtime.cudnn_benchmark,
            repro_runtime.strict_cuda,
        )
    search_state_file = resolve_search_runtime_source_file()
    search_runtime_bundle = build_search_runtime_bundle_from_state_file(
        search_state_file,
        device=device,
        enable_compile=config['online']['enable_compile'],
    )
    param_version = -1

    pts = np.array([90, 45, 0, -135])
    history_window = config['online']['history_window']
    history = []

    while True:
        while True:
            with remote_socket(
                remote,
                timeout_sec=connect_timeout_sec,
                retry_sec=connect_retry_sec,
                max_wait_sec=connect_max_wait_sec,
            ) as conn:
                msg = {
                    'type': 'get_param',
                    'param_version': param_version,
                }
                send_msg(conn, msg)
                rsp = recv_msg(conn, map_location=device)
                if rsp['status'] == 'ok':
                    param_version = rsp['param_version']
                    break
                time.sleep(3)
        runtime = rsp.get('runtime', {})
        runtime_actor_oracle = bool(runtime.get('actor_oracle_enabled', actor_oracle_enabled))
        if runtime_actor_oracle and not actor_oracle_enabled:
            raise RuntimeError(
                'trainer is publishing Oracle actor weights, but local '
                'oracle_guiding.actor_enabled=false; sync config first'
            )
        actor_oracle_keep_prob = float(
            runtime.get(
                'actor_oracle_keep_prob',
                1.0 if runtime_actor_oracle else 0.0,
            )
        )
        actor_oracle_input_mode = normalize_oracle_input_mode(
            runtime.get(
                'actor_oracle_source',
                'true' if runtime_actor_oracle else 'zero',
            ),
            field_name='runtime.actor_oracle_source',
        )
        load_brain_state_with_input_bridge(mortal, rsp['mortal'])
        dqn.load_state_dict(rsp['dqn'])
        if search_runtime_bundle is not None:
            search_runtime_bundle.load_payload(rsp.get('aux_payload'))
        logging.info(
            'param has been updated (actor_oracle=%s, keep_prob=%.4f)',
            runtime_actor_oracle,
            actor_oracle_keep_prob,
        )

        rankings, file_list = train_player.train_play(
            mortal,
            dqn,
            device,
            actor_oracle_enabled=runtime_actor_oracle,
            actor_oracle_keep_prob=actor_oracle_keep_prob,
            actor_oracle_input_mode=actor_oracle_input_mode,
            search_runtime_bundle=search_runtime_bundle,
        )
        avg_rank = rankings @ np.arange(1, 5) / rankings.sum()
        avg_pt = rankings @ pts / rankings.sum()

        history.append(np.array(rankings))
        if len(history) > history_window:
            del history[0]
        sum_rankings = np.sum(history, axis=0)
        ma_avg_rank = sum_rankings @ np.arange(1, 5) / sum_rankings.sum()
        ma_avg_pt = sum_rankings @ pts / sum_rankings.sum()

        logging.info(f'trainee rankings: {rankings} ({avg_rank:.6}, {avg_pt:.6}pt)')
        logging.info(f'last {len(history)} sessions: {sum_rankings} ({ma_avg_rank:.6}, {ma_avg_pt:.6}pt)')

        logs = {}
        for filename in file_list:
            with open(filename, 'rb') as f:
                logs[path.basename(filename)] = f.read()

        with remote_socket(
            remote,
            timeout_sec=connect_timeout_sec,
            retry_sec=connect_retry_sec,
            max_wait_sec=connect_max_wait_sec,
        ) as conn:
            send_msg(conn, {
                'type': 'submit_replay',
                'logs': logs,
                'param_version': param_version,
            })
            logging.info('logs have been submitted')
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
