"""Run a Mortal-compatible checkpoint on RiichiLab.

The bot token is read only from ``RIICHILAB_BOT_TOKEN`` so it never appears in
the command line or the checked-in configuration.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import socket
import struct
import time
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from mortal.eval.one_vs_three import load_mortal_engine
from mortal.eval.riichienv_compare import MortalAgent


ENDPOINTS = {
    'validate': 'wss://game.riichi.dev/ws/validate',
    'ranked': 'wss://game.riichi.dev/ws/ranked',
}
RATING_API_BASE = 'https://api.riichi.dev'


def _write_jsonl(path: Path | None, direction: str, message: dict) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        'time_unix': time.time(),
        'direction': direction,
        'message': message,
    }
    with path.open('a', encoding='utf-8') as output_file:
        output_file.write(json.dumps(record, ensure_ascii=False) + '\n')


def _append_jsonl(path: Path | None, record: dict) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as output_file:
        output_file.write(json.dumps(record, ensure_ascii=False) + '\n')


def _write_json_atomic(path: Path | None, payload: dict) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + '.tmp')
    temp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    temp_path.replace(path)


def _make_response(agent: MortalAgent, observation, request_id=None) -> dict:
    action = agent.act(observation)
    response = json.loads(action.to_mjai())
    if not isinstance(response, dict) or 'type' not in response:
        raise RuntimeError(f'RiichiEnv returned an invalid MJAI action: {response!r}')
    if request_id is not None:
        response['request_id'] = request_id
    return response


def _load_engine(checkpoint: Path, *, device: str, name: str):
    engine_cfg = {
        'state_file': str(checkpoint.resolve()),
        'device': device,
        'enable_compile': False,
        'enable_amp': device.startswith('cuda'),
        'enable_rule_based_agari_guard': True,
        'enable_metadata': False,
        'name': name,
    }
    return load_mortal_engine(engine_cfg, enable_metadata=False)


def _protocol_log_path(args, *, game_index: int, attempt_index: int) -> Path | None:
    if args.output_dir:
        return (
            Path(args.output_dir).resolve()
            / 'protocol'
            / f'game_{game_index:04d}_attempt_{attempt_index:03d}.jsonl'
        )
    return Path(args.output_jsonl).resolve() if args.output_jsonl else None


def _summary_log_path(args) -> Path | None:
    if args.summary_jsonl:
        return Path(args.summary_jsonl).resolve()
    if args.output_dir:
        return Path(args.output_dir).resolve() / 'games.jsonl'
    return None


def _progress_path(args) -> Path | None:
    if args.progress_json:
        return Path(args.progress_json).resolve()
    if args.output_dir:
        return Path(args.output_dir).resolve() / 'progress.json'
    return None


def _redact_error(exc: Exception, token: str) -> str:
    return str(exc).replace(token, '<redacted>')


def _network_path(args) -> dict:
    physical_direct_ip = getattr(args, 'physical_direct_ip', None)
    if physical_direct_ip:
        return {
            'mode': 'physical_direct',
            'direct_ip': physical_direct_ip,
            'source_address': getattr(args, 'physical_source_address', None),
            'interface_index': getattr(args, 'physical_interface_index', None),
        }
    proxy_url = getattr(args, 'proxy_url', None)
    if proxy_url:
        return {'mode': 'explicit_proxy', 'proxy_url': proxy_url}
    if getattr(args, 'no_proxy', False):
        return {'mode': 'no_explicit_proxy'}
    return {'mode': 'auto_proxy'}


def _stop_policy_path(args) -> Path | None:
    explicit_path = getattr(args, 'stop_policy_json', None)
    if explicit_path:
        return Path(explicit_path).resolve()
    if getattr(args, 'output_dir', None):
        return Path(args.output_dir).resolve() / 'stop_policy.json'
    return None


def _load_rating_stop_config(args) -> dict | None:
    policy_path = _stop_policy_path(args)
    policy = {}
    if policy_path is not None and policy_path.is_file():
        policy = json.loads(policy_path.read_text(encoding='utf-8'))
        if not isinstance(policy, dict):
            raise ValueError(f'RiichiLab stop policy must be a JSON object: {policy_path}')
    elif getattr(args, 'stop_policy_json', None):
        raise FileNotFoundError(policy_path)

    stop_rating_at = policy.get(
        'stop_rating_at',
        getattr(args, 'stop_rating_at', None),
    )
    if stop_rating_at is None:
        return None

    rating_bot_id = policy.get('rating_bot_id', getattr(args, 'rating_bot_id', None))
    if rating_bot_id is None:
        raise ValueError('rating stop requires --rating-bot-id or rating_bot_id in policy')

    activation_total_games = policy.get(
        'activation_total_games',
        getattr(args, 'rating_activation_total_games', None),
    )
    config = {
        'stop_rating_at': float(stop_rating_at),
        'rating_bot_id': int(rating_bot_id),
        'activation_total_games': (
            None if activation_total_games is None else int(activation_total_games)
        ),
        'rating_api_base': str(
            policy.get(
                'rating_api_base',
                getattr(args, 'rating_api_base', RATING_API_BASE),
            )
        ).rstrip('/'),
        'rating_request_timeout': float(
            policy.get(
                'rating_request_timeout',
                getattr(args, 'rating_request_timeout', 15.0),
            )
        ),
        'rating_poll_seconds': float(
            policy.get(
                'rating_poll_seconds',
                getattr(args, 'rating_poll_seconds', 2.0),
            )
        ),
        'policy_path': str(policy_path) if policy_path is not None else None,
    }
    if config['rating_bot_id'] <= 0:
        raise ValueError('rating_bot_id must be positive')
    if config['activation_total_games'] is not None and config['activation_total_games'] < 0:
        raise ValueError('activation_total_games must be non-negative')
    if config['rating_request_timeout'] <= 0:
        raise ValueError('rating_request_timeout must be positive')
    if config['rating_poll_seconds'] <= 0:
        raise ValueError('rating_poll_seconds must be positive')
    return config


def _fetch_bot_rating_snapshot(config: dict) -> dict:
    url = f"{config['rating_api_base']}/api/v1/bots/{config['rating_bot_id']}"
    request = Request(
        url,
        headers={
            'Accept': 'application/json',
            'User-Agent': 'MahjongAI-RiichiLab-runner/1',
        },
    )
    with urlopen(request, timeout=config['rating_request_timeout']) as response:
        payload = json.load(response)
    if not isinstance(payload, dict) or payload.get('ok') is not True:
        raise RuntimeError(f'RiichiLab rating API returned an error: {payload!r}')
    data = payload.get('data')
    if not isinstance(data, dict):
        raise RuntimeError(f'RiichiLab rating API omitted bot data: {payload!r}')
    return {
        'rating': float(data['rating']),
        'total_games': int(data['total_games']),
        'last_played_at': data.get('last_played_at'),
        'checked_at_unix': time.time(),
    }


async def _wait_for_bot_rating(
    config: dict,
    *,
    min_total_games: int | None = None,
) -> dict:
    while True:
        try:
            snapshot = await asyncio.to_thread(_fetch_bot_rating_snapshot, config)
        except Exception as exc:
            print(
                json.dumps(
                    {
                        'event': 'rating_check_retry',
                        'error_type': type(exc).__name__,
                        'error': str(exc),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        else:
            if min_total_games is None or snapshot['total_games'] >= min_total_games:
                return snapshot
            print(
                json.dumps(
                    {
                        'event': 'rating_api_pending',
                        'total_games': snapshot['total_games'],
                        'expected_total_games': min_total_games,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        await asyncio.sleep(config['rating_poll_seconds'])


def _rating_target_reached(config: dict, snapshot: dict) -> bool:
    activation_total_games = config['activation_total_games']
    return (
        activation_total_games is not None
        and snapshot['total_games'] > activation_total_games
        and snapshot['rating'] >= config['stop_rating_at']
    )


def _record_rating_snapshot(progress: dict, snapshot: dict) -> None:
    progress['latest_rating'] = snapshot['rating']
    progress['rating_total_games'] = snapshot['total_games']
    progress['rating_last_played_at'] = snapshot['last_played_at']
    progress['rating_checked_at_unix'] = snapshot['checked_at_unix']


async def _open_physical_direct_socket(args, endpoint: str) -> socket.socket:
    parsed = urlsplit(endpoint)
    if parsed.scheme != 'wss' or parsed.hostname is None:
        raise ValueError('physical direct mode requires a wss:// endpoint')
    port = parsed.port or 443
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.setsockopt(
            socket.IPPROTO_IP,
            getattr(socket, 'IP_UNICAST_IF', 31),
            struct.pack('!I', args.physical_interface_index),
        )
        if args.physical_source_address:
            sock.bind((args.physical_source_address, 0))
        sock.setblocking(False)
        await asyncio.wait_for(
            asyncio.get_running_loop().sock_connect(
                sock,
                (args.physical_direct_ip, port),
            ),
            timeout=args.open_timeout,
        )
        return sock
    except Exception:
        sock.close()
        raise


async def _run_single_game(
    args,
    *,
    token: str,
    engine,
    game_index: int,
    attempt_index: int,
) -> dict:
    try:
        import websockets
        from websockets.exceptions import ConnectionClosed
        from riichienv import Observation
    except ImportError as exc:
        raise RuntimeError(
            'RiichiLab requires riichienv and websockets; install them in the Mortal Python environment'
        ) from exc

    log_path = _protocol_log_path(
        args,
        game_index=game_index,
        attempt_index=attempt_index,
    )
    endpoint = args.url or ENDPOINTS[args.mode]
    headers = {'Authorization': f'Bearer {token}'}
    agent = None
    started_at_unix = time.time()
    summary = {
        'status': 'completed',
        'game_index': game_index,
        'attempt_index': attempt_index,
        'endpoint': endpoint,
        'mode': args.mode,
        'checkpoint': str(Path(args.checkpoint).resolve()),
        'network_path': _network_path(args),
        'websockets_version': getattr(websockets, '__version__', None),
        'started_at_unix': started_at_unix,
        'seat': None,
        'requests': 0,
        'accepted': 0,
        'defaulted': 0,
        'rejected': 0,
        'validation_passed': None,
        'end_game_received': False,
        'end_scores': None,
    }

    direct_sock = None
    connect_kwargs = {}
    if getattr(args, 'physical_direct_ip', None):
        direct_sock = await _open_physical_direct_socket(args, endpoint)
        connect_kwargs.update(sock=direct_sock, proxy=None)
    elif getattr(args, 'proxy_url', None):
        connect_kwargs['proxy'] = args.proxy_url
    elif getattr(args, 'no_proxy', False):
        connect_kwargs['proxy'] = None

    try:
        async with websockets.connect(
            endpoint,
            additional_headers=headers,
            open_timeout=args.open_timeout,
            ping_interval=20,
            ping_timeout=20,
            max_size=8 << 20,
            **connect_kwargs,
        ) as websocket:
            while True:
                message = json.loads(await websocket.recv())
                _write_jsonl(log_path, 'server', message)
                message_type = message.get('type')

                if message_type == 'start_game':
                    seat = int(message['id'])
                    summary['seat'] = seat
                    agent = MortalAgent(engine, seat)
                elif message_type == 'request_action':
                    if agent is None:
                        raise RuntimeError('request_action arrived before start_game')
                    observation = Observation.deserialize_from_base64(message['observation'])
                    response = _make_response(agent, observation, message.get('request_id'))
                    _write_jsonl(log_path, 'client', response)
                    await websocket.send(json.dumps(response, separators=(',', ':')))
                    summary['requests'] += 1
                elif message_type == 'action_ack':
                    status = message.get('status')
                    if status in ('accepted', 'defaulted', 'rejected'):
                        summary[status] += 1
                    if status in ('rejected', 'unparseable'):
                        raise RuntimeError(f'RiichiLab rejected an action: {message}')
                elif message_type == 'validation_result':
                    summary['validation_passed'] = bool(message.get('passed'))
                    break
                elif message_type == 'end_game':
                    summary['end_game_received'] = True
                    summary['end_scores'] = message.get('scores')
                    if args.mode == 'ranked':
                        break
                elif 'error' in message:
                    raise RuntimeError(f'RiichiLab server error: {message["error"]}')
    except ConnectionClosed as exc:
        if not summary['end_game_received'] and summary['validation_passed'] is None:
            raise RuntimeError(f'RiichiLab connection closed before completion: {exc}') from exc
    finally:
        if direct_sock is not None and direct_sock.fileno() != -1:
            direct_sock.close()

    if args.mode == 'validate' and summary['validation_passed'] is not True:
        raise RuntimeError(f'RiichiLab validation failed: {summary}')
    if args.mode == 'ranked' and not summary['end_game_received']:
        raise RuntimeError(f'RiichiLab ranked game ended without end_game: {summary}')

    finished_at_unix = time.time()
    summary['finished_at_unix'] = finished_at_unix
    summary['elapsed_seconds'] = finished_at_unix - started_at_unix
    return summary


async def _run_games(args, *, token: str, engine) -> dict:
    checkpoint = str(Path(args.checkpoint).resolve())
    summary_path = _summary_log_path(args)
    progress_path = _progress_path(args)
    rating_stop = _load_rating_stop_config(args)
    target_games = None if rating_stop is not None else args.games
    now = time.time()
    progress = {
        'schema_version': 1,
        'status': 'running',
        'mode': args.mode,
        'checkpoint': checkpoint,
        'network_path': _network_path(args),
        'target_games': target_games,
        'completed_games': 0,
        'failed_attempts': 0,
        'requests': 0,
        'accepted': 0,
        'defaulted': 0,
        'rejected': 0,
        'started_at_unix': now,
        'updated_at_unix': now,
        'latest_game': None,
        'last_error': None,
        'stop_reason': None,
    }

    if args.resume and progress_path is not None and progress_path.is_file():
        prior = json.loads(progress_path.read_text(encoding='utf-8'))
        if prior.get('mode') != args.mode or prior.get('checkpoint') != checkpoint:
            raise RuntimeError('resume progress does not match mode/checkpoint')
        progress.update(prior)
        progress['status'] = 'running'
        progress['target_games'] = target_games
        progress['network_path'] = _network_path(args)
        progress['updated_at_unix'] = time.time()
        progress['stop_reason'] = None
        progress.pop('finished_at_unix', None)

    rating_snapshot = None
    if rating_stop is not None:
        prior_activation = progress.get('rating_activation_total_games')
        if rating_stop['activation_total_games'] is None and prior_activation is not None:
            rating_stop['activation_total_games'] = int(prior_activation)
        progress.update(
            {
                'stop_rating_at': rating_stop['stop_rating_at'],
                'rating_bot_id': rating_stop['rating_bot_id'],
                'rating_activation_total_games': rating_stop['activation_total_games'],
                'rating_policy_path': rating_stop['policy_path'],
            }
        )
    else:
        for field in (
            'stop_rating_at',
            'rating_bot_id',
            'rating_activation_total_games',
            'rating_policy_path',
            'latest_rating',
            'rating_total_games',
            'rating_last_played_at',
            'rating_checked_at_unix',
        ):
            progress.pop(field, None)

    completed_games = int(progress['completed_games'])
    consecutive_errors = 0
    attempt_index = completed_games + int(progress['failed_attempts'])
    _write_json_atomic(progress_path, progress)

    if rating_stop is not None:
        rating_snapshot = await _wait_for_bot_rating(rating_stop)
        if rating_stop['activation_total_games'] is None:
            rating_stop['activation_total_games'] = rating_snapshot['total_games']
            progress['rating_activation_total_games'] = rating_snapshot['total_games']
        _record_rating_snapshot(progress, rating_snapshot)
        progress['updated_at_unix'] = time.time()
        _write_json_atomic(progress_path, progress)

    while True:
        if rating_stop is None:
            if completed_games >= args.games:
                progress['stop_reason'] = 'game_target_reached'
                break
        elif _rating_target_reached(rating_stop, rating_snapshot):
            progress['stop_reason'] = 'rating_target_reached'
            break

        game_index = completed_games + 1
        attempt_index += 1
        try:
            summary = await _run_single_game(
                args,
                token=token,
                engine=engine,
                game_index=game_index,
                attempt_index=attempt_index,
            )
        except Exception as exc:
            consecutive_errors += 1
            progress['failed_attempts'] = int(progress['failed_attempts']) + 1
            progress['updated_at_unix'] = time.time()
            progress['last_error'] = {
                'game_index': game_index,
                'attempt_index': attempt_index,
                'type': type(exc).__name__,
                'message': _redact_error(exc, token),
            }
            error_record = {
                'status': 'error',
                'time_unix': progress['updated_at_unix'],
                **progress['last_error'],
            }
            _append_jsonl(summary_path, error_record)
            _write_json_atomic(progress_path, progress)
            print(json.dumps({'event': 'retry', **error_record}, ensure_ascii=False), flush=True)

            if consecutive_errors >= args.max_consecutive_errors:
                progress['status'] = 'failed'
                _write_json_atomic(progress_path, progress)
                raise RuntimeError(
                    f'giving up after {consecutive_errors} consecutive errors'
                ) from exc

            if rating_stop is not None:
                rating_snapshot = await _wait_for_bot_rating(rating_stop)
                _record_rating_snapshot(progress, rating_snapshot)
                progress['updated_at_unix'] = time.time()
                _write_json_atomic(progress_path, progress)
                if _rating_target_reached(rating_stop, rating_snapshot):
                    progress['stop_reason'] = 'rating_target_reached'
                    break

            delay = min(
                args.retry_max_seconds,
                args.retry_min_seconds * (2 ** (consecutive_errors - 1)),
            )
            if delay > 0:
                await asyncio.sleep(delay)
            continue

        consecutive_errors = 0
        completed_games += 1
        progress['completed_games'] = completed_games
        for field in ('requests', 'accepted', 'defaulted', 'rejected'):
            progress[field] = int(progress[field]) + int(summary[field])
        progress['updated_at_unix'] = time.time()
        progress['latest_game'] = summary
        progress['last_error'] = None
        _append_jsonl(summary_path, summary)
        _write_json_atomic(progress_path, progress)

        if rating_stop is not None:
            expected_total_games = rating_snapshot['total_games'] + 1
            rating_snapshot = await _wait_for_bot_rating(
                rating_stop,
                min_total_games=expected_total_games,
            )
            _record_rating_snapshot(progress, rating_snapshot)
            progress['updated_at_unix'] = time.time()
            _write_json_atomic(progress_path, progress)

        print(
            json.dumps(
                {
                    'event': 'game_completed',
                    'completed_games': completed_games,
                    'target_games': target_games,
                    'stop_rating_at': (
                        None if rating_stop is None else rating_stop['stop_rating_at']
                    ),
                    'latest_rating': progress.get('latest_rating'),
                    'game': summary,
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    progress['status'] = 'complete'
    progress['finished_at_unix'] = time.time()
    progress['updated_at_unix'] = progress['finished_at_unix']
    _write_json_atomic(progress_path, progress)
    return progress


async def run_bot(args) -> dict:
    token = os.environ.get('RIICHILAB_BOT_TOKEN')
    if not token:
        raise RuntimeError('RIICHILAB_BOT_TOKEN is not set')

    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if args.games <= 0:
        raise ValueError('--games must be positive')
    if args.mode == 'validate' and args.games != 1:
        raise ValueError('validation mode supports exactly one game')
    rating_stop = _load_rating_stop_config(args)
    if rating_stop is not None and args.mode != 'ranked':
        raise ValueError('rating stop is supported only in ranked mode')
    if args.max_consecutive_errors <= 0:
        raise ValueError('--max-consecutive-errors must be positive')
    if not 0 <= args.retry_min_seconds <= args.retry_max_seconds:
        raise ValueError('retry seconds must satisfy 0 <= min <= max')
    physical_enabled = any(
        value is not None
        for value in (
            args.physical_direct_ip,
            args.physical_source_address,
            args.physical_interface_index,
        )
    )
    if physical_enabled and (
        args.physical_direct_ip is None or args.physical_interface_index is None
    ):
        raise ValueError(
            'physical direct mode requires --physical-direct-ip and '
            '--physical-interface-index'
        )
    if args.physical_interface_index is not None and args.physical_interface_index <= 0:
        raise ValueError('--physical-interface-index must be positive')

    engine = _load_engine(checkpoint, device=args.device, name=args.name)
    return await _run_games(args, token=token, engine=engine)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--mode', choices=sorted(ENDPOINTS), default='validate')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--name', default='S70')
    parser.add_argument('--url', help='Override the RiichiLab WebSocket endpoint')
    network_group = parser.add_mutually_exclusive_group()
    network_group.add_argument('--proxy-url', help='Force an HTTP/SOCKS proxy URL')
    network_group.add_argument('--no-proxy', action='store_true', help='Disable explicit proxies')
    network_group.add_argument(
        '--physical-direct-ip',
        help='Connect to this IPv4 address while preserving endpoint Host/SNI',
    )
    parser.add_argument('--physical-source-address')
    parser.add_argument('--physical-interface-index', type=int)
    parser.add_argument('--open-timeout', type=float, default=15.0)
    parser.add_argument('--games', type=int, default=1, help='Total games to complete')
    parser.add_argument(
        '--stop-rating-at',
        type=float,
        help='Ignore the game count and stop after a later completed game reaches this rating',
    )
    parser.add_argument('--rating-bot-id', type=int)
    parser.add_argument('--rating-activation-total-games', type=int)
    parser.add_argument('--rating-api-base', default=RATING_API_BASE)
    parser.add_argument('--rating-request-timeout', type=float, default=15.0)
    parser.add_argument('--rating-poll-seconds', type=float, default=2.0)
    parser.add_argument(
        '--stop-policy-json',
        help='Rating-stop policy JSON; defaults to OUTPUT_DIR/stop_policy.json when present',
    )
    parser.add_argument('--output-jsonl', help='Combined protocol log (mainly for one game)')
    parser.add_argument('--output-dir', help='Batch root with per-attempt protocol logs')
    parser.add_argument('--summary-jsonl', help='One summary record per game/failed attempt')
    parser.add_argument('--progress-json', help='Atomic cumulative progress snapshot')
    parser.add_argument('--resume', action='store_true', help='Resume from progress JSON')
    parser.add_argument('--max-consecutive-errors', type=int, default=20)
    parser.add_argument('--retry-min-seconds', type=float, default=5.0)
    parser.add_argument('--retry-max-seconds', type=float, default=300.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = asyncio.run(run_bot(args))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
