"""Compare TLS stability to RiichiLab across local network paths.

The probe never sends a bot token and never joins the ranked queue.  It only
performs a certificate-verified HTTPS request to the gameserver root.
"""

from __future__ import annotations

import argparse
import json
import socket
import ssl
import statistics
import struct
import time
from pathlib import Path


MODES = ('tun', 'proxy', 'physical_direct')


def _read_headers(sock: socket.socket) -> bytes:
    data = bytearray()
    while b'\r\n\r\n' not in data:
        chunk = sock.recv(4096)
        if not chunk:
            break
        data.extend(chunk)
        if len(data) > 64 * 1024:
            raise RuntimeError('response headers exceed 64 KiB')
    return bytes(data)


def _open_tcp(args, mode: str) -> tuple[socket.socket, str]:
    if mode == 'tun':
        sock = socket.create_connection((args.host, args.port), args.timeout)
        return sock, str(sock.getpeername()[0])

    if mode == 'proxy':
        sock = socket.create_connection((args.proxy_host, args.proxy_port), args.timeout)
        sock.settimeout(args.timeout)
        request = (
            f'CONNECT {args.host}:{args.port} HTTP/1.1\r\n'
            f'Host: {args.host}:{args.port}\r\n'
            'Proxy-Connection: keep-alive\r\n\r\n'
        )
        sock.sendall(request.encode('ascii'))
        response = _read_headers(sock)
        status_line = response.split(b'\r\n', 1)[0]
        if b' 200 ' not in status_line:
            raise RuntimeError(
                f'proxy CONNECT failed: {status_line.decode("ascii", "replace")}'
            )
        return sock, f'{args.proxy_host}:{args.proxy_port}'

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(args.timeout)
    ip_unicast_if = getattr(socket, 'IP_UNICAST_IF', 31)
    sock.setsockopt(
        socket.IPPROTO_IP,
        ip_unicast_if,
        struct.pack('!I', args.interface_index),
    )
    sock.bind((args.source_address, 0))
    sock.connect((args.direct_ip, args.port))
    return sock, str(sock.getpeername()[0])


def _probe_once(args, mode: str, attempt: int) -> dict:
    started = time.perf_counter()
    sock = None
    tls_sock = None
    try:
        sock, peer = _open_tcp(args, mode)
        connected = time.perf_counter()
        context = ssl.create_default_context()
        tls_sock = context.wrap_socket(sock, server_hostname=args.host)
        sock = None
        negotiated = time.perf_counter()
        request = (
            f'HEAD / HTTP/1.1\r\nHost: {args.host}\r\n'
            'Connection: close\r\n\r\n'
        )
        tls_sock.sendall(request.encode('ascii'))
        response = _read_headers(tls_sock)
        status_line = response.split(b'\r\n', 1)[0].decode('ascii', 'replace')
        if not status_line.startswith('HTTP/'):
            raise RuntimeError(f'invalid HTTP response: {status_line!r}')
        finished = time.perf_counter()
        return {
            'mode': mode,
            'attempt': attempt,
            'success': True,
            'peer': peer,
            'status_line': status_line,
            'tcp_seconds': connected - started,
            'tls_seconds': negotiated - connected,
            'total_seconds': finished - started,
        }
    except Exception as exc:
        return {
            'mode': mode,
            'attempt': attempt,
            'success': False,
            'error_type': type(exc).__name__,
            'error': str(exc),
            'total_seconds': time.perf_counter() - started,
        }
    finally:
        if tls_sock is not None:
            tls_sock.close()
        if sock is not None:
            sock.close()


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(len(ordered) * fraction))
    return ordered[index]


def _summarize(records: list[dict], mode: str) -> dict:
    selected = [record for record in records if record['mode'] == mode]
    successful = [record for record in selected if record['success']]
    totals = [record['total_seconds'] for record in successful]
    errors: dict[str, int] = {}
    for record in selected:
        if record['success']:
            continue
        error_type = record['error_type']
        errors[error_type] = errors.get(error_type, 0) + 1
    return {
        'mode': mode,
        'attempts': len(selected),
        'successes': len(successful),
        'failures': len(selected) - len(successful),
        'success_rate': len(successful) / len(selected) if selected else None,
        'median_total_seconds': statistics.median(totals) if totals else None,
        'p95_total_seconds': _percentile(totals, 0.95),
        'max_total_seconds': max(totals) if totals else None,
        'errors': errors,
        'peers': sorted({record['peer'] for record in successful}),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='game.riichi.dev')
    parser.add_argument('--port', type=int, default=443)
    parser.add_argument('--direct-ip', required=True)
    parser.add_argument('--source-address', required=True)
    parser.add_argument('--interface-index', type=int, required=True)
    parser.add_argument('--proxy-host', default='127.0.0.1')
    parser.add_argument('--proxy-port', type=int, default=7897)
    parser.add_argument('--attempts', type=int, default=50)
    parser.add_argument('--timeout', type=float, default=8.0)
    parser.add_argument('--interval', type=float, default=0.1)
    parser.add_argument('--output-json')
    parser.add_argument('--quiet-records', action='store_true')
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.attempts <= 0:
        raise ValueError('--attempts must be positive')

    records = []
    started_at_unix = time.time()
    for attempt in range(1, args.attempts + 1):
        for mode in MODES:
            record = _probe_once(args, mode, attempt)
            records.append(record)
            if not args.quiet_records:
                print(json.dumps(record, ensure_ascii=False), flush=True)
            if args.interval > 0:
                time.sleep(args.interval)

    payload = {
        'schema_version': 1,
        'started_at_unix': started_at_unix,
        'finished_at_unix': time.time(),
        'target': {
            'host': args.host,
            'port': args.port,
            'direct_ip': args.direct_ip,
            'source_address': args.source_address,
            'interface_index': args.interface_index,
            'proxy': f'{args.proxy_host}:{args.proxy_port}',
        },
        'summaries': [_summarize(records, mode) for mode in MODES],
        'records': records,
    }
    if args.output_json:
        output_path = Path(args.output_json).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding='utf-8',
        )
    print(
        json.dumps(
            {
                'event': 'summary',
                **{key: value for key, value in payload.items() if key != 'records'},
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == '__main__':
    main()
