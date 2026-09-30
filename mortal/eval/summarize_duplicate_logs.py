"""Summarize duplicate 1v3 game logs with seed-set confidence intervals."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

from libriichi.stat import Stat

from mortal.eval.riichienv_compare import summarize_games


def load_challenger_game(log_path, challenger_name):
    with gzip.open(log_path, 'rt', encoding='utf-8') as log_file:
        log_text = log_file.read()
    first_line = log_text.splitlines()[0]
    start_game = json.loads(first_line)
    names = start_game.get('names', [])
    matching_seats = [seat for seat, name in enumerate(names) if name == challenger_name]
    if len(matching_seats) != 1:
        raise ValueError(
            f'{log_path} has {len(matching_seats)} seats named {challenger_name!r}'
        )
    challenger_seat = matching_seats[0]
    rank = int(Stat.from_log(log_text, challenger_seat).avg_rank)
    seed = start_game.get('seed')
    seed_id = int(seed[0] if isinstance(seed, list) else seed)
    return {
        'log_path': str(log_path.resolve()),
        'seed': seed_id,
        'seed_key': int(seed[1]) if isinstance(seed, list) and len(seed) == 2 else None,
        'challenger_seat': challenger_seat,
        'challenger_rank': rank,
    }


def summarize_log_dir(log_dir, challenger_name):
    paths = sorted(Path(log_dir).rglob('*.json.gz'))
    if not paths:
        raise FileNotFoundError(f'no .json.gz game logs found under {log_dir}')
    games = [load_challenger_game(path, challenger_name) for path in paths]
    return {
        'challenger_name': challenger_name,
        'log_dir': str(Path(log_dir).resolve()),
        'games': games,
        'summary': summarize_games(games),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log-dir', required=True)
    parser.add_argument('--challenger-name', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    payload = summarize_log_dir(args.log_dir, args.challenger_name)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    print(json.dumps(payload['summary'], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
