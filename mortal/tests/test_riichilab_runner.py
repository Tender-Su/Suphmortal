import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from integrations.riichilab import run_mortal_bot as runner


class _Action:
    def to_mjai(self):
        return json.dumps({'type': 'none'})


class _Agent:
    def act(self, observation):
        self.observation = observation
        return _Action()


class RiichiLabRunnerTest(unittest.TestCase):
    def test_make_response_echoes_request_id(self):
        agent = _Agent()
        observation = object()

        response = runner._make_response(agent, observation, request_id=42)

        self.assertIs(observation, agent.observation)
        self.assertEqual({'type': 'none', 'request_id': 42}, response)


class RiichiLabBatchRunnerTest(unittest.IsolatedAsyncioTestCase):
    async def test_batch_retries_and_writes_atomic_progress(self):
        calls = []

        async def fake_run_single_game(
            args,
            *,
            token,
            engine,
            game_index,
            attempt_index,
        ):
            calls.append((game_index, attempt_index, token, engine))
            if len(calls) == 1:
                raise RuntimeError(f'temporary failure for {token}')
            return {
                'status': 'completed',
                'game_index': game_index,
                'attempt_index': attempt_index,
                'requests': 2,
                'accepted': 2,
                'defaulted': 0,
                'rejected': 0,
            }

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / 'batch'
            args = SimpleNamespace(
                checkpoint=str(Path(temp_dir) / 'model.pth'),
                mode='ranked',
                games=2,
                output_jsonl=None,
                output_dir=str(output_dir),
                summary_jsonl=None,
                progress_json=None,
                resume=False,
                max_consecutive_errors=3,
                retry_min_seconds=0,
                retry_max_seconds=0,
            )
            engine = object()
            with patch.object(runner, '_run_single_game', new=fake_run_single_game):
                progress = await runner._run_games(
                    args,
                    token='secret-token',
                    engine=engine,
                )

            self.assertEqual([(1, 1), (1, 2), (2, 3)], [(a, b) for a, b, _, _ in calls])
            self.assertTrue(all(token == 'secret-token' for _, _, token, _ in calls))
            self.assertTrue(all(loaded_engine is engine for _, _, _, loaded_engine in calls))
            self.assertEqual('complete', progress['status'])
            self.assertEqual(2, progress['completed_games'])
            self.assertEqual(1, progress['failed_attempts'])
            self.assertEqual(4, progress['accepted'])

            stored_progress = json.loads(
                (output_dir / 'progress.json').read_text(encoding='utf-8')
            )
            summaries = [
                json.loads(line)
                for line in (output_dir / 'games.jsonl').read_text(encoding='utf-8').splitlines()
            ]
            self.assertEqual('complete', stored_progress['status'])
            self.assertEqual(['error', 'completed', 'completed'], [row['status'] for row in summaries])
            self.assertNotIn('secret-token', (output_dir / 'games.jsonl').read_text(encoding='utf-8'))

    async def test_resume_uses_global_attempt_index_and_clears_old_finish_time(self):
        calls = []

        async def fake_run_single_game(
            args,
            *,
            token,
            engine,
            game_index,
            attempt_index,
        ):
            calls.append((game_index, attempt_index))
            return {
                'status': 'completed',
                'game_index': game_index,
                'attempt_index': attempt_index,
                'requests': 1,
                'accepted': 1,
                'defaulted': 0,
                'rejected': 0,
            }

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / 'batch'
            output_dir.mkdir()
            checkpoint = str((Path(temp_dir) / 'model.pth').resolve())
            prior_progress = {
                'schema_version': 1,
                'status': 'complete',
                'mode': 'ranked',
                'checkpoint': checkpoint,
                'target_games': 2,
                'completed_games': 2,
                'failed_attempts': 3,
                'requests': 2,
                'accepted': 2,
                'defaulted': 0,
                'rejected': 0,
                'started_at_unix': 1.0,
                'updated_at_unix': 2.0,
                'finished_at_unix': 2.0,
                'latest_game': None,
                'last_error': None,
            }
            (output_dir / 'progress.json').write_text(
                json.dumps(prior_progress),
                encoding='utf-8',
            )
            args = SimpleNamespace(
                checkpoint=checkpoint,
                mode='ranked',
                games=3,
                output_jsonl=None,
                output_dir=str(output_dir),
                summary_jsonl=None,
                progress_json=None,
                resume=True,
                max_consecutive_errors=3,
                retry_min_seconds=0,
                retry_max_seconds=0,
            )

            with patch.object(runner, '_run_single_game', new=fake_run_single_game):
                progress = await runner._run_games(
                    args,
                    token='secret-token',
                    engine=object(),
                )

            self.assertEqual([(3, 6)], calls)
            self.assertEqual('complete', progress['status'])
            self.assertEqual(3, progress['completed_games'])
            self.assertGreater(progress['finished_at_unix'], 2.0)

    async def test_rating_policy_ignores_game_cap_and_requires_later_reach(self):
        calls = []
        rating_checks = []

        async def fake_run_single_game(
            args,
            *,
            token,
            engine,
            game_index,
            attempt_index,
        ):
            calls.append(game_index)
            return {
                'status': 'completed',
                'game_index': game_index,
                'attempt_index': attempt_index,
                'requests': 1,
                'accepted': 1,
                'defaulted': 0,
                'rejected': 0,
            }

        snapshots = iter(
            [
                {
                    'rating': 1956.0,
                    'total_games': 100,
                    'last_played_at': 'before-activation',
                    'checked_at_unix': 1.0,
                },
                {
                    'rating': 1940.0,
                    'total_games': 101,
                    'last_played_at': 'game-one',
                    'checked_at_unix': 2.0,
                },
                {
                    'rating': 1956.0,
                    'total_games': 102,
                    'last_played_at': 'game-two',
                    'checked_at_unix': 3.0,
                },
            ]
        )

        async def fake_wait_for_bot_rating(config, *, min_total_games=None):
            rating_checks.append(min_total_games)
            return next(snapshots)

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / 'batch'
            output_dir.mkdir()
            (output_dir / 'stop_policy.json').write_text(
                json.dumps(
                    {
                        'schema_version': 1,
                        'stop_rating_at': 1956,
                        'rating_bot_id': 305,
                        'activation_total_games': 100,
                    }
                ),
                encoding='utf-8',
            )
            args = SimpleNamespace(
                checkpoint=str(Path(temp_dir) / 'model.pth'),
                mode='ranked',
                games=1,
                output_jsonl=None,
                output_dir=str(output_dir),
                summary_jsonl=None,
                progress_json=None,
                resume=False,
                max_consecutive_errors=3,
                retry_min_seconds=0,
                retry_max_seconds=0,
            )

            with (
                patch.object(runner, '_run_single_game', new=fake_run_single_game),
                patch.object(
                    runner,
                    '_wait_for_bot_rating',
                    new=fake_wait_for_bot_rating,
                ),
            ):
                progress = await runner._run_games(
                    args,
                    token='secret-token',
                    engine=object(),
                )

            self.assertEqual([1, 2], calls)
            self.assertEqual([None, 101, 102], rating_checks)
            self.assertIsNone(progress['target_games'])
            self.assertEqual('rating_target_reached', progress['stop_reason'])
            self.assertEqual(2, progress['completed_games'])
            self.assertEqual(1956.0, progress['latest_rating'])
            self.assertEqual(102, progress['rating_total_games'])


if __name__ == '__main__':
    unittest.main()
