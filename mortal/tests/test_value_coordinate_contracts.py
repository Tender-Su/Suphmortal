"""NumPy-only probes of production reward/GAE arithmetic, not training integration.

The optional torch/native dependencies are unavailable on the lightweight CPU
runner. Compile only the named, unchanged pure-NumPy production functions from
AST instead of installing fake modules. This deliberately does not test imports,
model inference, trajectory construction, PPO gradients, or the training loop.
Run: python -m unittest mortal.tests.test_value_coordinate_contracts -v
"""

import ast
from pathlib import Path
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def load_numpy_functions(relative_path, names, numpy_name):
    source_path = ROOT / relative_path
    tree = ast.parse(source_path.read_text(encoding='utf-8'), filename=str(source_path))
    selected = [node for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name in names]
    if {node.name for node in selected} != set(names):
        raise AssertionError(f'production functions missing from {relative_path}')
    namespace = {numpy_name: np}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(source_path), 'exec'), namespace)
    return namespace


ORACLE = load_numpy_functions('data/oracle_value.py', {
    'centered_rank_points', 'rank_array', 'terminal_rank_values_by_player',
    'rank_by_score', 'current_rank_values_by_kyoku',
    'terminal_rank_delta_values_by_kyoku', 'score_rank_delta_rewards_by_kyoku',
    'expand_kyoku_rewards_to_steps', 'discounted_returns_from_step_rewards',
    'normalize_oracle_return_mode', 'oracle_values_by_kyoku',
    'oracle_step_value_targets',
}, 'np')
ONLINE = load_numpy_functions('online/train_online.py', {
    'compute_gae_advantages', 'expand_sparse_kyoku_reward_to_steps',
    'compute_gae_advantages_from_step_rewards',
}, '_np')
GAE = ONLINE['compute_gae_advantages_from_step_rewards']


def td_residuals(rewards, values, gamma=1.0):
    next_values = np.concatenate((values[1:], np.zeros_like(values[:1])), axis=0)
    return rewards + gamma * next_values - values


def reference_blocked_gae(rewards, values, gamma, lam, ends):
    """Test-only reverse blocking, carrying next value AND next advantage.

    The production complete-game helper has no block/bootstrap API. This is an
    independent recurrence reference, not a claim that a production block path
    exists or has been exercised.
    """
    advantages = np.empty(len(rewards), dtype=np.float64)
    next_value = next_advantage = 0.0
    starts = [0, *ends[:-1]]
    for start, end in reversed(list(zip(starts, ends))):
        for t in range(end - 1, start - 1, -1):
            delta = float(rewards[t]) + gamma * next_value - float(values[t])
            next_advantage = delta + gamma * lam * next_advantage
            advantages[t] = next_advantage
            next_value = float(values[t])
    return advantages


class ValueCoordinateContractsTest(unittest.TestCase):
    def setUp(self):
        # Four actual kyoku, each with distinct ranks. Rank utility sums to zero.
        self.pts = [2, 1, 0, -3]
        self.features = np.zeros((4, 7), dtype=np.float32)
        self.features[:, 3:] = [
            [40, 30, 20, 10], [30, 40, 10, 20],
            [20, 30, 40, 10], [10, 20, 30, 40],
        ]
        self.final_ranks = [1, 0, 2, 3]
        self.potential = np.array([
            [2, 1, 0, -3], [1, 2, -3, 0],
            [0, 1, 2, -3], [-3, 0, 1, 2],
        ], dtype=np.float32)
        self.terminal = np.array([1, 2, 0, -3], dtype=np.float32)

    def rewards_and_values(self, at_kyoku):
        kyoku_rewards = ORACLE['score_rank_delta_rewards_by_kyoku'](
            self.features, self.final_ranks, self.pts)
        shaped = ORACLE['expand_kyoku_rewards_to_steps'](kyoku_rewards, at_kyoku)
        terminal_only = np.zeros_like(shaped)
        terminal_only[-1] = self.terminal
        potential = self.potential[at_kyoku]
        # Arbitrary, imperfect values: the identity must not require a perfect critic.
        terminal_value = np.arange(len(at_kyoku) * 4, dtype=np.float32).reshape(-1, 4) / 8 - 1
        return kyoku_rewards, shaped, terminal_only, terminal_value, potential

    def test_hand_computed_skipped_kyoku_and_trailing_no_decision_rewards(self):
        # This player's COMPLETE decision trace has no node in kyoku 1 or 3.
        at = np.array([0, 0, 2, 2])
        rewards, shaped, _, _, _ = self.rewards_and_values(at)
        np.testing.assert_array_equal(
            ORACLE['current_rank_values_by_kyoku'](self.features, self.pts), self.potential)
        np.testing.assert_array_equal(rewards, [
            [-1, 1, -3, 3], [-1, -1, 5, -3],
            [-3, -1, -1, 5], [4, 2, -1, -5],
        ])
        np.testing.assert_array_equal(shaped, [
            [0, 0, 0, 0], [-2, 0, 2, 0],
            [0, 0, 0, 0], [1, 1, -2, 0],
        ])
        expected_targets = self.terminal - self.potential[at]
        for mode in ('rank_delta', 'score_rank_mc'):
            np.testing.assert_array_equal(ORACLE['oracle_step_value_targets'](
                self.features, self.final_ranks, self.pts, at, mode, 1.0), expected_targets)
        for player in range(4):
            np.testing.assert_array_equal(ONLINE['expand_sparse_kyoku_reward_to_steps'](
                rewards[:, player], at), shaped[:, player])
        terminal_value = np.array([0.5, 0.75, -0.5, 1.25])
        delta_value = terminal_value - self.potential[at, 0]
        np.testing.assert_array_equal(td_residuals(shaped[:, 0], delta_value),
                                      [0.25, -1.25, 1.75, -0.25])
        np.testing.assert_array_equal(GAE(shaped[:, 0], delta_value, 1.0, 0.5),
                                      [0.03125, -0.4375, 1.625, -0.25])

    def test_gamma_one_coordinate_invariance_for_all_players_and_lambdas(self):
        # Includes normal boundaries, skipped kyoku, trailing empty kyoku,
        # and a player whose first decision happens only in a later kyoku.
        for at in ([0, 0, 1, 2, 2, 3], [0, 0, 2, 2], [2], [3, 3]):
            _, shaped, terminal_only, value, potential = self.rewards_and_values(at)
            next_potential = np.concatenate((potential[1:], np.zeros((1, 4))), axis=0)
            np.testing.assert_array_equal(shaped, terminal_only + next_potential - potential)
            np.testing.assert_array_equal(td_residuals(shaped, value - potential),
                                          td_residuals(terminal_only, value))
            for lam in (0.0, 0.2, 0.5, 0.95, 1.0):
                for player in range(4):
                    with self.subTest(at=at, lam=lam, player=player):
                        np.testing.assert_allclose(
                            GAE(shaped[:, player], value[:, player] - potential[:, player], 1.0, lam),
                            GAE(terminal_only[:, player], value[:, player], 1.0, lam), atol=1e-6)

    def test_value_only_conversion_changes_boundary_and_terminal_residuals(self):
        at = [0, 0, 2, 2]
        _, shaped, _, value, potential = self.rewards_and_values(at)
        correct = td_residuals(shaped, value - potential)
        wrong = td_residuals(shaped, value)  # Added U to V but retained shaped rewards.
        np.testing.assert_array_equal(wrong[:, 2] - correct[:, 2], [0, 2, 0, -2])
        self.assertFalse(np.allclose(GAE(shaped[:, 2], value[:, 2], 1.0, 0.95),
                                    GAE(shaped[:, 2], value[:, 2] - potential[:, 2], 1.0, 0.95)))

    def test_gamma_below_one_requires_discounted_potential_shaping(self):
        _, shaped, terminal_only, value, potential = self.rewards_and_values([0, 0, 2, 2])
        next_potential = np.concatenate((potential[1:], np.zeros((1, 4))), axis=0)
        gamma = 0.9
        discounted_shaping = terminal_only + gamma * next_potential - potential
        np.testing.assert_allclose(td_residuals(discounted_shaping, value - potential, gamma),
                                   td_residuals(terminal_only, value, gamma), atol=1e-6)
        # Current score_rank rewards are undiscounted U differences. Do not
        # extrapolate the gamma=1 equivalence to other configured discounts.
        self.assertFalse(np.allclose(td_residuals(shaped, value - potential, gamma),
                                     td_residuals(terminal_only, value, gamma)))

    def test_full_gae_matches_reference_blocks_only_with_carried_boundary_state(self):
        _, shaped, _, value, potential = self.rewards_and_values([0, 0, 1, 2, 2, 3])
        for gamma in (0.93, 1.0):
            for lam in (0.0, 0.5, 0.95, 1.0):
                for player in range(4):
                    r = shaped[:, player]
                    v = value[:, player] - potential[:, player]
                    full = GAE(r, v, gamma, lam)
                    for ends in ([6], [2, 4, 6], [1, 3, 5, 6], [1, 2, 3, 4, 5, 6]):
                        np.testing.assert_allclose(full, reference_blocked_gae(r, v, gamma, lam, ends),
                                                   atol=1e-6, rtol=1e-6)
        r, v = shaped[:, 0], value[:, 0] - potential[:, 0]
        incorrectly_reset = np.concatenate([GAE(r[:2], v[:2], 1.0, 0.95),
                                             GAE(r[2:], v[2:], 1.0, 0.95)])
        self.assertFalse(np.allclose(incorrectly_reset, GAE(r, v, 1.0, 0.95)))

    def test_lambda_one_matches_complete_return_minus_value(self):
        at = [0, 0, 2, 2]
        kyoku_rewards, shaped, _, value, potential = self.rewards_and_values(at)
        for gamma in (0.93, 1.0):
            returns = ORACLE['discounted_returns_from_step_rewards'](shaped, gamma)
            for player in range(4):
                v = value[:, player] - potential[:, player]
                np.testing.assert_allclose(ONLINE['compute_gae_advantages'](
                    kyoku_rewards[:, player], at, v, gamma, 1.0), returns[:, player] - v,
                    atol=1e-6)


if __name__ == '__main__':
    unittest.main()
