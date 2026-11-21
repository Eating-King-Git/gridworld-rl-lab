"""Mathematical checks independent of the value-iteration implementation.

The oracle is the shortest-path return: a reward received after d moves has
value gamma ** (d - 1). The absorbing goal itself has no future reward.
"""

import unittest

import numpy as np

from simple_grid_world import SimpleGridWorldEnv
from value_iteration import (
    action_value,
    bellman_optimality_residual,
    get_transition,
    value_iteration,
)


class ValueIterationTests(unittest.TestCase):
    def make_env(self, size=5, max_steps=50):
        env = SimpleGridWorldEnv(size=size, max_steps=max_steps)
        self.addCleanup(env.close)
        return env

    def test_matches_shortest_path_return_for_multiple_grids(self):
        for size in (2, 5, 8):
            for gamma in (0.0, 0.5, 0.9):
                with self.subTest(size=size, gamma=gamma):
                    env = self.make_env(size)
                    actual, iterations, delta = value_iteration(env, gamma)
                    expected = np.zeros((size, size), dtype=np.float64)
                    goal = np.array([size - 1, size - 1])
                    for state in np.ndindex(expected.shape):
                        distance = int(np.abs(np.array(state) - goal).sum())
                        if distance:
                            expected[state] = gamma ** (distance - 1)

                    np.testing.assert_allclose(
                        actual, expected, rtol=0, atol=1e-12
                    )
                    self.assertEqual(actual[-1, -1], 0.0)
                    self.assertEqual(actual[-1, -2], 1.0)
                    self.assertGreater(iterations, 0)
                    self.assertTrue(np.isfinite(actual).all())
                    self.assertLess(delta, 1e-12)
                    self.assertLessEqual(
                        bellman_optimality_residual(env, actual, gamma), 1e-12
                    )

    def test_terminal_goal_is_absorbing_without_repeated_reward(self):
        env = self.make_env()
        for action in range(4):
            with self.subTest(action=action):
                self.assertEqual(
                    get_transition(env, (4, 4), action), ((4, 4), 0.0, True)
                )

    def test_goal_entry_pays_once_without_bootstrapping(self):
        env = self.make_env()
        # A deliberately wrong goal estimate must not leak through a terminal
        # transition into the action value of its predecessor.
        values = np.full((5, 5), 123.0)
        self.assertEqual(get_transition(env, (4, 3), 1), ((4, 4), 1.0, True))
        self.assertEqual(action_value(env, values, (4, 3), 1, 0.9), 1.0)

    def test_wall_transition_stays_in_place_without_reward(self):
        env = self.make_env()
        self.assertEqual(get_transition(env, (0, 0), 0), ((0, 0), 0.0, False))
        self.assertEqual(get_transition(env, (0, 0), 3), ((0, 0), 0.0, False))

    def test_residual_detects_an_unsolved_value_table(self):
        env = self.make_env()
        # At a goal-adjacent state, the right action pays one immediately.
        # Therefore the Bellman residual of the zero table is exactly one.
        values = np.zeros((5, 5))
        self.assertEqual(bellman_optimality_residual(env, values, 0.9), 1.0)

    def test_rollout_time_limit_does_not_change_the_planning_mdp(self):
        short = self.make_env(max_steps=1)
        long = self.make_env(max_steps=50)
        short_values, _, _ = value_iteration(short, 0.9)
        long_values, _, _ = value_iteration(long, 0.9)
        np.testing.assert_array_equal(short_values, long_values)
        # This is an infinite-horizon discounted value, not a claim that the
        # goal can be reached within the one-step rollout budget.
        self.assertAlmostEqual(short_values[0, 0], 0.9 ** 7, places=12)

    def test_invalid_discount_is_rejected(self):
        env = self.make_env()
        for gamma in (-0.1, 1.0, 1.1, float("inf"), -float("inf"), float("nan")):
            with self.subTest(gamma=gamma), self.assertRaises(ValueError):
                value_iteration(env, gamma)

    def test_invalid_tolerance_is_rejected(self):
        env = self.make_env()
        for tolerance in (0, -1e-6, float("inf"), -float("inf"), float("nan")):
            with self.subTest(tolerance=tolerance), self.assertRaises(ValueError):
                value_iteration(env, 0.9, tolerance=tolerance)

    def test_nonpositive_iteration_budget_is_rejected(self):
        env = self.make_env()
        for limit in (0, -1):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                value_iteration(env, 0.9, max_iterations=limit)

    def test_insufficient_iteration_budget_reports_nonconvergence(self):
        env = self.make_env()
        with self.assertRaises(RuntimeError):
            value_iteration(env, 0.9, max_iterations=1)


if __name__ == "__main__":
    unittest.main()
