"""Environment regression tests; run directly or via unittest discovery."""

import os
import unittest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np
from gymnasium.utils.env_checker import check_env

from simple_grid_world import SimpleGridWorldEnv


class EnvironmentTests(unittest.TestCase):
    def make_env(self, **kwargs):
        env = SimpleGridWorldEnv(**kwargs)
        self.addCleanup(env.close)
        return env

    def test_gymnasium_api(self):
        check_env(self.make_env(), skip_render_check=True)

    def test_reset_and_observation_copies(self):
        env = self.make_env()
        obs, info = env.reset(seed=42)
        self.assertTrue(env.observation_space.contains(obs))
        np.testing.assert_array_equal(obs["agent"], [0, 0])
        np.testing.assert_array_equal(obs["target"], [4, 4])
        self.assertEqual(info, {"distance": 8, "steps": 0})
        obs["agent"][:] = 3
        obs["target"][:] = 0
        obs, _, _, _, _ = env.step(1)
        np.testing.assert_array_equal(obs["agent"], [0, 1])
        np.testing.assert_array_equal(obs["target"], [4, 4])

    def test_all_four_boundaries(self):
        env = self.make_env(size=3)
        env.reset()
        for action, expected in (
            (0, [0, 0]), (3, [0, 0]), (1, [0, 1]), (1, [0, 2]),
            (1, [0, 2]), (3, [0, 1]), (3, [0, 0]), (2, [1, 0]),
            (2, [2, 0]), (2, [2, 0]),
        ):
            obs, reward, terminated, truncated, _ = env.step(action)
            np.testing.assert_array_equal(obs["agent"], expected)
            self.assertEqual((reward, terminated, truncated), (0.0, False, False))
        self.assertEqual(env._get_info()["steps"], 10)

    def test_goal_and_end_of_episode(self):
        env = self.make_env()
        env.reset()
        for action in [1] * 4 + [2] * 3:
            _, reward, terminated, truncated, _ = env.step(action)
            self.assertEqual((reward, terminated, truncated), (0.0, False, False))
        obs, reward, terminated, truncated, info = env.step(2)
        np.testing.assert_array_equal(obs["agent"], [4, 4])
        self.assertEqual((reward, terminated, truncated), (1.0, True, False))
        self.assertEqual(info, {"distance": 0, "steps": 8})
        with self.assertRaises(RuntimeError):
            env.step(0)
        env.reset()
        self.assertEqual(env.step(1)[4]["steps"], 1)

    def test_truncation_requires_reset(self):
        env = self.make_env(max_steps=2)
        env.reset()
        self.assertFalse(env.step(0)[3])
        result = env.step(0)
        self.assertEqual(result[1:4], (0.0, False, True))
        with self.assertRaises(RuntimeError):
            env.step(1)

    def test_goal_at_limit_is_not_truncated(self):
        env = self.make_env(size=2, max_steps=2)
        env.reset()
        env.step(1)
        self.assertEqual(env.step(2)[1:4], (1.0, True, False))

    def test_step_before_reset(self):
        with self.assertRaises(RuntimeError):
            self.make_env().step(0)

    def test_invalid_actions_do_not_consume_a_step(self):
        env = self.make_env()
        env.reset()
        for action in (-1, 4, 1.5, "1", None, True, np.bool_(False), [1]):
            with self.subTest(action=action), self.assertRaises(ValueError):
                env.step(action)
        self.assertEqual(env._get_info()["steps"], 0)
        np.testing.assert_array_equal(env.step(np.int64(1))[0]["agent"], [0, 1])

    def test_invalid_constructor_arguments(self):
        for arguments in (
            {"size": 1}, {"size": 2.5}, {"size": True},
            {"max_steps": 0}, {"max_steps": 2.5}, {"max_steps": False},
            {"window_size": 127}, {"window_size": 200.5},
            {"render_mode": "ansi"},
        ):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                SimpleGridWorldEnv(**arguments)

    def test_rgb_shapes_and_panel_position(self):
        for width in (128, 300, 600):
            with self.subTest(width=width):
                env = self.make_env(render_mode="rgb_array", window_size=width)
                env.reset()
                frame = env.render()
                self.assertEqual(frame.shape, (width + 100, width, 3))
                self.assertEqual(frame.dtype, np.uint8)
                panel = frame[width:]
                np.testing.assert_array_equal(panel[-1, -1], [30, 41, 59])
                # Each status line must be visible below boards of any size.
                for start in (10, 38, 66):
                    self.assertTrue(np.any(panel[start:start + 22] > 200))
                frame[:] = 0
                self.assertTrue(np.any(env.render() > 0))

    def test_no_render_mode_and_idempotent_close(self):
        env = self.make_env()
        env.reset()
        self.assertIsNone(env.render())
        env.close()
        env.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
