"""Keyboard, reset and exit tests using the real demo event handler."""

import os
import unittest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np
import pygame

from demo import DemoController
from simple_grid_world import SimpleGridWorldEnv


def key_event(key):
    return pygame.event.Event(pygame.KEYDOWN, key=key)


class DemoTests(unittest.TestCase):
    def setUp(self):
        self.env = SimpleGridWorldEnv()
        self.addCleanup(self.env.close)
        self.controller = DemoController(self.env)

    def test_arrows_and_reset(self):
        c = self.controller
        for key in (pygame.K_RIGHT, pygame.K_DOWN, pygame.K_LEFT, pygame.K_UP):
            c.handle_event(key_event(key))
        np.testing.assert_array_equal(c.observation["agent"], [0, 0])
        self.assertEqual(c.info["steps"], 4)
        c.handle_event(key_event(pygame.K_r))
        self.assertEqual(c.info["steps"], 0)
        self.assertEqual(self.env._last_action, 1)
        self.assertFalse(c.episode_finished)

    def test_goal_ignores_arrows_until_reset(self):
        c = self.controller
        for key in [pygame.K_RIGHT] * 4 + [pygame.K_DOWN] * 4:
            c.handle_event(key_event(key))
        self.assertTrue(c.episode_finished)
        c.handle_event(key_event(pygame.K_LEFT))
        np.testing.assert_array_equal(c.observation["agent"], [4, 4])
        self.assertEqual(c.info["steps"], 8)
        c.handle_event(key_event(pygame.K_r))
        c.handle_event(key_event(pygame.K_RIGHT))
        np.testing.assert_array_equal(c.observation["agent"], [0, 1])

    def test_truncation_ignores_arrows_until_reset(self):
        env = SimpleGridWorldEnv(max_steps=1)
        self.addCleanup(env.close)
        c = DemoController(env)
        c.handle_event(key_event(pygame.K_RIGHT))
        self.assertTrue(c.episode_finished)
        c.handle_event(key_event(pygame.K_DOWN))
        self.assertEqual(c.info["steps"], 1)
        c.handle_event(key_event(pygame.K_r))
        self.assertFalse(c.episode_finished)
        self.assertEqual(c.info["steps"], 0)

    def test_escape_stops_later_events(self):
        self.controller.handle_event(key_event(pygame.K_ESCAPE))
        self.controller.handle_event(key_event(pygame.K_RIGHT))
        self.assertFalse(self.controller.running)
        self.assertEqual(self.controller.info["steps"], 0)

    def test_window_close(self):
        self.controller.handle_event(pygame.event.Event(pygame.QUIT))
        self.assertFalse(self.controller.running)

    def test_unrelated_events_do_not_move_agent(self):
        for event in (key_event(pygame.K_a), pygame.event.Event(pygame.MOUSEMOTION)):
            self.assertIsNone(self.controller.handle_event(event))
        self.assertEqual(self.controller.info["steps"], 0)

    def test_human_render_event_queue_and_cleanup(self):
        # Actual SDL queue with a dummy video driver, not a physical UI test.
        env = SimpleGridWorldEnv(render_mode="human", window_size=300)
        self.addCleanup(env.close)
        c = DemoController(env)
        pygame.event.clear()
        for key in (pygame.K_RIGHT, pygame.K_r, pygame.K_DOWN, pygame.K_ESCAPE):
            pygame.event.post(key_event(key))
        for event in pygame.event.get():
            c.handle_event(event)
        np.testing.assert_array_equal(c.observation["agent"], [1, 0])
        self.assertFalse(c.running)
        self.assertEqual(c.info["steps"], 1)
        env.close()
        self.assertIsNone(env.window)
        self.assertFalse(pygame.display.get_init())


if __name__ == "__main__":
    unittest.main(verbosity=2)
