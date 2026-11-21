"""A deterministic Gymnasium grid world with an asset-free renderer.

The agent starts at the top-left corner. Entering the bottom-right goal earns
1; all other transitions earn 0. This module defines the environment, not an
agent or a learning algorithm.
"""

from __future__ import annotations

from math import cos, pi, sin
from numbers import Integral
from typing import Any

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces


class SimpleGridWorldEnv(gym.Env):
    """Square grid; actions 0/1/2/3 mean up/right/down/left.

    Coordinates are ``[row, column]``. A move outside the grid leaves the agent
    in place, but still consumes a step. Call ``reset`` before the first action
    and after either termination or truncation.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 6}
    ACTION_NAMES = {0: "Up", 1: "Right", 2: "Down", 3: "Left"}

    def __init__(
        self,
        size: int = 5,
        render_mode: str | None = None,
        max_steps: int = 50,
        window_size: int = 600,
    ) -> None:
        """Create a grid and an optional renderer.

        ``max_steps`` is an external episode limit, not part of the state used
        by the discounted value-iteration example. ``window_size`` controls
        the square board; the status panel adds 100 pixels to its height.
        """
        super().__init__()
        for name, value, minimum in (
            ("size", size, 2),
            ("max_steps", max_steps, 1),
            ("window_size", window_size, 128),
        ):
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise ValueError(f"{name} must be an integer")
            if value < minimum:
                raise ValueError(f"{name} must be at least {minimum}")
        if render_mode not in (None, "human", "rgb_array"):
            raise ValueError("render_mode must be None, 'human', or 'rgb_array'")

        self.size = int(size)
        self.max_steps = int(max_steps)
        self.window_size = int(window_size)
        self.render_mode = render_mode
        self.panel_height = 100
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Dict(
            {
                name: spaces.Box(0, self.size - 1, shape=(2,), dtype=np.int64)
                for name in ("agent", "target")
            }
        )
        self._action_to_direction = {
            0: np.array([-1, 0], dtype=np.int64),
            1: np.array([0, 1], dtype=np.int64),
            2: np.array([1, 0], dtype=np.int64),
            3: np.array([0, -1], dtype=np.int64),
        }
        self._agent_location = np.zeros(2, dtype=np.int64)
        self._target_location = np.full(2, self.size - 1, dtype=np.int64)
        self._step_count = 0
        self._last_reward = 0.0
        self._last_action = 1
        self._status = "Ready: call reset()"
        self._needs_reset = True
        self.window: pygame.Surface | None = None
        self.clock: pygame.time.Clock | None = None
        self._font: pygame.font.Font | None = None

    def _get_obs(self) -> dict[str, np.ndarray]:
        # Return copies so a caller cannot mutate the underlying environment.
        return {
            "agent": self._agent_location.copy(),
            "target": self._target_location.copy(),
        }

    def _get_info(self) -> dict[str, int]:
        distance = int(np.abs(self._agent_location - self._target_location).sum())
        return {"distance": distance, "steps": self._step_count}

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, int]]:
        """Start a fresh episode at [0, 0]; the map itself is deterministic."""
        super().reset(seed=seed)
        self._agent_location = np.zeros(2, dtype=np.int64)
        self._target_location = np.full(2, self.size - 1, dtype=np.int64)
        self._step_count = 0
        self._last_reward = 0.0
        self._last_action = 1
        self._status = "Running"
        self._needs_reset = False
        if self.render_mode == "human":
            self._render_frame()
        return self._get_obs(), self._get_info()

    def step(
        self, action: int
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, int]]:
        """Take one action; goal completion takes priority over the step limit."""
        if self._needs_reset:
            raise RuntimeError("Call reset() before stepping a new or finished episode")
        if isinstance(action, (bool, np.bool_)) or not self.action_space.contains(action):
            raise ValueError(f"invalid action {action!r}; expected an integer 0-3")

        self._last_action = int(action)
        self._agent_location = np.clip(
            self._agent_location + self._action_to_direction[int(action)],
            0,
            self.size - 1,
        ).astype(np.int64)
        self._step_count += 1
        terminated = bool(np.array_equal(self._agent_location, self._target_location))
        truncated = self._step_count >= self.max_steps and not terminated
        reward = 1.0 if terminated else 0.0
        self._last_reward = reward
        self._needs_reset = terminated or truncated
        self._status = (
            "Goal reached!" if terminated else "Step limit reached" if truncated
            else f"Last action: {self.ACTION_NAMES[int(action)]}"
        )
        if self.render_mode == "human":
            self._render_frame()
        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def render(self) -> np.ndarray | None:
        """Return an RGB array, display a window, or do nothing in None mode."""
        return None if self.render_mode is None else self._render_frame()

    def _render_frame(self) -> np.ndarray | None:
        # RGB rendering needs only a Surface and fonts, not an audio device.
        if not pygame.font.get_init():
            pygame.font.init()
        canvas = pygame.Surface((self.window_size, self.window_size + self.panel_height))
        canvas.fill((245, 247, 250))
        cell = self.window_size / self.size

        def center(position: np.ndarray) -> tuple[int, int]:
            row, column = position
            return round((column + 0.5) * cell), round((row + 0.5) * cell)

        # Both symbols are drawn from geometry; no external artwork is loaded.
        gx, gy = center(self._target_location)
        pygame.draw.circle(canvas, (255, 244, 190), (gx, gy), max(1, round(cell * 0.32)))
        star = [
            (
                gx + cos(-pi / 2 + i * pi / 5) * cell * (0.25 if i % 2 == 0 else 0.12),
                gy + sin(-pi / 2 + i * pi / 5) * cell * (0.25 if i % 2 == 0 else 0.12),
            )
            for i in range(10)
        ]
        pygame.draw.polygon(canvas, (232, 165, 29), star)
        ax, ay = center(self._agent_location)
        radius = max(1, round(cell * 0.30))
        pygame.draw.circle(canvas, (48, 117, 217), (ax, ay), radius)
        row_delta, col_delta = self._action_to_direction[self._last_action]
        forward = np.array([col_delta, row_delta], dtype=float)
        sideways = np.array([-row_delta, col_delta], dtype=float)
        for side in (-1, 1):
            eye = np.array([ax, ay]) + radius * (0.34 * forward + 0.37 * side * sideways)
            pupil = eye + radius * 0.11 * forward
            pygame.draw.circle(canvas, (255, 255, 255), eye, max(1, round(radius * 0.22)))
            pygame.draw.circle(canvas, (22, 40, 74), pupil, max(1, round(radius * 0.11)))

        for index in range(self.size + 1):
            point = min(self.window_size - 1, round(index * cell))
            pygame.draw.line(canvas, (94, 105, 122), (0, point), (self.window_size, point), 2)
            pygame.draw.line(canvas, (94, 105, 122), (point, 0), (point, self.window_size), 2)

        pygame.draw.rect(
            canvas, (30, 41, 59), (0, self.window_size, self.window_size, self.panel_height)
        )
        if self._font is None:
            self._font = pygame.font.Font(None, 24)
        lines = (
            f"SimpleGridWorld   Step: {self._step_count}/{self.max_steps}",
            f"Agent: {tuple(map(int, self._agent_location))}   "
            f"Goal: {tuple(map(int, self._target_location))}   Reward: {self._last_reward:g}",
            f"Status: {self._status}",
        )
        for index, text in enumerate(lines):
            surface = self._font.render(text, True, (238, 242, 248))
            available_width = self.window_size - 24
            if surface.get_width() > available_width:
                height = max(1, round(surface.get_height() * available_width / surface.get_width()))
                surface = pygame.transform.smoothscale(surface, (available_width, height))
            canvas.blit(surface, (12, self.window_size + 10 + index * 28))

        if self.render_mode == "human":
            if self.window is None:
                pygame.display.init()
                self.window = pygame.display.set_mode(canvas.get_size())
                pygame.display.set_caption("GridWorld RL Lab")
                self.clock = pygame.time.Clock()
            self.window.blit(canvas, (0, 0))
            pygame.event.pump()
            pygame.display.update()
            assert self.clock is not None
            self.clock.tick(self.metadata["render_fps"])
            return None
        return np.transpose(pygame.surfarray.array3d(canvas), (1, 0, 2)).copy()

    def close(self) -> None:
        """Close the owned display. Calling close more than once is harmless."""
        if self.window is not None:
            pygame.display.quit()
        self.window = None
        self.clock = None
        self._font = None
