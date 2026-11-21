"""Play GridWorld with the arrow keys; R resets and Esc exits."""

from __future__ import annotations

import pygame

from simple_grid_world import SimpleGridWorldEnv


KEY_TO_ACTION = {
    pygame.K_UP: 0,
    pygame.K_RIGHT: 1,
    pygame.K_DOWN: 2,
    pygame.K_LEFT: 3,
}


class DemoController:
    """The same event handler is used by the interactive loop and event tests."""

    def __init__(self, env: SimpleGridWorldEnv) -> None:
        self.env = env
        self.running = True
        self.episode_finished = False
        self.observation, self.info = env.reset(seed=42)

    def handle_event(self, event: pygame.event.Event) -> str | None:
        if not self.running:
            return None
        if event.type == pygame.QUIT:
            self.running = False
            return "Demo closed"
        if event.type != pygame.KEYDOWN:
            return None
        if event.key == pygame.K_ESCAPE:
            self.running = False
            return "Demo closed"
        if event.key == pygame.K_r:
            self.observation, self.info = self.env.reset()
            self.episode_finished = False
            return "Episode reset"
        if event.key not in KEY_TO_ACTION or self.episode_finished:
            return None

        action = KEY_TO_ACTION[event.key]
        self.observation, reward, terminated, truncated, self.info = self.env.step(action)
        self.episode_finished = terminated or truncated
        if terminated:
            return "Goal reached! Press R to play again."
        if truncated:
            return "Step limit reached. Press R to play again."
        return (
            f"action={self.env.ACTION_NAMES[action]} | "
            f"agent={tuple(map(int, self.observation['agent']))} | "
            f"reward={reward:g} | distance={self.info['distance']}"
        )


def main() -> None:
    env = SimpleGridWorldEnv(render_mode="human")
    try:
        controller = DemoController(env)
        print("Arrow keys: move | R: reset | Esc: quit")
        clock = pygame.time.Clock()
        while controller.running:
            for event in pygame.event.get():
                message = controller.handle_event(event)
                if message:
                    print(message)
            clock.tick(60)
    finally:
        env.close()
        pygame.quit()


if __name__ == "__main__":
    main()
