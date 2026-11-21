"""Render a greedy value-iteration rollout as PNG, GIF and MP4 without a window."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')
import imageio.v2 as imageio
import numpy as np
from simple_grid_world import SimpleGridWorldEnv
from value_iteration import action_value, value_iteration

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path,
                        default=Path(__file__).resolve().parent / 'docs' / 'media')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    env = SimpleGridWorldEnv(render_mode='rgb_array')
    try:
        values, _, _ = value_iteration(env, gamma=0.9)
        observation, _ = env.reset(seed=42)
        snapshots = [env.render()]
        path = [tuple(map(int, observation['agent']))]
        for _ in range(env.max_steps):
            state = tuple(map(int, observation['agent']))
            action = int(np.argmax([action_value(env, values, state, a, 0.9)
                                    for a in range(env.action_space.n)]))
            observation, reward, terminated, truncated, _ = env.step(action)
            path.append(tuple(map(int, observation['agent'])))
            snapshots.append(env.render())
            if terminated or truncated:
                break
        if not terminated or reward != 1.0 or len(path) - 1 != 8:
            raise RuntimeError('Expected an eight-step optimal rollout reaching the goal')
        imageio.imwrite(args.output_dir / 'preview.png', snapshots[0])
        imageio.mimsave(args.output_dir / 'demo.gif', snapshots,
                       duration=[1000] + [500] * 7 + [1700], loop=0)
        with imageio.get_writer(args.output_dir / 'demo.mp4', fps=12,
                                codec='libx264', quality=8,
                                macro_block_size=None) as writer:
            for index, frame in enumerate(snapshots):
                repetitions = 12 if index == 0 else 20 if index == len(snapshots) - 1 else 6
                for _ in range(repetitions):
                    writer.append_data(frame)
        print(f'Verified optimal rollout: {path}; reward={reward:g}')
        print(f'Media saved to {args.output_dir}')
    finally:
        env.close()

if __name__ == '__main__':
    main()
