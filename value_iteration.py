"""使用价值迭代计算 SimpleGridWorld 的最优状态价值函数。

本程序使用 5×5 SimpleGridWorld 的无限时域折扣模型（不把 rollout 截断当成终止）：

- Agent 从任意非终止状态出发，可以执行上、右、下、左四种动作；
- 试图走出边界时停留在原位置；
- 进入右下角目标状态时立即获得奖励 1，并结束当前回合；
- 其他状态转移的即时奖励均为 0。

价值迭代使用 Bellman 最优方程：

    V*(s) = max_a [R(s,a,s') + gamma * V*(s')]

程序分别计算 gamma=0、0.5、0.9 时的 V*，并把结果保存为文本文件。
"""

from __future__ import annotations

import argparse
from numbers import Integral
from pathlib import Path

import numpy as np

from simple_grid_world import SimpleGridWorldEnv


# 默认演示的三个折扣率。
GAMMAS = (0.0, 0.5, 0.9)


def get_transition(
    env: SimpleGridWorldEnv,
    state: tuple[int, int],
    action: int,
) -> tuple[tuple[int, int], float, bool]:
    """根据网格世界的运动规则，计算一次确定性状态转移。

    参数：
        env: 第一次作业实现的 SimpleGridWorld 环境。
        state: 当前状态，用 (行, 列) 表示。
        action: 动作编号，0=上、1=右、2=下、3=左。

    返回：
        next_state: 执行动作后的状态。
        reward: 进入目标时为 1，否则为 0。
        terminated: 是否已经进入目标状态。
    """

    if not env.action_space.contains(action):
        raise ValueError("action must be 0, 1, 2, or 3")
    if len(state) != 2 or any(not isinstance(x, Integral) or not 0 <= x < env.size for x in state):
        raise ValueError("state must be an in-bounds (row, column) pair")
    goal = (env.size - 1, env.size - 1)
    if tuple(state) == goal:
        return goal, 0.0, True

    # 将元组坐标转换成 NumPy 数组，便于和方向向量相加。
    location = np.array(state, dtype=np.int64)

    # 复用环境中“动作编号 -> 方向向量”的对应关系。
    direction = env._action_to_direction[action]

    # np.clip 把行列坐标限制在 0 到 size-1 之间，防止走出边界。
    next_location = np.clip(location + direction, 0, env.size - 1)
    next_state = (int(next_location[0]), int(next_location[1]))

    # 目标位于右下角。Agent 进入目标状态时获得唯一的即时奖励 1。
    goal_state = (env.size - 1, env.size - 1)
    terminated = next_state == goal_state
    reward = 1.0 if terminated else 0.0

    return next_state, reward, terminated


def action_value(
    env: SimpleGridWorldEnv,
    values: np.ndarray,
    state: tuple[int, int],
    action: int,
    gamma: float,
) -> float:
    """计算给定状态和动作的 Bellman 单步回报 Q(s,a)。"""

    next_state, reward, terminated = get_transition(env, state, action)

    # 到达终点后回合已经结束，因此不再计算终点之后的未来价值。
    future_value = 0.0 if terminated else values[next_state]

    # Bellman 单步回报 = 即时奖励 + 折扣率 × 下一状态价值。
    return float(reward + gamma * future_value)


def value_iteration(
    env: SimpleGridWorldEnv,
    gamma: float,
    tolerance: float = 1e-12,
    max_iterations: int = 100_000,
) -> tuple[np.ndarray, int, float]:
    """执行同步价值迭代，求解最优状态价值函数 V*。

    同步迭代表示：第 k+1 轮的所有状态，都只使用第 k 轮的旧价值进行更新。

    返回：
        values: 形状为 size×size 的最优状态价值矩阵。
        iteration: 实际迭代轮数。
        delta: 最后一轮更新前后的最大绝对差。
    """

    if not 0.0 <= gamma < 1.0:
        raise ValueError("折扣率 gamma 必须满足 0 <= gamma < 1")

    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive")
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, Integral) or max_iterations < 1:
        raise ValueError("max_iterations must be a positive integer")

    # 初始时不知道任何状态价值，因此全部初始化为 0。
    values = np.zeros((env.size, env.size), dtype=np.float64)
    goal_state = (env.size - 1, env.size - 1)

    for iteration in range(1, max_iterations + 1):
        # copy 保存第 k 轮价值，new_values 用于计算第 k+1 轮价值。
        old_values = values.copy()
        new_values = old_values.copy()

        for row in range(env.size):
            for column in range(env.size):
                state = (row, column)

                # 目标是终止状态，到达后没有未来奖励，所以其价值固定为 0。
                if state == goal_state:
                    new_values[state] = 0.0
                    continue

                # 计算当前状态执行四种动作时的单步回报。
                q_values = [
                    action_value(env, old_values, state, action, gamma)
                    for action in range(env.action_space.n)
                ]

                # Bellman 最优算子取所有动作价值中的最大值。
                new_values[state] = max(q_values)

        # delta 是这一轮中变化最大的状态，用于判断算法是否已经收敛。
        delta = float(np.max(np.abs(new_values - old_values)))
        values = new_values

        if delta < tolerance:
            return values, iteration, delta

    raise RuntimeError(f"价值迭代在 {max_iterations} 轮内没有收敛")


def bellman_optimality_residual(
    env: SimpleGridWorldEnv,
    values: np.ndarray,
    gamma: float,
) -> float:
    """计算 Bellman 最优方程残差；越接近 0，结果越可靠。"""

    goal_state = (env.size - 1, env.size - 1)
    residual = 0.0

    for row in range(env.size):
        for column in range(env.size):
            state = (row, column)
            if state == goal_state:
                bellman_value = 0.0
            else:
                bellman_value = max(
                    action_value(env, values, state, action, gamma)
                    for action in range(env.action_space.n)
                )

            residual = max(residual, abs(float(values[state]) - bellman_value))

    return residual


def theoretical_values(size: int, gamma: float) -> np.ndarray:
    """根据最短距离公式计算理论值，用来交叉验证迭代结果。

    当前环境只有到达目标时获得奖励 1。若某状态到目标的最短距离为 d，
    奖励会在第 d 步获得，因此该状态的最优价值为 gamma^(d-1)。
    目标状态本身是终止状态，价值规定为 0。
    """

    values = np.zeros((size, size), dtype=np.float64)
    goal_row = size - 1
    goal_column = size - 1

    for row in range(size):
        for column in range(size):
            distance = (goal_row - row) + (goal_column - column)
            if distance > 0:
                values[row, column] = gamma ** (distance - 1)

    return values


def format_matrix(values: np.ndarray) -> str:
    """把价值矩阵格式化为保留 6 位小数的可读字符串。"""

    return np.array2string(
        values,
        formatter={"float_kind": lambda number: f"{number:.6f}"},
    )


def main() -> None:
    """计算三个折扣率的 V*，打印、验证并保存结果。"""

    parser = argparse.ArgumentParser(description="Compute and verify GridWorld optimal values.")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "examples" / "value_iteration_results.txt")
    args = parser.parse_args()
    env = SimpleGridWorldEnv(render_mode=None)
    output_lines = [
        "SimpleGridWorld 最优状态价值函数 V*",
        "矩阵中的行、列与网格中的状态坐标一致；右下角为终止状态。",
        "奖励规则：进入目标奖励 1，其他转移奖励 0。",
        "",
    ]

    for gamma in GAMMAS:
        values, iterations, delta = value_iteration(env, gamma)
        residual = bellman_optimality_residual(env, values, gamma)
        expected = theoretical_values(env.size, gamma)

        # 如果迭代结果和理论公式不一致，allclose 会返回 False。
        if not np.allclose(values, expected, atol=1e-10):
            raise AssertionError(f"gamma={gamma} 的迭代结果未通过理论值验证")

        title = f"gamma = {gamma:.1f}"
        matrix_text = format_matrix(values)

        print("=" * 70)
        print(title)
        print(f"迭代轮数: {iterations}")
        print(f"最后一次最大变化量 delta: {delta:.3e}")
        print(f"Bellman 最优方程残差: {residual:.3e}")
        print("V* =")
        print(matrix_text)

        output_lines.extend(
            [
                "=" * 70,
                title,
                f"迭代轮数: {iterations}",
                f"最后一次最大变化量 delta: {delta:.3e}",
                f"Bellman 最优方程残差: {residual:.3e}",
                "V* =",
                matrix_text,
                "",
            ]
        )

    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(output_lines), encoding="utf-8")
    env.close()

    print("=" * 70)
    print(f"结果已保存到：{output_path}")


if __name__ == "__main__":
    main()
