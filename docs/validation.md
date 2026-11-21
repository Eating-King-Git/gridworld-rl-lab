# 验证记录

日期：2026-11-21。对应 0.1.0 首次公开整理。

## 本地已执行

系统：Windows x64；Python 3.10.11。创建新的 venv，从官方 PyPI 安装 `requirements.txt`，未复用原课程工程的虚拟环境。

| 检查 | 结果 |
|---|---|
| `python -m pip check` | 无依赖冲突 |
| `python -m unittest discover -v` | 28 项测试全部通过（另有多参数子测试） |
| Gymnasium `check_env` | 通过；包含在环境测试中 |
| 环境边界、奖励、截断、重置、观察副本、非法输入 | 通过 |
| 默认和非默认画面尺寸、信息面板位置 | 通过 |
| 演示方向键、R、Esc、关闭窗口和终局输入 | dummy SDL 事件控制测试通过 |
| 2×2、5×5、8×8 与 gamma=0/0.5/0.9 | 与独立解析解一致；残差通过阈值检查 |
| 默认三组价值计算 | 收敛；Bellman 残差均为 0 |
| 在仓库之外作为工作目录运行环境测试和价值计算 | 通过，输出文本与样例一致 |
| 从价值表生成演示 | 验证 8 步到达目标，最后奖励为 1 |
| PNG / GIF / MP4 | 600×700；GIF 9 帧，MP4 12 fps、约 6.17 秒；解码检查通过 |
| 预览图片 | 已检查角色、目标、网格和状态面板可见 |

## 复现命令

```text
python -m pip install -r requirements.txt
python -m pip check
python -m unittest discover -v
python value_iteration.py
python make_demo_video.py
```

`test_env.py` 和 `test_demo.py` 使用无窗口 SDL 驱动。事件测试走的是与交互演示相同的处理函数，并包含 SDL 事件队列验证；这不等于真实键盘、显示器和窗口管理器上的人工验收。本次没有声称完成第二台物理电脑或 macOS 的桌面验证。

## 云端验证

工作流配置为 Windows / Python 3.10 和 Ubuntu / Python 3.12，执行依赖检查、28 项测试、价值迭代，并比较重算结果是否改变已提交的样例。

云端的每次运行结论以 [GitHub Actions](https://github.com/Eating-King-Git/gridworld-rl-lab/actions/workflows/tests.yml) 为准；固定文档不预先承诺未来提交一定通过。

## 首版范围

环境与价值迭代、几何绘制、交互事件、可复现结果和演示素材在本次范围内。SARSA、Q-learning、神经网络、真实研究数据和私人课程提交包均不在本仓库内。
