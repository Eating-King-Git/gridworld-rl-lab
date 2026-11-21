# 来源与第三方声明

本仓库来自个人课程工程的公开整理，使用 AI 辅助开发与复核。网格世界与价值迭代是经典教学内容，不主张算法原创性。

环境的基础接口与网格结构参考 Gymnasium 的官方自定义环境教程：
https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/

本项目采用固定起点/目标、明确的上右下左动作编码、截断规则、独立绘制和额外测试；保留 Gymnasium 上游的 MIT 声明如下。Gymnasium 本身由依赖安装，不将其完整源码复制进本仓库。

角色、目标与演示素材由本仓库中的几何绘制和生成脚本产生，没有附带历史工程的第三方角色图片、字体文件或课程资料。

## 依赖

requirements.txt 记录测试环境的完整依赖版本。Gymnasium、NumPy、Pygame、ImageIO、imageio-ffmpeg、Pillow 及其依赖保留各自许可证；本仓库的 MIT 许可证不改变依赖的许可。软件包与 FFmpeg 编码器通过官方 PyPI 安装，本仓库不附带其二进制分发包。

## Gymnasium 上游声明

The MIT License

Copyright (c) 2016 OpenAI
Copyright (c) 2022 Farama Foundation

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
