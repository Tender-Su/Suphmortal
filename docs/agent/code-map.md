# 代码地图与整理边界

这份文档记录当前代码整理后的职责边界。它不替代 `handoff.md` 的当前结论，只回答“代码在哪里、该往哪里放”。

## 根目录

| 路径 | 职责 |
| --- | --- |
| `libriichi/` | Rust 牌局引擎、状态机、特征提取、PyO3 模块 |
| `mortal/` | Python 训练、评测、实验编排 |
| `exe-wrapper/` | 小型 Rust helper binary crate |
| `scripts/` | 仓库内 Windows 入口脚本和机器操作脚本 |
| `docs/` | 当前状态、证据、接手说明和历史归档 |
| `checkpoints/`, `logs/`, `target/` | 本地产物，不参与源码整理 |

## `libriichi/` 边界

- PyO3 对外模块包括 `consts`、`state`、`dataset`、`arena`、`stat`、`mjai`。
- `consts` 提供 Python 侧直接导入的形状和动作空间常数，改动时必须同步 Rust 特征提取、Python 模型输入和相关测试。
- `dataset` 负责从日志解析训练样本，核心链路是 Rust loader -> Python `FileDatasetsIter` / `SupervisedFileDatasetsIter` -> PyTorch `DataLoader`。
- `state` / `arena` 的测试通常用 mjai 事件驱动状态更新，Rust 测试优先放在同模块 `#[cfg(test)]`。

## `mortal/` 分层

| 路径 | 放什么 | 不放什么 |
| --- | --- | --- |
| `mortal/core/` | 模型、checkpoint、配置、复现、通用训练小工具 | 阶段特定实验编排 |
| `mortal/data/` | 数据集迭代器、加载器、奖励/标签构造 | 训练主循环 |
| `mortal/supervised/` | 监督学习训练入口、P0/P1/formal/fidelity 编排、SL 选择逻辑 | 在线 RL 专用逻辑 |
| `mortal/online/` | 在线 RL 训练、server/client、角色启动、机器模式配置 | 监督学习 A/B 编排 |
| `mortal/eval/` | `1v3`、mjai engine、Oracle 评测、搜索运行时 | 训练主循环 |
| `mortal/research/` | 一次性探针、审计脚本、历史 A/B 辅助脚本 | 当前默认训练入口 |
| `mortal/tests/` | Python 测试 | 训练产物、临时日志 |

## 关键 Python 模块

| 文件 | 职责 |
| --- | --- |
| `mortal/config.py` | 读取 `mortal/config.toml` 或 `MORTAL_CFG` 指向的 TOML，不做完整 schema 校验，类型安全由调用方负责 |
| `mortal/core/prelude.py` | 进程级预设：日志、warning、UTF-8 stdin、CPU affinity opt-in 等 |
| `mortal/core/model.py` | `Brain`、policy、value / aux head、`GRP` 等模型定义 |
| `mortal/data/dataloader.py` | 监督学习 / 在线 RL 训练数据迭代器和 worker 初始化 |
| `mortal/online/server.py` | 本机 `ThreadingTCPServer` 参数分发与 replay buffer 管理 |
| `mortal/online/client.py` | self-play worker，拉取参数、运行 `TrainPlayer.train_play()`、回传 replay |
| `mortal/online/train_online.py` | 在线 RL trainer，drain replay、PPO 更新、发布新参数 |
| `mortal/eval/engine.py` | 模型推理 wrapper，`MortalEngine.react_batch()` 接 obs / mask 并返回动作 |
| `mortal/eval/one_vs_three.py` | `1v3` challenger vs champion 评测入口 |

## 模型结构速查

- 当前主线 encoder 配置来自 `[resnet]`：`conv_channels=192`、`num_blocks=40`，实际入口显式使用 `Norm="GN"`。
- `Brain` 是 1D ResNet encoder；V3/V4 是 pre-activation ResBlock，每个 ResBlock 都有 SE-style `ChannelAttention`。
- `CategoricalPolicy`：`Linear(1024 -> 256)` + `tanh` + `Linear(256 -> ACTION_SPACE)`。
- `DQN`：dueling value / advantage；V4 形态是 `Linear(1024 -> 1 + ACTION_SPACE)`。
- `AuxNet` / 后续 aux heads 只属于辅助监督或诊断接口，不应混进评测入口。
- `GRP`：GRU 输入维度是 `GRP_SIZE=7`，当前默认主线配置是 `hidden_size=384`、`num_layers=3`、`dtype=float32`。

## 当前入口

仓库内入口统一从 `scripts/` 进入：

| 阶段 | 入口 |
| --- | --- |
| Rust/PyO3 构建 | `scripts/build_libriichi.bat` |
| GRP | `scripts/run_grp.bat` |
| 监督学习主线 | `scripts/run_supervised.bat` |
| 手动 P1 | `scripts/run_sl_p1_only.bat` |
| 在线 RL | `scripts/run_online.bat` |
| 在线 RL fidelity | `scripts/run_online_fidelity.bat` |
| Oracle dependency eval | `scripts/run_oracle_dependency_eval.bat` |

## 迁移纪律

- 入口脚本和测试必须跟随文件移动同步更新。
- 训练产物路径不随代码目录移动自动改名，避免破坏已存在 checkpoint 和日志。
- 大训练文件优先拆出无副作用工具函数；主循环最后拆。
- `docs/status/` 的当前结论优先级高于 `docs/research/` 和 `docs/archive/`。
- 新的一次性实验脚本默认进入 `mortal/research/`，只有成为稳定入口后才移入 `supervised/`、`online/` 或 `eval/`。
