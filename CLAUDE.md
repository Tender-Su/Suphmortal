# Repository Guidelines

这份文件只放所有 agent 必须先知道的规则。当前训练结论、机器参数和长流程不要在这里复写，按入口文档读取。

## 先读哪里

- 接手入口：`docs/agent/README.md`
- 当前状态：`docs/agent/handoff.md`
- 监督学习主线：`docs/status/supervised-mainline.md`
- 在线 RL 主线：`docs/status/online-rl-mainline.md`
- 机器与 loader / `1v3` 默认：`docs/status/machine-benchmarks.md`
- 命令与脚本：`docs/agent/workflows.md`
- 双机与远程：`docs/agent/remote-ops.md`
- 代码位置：`docs/agent/code-map.md`
- 文档整理：`docs/agent/doc-maintenance.md`

冲突时按 `docs/status/` > `docs/agent/` > `docs/research/` 判断；`docs/archive/` 和 `docs/reflections/` 默认不作为当前运行依据。
窄代码修改或定点排障可以先用 `rg` / 本地文件定位，再只打开相关文档；不要从研究长文或归档里拼当前默认。

## 项目边界

- `libriichi/`：Rust 牌局引擎、状态机、特征提取、PyO3 模块。
- `mortal/`：Python 训练、评测、实验编排；子目录职责看 `docs/agent/code-map.md`。
- `exe-wrapper/`：小型 Rust helper binary crate。
- `scripts/`：Windows 入口脚本。
- `docs/`：当前状态、证据、接手说明和历史归档。
- `checkpoints/`、`logs/`、`target/`：本地产物，默认不提交。

## 工作原则

- 总目标是最强模型，训练效率和便利性排在最终强度之后。
- 默认用中文说明；保留项目内已稳定使用的术语，例如 `GRP`、`Oracle critic`、`value / GAE`、`1v3`。
- 代码改动要小而清晰，优先删除旧路径或旧逻辑，避免只堆新增；逻辑转折处才加短注释。
- 不要改动用户已有的无关变更；当前工作树可能本来就是 dirty。
- 查找文件和文本优先用 `rg` / `rg --files`。
- 长输出外部命令按 `RTK.md` 使用 `rtk`，PowerShell 内建、短探针和管道直接运行。
- 本地 Codex shell 正常用工具层 `workdir` 控制目录；只有目录异常、嵌套 shell、远程 shell 或命令本身需要时才显式 `Set-Location`。
- 面向用户解释时优先保留项目内已有术语；引入论文或外部概念时先用中文说明它和本项目的关系，再给原名。

## 环境与常用命令

非交互 Python 优先使用：

```powershell
C:\ProgramData\anaconda3\envs\mortal\python.exe
```

常用入口：

```powershell
.\scripts\build_libriichi.bat
.\scripts\run_grp.bat
.\scripts\run_supervised.bat
.\scripts\run_sl_p1_only.bat
.\scripts\run_online.bat
.\scripts\run_online_fidelity.bat
.\scripts\run_oracle_critic_pretrain.bat
.\scripts\run_oracle_dependency_eval.bat
```

Rust / PyO3 测试前固定解释器和 DLL 路径：

```powershell
$env:PYO3_PYTHON="C:\ProgramData\anaconda3\envs\mortal\python.exe"
$env:PATH="C:\ProgramData\anaconda3\envs\mortal;C:\ProgramData\anaconda3\envs\mortal\Library\bin;C:\ProgramData\anaconda3\envs\mortal\Scripts;$env:PATH"
cargo test -p libriichi state::test
```

Python smoke：

```powershell
C:\ProgramData\anaconda3\envs\mortal\python.exe -m mortal.tests.test_greedy
```

`conda` 在新 PowerShell 中不一定在 `PATH`；能用绝对解释器时不要假设 `conda run` 可用。

## 代码约束

- Rust：Edition 2024；`libriichi/src/lib.rs` 有严格 clippy deny；提交前运行 `cargo fmt`。
- Python：4 空格缩进，函数和变量用 `snake_case`，类用 `PascalCase`。
- 不要改特征通道数，除非同步更新 Rust 特征提取和 Python 模型输入。关键常数：`ACTION_SPACE=46`，`obs_shape(v4)=(1012,34)`，`oracle_obs_shape(v4)=(217,34)`，`GRP_SIZE=7`，`MAX_VERSION=4`。
- `Brain.__init__` 默认是 `"BN"`，但当前配置和评测入口使用 `"GN"`；实例化时显式传 norm。
- V3/V4 是 pre-activation ResBlock；每个 ResBlock 都有 SE-style channel attention，不要随手移除。
- `mortal/config.toml` 可由 `MORTAL_CFG` 覆盖；不要提交真实数据路径或凭据。
- Windows PowerShell 写 TOML / 配置时优先 `apply_patch`，避免 `Set-Content`、`Out-File`、`>` 产生编码问题。
- CPU affinity 现在是 opt-in；正常训练默认不设置 `MORTAL_CPU_AFFINITY`。

## 训练与产物口径

- 当前阶段摘要看 `docs/agent/handoff.md`；不要从旧研究文档拼默认结论。
- `GRP` checkpoint 分三类：`best_loss` 默认下游使用，`best_acc` 只做受控对照，`latest` 只用于续训。
- 在线 RL 启动优先级：`[control].state_file` -> `[online].init_state_file` -> `[supervised].best_loss_state_file` -> `[supervised].best_state_file`。
- `1v3` challenger 路径是 `[1v3.challenger].state_file`，不是 `[control].state_file`。
- 训练阶段默认关闭 `search`；推理期增强要单独 A/B。

## 双机纪律

- 台式机 `main` 工作树是源码真源。
- 笔记本是独立实验 runner，不默认共享梯度、replay buffer 或 checkpoint。
- 双机同时跑同一阶段时，run name、输出目录和 checkpoint 路径必须带机器区分。
- 笔记本 IP 可能变化，远程命令前先按 `docs/agent/remote-ops.md` 重新确认。

## 测试与提交

- Rust 测试放同模块 `#[cfg(test)]`，优先跑 targeted test。
- Python 测试放 `mortal/tests/` 或相关模块附近，新增行为至少有可复现实测。
- 安全注意：`mortal/core/common.py` 的 TCP / pickle 通信只面向本机可信输入。
- commit subject 用短祈使句并带 scope，例如 `mortal: fix oracle dropout schedule`。

## 相关本地说明

@RTK.md

`RTK.md` 是命令输出压缩规则；写代码时也遵循其中“减少改动范围、重视可读性和运行效率”的约束。

工程取舍上想想 Linus 会怎么做。
