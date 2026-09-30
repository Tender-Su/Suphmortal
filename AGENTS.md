# Repository Guidelines

这里维护所有 agent 共用的长期规则。阶段、机器参数和实验结论只在对应状态页维护。

## 阅读入口

- 按需查阅：[任务入口](docs/agent/README.md)、[接手摘要](docs/agent/handoff.md)。
- 阶段：[GRP](docs/status/grp-mainline.md)、[SL](docs/status/supervised-mainline.md)、[Oracle critic](docs/status/oracle-critic-mainline.md)、[在线 RL](docs/status/online-rl-mainline.md)。
- 操作：[运行流程](docs/agent/workflows.md)、[双机与远程](docs/agent/remote-ops.md)、[机器与资源](docs/status/machine-benchmarks.md)。
- 开发：[代码地图](docs/agent/code-map.md)、[重构与验证](docs/agent/code-health.md)、[文档维护](docs/agent/doc-maintenance.md)。

文档按任务需要读取和维护。当前事实优先依据可复核的源码、有效配置和原始产物；文档间按 `docs/status/` > `docs/agent/` > `docs/research/` 判断。`docs/archive/` 和 `docs/reflections/` 用于追溯历史，不指导当前运行。

## 工作原则

- 最终模型强度优先于吞吐和便利性；默认中文，保留 `GRP`、`Oracle critic`、`value / GAE`、`1v3`、`all_players`、`score_rank` 等术语。
- 在用户目标和授权范围内自主完成工作，具体方法自行决定。
- 2026-09-30 用户进一步明确：实验取舍根本依据投入产出比。每次实验要明确关键疑问、可能改变的决定、已有证据及更低成本替代；必要实验做足，可有可无的优化或省略。不为填满授权时窗或追求设备利用率而增加实验。算力截止仍是权限边界，不能自行延长。
- 以正确、可维护的代码为目标，按改动选择有效的[验证方式](docs/agent/code-health.md#按改动验证)；保留用户已有改动，避免无关变更。
- 查找优先 `rg` / `rg --files`；长输出外部命令按 [RTK.md](RTK.md) 执行。PowerShell 内建、短探针和管道直接运行。
- `logs/`、`checkpoints/`、`target/` 是产物，不随源码整理搬动或清理；不提交真实数据路径和凭据。
- 活跃 worker 在 spawn、恢复、切换阶段时可能重读源码。遵守 [活跃训练边界](docs/agent/code-health.md#活跃训练边界)，不原地覆盖其源码、原生扩展或 checkpoint。

## 环境与代码契约

- 非交互 Python 优先 `C:\ProgramData\anaconda3\envs\mortal\python.exe`；不假设 `conda` 或裸 `python` 指向正确环境。命令统一在 [运行流程](docs/agent/workflows.md) 维护。
- Rust 使用 Edition 2024；遵守 [libriichi/src/lib.rs](libriichi/src/lib.rs) 的 clippy deny。
- Python 使用 4 空格、`snake_case` 函数/变量、`PascalCase` 类。
- 特征常数：`ACTION_SPACE=46`、`obs_shape(v4)=(1012,34)`、`oracle_obs_shape(v4)=(217,34)`、`GRP_SIZE=7`、`MAX_VERSION=4`。变更必须同步 Rust、Python 与测试。
- `Brain` 默认 `BN`，当前训练/评测入口使用 `GN`，实例化时显式传 norm。V3/V4 为 pre-activation ResBlock，保留每块的 SE-style channel attention。
- [mortal/config.py](mortal/config.py) 默认读取 `mortal/config.toml`，可被 `MORTAL_CFG` 覆盖。PowerShell 写 TOML 优先 `apply_patch`，避免编码变化。
- CPU affinity 是 opt-in；正常训练不设置 `MORTAL_CPU_AFFINITY`。
- [mortal/core/common.py](mortal/core/common.py) 的 TCP / pickle 仅面向本机可信输入。

## 训练与评测

- `GRP best_loss` 默认供下游使用，`best_acc` 只做受控对照，`latest` 只用于续训。
- RL 启动权重优先级：`[control].state_file` → `[online].init_state_file` → `[supervised].best_loss_state_file` → `[supervised].best_state_file`。
- `1v3` challenger 使用 `[1v3.challenger].state_file`；训练默认关闭 `search`，推理增强单独 A/B。
- Oracle critic 保留 `all_players` 与真实 outcome / `score_rank` / return-to-go 标签；不为方便改成 GRP 伪标签或缩窄输出。
- 每轮选择依据该轮预声明目标和判据；旧实验的 guardrail 仅用于解释旧结论，不自动成为新研究不可变的要求。用户明确要求重新验证指标与实际牌力的关系；MAE 及按事后回报分组的误差不能未经论证一票否决条件均值预测器。保留历史结果，不事后改门槛宣布旧实验成功。平局未决，不默认选择更大权重，不按已实现的结果幅度加权样本。offline finalist 决策前不打开 sealed test。

## 双机与提交

- 2026-09-30 用户更新：本轮研究的源码统一在云端工作副本修改、测试和提交；ModelKits（RTX 5070 Ti）与 ABANDON（RTX 4060 Laptop）仅通过 Git 同步到独立、固定 commit 的运行工作树，不在两台 runner 上直接编辑源码。保留既有用户改动及正在运行的旧版本。
- 本轮算力授权截止为北京时间 2026-10-08 19:00（UTC 11:00）。所有本轮进程须有本机独立截止管理、明确归属及保存/停止余量；不得越过截止，不影响其他进程。
- 远程操作与本机采用同等授权，不因 SSH 额外要求确认；操作参考[远程流程](docs/agent/remote-ops.md)。
- 笔记本是独立 runner，默认不共享梯度、replay 或 checkpoint；同阶段双机运行的 run name、输出目录与 checkpoint 路径带机器区分。
- 后续所有笔记本计算任务都须在避免 RAM / VRAM OOM、保持训练与评测语义的前提下，最大限度利用计算资源，提高持续有效吞吐。按[性能与容量验收](docs/agent/remote-ops.md#性能与容量验收)验证峰值、并存负载与安全余量；持续低利用率须定位并优化，不能仅凭进程存活或未 OOM 判定调优完成。
- commit subject 用简短祈使句并带 scope，例如 `mortal: fix oracle dropout schedule`。
