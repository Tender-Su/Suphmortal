# MahjongAI

日本立直麻将 AI 的训练、评测与接入仓库。目标是最终牌力；模型选择需要可复现的对局证据。

## 从这里开始

| 要做什么 | 入口 |
| --- | --- |
| 看当前进度与下一步 | [接手摘要](docs/agent/handoff.md) |
| 查全部文档 | [文档地图](docs/README.md) |
| 安装、训练、评测 | [运行流程](docs/agent/workflows.md) |
| 修改代码 | [代码地图](docs/agent/code-map.md) · [重构与验证](docs/agent/code-health.md) |
| 让 agent 接手 | [仓库规则](AGENTS.md) · [按任务阅读](docs/agent/README.md) |

## 仓库结构

| 路径 | 职责 |
| --- | --- |
| [libriichi/](libriichi/) | Rust 牌局引擎、状态与特征、PyO3 扩展 |
| [mortal/](mortal/README.md) | GRP、SL、Oracle critic、在线 RL 与评测 |
| [integrations/](integrations/) | [雀魂连接](integrations/majsoul/README.md)、[RiichiLab 客户端](integrations/riichilab/README.md) |
| [MahjongCopilot/](MahjongCopilot/readme.md) | 独立宿主；加载器与运行环境的边界见其自身说明 |
| [exe-wrapper/](exe-wrapper/) | Rust helper binary |
| [scripts/](scripts/) | Windows 启动、机器操作和审计工具 |
| [docs/](docs/README.md) | 当前状态、操作说明、证据、历史与复盘 |

模型、数据、凭据和实验日志通常不随源码分发。首次使用按 [运行流程](docs/agent/workflows.md) 配置环境和数据；不要把示例配置当成当前实验配方。

当前结论只在 `docs/status/` 维护。研究报告解释证据，归档保留历史；根 README 不复制训练阶段、winner 或机器参数。
