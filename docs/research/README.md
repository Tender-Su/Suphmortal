# 研究文档索引

这里放“为什么这样设计”和“证据从哪里来”。它不是当前默认手册；当前默认先看 `docs/agent/` 和 `docs/status/`。

## 主题入口

| 主题 | 入口 |
| --- | --- |
| Stage 0 / GRP | `stage0/grp-experience.md` |
| 监督学习演进总览 | `supervised-evolution.md` |
| 监督学习工程与 selector 证据 | `supervised/` |
| 在线 RL 研究、Oracle、baseline、opponent pool | `online-rl/README.md` |

## 监督学习材料

| 文件 | 用途 |
| --- | --- |
| `supervised-evolution.md` | 监督学习阶段从探索到正式 winner 的演进记录 |
| `supervised/engineering-playbook.md` | 当前仍可复用的工程经验、运行纪律与排障结论 |
| `supervised/selector-stat-audit.md` | selector 参数的统计支持和启发式边界 |
| `supervised/p1-aux-adjustment-2026-03-22.md` | `P1` auxiliary 修正背景与落地记录 |
| `supervised/a2y-aux-shape-freeze-2026-03-25.md` | `A2y` 主线下三类辅助头内部 shape 的冻结结论 |

## 在线 RL 材料

在线 RL 研究备忘集中在 `online-rl/`，入口见 `online-rl/README.md`。常用文件：

- `online-rl/rl-ppo-improvement-plan.md`
- `online-rl/oracle-critic-literature-observations-2026-04-16.md`
- `online-rl/online-baseline-refresh-and-reward-target-report-2026-04-13.md`
- `online-rl/oracle-ablation-protocol-2026-04-10.md`

## 使用规则

- 研究文档可以解释来源和边界，不能覆盖 `status/` 的当前结论。
- 新的在线 RL 调研放进 `online-rl/`；新的监督学习证据放进 `supervised/`。
- 新研究文档默认写成短备忘；需要保留长证据时，开头必须说明是否仍影响当前主线。
