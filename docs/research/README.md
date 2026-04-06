# 研究文档索引

这里放的是“为什么这样设计”和“有哪些证据支持”的文档，不是当前默认手册。

## 按主题划分

### `supervised-evolution.md`

- 监督学习阶段从实验探索到当前结构的统一演进记录

### `stage0/`

- `grp-experience.md`
  - `GRP` 当前默认、代表性结果、仍值得继续探索的候选

### `supervised/`

- `engineering-playbook.md`
  - 当前监督学习工程经验、运行纪律与排障结论
- `selector-stat-audit.md`
  - selector 哪些参数已经有统计支持，哪些仍保留启发式
- `p1-aux-adjustment-2026-03-22.md`
  - `P1` auxiliary 修正的统计背景与代码落地记录
- `a2y-aux-shape-freeze-2026-03-25.md`
  - `A2y` 主线下三类辅助头内部 shape 的冻结结论

### `rl-ppo-improvement-plan.md`

- 强化学习阶段 PPO 的研究性改进方案草稿

## 怎么读

- 只想知道当前默认：先看 `docs/agent/` 和 `docs/status/`
- 想理解监督学习阶段是如何形成当前结构的：看 `supervised-evolution.md`
- 想理解当前工程经验：看 `supervised/engineering-playbook.md`
- 想看 selector 统计证据：看 `supervised/selector-stat-audit.md`
- 想看 RL 方向的后续研究：看 `rl-ppo-improvement-plan.md`
