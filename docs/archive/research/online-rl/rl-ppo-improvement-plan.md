# 在线 RL PPO 改进备忘

> 历史归档 · 归档整理：2026-09-05。正文保留当时的事实、判断和命令，不作为当前运行依据。当前入口见 [文档地图](../../../README.md)；旧 SL / RL 强度及 Oracle 验证结论须结合 [独立审计](../../../research/sl-rl-audit-2026-09-05.md) 阅读。

> 原始长文写于 `2026-04-01`，用于规划强化学习阶段。  
> 当前默认以 `docs/status/online-rl-mainline.md` 为准；本文只保留仍有用的研究结论和后续候选。

## 当前结论

- `logit_thres`、单边 `entropy_floor`、Step-Level GAE、版本级 replay importance sampling、Oracle critic、local belief search / planner 都已经有代码路径。
- 当前短窗结果显示：三辅助头 `rank / opponent / danger` 有微正信号；`value / GAE / replay IS` 这组还没有稳定放大。
- `RL-2` 的早期 smoke 变差不能直接判死完整 Oracle curriculum，但也不能作为当前第一优先级。
- 当前不该再按原长文的“大而全 PPO 改造清单”推进；应该围绕 `add_rank_opp_danger`、critic 稳定化和正式 `1v3` 过门收敛。

## 已落地能力

| 能力 | 当前状态 | 备注 |
| --- | --- | --- |
| chosen-action `logit_thres` | 已实现 | 梯度门控，不再做前向 logits clamp |
| 单边 `entropy_floor` | 已实现 | 熵低于下限时增强熵项 |
| Step-Level GAE | 已实现 | 但和 value / IS 组合仍需稳定性复查 |
| replay importance sampling / V-trace | 已实现 | `vtrace_mode=auto` 时只对足够陈旧 replay 启用 |
| Oracle critic | 已实现 | 当前风险在初始化和过早支配 actor |
| local belief search / planner | 已实现 | 当前训练阶段默认关闭，只作为推理期增强单独评估 |
| `search_distill` | 已接线 | 不作为当前默认训练底座 |

## 保留的研究判断

### LuckyJ / ACH

- LuckyJ 公开信息不足以复刻完整 recipe。
- ACH / NW-CFR 的价值在于提醒我们：降低 sampled update 方差和改进策略更新很重要。
- 但 ACH 的两人零和理论保证不能直接搬到四人日本麻将；当前不建议把 Hedge 或完整 CFR 当近期开工主线。

### RVR

- RVR 真正有价值的是把高随机性终局目标替换成更稳定的 value / expected reward 估计。
- 这支持我们继续研究 critic、expected reward、Oracle critic 和 search，而不是继续把 `grp.label_smoothing` 当主线减方差方案。

### 局部后悔度辅助头

- 原长文列出的牌效率、立直改良、默听、副露、见逃、押引、杠等 regret-style 标签仍可作为长期候选。
- 近期不要一次性做完整“局部后悔度系统”；优先从最容易验证、最贴近现有数据的局部标签开始。
- 如果要接入 PPO，先作为辅助监督或诊断头，不要直接强行改 advantage。

## 当前优先级

1. 继续扩 `add_rank_opp_danger` 到 `1500 -> 3000`。
2. 回头查 `value / GAE / replay IS` 为什么短窗正向不能持续。
3. 对 Oracle critic 做轻量 `critic influence ramp`，让 value loss 可从 step 0 学，但 critic 对 actor / GAE 的影响逐步放大。
4. `search` 只做推理期增强 A/B；训练期保持关闭，避免把 actor 学习收益和 planner 收益混在一起。
5. regret-style 方向保留为长期研究，不抢当前主线资源。

## 不再保留的旧计划

- 不再按 P0/P1/P2/P3 的旧大表逐项推进。
- 不再把完整 CFR、Hedge 替代 softmax、全量局部后悔度标签构造当近期工程计划。
- 不再在本文内维护代码级 patch 片段；实现细节以源码和测试为准。

## 参考

- `docs/status/online-rl-mainline.md`
- `docs/archive/research/online-rl/luckyj-variance-reduction-survey-2026-04-12.md`
- `docs/archive/research/online-rl/oracle-critic-literature-observations-2026-04-16.md`
- `docs/archive/research/online-rl/online-baseline-refresh-and-reward-target-report-2026-04-13.md`
