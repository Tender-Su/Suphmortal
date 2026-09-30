# Oracle critic 与训练期特权信息备忘

> 历史归档 · 归档整理：2026-09-05。正文保留当时的事实、判断和命令，不作为当前运行依据。当前入口见 [文档地图](../../../README.md)；旧 SL / RL 强度及 Oracle 验证结论须结合 [独立审计](../../../research/sl-rl-audit-2026-09-05.md) 阅读。

> 原始文献记录写于 `2026-04-16`。  
> 当前只保留文献边界、项目风险和方案收束。

## 当前问题

当前 `Oracle critic` 的风险不在于 Oracle 信息没价值，而在于初始化：

- `oracle_brain` 可能由 visible-only SL checkpoint bridge 而来。
- 这个模型一开始没学过如何使用 Oracle 输入。
- 如果它立刻通过 value / GAE / advantage 强力影响 actor，可能把策略更新带偏。

所以短期关键不是“要不要 Oracle critic”，而是“如何让未成熟 critic 不过早支配 actor”。

一句话口径：不是要防止 critic 使用 Oracle，而是要防止未成熟、未对齐的 critic 过早影响 actor。成熟且对齐的 Oracle critic 应该完整使用。

## 文献观测边界

| 来源 | 能支持什么 | 不能支持什么 |
| --- | --- | --- |
| Suphx `rl.tex` | 正式做过 actor 侧 Oracle guiding + perfect-feature dropout | 没给出 Oracle critic 的完整 recipe |
| Suphx `remark.tex` | 提到 Oracle critic 是可考虑方向 | 没有初始化、warmup、loss 配比和消融细节 |
| Suphx RL-2 | 完整 oracle guiding recipe 在其设置下优于 RL-1 | 不证明本项目 Oracle critic 单独有效 |
| Mortal-Policy | 强调 offline -> online，online-only 不推荐 | 没有 Oracle critic 方案 |
| 不对称 actor-critic 文献 | critic 可以看更多训练期信息来降噪 | 前提是特权信息确实改善 return 估计 |
| PPG | 多目标干扰可用分阶段训练缓解 | 不直接给出 Oracle critic 接法 |

## 对当前项目的推论

- Oracle critic 是合理候选，但不是已经被 Suphx 验证好的固定做法。
- 直接从 visible-only bridge 初始化后马上让它影响 actor，风险偏高。
- 如果短窗负收益，不能直接判 Oracle 信息无用；更可能是 critic 接入节奏太猛。

## 方案收束

当前最推荐：

- `value_loss` 可以从 step 0 开始训练。
- critic 对 GAE / actor 的影响使用 ramp，从小到大。
- 保持 actor visible-only，先让 critic 学会稳定使用 Oracle。

后续备选：

- hard `oracle_critic_warmup`：先只训 critic，不让它影响 actor。
- sidecar / late fusion：把 Oracle 分支单独建模，再融合到 value。
- Oracle Value Predictor 预训练：先让 Oracle value 学稳定，再接 PPO。
- 分阶段接回 actor：只有 critic 稳定后再考虑更强的 actor/teacher 互动。

## 当前不建议

- 不建议把 Suphx RL-2 直接等同于“Oracle critic 已被验证”。
- 不建议在 `protocol_decide` 形式 winner 不显著时继续推 `winner_refine`。
- 不建议用训练内 `test_play` 替代正式 `1v3` 判断。

## 参考

- `docs/status/online-rl-mainline.md`
- `docs/archive/research/online-rl/auxiliary-heads-and-privileged-information-survey-2026-04-15.md`
- `docs/archive/research/online-rl/oracle-ablation-protocol-2026-04-10.md`
