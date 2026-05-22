# LuckyJ 与麻将 RL 减方差备忘

> 原始调研写于 `2026-04-12`。  
> 本文保留对当前项目仍有用的结论，不再展开完整文献综述。

## 当前结论

- `grp.label_smoothing` 不是主线减方差方案。它只是把终局标签变软，没有解决末盘随机性和部分可观测导致的 value 估计噪声。
- LuckyJ 公开口径更接近“强化学习 + 遗憾值最小化 + 乐观价值估计搜索”，但公开信息不足以复刻其训练细节。
- RVR、Suphx、ACH、LuckyJ 的共同点不是“标签平滑”，而是降低与决策无关的随机性或降低更新估计噪声。
- 当前项目的减方差主线应放在 `critic / value / expected reward / search / regret-style update`，不是继续加大 `label_smoothing`。

## 为什么 `label_smoothing` 不够

它能做：

- 让终局 rank 标签没那么尖。
- 让局部梯度更平。

它不能做：

- 建模牌山、对手手牌、末盘摸牌顺序带来的随机性。
- 让 critic 更准确地区分“决策导致的收益”和“运气导致的收益”。
- 替代 expected reward 或 Oracle critic。

所以它最多是 cheap smoothing fallback，不是 RVR 或 LuckyJ 式减方差。

## 公开工作给我们的启发

| 来源 | 可借鉴点 | 当前边界 |
| --- | --- | --- |
| RVR | 用全局信息 value / expected reward 降低终局奖励方差 | 中国标准麻将设置，不可直接照搬 |
| Suphx | global reward prediction、Oracle guiding、feature dropout | Oracle actor 证据不等于 Oracle critic 已验证 |
| ACH / NW-CFR | 避免高方差 counterfactual regret，改用 sampled advantage | 两人零和理论，四麻只能借鉴思想 |
| LuckyJ | 强调 RL、遗憾值、乐观价值估计搜索 | 公开细节很少，不能当作可复刻 recipe |

## 对当前项目的落地顺序

1. 固定 reward target 尺度，避免 online running 标准化和 batch 级 `v_target` 归一化。
2. 让 critic 真正成为低方差估计器，再谈 value / GAE / replay IS 放大。
3. Oracle 信息优先用于 critic / teacher / planner，而不是直接塑形最终 actor。
4. `search` 保持训练外评估，作为推理期系统增强单独 A/B。
5. regret-style update 或局部后悔度标签作为长期候选，从小标签开始验证。

## 当前不再建议

- 不再把 `grp.label_smoothing` 解释成正式 variance reduction。
- 不再把 LuckyJ 公开宣传语直接翻译成一套工程 recipe。
- 不再在训练期混合 `search` 收益和 actor 学习收益。

## 参考线索

- `docs/research/online-rl/online-baseline-refresh-and-reward-target-report-2026-04-13.md`
- `docs/research/online-rl/oracle-critic-literature-observations-2026-04-16.md`
- RVR: `Speedup Training Artificial Intelligence for Mahjong via Reward Variance Reduction`
- ACH: `Actor-Critic Policy Optimization in a Large-Scale Imperfect-Information Game`
