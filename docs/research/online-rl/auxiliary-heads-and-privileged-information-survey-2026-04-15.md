# 辅助头与训练期特权信息备忘

> 原始调研写于 `2026-04-15`。  
> 本文保留对当前 online RL 设计仍有用的结论。

## 当前结论

- `rank / opponent / danger` 是可见信息辅助头，默认继续和 actor 共享主干共训。
- 如果辅助头和主策略互相干扰，优先考虑分阶段更新或调权，不要第一反应就是全冻结。
- `Oracle` 属于训练期特权信息，不建议优先直接塑形最终要部署的 actor。
- Oracle 更适合先放在 critic / teacher / planner 这些不直接部署的路径上。
- 训练阶段固定关闭 `search`；`search=true` 作为推理期系统增强单独 A/B。

## 为什么可见辅助头不默认冻结

`rank / opponent / danger` 学的是 actor 实战中可见或可从公开信息推断的结构：

- 局况压力和点差语境。
- 对手向听 / 听牌等公开推断目标。
- 各类可打牌的危险度。

这些头的价值是逼共享主干读懂公开牌局。只要没有明确干扰证据，共训比冻结更自然。

公开工作的启发：

- `UNREAL` 支持“辅助任务给共享表示补训练信号”。
- `PPG` 提醒多目标可能互相干扰，但解决方案是分 phase 或隔离更新，不是默认否定共享。
- `What makes useful auxiliary tasks in reinforcement learning` 的关键提醒是：辅助任务必须和主任务共享有用信息，否则会变成噪声。

## 为什么 Oracle 不优先直接喂 actor

Oracle hidden info 在部署时不可见。如果直接喂 actor，风险是：

- actor 学到部署时拿不到的捷径。
- continuation 后看似可部署，但收益可能只是“加噪再删噪”。
- shuffled Oracle 若表现接近 true Oracle，就说明真实 hidden info 价值并不清楚。

更稳的用法是：

- critic 用 Oracle 降低 value 估计噪声。
- teacher / distillation 用 Oracle 提供训练信号。
- planner / search 用 belief 或采样在推理期单独评估。

## search 的位置

- 训练阶段默认 `search=false`。
- `search` 作为推理期系统增强，不和 actor 训练收益混算。
- 如果要做 `search_distill`，必须单独标注为 teacher/refine 路线。

## 当前实验顺序

1. 先验证 `rank / opponent / danger` 这组可见辅助头是否能稳定过门。
2. 再处理 `value / GAE / replay IS` 的稳定性。
3. Oracle critic 先做 influence ramp，避免未成熟 critic 过早主导 actor。
4. actor Oracle 四臂对照后置到共享优化栈更稳定之后。

## 参考

- `docs/status/online-rl-mainline.md`
- `docs/research/online-rl/oracle-ablation-protocol-2026-04-10.md`
- `docs/research/online-rl/oracle-critic-literature-observations-2026-04-16.md`
