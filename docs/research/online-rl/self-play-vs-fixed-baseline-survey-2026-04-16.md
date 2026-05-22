# 自博弈、历史快照池与固定 baseline 备忘

> 原始综述写于 `2026-04-16`。  
> 本文只保留对当前 opponent pool 设计仍有用的判断。

## 当前结论

- 纯自博弈不是唯一正统；它更适合近似可传递、对称、零和的环境。
- 历史快照池 / 联盟训练在复杂多智能体、非传递博弈里很常见，不是权宜之计。
- 固定 baseline 更适合早期起步、课程学习、可控 benchmark 和因果验证。
- 当前项目用 `1v3` baseline 分布做排因是合理的；长期最强训练应逐步走向 champion + anchor + history 的 opponent pool。

## 三类对手分布

| 分布 | 优点 | 风险 | 当前用途 |
| --- | --- | --- | --- |
| 纯自博弈 | 追求长期上限，减少固定对手过拟合 | 多人不完美信息下容易循环和遗忘 | 暂不作为当前第一步 |
| 历史快照池 / 联盟 | 提高鲁棒性，防止只克上一代 | 管理复杂，需要晋升和采样规则 | `formal` pool 的长期方向 |
| 固定 / 半固定 baseline | 稳定、可比、便于排因 | 长期可能过拟合对手 | `validation` pool 和正式 `1v3` |

## 公开工作的启发

- AlphaZero：纯自博弈在对称零和场景自然有效。
- FSP：不完美信息博弈里，平均策略 / 群体策略是正统路线之一。
- AlphaStar：复杂多智能体场景依赖联盟和多样对手。
- OpenAI Five：自博弈骨架外，仍需要定期面对历史版本防塌陷。
- 固定 baseline / built-in AI：常用于课程学习和可控 benchmark。

## 对当前项目的工程判断

- `validation` opponent pool 用于干净因果验证，不追求最终训练分布最强。
- `formal` opponent pool 用于更长窗口和冲上限，应该保留 champion、anchor、history 的混合。
- `baseline.test` 保持稳定，避免评测口径漂移。
- 只有 recipe 本身站稳后，才值得把训练进一步推向更像自博弈的人口式训练。

## V-trace 边界

V-trace 解决的是“样本来自旧策略 / 异步行为策略”的校正问题，不负责设计对手分布。不要把 replay 校正和 opponent pool 设计混为一谈。

## 参考

- `docs/status/online-rl-mainline.md`
- `docs/research/online-rl/online-baseline-refresh-and-reward-target-report-2026-04-13.md`
