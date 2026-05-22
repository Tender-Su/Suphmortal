# 在线 RL baseline 与 reward target 备忘

> 原始报告写于 `2026-04-13`。  
> 本文只保留当前仍影响代码和实验判读的结论。

## 当前结论

- `reward_calculator.py` 已去掉 online running reward 标准化，reward target 保持固定 `delta_pt` 尺度。
- `train_online.py` 已去掉把 `v_target` 跟当前 batch advantage 一起归一化的逻辑。
- 当前 online rollout 支持 `baseline.test` 固定评测基线和 `baseline.train` opponent pool。
- 训练期 opponent pool 目前是 session 级抽样：每个 `train_play session` 抽一个 checkpoint，该 session 三家对手共用它。

## 为什么去掉两类归一化

| 机制 | 问题 | 当前处理 |
| --- | --- | --- |
| reward online running 标准化 | 训练分布会随模型变化，running mean/std 会让目标尺度漂移 | 直接用固定 `GRP delta_pt` 尺度 |
| batch 级 `v_target` 归一化 | 每个 batch 都改 target 尺度，critic 学不到稳定绝对值 | 去掉，保持 raw target |

核心判断：critic 要学习稳定的回报尺度，不该每个 batch 都被重新标尺。

## baseline / opponent pool 口径

- `baseline.test`：固定评测基线，用于可比的 `1v3`。
- `baseline.train`：训练对手分布，可以是固定 baseline，也可以是小型 opponent pool。
- 当前 `validation` pool 用于排因、Oracle 对照、短窗门测。
- 当前 `formal` pool 用于更长窗口和冲上限训练。

不要把训练对手更新和评测基线更新混成一个概念。评测需要稳定，训练可以逐步变强。

## 公开资料边界

| 来源 | 能确认 | 不能确认 |
| --- | --- | --- |
| Suphx | 异步分布式自博弈，worker 定期拉最新策略 | 公开文本没有给出类似本项目的显式 baseline 晋升器 |
| Mortal-Policy | 训练和测试读取固定 baseline checkpoint | 开源代码没有自动刷新 baseline 实现 |
| LuckyJ | 公开强调自博弈、遗憾值、乐观价值估计搜索 | 没公开 opponent pool / champion refresh 细节 |

## 当前建议

1. 继续保持 `baseline.test` 稳定，用正式 `1v3` 做过门判断。
2. 训练对手池按 `validation` / `formal` 两档使用，不手改 `baseline.train`。
3. 若未来做动态 champion 晋升，必须把晋升规则、冷却周期和回滚规则写清楚。
4. 不要每出现一个短窗更优 checkpoint 就立刻替换训练 baseline。

## 参考

- `docs/status/online-rl-mainline.md`
- `docs/research/online-rl/self-play-vs-fixed-baseline-survey-2026-04-16.md`
- `docs/research/online-rl/luckyj-variance-reduction-survey-2026-04-12.md`
