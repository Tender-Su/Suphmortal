# 在线 RL 研究索引

这里放在线 RL 相关的研究备忘、论文调研和协议设计。当前默认仍以 `docs/status/online-rl-mainline.md` 为准。

## 当前最常用

| 文件 | 用途 |
| --- | --- |
| `rl-ppo-improvement-plan.md` | PPO 改进状态、保留候选和当前优先级 |
| `oracle-critic-literature-observations-2026-04-16.md` | Oracle critic 文献边界与接入风险 |
| `online-baseline-refresh-and-reward-target-report-2026-04-13.md` | reward target、baseline、opponent pool 口径 |
| `self-play-vs-fixed-baseline-survey-2026-04-16.md` | 自博弈、历史快照池和固定 baseline 的取舍 |

## 专题材料

| 文件 | 用途 |
| --- | --- |
| `oracle-ablation-protocol-2026-04-10.md` | Oracle 四臂对照与依赖度评测协议 |
| `luckyj-variance-reduction-survey-2026-04-12.md` | LuckyJ、RVR、Suphx 等减方差思路对照 |
| `auxiliary-heads-and-privileged-information-survey-2026-04-15.md` | 辅助头、特权信息和 critic / teacher 位置判断 |

## 使用规则

- 先读 `docs/status/online-rl-mainline.md`，再来这里追原因。
- 这里的旧计划不能覆盖当前状态页。
- 新的在线 RL 调研默认放在本目录，不再堆到 `docs/research/` 顶层。
- 新文件优先写短备忘；只有原始证据不可压缩时才保留长文。
