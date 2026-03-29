# Selector 统计审计

> 更新说明：本文里的多 seed 噪声证据最初来自历史 `p1_solo_round`。
> 这些统计仍可作为 selector 噪声量级参考，但不再代表当前主线 round 结构。
> 当前默认 P1 主线 round 是 `p1_protocol_decide_round / p1_winner_refine_round / p1_ablation_round`。

这份文档记录 `2026-03-26` 对当前 selector 口径做的一次“哪些可以统计化、哪些仍需保留启发式”的核对结果。对应脚本是：

- `mortal/analyze_selection_heuristics.py`
- 输出：
  - `logs/selection_heuristic_audit.json`
  - `logs/selection_heuristic_audit.md`

## 已跑统计

- 数据样本：`2009-2026` 共 `3240` 个日志文件
- 覆盖状态数：`2,133,322`
- 多 seed selector 证据：当前仓库里可直接用于 `P1` selector 噪声估计的是 `s05_fidelity_main/p1_solo_round.json`

## 结论

### 1. 可以改成统计值的

- `policy_loss_epsilon`
  - 统计来源：多 seed 同 arm 的 `recent_policy_loss` 抖动
  - 当前分布：`p90 abs diff = 0.002873...`
  - 推荐值：`0.003`
  - 结论：维持当前默认，不需要改

- `old_regression_policy_loss_epsilon`
  - 统计来源：多 seed 同 arm 的 `old_regression_policy_loss` 抖动
  - 当前分布：`p90 abs diff = 0.003444...`
  - 推荐值：`0.0035`
  - 结论：不应继续和 `policy epsilon` 共用同一个 `0.003`

### 2. 仍保留启发式的

- `comparison_recent_loss = recent_policy_loss`
  - 这是实验目标定义，不是频率统计问题

- `eligibility_group_key = protocol_arm`
  - 这是公平对照约束，不是频率统计问题

- `selection_quality_score = action + scenario_factor * scenario`
  - 它仍然不是可由独立真实牌力标签唯一拟合出的“纯统计常数”
  - 但在 `selection_tiebreak_key` 收敛到 `selection_quality_score -> -recent_policy_loss -> -old_regression_policy_loss` 后，可以做稳定性导向的细搜
  - 当前正式搜索带：`0.0-0.25`
  - 推荐准则：先看 `pairwise_agreement`，再用 `aggregate_winner_match_rate` 做次级筛选；旧 `0.5` 只保留为对照，不直接参与默认值推荐
  - 当前多 seed 细搜最佳点：`0.199`
  - 运行时默认值：`0.20`
  - 额外解释：按当前 `P1 solo` 候选分布，`0.20` 让 `scenario` 的跨度贡献约为 `action` 跨度的 `52.9%`；旧 `0.5` 则达到 `132%`
  - 结论：不把 `scenario_factor` 视为纯统计常数，但把默认值从旧 `0.5` 收敛到新的 `0.20`

- `ACTION_SCORE_WEIGHTS / SCENARIO_SCORE_WEIGHTS`
  - 决策切片每个状态同时激活很多项：
    - `decision active slices per state`: `median = 10`, `p90 = 14`
    - `discard active slices per state`: `median = 3`, `p90 = 8`
  - 因而频率本身无法唯一识别“哪一项该分多少信用”
  - 结论：继续保留启发式 + shrinkage

- `selection_tiebreak_key` 的尾部字典序
  - 当前没有独立真实标签来把这些尾部字段拟合成唯一正确顺序
  - `scenario_quality_score / action_quality_score` 已经被并入 `selection_quality_score`，而现有产物审计里各类 `acc` 也没有实际参与过决胜
  - 结论：排序键只保留 `selection_quality_score -> -recent_policy_loss -> -old_regression_policy_loss`；各类 `acc` 改为诊断字段

- `count-shrinkage priors`
  - 当前仓库没有专门为 prior 做的多 seed 受控重复实验
  - 结论：保留启发式

## 当前默认

- `recent_policy_loss <= group_best + 0.003`
- `old_regression_policy_loss <= group_best_old + 0.0035`
- `selection_quality_score = action_quality_score + 0.20 * scenario_quality_score`
- `selection_tiebreak_key = selection_quality_score -> -recent_policy_loss -> -old_regression_policy_loss`
