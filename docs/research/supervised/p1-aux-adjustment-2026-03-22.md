# 2026-03-22 P1 辅助头搜索调整备忘

> 原始记录用于说明 `P1` auxiliary search 调整过程。  
> 当前监督学习主线以 `docs/status/supervised-mainline.md` 为准；本文只保留仍有价值的证据和结论。

## 这次调整解决的问题

- 旧的 `rank / opponent / danger` turn weighting 过于经验化。
- 三类辅助头 raw loss 不可直接比较，需要统一有效预算口径。
- 旧 `P1 solo` 最小预算已经太重，导致辅助族过早压坏 policy loss。
- `solo -> pairwise` 不能让已经输给协议内 `CE-only` 的 family 继续晋级。

## turn bucket 结论

本地样本：

- `18` 年分层。
- `60` 文件 / 年。
- `1080` 游戏。
- `707,930` supervised states。

最终 turn bucket：

- early: `0-4`
- mid: `5-11`
- late: `12+`

关键观察：

- `rank_match_rate = 0.4719 / 0.4676 / 0.4643`
- `opp_any_tenpai_rate = 0.0342 / 0.4078 / 0.8303`
- `danger_state_has_any_rate = 0.0100 / 0.1650 / 0.4117`
- `danger_positive_discard_rate_given_valid = 0.0014 / 0.0291 / 0.0909`

选定默认：

- `rank = 1.00 / 1.05 / 1.15`
- `opp = 0.20 / 1.00 / 1.60`
- `danger = 0.05 / 1.00 / 2.50`

解释：

- `rank` 全局都有用，只随巡目轻微增强。
- `opp` 中后盘迅速变重要。
- `danger` 前期几乎无用，后期权重大幅上升。

## 年份漂移结论

本地 year-trend 样本：

- `18` 年。
- `120` 文件 / 年。
- 每年约 `76k-82k` states。

早期五年 `2010-2014` vs 近期五年 `2021-2025`：

- `turn 7 opp_any_tenpai_rate: 35.9% -> 34.1%`
- `turn 10 opp_any_tenpai_rate: 64.0% -> 63.3%`
- `mid 5-11 opp_any_tenpai_rate: 41.34% -> 40.35%`
- `late 12+ danger_positive_discard_rate_given_valid: 9.05% -> 9.23%`

结论：

- 没有证据支持按年份改 turn bucket。
- `0-4 / 5-11 / 12+` 在跨年份上仍然稳。

## cross-head calibration

raw loss 不能直接比较，因为：

- `rank` 有 sample-dependent weight template。
- `opp` 和 `danger` target 结构不同、稀疏度不同。
- head 对 trunk 的梯度压力也不同。

校准逻辑：

- 记录 loss-based effective budget。
- 记录 `phi_grad_rms` 表示 trunk-side gradient pressure。
- loss 轴和 grad 轴用几何平均混合。

关键校准值：

- `rank_effective_base = 0.0685`
- `opp_effective_per_unit = 1.1051`
- `danger_effective_per_unit = 0.2655`
- `rank_grad_effective_base = 1.09e-6`
- `opp_grad_effective_per_unit = 1.64e-5`
- `danger_grad_effective_per_unit = 9.55e-6`
- `opp_weight_per_budget_unit = 0.064`
- `danger_weight_per_budget_unit = 0.144`
- `joint_combo_factor = 0.883`

解释：

- `danger` 需要更大的显式 head weight 才能达到相近有效预算。
- 这不代表 `danger` 策略重要性一定高于 `opp`。

## `P1 solo` 调整

旧最小预算已经太重：

- `rank @ 0.25`: mean delta `+0.01693`
- `opp @ 0.25`: mean delta `+0.01971`
- `danger @ 0.25`: mean delta `+0.00898`

新的 solo budget ranges：

- `rank: [0.03, 0.06, 0.10, 0.15]`
- `opp: [0.03, 0.06, 0.10, 0.15]`
- `danger: [0.05, 0.10, 0.20, 0.30]`

映射后大致 head-weight bands：

- `opp: 0.0019 / 0.0038 / 0.0064 / 0.0096`
- `danger: 0.0072 / 0.0144 / 0.0288 / 0.0432`

## survivor rule

`solo -> pairwise` 只允许满足 canonical `policy_quality` gate 的 family 继续：

- arm 必须 valid。
- `comparison_recent_loss = recent_policy_loss` 必须在协议内门槛内。
- 如果有 `old_regression_policy_loss`，还必须满足 `0.0035` guardrail。
- family winner 不能明显输给同协议 `CE-only`。

## 当前保留价值

- turn bucket 和 turn weights 可作为当前训练默认解释。
- cross-head calibration 解释为什么不同 family 的 raw coefficient 不能直接比较。
- 旧 solo 结果解释为什么当前搜索预算更小、更偏 micro-budget。
- 监督学习最终 winner 已经冻结；本文不再指导重开主线，只作为证据记录。
