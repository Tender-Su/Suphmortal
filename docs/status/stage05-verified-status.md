# Stage 0.5 当前核对状态

这份文档只放“现在真实默认到哪里了”。它由人工维护，不让自动流程覆盖。

## 当前结论

- 核对日期：`2026-03-29`
- 当前主线阶段：`P1 protocol_decide`
- 当前活跃 run：`logs/stage05_fidelity/s05_fidelity_p1_top3_cali_slim_20260329_001413/`
- 当前状态字段：`running_p1_protocol_decide`

## 当前已冻结的事实

- `P0` 官方入选 `top3`：
  - `C_A2y_cosine_broad_to_recent_strong_12m_6m`
  - `C_A2x_cosine_broad_to_recent_strong_24m_12m`
  - `C_A1x_cosine_broad_to_recent_mild_24m_12m`
- `B-side` 当前不进入 `P1`
- 三类辅助内部 shape 已冻结：
  - `rank = 18K_ROUND_ONLY`
  - `opp = HYBRID_GRAD`
  - `danger = 18K_STAT`
- `P1` 当前唯一主线结构：
  - `calibration -> protocol_decide -> winner_refine -> ablation`
- `P1` 当前唯一有效选模口径：
  - `docs/status/p1-selection-canonical.md`

## 当前在跑什么

- 当前 `calibration` 已完成瘦版定标，后续不需要先重跑 `cali`
- 当前这轮 `calibration_mode = combo_only` 的含义是：
  沿用 `2026-03-25` 那轮旧 `single-head cali` 数值，
  不在本轮重算纯单头探针，只补 `pairwise / triple combo factor`
- 当前正在跑 `protocol_decide`
- 当前默认网格：
  - `total_budget_ratios = [0.09, 0.12]`
  - `anchor = 0.43 / 0.21 / 0.36`
  - `rank_lean = 0.53 / 0.16 / 0.31`
  - `opp_lean = 0.38 / 0.31 / 0.31`
  - `danger_lean = 0.38 / 0.16 / 0.46`

## 当前停点

- `protocol_decide` 结束后停下来
- 不自动进入 `winner_refine`
- 用户确认方向后，才继续下一阶段

## 怎么用这份文档

- 问“项目现在真实停在哪一步”，看这里
- 问“`P1` 应该按什么口径解释 winner”，看 `p1-selection-canonical.md`
- 问“自动流程刚刚写出了什么 run 快照”，看 `stage05-fidelity-results.md`
- 问“旧阶段当时是怎么判断的”，去 `docs/archive/`
