# Stage 0.5 保真版 A/B 实时结果

> Historical snapshot only. This file was generated before the `2026-03-28` P1 redesign.
> Current default P1 structure is `calibration -> protocol_decide -> winner_refine -> ablation`.
> Do not use the old `p1_solo / pairwise / joint_refine` layout in this snapshot as the current default.

- 运行目录：`C:\Users\numbe\Desktop\MahjongAI\logs\stage05_fidelity\s05_fidelity_main`
- 更新时间：`2026-03-24 01:33:01`
- 当前状态（historical snapshot）：`stopped_after_p1_solo`
- 自动串联范围：`P0 + P1 + P2 + Stage 0.5 formal`
- 说明：`P3(Stage 1 transfer)` 不作为本轮 `Stage 0.5 formal` 启动前置条件，避免把 Stage 1 下游转移实验混入 0.5 阶段主协议定型。

## 当前结论

- P0 下游种子 top4：`C_A1x_cosine_broad_to_recent_mild_24m_12m, C_A3x_cosine_broad_to_recent_two_stage_24m_12m, C_B2z_cosine_recent_broad_recent_strong_6m_6m, C_B2x_cosine_recent_broad_recent_strong_24m_12m`
- P0 round3 winner：`TBD`
- P1 总胜者：`TBD`
- P2 默认 checkpoint：`TBD`
- 正式训练：`pending`

## P0

### round0

| rank | arm | cmp_loss | full_loss | selection | action | scenario | rank_acc | eligible |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `C_B1z_cosine_recent_broad_recent_mild_6m_6m` | 0.6411 | 0.6411 | -0.3395 | -0.2170 | -0.2451 | 0.4598 | True |
| 2 | `C_A2x_cosine_broad_to_recent_strong_24m_12m` | 0.6414 | 0.6414 | -0.3404 | -0.2175 | -0.2458 | 0.4583 | True |
| 3 | `C_B1x_cosine_recent_broad_recent_mild_24m_12m` | 0.6414 | 0.6414 | -0.3404 | -0.2174 | -0.2462 | 0.4650 | True |
| 4 | `C_A2z_cosine_broad_to_recent_strong_6m_6m` | 0.6425 | 0.6425 | -0.3406 | -0.2173 | -0.2465 | 0.4633 | True |
| 5 | `C_A2y_cosine_broad_to_recent_strong_12m_6m` | 0.6421 | 0.6421 | -0.3407 | -0.2171 | -0.2471 | 0.4595 | True |
| 6 | `C_B2x_cosine_recent_broad_recent_strong_24m_12m` | 0.6420 | 0.6420 | -0.3407 | -0.2181 | -0.2452 | 0.4617 | True |
| 7 | `C_B3x_cosine_recent_broad_recent_two_stage_24m_12m` | 0.6437 | 0.6437 | -0.3408 | -0.2177 | -0.2464 | 0.4613 | True |
| 8 | `C_B2z_cosine_recent_broad_recent_strong_6m_6m` | 0.6412 | 0.6412 | -0.3409 | -0.2174 | -0.2471 | 0.4607 | True |

### round1

| rank | arm | cmp_loss | full_loss | selection | action | scenario | rank_acc | eligible |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `C_B3x_cosine_recent_broad_recent_two_stage_24m_12m` | 0.6042 | 0.6042 | -0.3317 | -0.2107 | -0.2420 | 0.4651 | True |
| 2 | `C_A3x_cosine_broad_to_recent_two_stage_24m_12m` | 0.6039 | 0.6039 | -0.3320 | -0.2107 | -0.2425 | 0.4676 | True |
| 3 | `C_B2x_cosine_recent_broad_recent_strong_24m_12m` | 0.6042 | 0.6042 | -0.3326 | -0.2108 | -0.2435 | 0.4637 | True |
| 4 | `C_A1x_cosine_broad_to_recent_mild_24m_12m` | 0.6039 | 0.6039 | -0.3327 | -0.2112 | -0.2430 | 0.4647 | True |
| 5 | `C_B3y_cosine_recent_broad_recent_two_stage_12m_6m` | 0.6023 | 0.6023 | -0.3332 | -0.2117 | -0.2430 | 0.4690 | True |
| 6 | `C_B2z_cosine_recent_broad_recent_strong_6m_6m` | 0.6037 | 0.6037 | -0.3339 | -0.2122 | -0.2434 | 0.4687 | True |
| 7 | `C_B3z_cosine_recent_broad_recent_two_stage_6m_6m` | 0.6037 | 0.6037 | -0.3348 | -0.2126 | -0.2444 | 0.4682 | True |
| 8 | `P_B3z_plateau_recent_broad_recent_two_stage_6m_6m` | 0.6196 | 0.6196 | -0.3373 | -0.2142 | -0.2463 | 0.4663 | False |

### round2

| rank | arm | cmp_loss | full_loss | selection | action | scenario | rank_acc | eligible |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `C_A1x_cosine_broad_to_recent_mild_24m_12m` | 0.5810 | 0.5810 | -0.3201 | -0.2032 | -0.2339 | 0.4548 | True |
| 2 | `C_A3x_cosine_broad_to_recent_two_stage_24m_12m` | 0.5794 | 0.5794 | -0.3201 | -0.2028 | -0.2346 | 0.4525 | True |
| 3 | `C_B2z_cosine_recent_broad_recent_strong_6m_6m` | 0.5797 | 0.5797 | -0.3202 | -0.2026 | -0.2352 | 0.4547 | True |
| 4 | `C_B2x_cosine_recent_broad_recent_strong_24m_12m` | 0.5816 | 0.5816 | -0.3203 | -0.2031 | -0.2343 | 0.4551 | True |
| 5 | `C_B3z_cosine_recent_broad_recent_two_stage_6m_6m` | 0.5807 | 0.5807 | -0.3206 | -0.2028 | -0.2356 | 0.4537 | True |
| 6 | `C_B3x_cosine_recent_broad_recent_two_stage_24m_12m` | 0.5814 | 0.5814 | -0.3211 | -0.2036 | -0.2350 | 0.4554 | True |

## P1

### calibration

- `mapping_mode = hybrid_loss_grad_geomean`
- `rank_effective_base = 0.071455`
- `opp_effective_per_unit = 0.984633`
- `danger_effective_per_unit = 0.170887`
- `rank_grad_effective_base = 0.000001`
- `opp_grad_effective_per_unit = 0.000018`
- `danger_grad_effective_per_unit = 0.000005`
- `budget_ratios = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]`
- `opp_weight_per_budget_unit_loss = 0.073000`
- `danger_weight_per_budget_unit_loss = 0.180000`
- `opp_weight_per_budget_unit_grad = 0.061000`
- `danger_weight_per_budget_unit_grad = 0.180000`
- `opp_weight_per_budget_unit = 0.067000`
- `danger_weight_per_budget_unit = 0.180000`
- `joint_combo_factor = 0.911000`
- `joint_combo_factor_loss = 0.990000`
- `joint_combo_factor_grad = 0.838500`
- `grad_probe_batches = 8`
- `fallback_used = False`

### calibration_round

- `actual_seeds = [20261716]`

| rank | arm | cmp_loss | full_loss | selection | action | scenario | rank_acc | eligible |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `C_B2z_cosine_recent_broad_recent_strong_6m_6m__CAL_rank_only` | 0.6463 | 0.6463 | -0.3441 | -0.2124 | -0.2634 | 0.4368 | True |
| 2 | `C_A3x_cosine_broad_to_recent_two_stage_24m_12m__CAL_rank_only` | 0.6483 | 0.6483 | -0.3450 | -0.2135 | -0.2630 | 0.4388 | True |
| 3 | `C_A1x_cosine_broad_to_recent_mild_24m_12m__CAL_rank_only` | 0.6479 | 0.6479 | -0.3458 | -0.2134 | -0.2649 | 0.4391 | True |
| 4 | `C_B2x_cosine_recent_broad_recent_strong_24m_12m__CAL_rank_only` | 0.6471 | 0.6471 | -0.3470 | -0.2137 | -0.2668 | 0.4384 | True |
| 5 | `C_B2x_cosine_recent_broad_recent_strong_24m_12m__CAL_both_probe` | 0.7158 | 0.7158 | -0.3429 | -0.2123 | -0.2613 | 0.4313 | False |
| 6 | `C_B2z_cosine_recent_broad_recent_strong_6m_6m__CAL_danger_probe` | 0.6555 | 0.6555 | -0.3433 | -0.2122 | -0.2623 | 0.4343 | False |
| 7 | `C_A1x_cosine_broad_to_recent_mild_24m_12m__CAL_danger_probe` | 0.6576 | 0.6576 | -0.3434 | -0.2122 | -0.2623 | 0.4393 | False |
| 8 | `C_A3x_cosine_broad_to_recent_two_stage_24m_12m__CAL_both_probe` | 0.7170 | 0.7170 | -0.3439 | -0.2130 | -0.2619 | 0.4319 | False |

### solo_round

- `actual_seeds = [20261817, 20262826]`
- `seed_strategy = progressive_probe_then_expand`
- `probe_selector = solo_family_top2`
- `probe_candidate_count = 28`
- `decision_candidate_count = 52`
- `expanded_groups = ['C_A1x_cosine_broad_to_recent_mild_24m_12m', 'C_A3x_cosine_broad_to_recent_two_stage_24m_12m', 'C_B2x_cosine_recent_broad_recent_strong_24m_12m', 'C_B2z_cosine_recent_broad_recent_strong_6m_6m']`

| rank | arm | cmp_loss | full_loss | selection | action | scenario | rank_acc | eligible |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `C_B2x_cosine_recent_broad_recent_strong_24m_12m__B_r000_o006_d000` | 0.5734 | 0.5788 | -0.3443 | -0.2129 | -0.2630 | 0.2505 | True |
| 2 | `C_B2z_cosine_recent_broad_recent_strong_6m_6m__B_r000_o000_d000` | 0.5726 | 0.5726 | -0.3446 | -0.2125 | -0.2642 | 0.2466 | True |
| 3 | `C_B2x_cosine_recent_broad_recent_strong_24m_12m__B_r000_o000_d005` | 0.5739 | 0.5755 | -0.3446 | -0.2127 | -0.2637 | 0.2518 | True |
| 4 | `C_B2x_cosine_recent_broad_recent_strong_24m_12m__B_r000_o015_d000` | 0.5735 | 0.5859 | -0.3447 | -0.2130 | -0.2634 | 0.2513 | True |
| 5 | `C_B2x_cosine_recent_broad_recent_strong_24m_12m__B_r000_o000_d030` | 0.5738 | 0.5831 | -0.3447 | -0.2127 | -0.2639 | 0.2500 | True |
| 6 | `C_A3x_cosine_broad_to_recent_two_stage_24m_12m__B_r000_o010_d000` | 0.5738 | 0.5831 | -0.3448 | -0.2129 | -0.2638 | 0.2521 | True |
| 7 | `C_B2z_cosine_recent_broad_recent_strong_6m_6m__B_r000_o010_d000` | 0.5739 | 0.5831 | -0.3448 | -0.2127 | -0.2642 | 0.2501 | True |
| 8 | `C_A1x_cosine_broad_to_recent_mild_24m_12m__B_r000_o000_d010` | 0.5739 | 0.5771 | -0.3448 | -0.2129 | -0.2639 | 0.2501 | True |

## 路径

- 状态文件：`C:\Users\numbe\Desktop\MahjongAI\logs\stage05_fidelity\s05_fidelity_main\state.json`
- 文档文件（归档后）：`C:\Users\numbe\Desktop\MahjongAI\docs\archive\status\stage05-fidelity-results-root-legacy.md`
