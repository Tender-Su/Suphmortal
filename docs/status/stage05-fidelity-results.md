# Stage 0.5 淇濈湡鐗?A/B 瀹炴椂缁撴灉

- 运行目录：`C:\Users\numbe\Desktop\MahjongAI\logs\stage05_fidelity\s05_fidelity_p1_top3_cali_slim_20260329_001413`
- 更新时间：`2026-03-29 04:31:59`
- 当前状态：`running_p1_protocol_decide`
- 自动串联范围：`P0 + P1 + P2 + Stage 0.5 formal`
- 说明：`P3(Stage 1 transfer)` 不作为本轮 `Stage 0.5 formal` 启动前置条件，避免把 Stage 1 下游转移实验混入 0.5 阶段主协议定型。

> run-scoped snapshot: if this file conflicts with the verified/canonical status docs, prefer those docs for current defaults.

## 当前结论

- P0 下游种子 top4：``
- P0 round3 winner：`TBD`
- P1 总胜者：`TBD`
- P2 默认 checkpoint：`TBD`
- 正式训练：`pending`

## P1

### canonical_selection

- `selector = policy_quality`
- `适用范围 = A2y internal-shape micro AB / p1_protocol_decide_round / p1_winner_refine_round / p1_ablation_round`
- `比较字段 = comparison_recent_loss = recent_policy_loss`
- `eligible 分组 = protocol_arm`；`recent_policy_loss <= group_best + 0.003`
- `old_regression` 可用时，再要求 `old_regression_policy_loss <= group_best_old + 0.0035`
- `selection_quality_score = action_quality_score + 0.200 * scenario_quality_score`
- `主排序 = selection_quality_score -> -recent_policy_loss -> -old_regression_policy_loss`
- `family guardrail = historical family-survivor guardrail: family survivor must not lose clearly to ce_only under the same policy_quality gate`
- `ce_only anchor = ce_only is kept as a diagnostic anchor; mainline P1 no longer uses family survivors as the protocol gate`
- `full_recent_loss = full_recent_loss` 仅作 aux tax / 总 loss 诊断，不是 `P1` family winner 的主判胜字段
- `calibration note = p1_calibration is a mapping step for single-head units plus pairwise/triple combo factors; it must not be used to declare family winners`

### calibration

- `mapping_mode = hybrid_loss_grad_geomean`
- `combo_scheme = single_head_mapping_plus_pairwise_triple_combo`
- `inherited_single_head = True`
- `inherited_single_head_source = 2026-03-25 post-shape top3 calibration on the frozen A2y/A2x/A1x pool`
- `rank_effective_base = 0.051215`
- `opp_effective_per_unit = 0.899069`
- `danger_effective_per_unit = 0.049248`
- `rank_grad_effective_base = 0.000001`
- `opp_grad_effective_per_unit = 0.000018`
- `danger_grad_effective_per_unit = 0.000002`
- `budget_ratios = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]`
- `opp_weight_per_budget_unit_loss = 0.057000`
- `danger_weight_per_budget_unit_loss = 0.180000`
- `opp_weight_per_budget_unit_grad = 0.047000`
- `danger_weight_per_budget_unit_grad = 0.180000`
- `opp_weight_per_budget_unit = 0.052000`
- `danger_weight_per_budget_unit = 0.180000`
- `rank_opp_combo_factor = 0.968000`
- `rank_opp_combo_factor_loss = 1.250000`
- `rank_opp_combo_factor_grad = 0.750000`
- `rank_danger_combo_factor = 1.065000`
- `rank_danger_combo_factor_loss = 1.250000`
- `rank_danger_combo_factor_grad = 0.907000`
- `opp_danger_combo_factor = 0.969000`
- `opp_danger_combo_factor_loss = 0.996000`
- `opp_danger_combo_factor_grad = 0.943000`
- `triple_combo_factor = 0.968000`
- `triple_combo_factor_loss = 1.250000`
- `triple_combo_factor_grad = 0.750000`
- `joint_combo_factor(legacy_opp_danger_alias) = 0.969000`
- `grad_probe_batches = 8`
- `fallback_used = False`

### search_space

- `combo_scheme = single_head_mapping_plus_pairwise_triple_combo`
- `protocol_decide_total_budget_ratios = [0.09, 0.12]`
- `protocol_decide_mixes = [{'name': 'anchor', 'rank_share': 0.43, 'opp_share': 0.21, 'danger_share': 0.36}, {'name': 'rank_lean', 'rank_share': 0.53, 'opp_share': 0.16, 'danger_share': 0.31}, {'name': 'opp_lean', 'rank_share': 0.38, 'opp_share': 0.31, 'danger_share': 0.31}, {'name': 'danger_lean', 'rank_share': 0.38, 'opp_share': 0.16, 'danger_share': 0.46}]`
- `rank_opp_combo_factor = 0.968`
- `rank_danger_combo_factor = 1.065`
- `opp_danger_combo_factor = 0.969`
- `triple_combo_factor = 0.968`
- `protocol_decide_probe_keep_per_protocol = 4`
- `winner_refine_total_scale_factors = [0.85, 1.0, 1.15]`
- `winner_refine_transfer_delta = 0.01`

### calibration_round

- `actual_seeds = [20261716]`
- `ranking_mode = policy_quality`

- `cmp_policy = comparison_recent_loss = recent_policy_loss`
- `full_loss(diag) = full_recent_loss`

| rank | arm | cmp_policy | full_loss(diag) | selection | action | scenario | rank_acc | eligible |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `C_A2y_cosine_broad_to_recent_strong_12m_6m__CAL_rank_danger_probe` | 0.5712 | 0.6237 | -0.2636 | -0.2114 | -0.2610 | 0.4361 | True |
| 2 | `C_A2y_cosine_broad_to_recent_strong_12m_6m__CAL_triple_probe` | 0.5727 | 0.6788 | -0.2648 | -0.2120 | -0.2636 | 0.4396 | True |
| 3 | `C_A2y_cosine_broad_to_recent_strong_12m_6m__CAL_rank_opp_probe` | 0.5732 | 0.6766 | -0.2653 | -0.2126 | -0.2634 | 0.4345 | True |
| 4 | `C_A2y_cosine_broad_to_recent_strong_12m_6m__CAL_opp_danger_probe` | 0.5743 | 0.6308 | -0.2665 | -0.2134 | -0.2653 | 0.2446 | False |

## 路径

- 状态文件：`C:\Users\numbe\Desktop\MahjongAI\logs\stage05_fidelity\s05_fidelity_p1_top3_cali_slim_20260329_001413\state.json`
- 文档文件：`C:\Users\numbe\Desktop\MahjongAI\docs\status\stage05-fidelity-results.md`
