# Stage 0.5 保真版 A/B 实时结果 (historical snapshot)

- 运行目录：`C:\Users\numbe\Desktop\MahjongAI\logs\stage05_fidelity\s05_fidelity_p1_a2x_budget018_probe3_20260402_195712`
- 更新时间：`2026-04-02 21:55:46`
- 运行态状态：`paused_p1_protocol_decide_seed2`
- 人工解释：`manual_closeout_after_seed1_negative_probe`
- 自动串联范围：`P0 + P1 + P2 + Stage 0.5 formal`
- 说明：这份 snapshot 记录的是一次 `A2x @ 0.18` 三臂定向 probe，不是当前主线 run；当前默认仍以 `current-plan.md / mainline.md / stage05-verified-status.md / p1-selection-canonical.md` 为准。

> historical snapshot: this file records one run output and may not match the current default search space.
> The search_space section below may contain retired keys or retired ambiguity settings from that run.
> For current defaults, prefer `docs/agent/current-plan.md`, `docs/agent/mainline.md`, `docs/status/stage05-verified-status.md`, and `docs/status/p1-selection-canonical.md`.

## 当前结论

- 当前主线协议 winner：`C_A2x_cosine_broad_to_recent_strong_24m_12m`
- 当前主线 winner 点位：`0.12 + A2x`
- 本轮 run 类型：`A2x-only / budget 0.18 / anchor + rank_lean + danger_lean`
- 本轮 run 结论：`seed1-only negative probe`
- 本轮对主线的影响：`none`

## P1

### canonical_selection

- `selector = policy_quality`
- `适用范围 = A2y internal-shape micro AB / p1_protocol_decide_round / p1_winner_refine_round / p1_ablation_round`
- `比较字段 = comparison_recent_loss = recent_policy_loss`
- `eligible 分组 = protocol_arm`；`recent_policy_loss <= group_best + 0.003`
- `old_regression` 可用时，再要求 `old_regression_policy_loss <= group_best_old + 0.0035`
- `selection_quality_score = action_quality_score + 0.200 * scenario_quality_score`
- `主排序 = selection_quality_score -> -recent_policy_loss -> -old_regression_policy_loss`

### search_space

- `calibration_mode = combo_only`
- `combo_scheme = single_head_mapping_plus_pairwise_triple_combo`
- `inherited_single_head_source = 2026-03-25 post-shape top3 calibration on the frozen A2y/A2x/A1x pool`
- `protocol_decide_total_budget_ratios = [0.18]`
- `protocol_decide_mixes = anchor / rank_lean / danger_lean`
- `protocol_decide_probe_keep_per_protocol = 4`
- `protocol_decide_seed2_actual_trigger = mechanical`
- `mechanical_trigger_reason = candidate_count = 3 < probe_keep_per_protocol = 4`
- `ambiguity_evidence = none`

### calibration_round

- `actual_seeds = [20261716]`
- `ranking_mode = policy_quality`

### protocol_decide_seed1

- `actual_seeds = [20261817]`
- `candidate_count = 3`
- `ranking_mode = policy_quality`

| rank | arm | mix | cmp_policy | full_loss(diag) | selection | action | scenario | rank_acc | eligible |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `C_A2x_cosine_broad_to_recent_strong_24m_12m__W_r00774_o000204_d001205` | `anchor_18` | 0.533689 | 0.540900 | -0.253143 | -0.202575 | -0.252838 | 0.3986 | True |
| 2 | `C_A2x_cosine_broad_to_recent_strong_24m_12m__W_r00684_o000155_d001539` | `danger_lean_18` | 0.536087 | 0.542388 | -0.253313 | -0.202882 | -0.252154 | 0.4040 | True |
| 3 | `C_A2x_cosine_broad_to_recent_strong_24m_12m__W_r00954_o000155_d001037` | `rank_lean_18` | 0.535610 | 0.543044 | -0.253556 | -0.202849 | -0.253538 | 0.4081 | True |

### interpretation

- `anchor_18` 是这轮 probe 内部的 `seed1` front runner
- `anchor_18` 相比 `anchor_15` 有所回升，但在 canonical selection 下仍不足以取代当前主线的 `0.12 + A2x`
- `rank_lean_18` 相比 `rank_lean_15` 明显回落
- `danger_lean_18` 相比 `danger_lean_15` 也没有继续改善
- 这说明 `0.18` 更像是一个局部高风险 surface，而不是适合继续下游 `winner_refine` 的稳定入口
- 因此本轮 run 按 `seed1-only negative probe` 人工收口，不恢复 `seed2`

## 路径

- 状态文件：`C:\Users\numbe\Desktop\MahjongAI\logs\stage05_fidelity\s05_fidelity_p1_a2x_budget018_probe3_20260402_195712\state.json`
- 调度状态：`C:\Users\numbe\Desktop\MahjongAI\logs\stage05_fidelity\s05_fidelity_p1_a2x_budget018_probe3_20260402_195712\distributed\protocol_decide_dispatch\dispatch_state.json`
- `seed1` 轮次摘要：`C:\Users\numbe\Desktop\MahjongAI\logs\stage05_fidelity\s05_fidelity_p1_a2x_budget018_probe3_20260402_195712\p1_protocol_decide_round__s20261817.json`
- 文档文件：`C:\Users\numbe\Desktop\MahjongAI\docs\status\stage05-fidelity-results.md`
