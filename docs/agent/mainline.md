# 当前主线与稳定默认

这份文档汇总当前仍然有效、且会影响后续实验判断的默认结论。它不是逐日日志，也不记录所有历史试错。

## 口径优先级

1. `AGENTS.md`
2. `docs/agent/current-plan.md`
3. `docs/status/stage05-verified-status.md`
4. `docs/status/p1-selection-canonical.md`
5. `docs/status/stage05-fidelity-results.md` 仅作 run 快照

## 总目标

- 目标是最强麻将 AI，不是最省时间的麻将 AI。
- 当两个方案强度接近时，优先更小更快。
- 当强度差异明确时，优先更强方案。

## Stage 0

- 默认主线：`384x3 fp32`
- 默认下游 checkpoint：`best_loss`
- 受控对照：`best_acc`
- `latest` 只用于续训
- 仍值得继续探索的更大候选：`512x4`
- 研究入口：`docs/research/stage0/grp-experience.md`

## Stage 0.5

- 默认正式训练快路径：
  - train：`num_workers = 4`、`file_batch_size = 10`、`prefetch_factor = 3`
  - val：`val_file_batch_size = 8`、`val_prefetch_factor = 5`
- 当前 `P0` 官方入选 `top3`：
  - `C_A2y_cosine_broad_to_recent_strong_12m_6m`
  - `C_A2x_cosine_broad_to_recent_strong_24m_12m`
  - `C_A1x_cosine_broad_to_recent_mild_24m_12m`
- `B-side` 当前是资源冻结，不进入当前 `P1`
- `P0 / P2 / formal` 的默认主比较字段仍是 `full_recent_loss`

## P1 主线

- 当前结构固定为：`calibration -> protocol_decide -> winner_refine -> ablation`
- `calibration` 的职责是定标，不是宣告 winner
- 当前默认使用瘦版 `calibration`：
  - 代表协议固定为 `A2y`
  - 单头量纲直接沿用 `2026-03-25` post-shape `top3 calibration` 的旧数值
  - 这意味着当前 `combo_only` 不是“本轮重新估单头，只少跑几项”，而是复用旧 `single-head cali`
  - 当前 run 只补 `pairwise / triple combo factor`
- 当前冻结的三头内部 shape：
  - `rank = 18K_ROUND_ONLY`
    - `south_factor = 1.59`
    - `all_last_factor = 1.617`
    - `gap_focus_points = 4000`
    - `gap_close_bonus = 0.0`
  - `opp = HYBRID_GRAD`
    - `shanten = 0.8506568408`
    - `tenpai = 1.1493431592`
  - `danger = 18K_STAT`
    - `any = 0.0904217947`
    - `value = 0.8180402859`
    - `player = 0.0915379194`
- 当前 `protocol_decide` 默认网格：
  - `total_budget_ratios = [0.09, 0.12]`
  - `anchor = 0.43 / 0.21 / 0.36`
  - `rank_lean = 0.53 / 0.16 / 0.31`
  - `opp_lean = 0.38 / 0.31 / 0.31`
  - `danger_lean = 0.38 / 0.16 / 0.46`
- `P1` 选模、门槛和解释规则只看：`docs/status/p1-selection-canonical.md`

## 选择器当前默认

- `selection_quality_score = action_quality_score + 0.20 * scenario_quality_score`
- `recent_policy_loss` 门槛：`+ 0.003`
- `old_regression_policy_loss` 门槛：`+ 0.0035`
- `selection_tiebreak_key = selection_quality_score -> -recent_policy_loss -> -old_regression_policy_loss`
- 各类 `acc` 只保留为诊断字段
- 统计审计入口：`docs/research/stage05/selector-stat-audit.md`

## Stage 1 与 Stage 2

- `Stage 1` 默认主线：`Oracle Dropout Supervised Refinement`
- 它从 `Stage 0.5` winner 协议和配比继续，而不是回到旧 `GRP-AWR`
- `Stage 2` 在 `Stage 1` 稳定后再推进，不提前抢跑

## 当前活跃 run

- run 目录：`logs/stage05_fidelity/s05_fidelity_p1_top3_cali_slim_20260329_001413/`
- 当前状态：`running_p1_protocol_decide`
- 当前人工停点：`protocol_decide` 结束即停，等待确认

## 进一步阅读

- 当前真实进度：`docs/status/stage05-verified-status.md`
- 运行流程与停点：`docs/agent/experiment-workflow.md`
- 工程方法与有效经验：`docs/research/stage05/engineering-playbook.md`
- 历史长入口：`docs/archive/agent/current-plan-legacy-2026-03-29.md`
