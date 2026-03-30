# 当前主线与稳定默认

这份文档汇总当前仍有效、会直接影响后续实验判断的默认结论。它不是历史日志，也不负责保存所有旧方案。

## 口径优先级

1. `AGENTS.md`
2. `docs/agent/current-plan.md`
3. `docs/status/stage05-verified-status.md`
4. `docs/status/p1-selection-canonical.md`
5. `docs/status/stage05-fidelity-results.md`

如果 run snapshot 与上面这些文档冲突，以上面这些文档为准。

## 总目标

- 目标是最强麻将 AI，不是最省时间的麻将 AI。
- 强度差异明确时，优先更强方案。
- 只有强度接近时，才让效率和工程成本参与取舍。

## 资源与训练快路径

- 主节点：`i5-13600KF + RTX 5070 Ti`
- 副节点：`i9-13900HX + RTX 4060 Laptop 8GB + 32GB DDR5`
- 当前代码主分支：`main`
- 双机代码同步细节：`docs/agent/code-sync.md`
- 当前台式机默认 `Stage 0.5` 训练快路径：
  - train：`num_workers = 4`、`file_batch_size = 10`、`prefetch_factor = 3`
  - val：`val_file_batch_size = 8`、`val_prefetch_factor = 5`
- 当前笔记本独立 benchmark 操作默认：
  - train：`6 / 7 / 3`
  - val：`7 / 6`
  - close fallback：`7 / 5`

## Stage 0.5 当前冻结事实

- `P0` 事实 `top3` 顺序固定为：
  - `C_A2y_cosine_broad_to_recent_strong_12m_6m`
  - `C_A2x_cosine_broad_to_recent_strong_24m_12m`
  - `C_A1x_cosine_broad_to_recent_mild_24m_12m`
- `P1` 当前唯一主线固定为：
  - `calibration -> protocol_decide -> winner_refine -> ablation`
- 当前 `protocol_decide` winner，也是当前下游默认协议：
  - `C_A2x_cosine_broad_to_recent_strong_24m_12m`
- 当前瘦版 `calibration` 仍固定用 `A2y-only + combo_only`
  - 这是定标角色，不是下游 winner
- 当前三类辅助头内部 shape 已冻结：
  - `rank = 18K_ROUND_ONLY`
  - `opp = HYBRID_GRAD`
  - `danger = 18K_STAT`

## P1 当前默认

### 统一选模

- 当前 `P1` winner 解释只看 `docs/status/p1-selection-canonical.md`
- 当前排序核心固定为：
  - `selection_quality_score = action_quality_score + 0.20 * scenario_quality_score`
  - 先过 `recent_policy_loss + 0.003`
  - 如有旧回归集，再过 `old_regression_policy_loss + 0.0035`
  - 通过门槛后按 `selection_quality_score -> -recent_policy_loss -> -old_regression_policy_loss`
- 各类 `acc` 只保留为诊断字段，不参与排序

### protocol_decide

- 当前默认网格固定为：
  - `total_budget_ratios = [0.09, 0.12]`
  - `mixes = anchor / rank_lean / opp_lean / danger_lean`
- 当前默认 seed2 扩展规则固定为：
  - `ambiguity_mode = flip_or_gap`
  - `gap_threshold = 0.001`
- 历史旧 `ambig` 规则只允许作为旧 run 兼容字段读取，不再是当前默认

### winner_refine

- 当前默认只在 `A2x` 内部继续
- 当前默认不是自动取聚合榜前 `k` 名 center
- 当前默认冻结三中心：
  - `C_A2x_cosine_broad_to_recent_strong_24m_12m__B_r0046_o0037_d0037`
  - `C_A2x_cosine_broad_to_recent_strong_24m_12m__B_r0034_o0014_d0041`
  - `C_A2x_cosine_broad_to_recent_strong_24m_12m__B_r0052_o0025_d0043`
- 当前局部细搜规则固定为：
  - `total_scale_factors = [0.85, 1.0, 1.15]`
  - `transfer_delta = 0.01`
  - `step_scale = 1.5`

## 当前活跃 run

- run 目录：
  - `logs/stage05_fidelity/s05_fidelity_p1_top3_cali_slim_20260329_001413/`
- 当前状态：
  - `stopped_after_p1_protocol_decide`
- 当前人工停点：
  - `protocol_decide` 已结束并完成失败臂补跑
  - 等待确认后再进入 `winner_refine`

## 当前推荐阅读

1. `docs/status/stage05-verified-status.md`
2. `docs/status/p1-selection-canonical.md`
3. 本文档
4. `docs/research/stage05/engineering-playbook.md`
5. 需要更长背景时，再看 `docs/archive/`
