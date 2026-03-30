# 当前主线与稳定默认

这份文档只保留仍然有效、会直接影响后续实验判断的稳定默认。  
实时状态看 `docs/agent/current-plan.md` 和 `docs/status/stage05-verified-status.md`。  
运行步骤看 `docs/agent/experiment-workflow.md`。

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

## 代码与机器默认

- 当前代码主分支：`main`
- 当前同步细节入口：`docs/agent/code-sync.md`
- 主节点：`i5-13600KF + RTX 5070 Ti`
- 副节点：`i9-13900HX + RTX 4060 Laptop 8GB + 32GB DDR5`
- 台式机默认 `Stage 0.5` 训练快路径：
  - train：`4 / 10 / 3`
  - val：`8 / 5`
- 笔记本当前操作默认：
  - train：`6 / 7 / 3`
  - val：`7 / 6`
  - close fallback：`7 / 5`

## Stage 0 默认

- 默认主线：`384x3 fp32`
- 默认下游 checkpoint：`best_loss`
- `best_acc` 只作受控对照
- `latest` 只用于续训

## Stage 0.5 / P1 当前冻结默认

- `P0` 事实 `top3` 顺序固定为：
  - `C_A2y_cosine_broad_to_recent_strong_12m_6m`
  - `C_A2x_cosine_broad_to_recent_strong_24m_12m`
  - `C_A1x_cosine_broad_to_recent_mild_24m_12m`
- 当前 `P1` 唯一主线固定为：
  - `calibration -> protocol_decide -> winner_refine -> ablation`
- 当前 downstream 协议 winner：
  - `C_A2x_cosine_broad_to_recent_strong_24m_12m`
- 当前瘦版 `calibration` 仍固定用：
  - `A2y-only + combo_only`
- 当前三类辅助头内部 shape 已冻结：
  - `rank = 18K_ROUND_ONLY`
  - `opp = HYBRID_GRAD`
  - `danger = 18K_STAT`

## P1 选模与搜索默认

- 当前 `P1` winner 解释只看：`docs/status/p1-selection-canonical.md`
- 当前排序核心固定为：
  - `selection_quality_score = action_quality_score + 0.20 * scenario_quality_score`
  - 先过 `recent_policy_loss + 0.003`
  - 如有旧回归集，再过 `old_regression_policy_loss + 0.0035`
  - 通过门槛后按 `selection_quality_score -> -recent_policy_loss -> -old_regression_policy_loss`
- 各类 `acc` 只保留为诊断字段，不参与排序
- 当前 `protocol_decide` 默认网格：
  - `total_budget_ratios = [0.09, 0.12]`
  - `mixes = anchor / rank_lean / opp_lean / danger_lean`
- 当前 `protocol_decide` 默认 seed2 扩展规则：
  - `ambiguity_mode = flip_or_gap`
  - `gap_threshold = 0.001`
- 历史旧 `ambig` 规则只允许作为旧 run 兼容字段读取

## winner_refine 当前冻结默认

- 当前只允许在 `A2x` 内部继续
- 当前默认不是自动取聚合榜前 `k` 名 center
- 当前冻结三中心：
  - `C_A2x_cosine_broad_to_recent_strong_24m_12m__B_r0046_o0037_d0037`
  - `C_A2x_cosine_broad_to_recent_strong_24m_12m__B_r0034_o0014_d0041`
  - `C_A2x_cosine_broad_to_recent_strong_24m_12m__B_r0052_o0025_d0043`
- 当前局部细搜规则固定为：
  - `total_scale_factors = [0.85, 1.0, 1.15]`
  - `transfer_delta = 0.01`
  - `step_scale = 1.5`
- 如果让笔记本参与当前这轮 `winner_refine`，不要让两台机器共同写同一个 run
- 当前推荐入口是：
  - `python mortal/run_stage05_winner_refine_distributed.py dispatch --run-name <run_name>`
- 这条入口只改变执行方式：
  - `seed1` 全量
  - `seed2` 只补 `seed1` 后仍处在竞争带里的候选
- 它不改变当前 `winner_refine` 的 center、局部搜索点或最终 winner 解释口径

## 下游原则

- `Stage 1` 默认主线：`Oracle Dropout Supervised Refinement`
- `Stage 2` 只在新 `Stage 1` 稳定后再推进
