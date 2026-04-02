# 当前主线与冻结默认

这份文档只保留仍然有效、会直接影响后续判断的稳定默认。  
实时停点看 `docs/agent/current-plan.md`。  
当前核对状态看 `docs/status/stage05-verified-status.md`。  
运行顺序看 `docs/agent/experiment-workflow.md`。

## 总目标

- 目标是最强麻将 AI，不是最省时间的麻将 AI
- 强度差异明确时，优先更强方案
- 只有强度接近时，才让效率和工程成本参与取舍

## 代码与机器默认

- 当前开发分支：`main`
- 当前源码真源：台式机 `main` 工作树
- Git 同步入口：`docs/agent/code-sync.md`
- 主节点：`i5-13600KF + RTX 5070 Ti`
- 副节点：`i9-13900HX + RTX 4060 Laptop GPU 8GB + 32GB DDR5`

### Stage 0.5 loader 默认

- 台式机：
  - train：`4 / 10 / 3`
  - val：`8 / 5`
- 笔记本：
  - train：`4 / 10 / 4`
  - val：`7 / 5`
- 笔记本旧 `6 / 7 / 3`、`7 / 6` 口径已退役，不再作为当前默认

### 1v3 默认

- 台式机 `RTX 5070 Ti`：
  - `seed_count = 1024`
  - `shard_count = 4`
- 笔记本 `RTX 4060 Laptop GPU`：
  - `seed_count = 640`
  - `shard_count = 3`

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
- `P1` 当前唯一主线：
  - `calibration -> protocol_decide -> winner_refine -> ablation`
- 当前 downstream 协议 winner：
  - `C_A2x_cosine_broad_to_recent_strong_24m_12m`
- 当前人工确认的 winner 点位：
  - `0.12 + A2x`
- 当前瘦版 `calibration` 固定为：
  - `A2y-only + combo_only`
- 当前三类辅助头内部 shape 已冻结：
  - `rank = 18K_ROUND_ONLY`
  - `opp = HYBRID_GRAD`
  - `danger = 18K_STAT`

## P1 选模与搜索默认

- `P1` winner 解释只看：
  - `docs/status/p1-selection-canonical.md`
- 当前排序核心：
  - `selection_quality_score = action_quality_score + 0.20 * scenario_quality_score`
  - 先过 `recent_policy_loss + 0.003`
  - 如有旧回归集，再过 `old_regression_policy_loss + 0.0035`
  - 通过门槛后按 `selection_quality_score -> -recent_policy_loss -> -old_regression_policy_loss`
- 各类 `acc` 只保留为诊断字段，不参与排序

### protocol_decide 默认

- 坐标模式：
  - `projected_effective_from_budget_grid_v2`
- 搜索网格：
  - `total_budget_ratios = [0.09, 0.12]`
  - `mixes = anchor / rank_lean / opp_lean / danger_lean`
- seed2 扩展规则：
  - `ambiguity_mode = flip_or_gap`
  - `gap_threshold = 0.001`
- `2026-04-02` 的 `A2x @ 0.18` 三臂 probe 已收口为：
  - `seed1-only negative probe`
  - 它不改变默认网格
  - 它不改变当前协议 winner
  - 它不改变当前 winner 点位

## winner_refine 当前冻结默认

- 当前只允许在 `A2x` 协议内部继续
- 当前 `winner_refine` 不是“从任意旧 run 自动取 top-k”
- 当前冻结规则是：
  - 只对当前主线 run 的 `protocol_decide` effective-coordinate 排名使用 `top_ranked_keep`
  - `center_mode = top_ranked_keep`
  - `center_keep = 4`
- 当前四个 center：
  - `C_A2x_cosine_broad_to_recent_strong_24m_12m__W_r00516_o000135_d000804`
  - `C_A2x_cosine_broad_to_recent_strong_24m_12m__W_r00456_o000199_d000692`
  - `C_A2x_cosine_broad_to_recent_strong_24m_12m__W_r00636_o000103_d000692`
  - `C_A2x_cosine_broad_to_recent_strong_24m_12m__W_r00456_o000103_d001027`
- 当前局部细搜规则：
  - `total_scale_factors = [0.85, 1.0, 1.15]`
  - `transfer_delta = 0.01`
  - `step_scale = 1.5`
- 当前 distributed `winner_refine` seed2 selector：
  - `min_keep = 4`
  - `selection_gap = 0.001`
  - `max_keep = 12`
  - 每个 center 至少保 `1` 个有效点，再按全局竞争带补齐
- 如果让笔记本参与，推荐入口：
  - `python mortal/run_stage05_winner_refine_distributed.py dispatch --run-name <run_name>`
- 这条入口只改变调度方式，不改变候选空间和最终 winner 解释口径

## 下游原则

- `Stage 1` 默认主线：`Oracle Dropout Supervised Refinement`
- `Stage 2` 只在新的 `Stage 1` 稳定后再推进
