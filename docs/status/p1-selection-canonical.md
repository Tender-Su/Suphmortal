# P1 统一评估口径

这份文档是当前项目里 `P1` 的唯一有效评估规范。任何脚本、自动摘要、人工总结或无上下文 agent，只要要回答 `P1` 的排序、第一梯队或 downstream 入口，都以这里为准。

## 结果边界

- `protocol_decide` 负责选出协议 winner
- `winner_refine` 负责在 winner 协议内部给出 pre-formal 第一梯队
- `formal triplet playoff -> formal_1v3` 负责产生官方 supervised winner
- 当前官方 supervised winner：
  - `anchor*1.0`
- 当前第一替补：
  - `opp_lean*0.85`
- downstream 证据入口：
  - `docs/status/supervised-formal-triplet-playoff-canonical.md`

## 适用范围

- `A2y internal-shape micro AB`
- `p1_protocol_decide_round`
- `p1_winner_refine_round`
- `p1_ablation_round`

## 排名核心

1. `ranking_mode` 固定为 `policy_quality`
2. 主比较字段固定为 `comparison_recent_loss = recent_policy_loss`
3. `eligible` 必须按 `protocol_arm` 分组判断，不能跨协议混排
4. 每个 `protocol_arm` 组内先过门槛
   - `recent_policy_loss <= group_best_recent_policy_loss + 0.003`
   - 如果有 `old_regression_policy_loss`，再要求 `old_regression_policy_loss <= group_best_old_regression_policy_loss + 0.0035`
5. 进入 `eligible` 后，再按以下顺序排序
   - `selection_quality_score`
   - `-recent_policy_loss`
   - `-old_regression_policy_loss`

## 字段解释

- `selection_quality_score = action_quality_score + 0.20 * scenario_quality_score`
- 各类 `acc` 作为诊断字段
- `full_recent_loss` 作为总 loss / aux tax 诊断字段
- 自动摘要里的 `cmp_policy` 对应 `recent_policy_loss`
- 自动摘要里的 `full_loss(diag)` 对应 `full_recent_loss`

## 各轮职责

### `p1_calibration`

- 结构：
  - `A2y-only + combo_only`
- 职责：
  - budget mapping
  - combo factor 定标

### `p1_protocol_decide_round`

- 职责：
  - 在统一三头脚手架下选出协议 winner
- 搜索默认：
  - `coordinate_mode = projected_effective_from_budget_grid_v2`
  - `total_budget_ratios = [0.09, 0.12]`
  - `mixes = anchor / rank_lean / opp_lean / danger_lean`
  - `ambiguity_mode = flip_or_gap`
  - `gap_threshold = 0.001`
- 当前人工确认的 winner 点位：
  - `0.12 + A2x`
- 当前协议 winner：
  - `C_A2x_cosine_broad_to_recent_strong_24m_12m`

### `p1_winner_refine_round`

- 协议范围：
  - `A2x`
- center 规则：
  - `center_mode = top_ranked_keep`
  - `center_keep = 4`
- center 集合：
  - `anchor`
  - `opp_lean`
  - `rank_lean`
  - `danger_lean`
- 局部扰动规则：
  - `total_scale_factors = [0.85, 1.0, 1.15]`
  - `transfer_delta = 0.01`
  - `step_scale = 1.5`
- 职责：
  - 给出 pre-formal 第一梯队
  - 保留内部排序证据

### `p1_ablation_round`

- 职责：
  - 比较 `all_three / drop_* / ce_only` 的边际贡献
- 角色：
  - 手动诊断轮

## 命名口径

- center 只写：
  - `anchor / rank_lean / opp_lean / danger_lean`
- 全头统一缩放只写：
  - `*0.85 / *1.0 / *1.15`
- center 内部再分配只写：
  - `rank+/rank++/opp-/danger++`
- canonical 文档使用结构别名

## calibration 输出如何被读取

- `protocol_decide / winner_refine` 读取 `triple_combo_factor`
- `drop_rank` 读取 `opp_danger_combo_factor`
- `drop_opp` 读取 `rank_danger_combo_factor`
- `drop_danger` 读取 `rank_opp_combo_factor`
- `joint_combo_factor` 作为 `opp_danger_combo_factor` 的别名可被读取

## 使用规则

- `calibration` 只负责定标
- `winner_refine` 只负责 pre-formal 第一梯队
- 官方 supervised winner 只由 `formal triplet playoff -> formal_1v3` 产生
- canonical 文档统一使用结构别名，不手写原始 `W_r..._o..._d...` 名字
