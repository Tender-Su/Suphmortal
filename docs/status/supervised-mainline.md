# 监督学习主线结论

这份文档合并原来的监督学习核对状态、`P1` 统一口径和 formal triplet 结论，只保留当前仍然有效的单一真相。

## 当前结论

- 核对日期：`2026-04-06`
- 监督学习阶段已经完成
- 当前 `P1` 协议 winner：
  - `C_A2x_cosine_broad_to_recent_strong_24m_12m`
- 当前已验证 winner 点位：
  - `0.12 + A2x`
- 当前正式 supervised winner：
  - `anchor*1.0`
- 当前第一替补：
  - `opp_lean*0.85`
- 当前 canonical supervised checkpoint：
  - `./checkpoints/sl_canonical.pth`
- `2026-04-12` 的补充核查：
  - `sl_canonical.pth` 对旧 `baseline.pth` 的 `3000` 局 `1v3`
  - `avg_rank=2.4917`
  - `avg_pt=-0.075`
  - 结论：当前 `sl_canonical` 至少可以认为与旧 `baseline` 基本接近，但还不是明显压制

## checkpoint 语义

- `state_file`
  - 当前 run 的 live training / resume 状态
- `best_state_file`
  - canonical supervised winner 的导出别名
- `best_loss_state_file`
  - 当前默认下游种子别名
  - 在当前 canonical 口径下应与 `best_state_file` 一起落到正式 winner
- `best_acc_state_file`
  - 作为 secondary candidate 保留
- `best_rank_state_file`
  - 作为 secondary candidate 保留
- formal child run 内部仍保留 `best_loss / best_acc / best_rank`
- 在线 RL 如果没有单独配置 `[online].init_state_file`，会继续回退到 `[supervised].best_loss_state_file -> [supervised].best_state_file`

## 已冻结事实

### `P0`

官方 `top3` 顺序固定为：

1. `C_A2y_cosine_broad_to_recent_strong_12m_6m`
2. `C_A2x_cosine_broad_to_recent_strong_24m_12m`
3. `C_A1x_cosine_broad_to_recent_mild_24m_12m`

### 三类辅助头内部 shape

- `rank = 18K_ROUND_ONLY`
- `opp = HYBRID_GRAD`
- `danger = 18K_STAT`

### `P1` 主线结构

- `calibration -> protocol_decide -> winner_refine`
- `ablation` 只保留为手动诊断轮

## `P1` 唯一有效评估口径

### 结果边界

- `protocol_decide` 负责协议 winner
- `winner_refine` 负责 winner 协议内的 pre-formal 第一梯队
- 官方 supervised winner 只由 `formal triplet -> formal_1v3` 产生

### 排名核心

1. `ranking_mode = policy_quality`
2. 主比较字段固定为 `recent_policy_loss`
3. `eligible` 必须按 `protocol_arm` 组内判断，不能跨协议混排
4. 组内先过门槛：
   - `recent_policy_loss <= group_best_recent_policy_loss + 0.003`
   - 如果存在 `old_regression_policy_loss`，再要求 `<= group_best_old_regression_policy_loss + 0.0035`
5. 进入 `eligible` 后，再按以下顺序排序：
   - `selection_quality_score`
   - `-recent_policy_loss`
   - `-old_regression_policy_loss`

### 关键字段

- `selection_quality_score = action_quality_score + 0.20 * scenario_quality_score`
- 自动摘要里的 `cmp_policy` 对应 `recent_policy_loss`
- 自动摘要里的 `full_loss(diag)` 对应 `full_recent_loss`

### 当前 `P1` 冻结配置

- `calibration`
  - `A2y-only + combo_only`
- `protocol_decide`
  - `coordinate_mode = projected_effective_from_budget_grid_v2`
  - `total_budget_ratios = [0.09, 0.12]`
  - `mixes = anchor / rank_lean / opp_lean / danger_lean`
  - `ambiguity_mode = flip_or_gap`
  - `gap_threshold = 0.001`
- `winner_refine`
  - 协议范围：`A2x`
  - `center_mode = top_ranked_keep`
  - `center_keep = 4`
  - center：`anchor / rank_lean / opp_lean / danger_lean`
  - `total_scale_factors = [0.85, 1.0, 1.15]`
  - `transfer_delta = 0.01`
  - `step_scale = 1.5`

### 命名口径

- center 只写：
  - `anchor / rank_lean / opp_lean / danger_lean`
- 全头统一缩放只写：
  - `*0.85 / *1.0 / *1.15`
- center 内部再分配只写：
  - `rank+ / rank++ / opp- / danger++`
- canonical 文档统一使用结构别名，不手写原始 `W_r..._o..._d...` 名字

### calibration 输出如何被读取

- `protocol_decide / winner_refine` 读取 `triple_combo_factor`
- `drop_rank` 读取 `opp_danger_combo_factor`
- `drop_opp` 读取 `rank_danger_combo_factor`
- `drop_danger` 读取 `rank_opp_combo_factor`
- `joint_combo_factor` 仍保留为 `opp_danger_combo_factor` 的 legacy alias

## formal triplet -> 官方 winner

### 当前 triplet

送入 `formal_train` 的三个候选是：

1. `opp_lean*0.85`
2. `anchor*1.0`
3. `opp_lean(rank--/danger++)`

### child formal 结果

- `3 / 3` child formal 已完成
- `3 / 3` 的 `offline_checkpoint_winner` 都是 `best_loss`
- cross-run offline 顺序：
  1. `opp_lean*0.85`
  2. `opp_lean(rank--/danger++)`
  3. `anchor*1.0`
- 关键标量：
  - `opp_lean*0.85`
    - `best_full_recent_loss = 0.480049`
    - offline front-runner
  - `opp_lean(rank--/danger++)`
    - `best_full_recent_loss = 0.480872`
    - hedge challenger
  - `anchor*1.0`
    - `best_full_recent_loss = 0.480868`
    - `rank_acc` 最强

### `formal_1v3` 最终顺序

判据：

- `avg_pt` 为主
- `avg_rank` 为辅
- 位次分：`90 / 45 / 0 / -135`

最终顺序：

1. `anchor*1.0`
2. `opp_lean*0.85`
3. `opp_lean(rank--/danger++)`

### 运行长度

- `phase_a / phase_b / phase_c = 45000 / 30000 / 15000`
- `2026-04-05` 实测 wall-clock：
  - 台式机单条约 `4.5 h`
  - 笔记本单条约 `11.2 h`

## 当前证据路径

- 当前活跃监督学习 source run：
  - `logs/sl_fidelity/sl_fidelity_p1_top3_cali_slim_20260329_001413/`
- downstream coordinator run：
  - `logs/sl_fidelity/sl_formal_triplet_20260405/`
- downstream playoff run：
  - `logs/sl_fidelity/sl_formal_triplet_20260405_winner_playoff_1v3/`
- 自动 snapshot：
  - `docs/status/supervised-fidelity-results.md`
