# P1 统一选模口径

这份文档是当前项目里 `P1` 的唯一有效选模规范。  
任何脚本、自动摘要、人工总结或无上下文新 agent，只要要回答 `P1` winner 相关问题，都以这里为准。

## 适用范围

- `A2y internal-shape micro AB`
- `p1_protocol_decide_round`
- `p1_winner_refine_round`
- `p1_ablation_round`

历史 `p1_solo_round / p1_pairwise_round / p1_joint_refine_round` 仍可作诊断证据，但已经不是当前主线结构。

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
- 各类 `acc` 只保留为诊断字段，不参与排序
- `full_recent_loss` 只保留为总 loss / aux tax 诊断字段
- 自动摘要里的 `cmp_policy` 就是 `recent_policy_loss`
- 自动摘要里的 `full_loss(diag)` 就是 `full_recent_loss`

## 各轮职责

- `p1_calibration`
  - 负责 budget mapping 和 combo factor 定标
  - 不能单独宣布 winner
- `p1_protocol_decide_round`
  - 负责在统一三头脚手架下选出协议 winner
- `p1_winner_refine_round`
  - 负责只在 winner 协议内部细调三头全开配比
- `p1_ablation_round`
  - 负责比较 `all_three / drop_* / ce_only` 的边际贡献

## 当前 protocol_decide 默认

- 搜索网格固定为：
  - `total_budget_ratios = [0.09, 0.12]`
  - `mixes = anchor / rank_lean / opp_lean / danger_lean`
- 当前 seed2 扩展规则固定为：
  - `ambiguity_mode = flip_or_gap`
  - `gap_threshold = 0.001`
- 含义：
  - 如果 probe 后 winner 发生翻转，展开该协议的完整 seed2
  - 如果 probe 后 winner 未翻转，但 `top1-top2 recent_policy_loss gap <= 0.001`，也展开完整 seed2
  - 否则不展开
- 当前人工确认的 winner 点位是：
  - `0.12 + A2x`
- `2026-04-02` 的 `A2x @ 0.18` 三臂 probe 只作为：
  - `seed1-only negative probe`
  - 它不改默认网格
  - 它不改当前 winner 解释
  - 它不构成新的下游入口

## 当前 winner_refine 默认

- 当前只在 `A2x` 协议内部继续
- 当前规则不是“从任意旧 run 自动取 top-k”
- 当前冻结规则是：
  - 对当前主线 run 的 `protocol_decide` effective-coordinate 排名使用 `top_ranked_keep`
  - `center_mode = top_ranked_keep`
  - `center_keep = 4`
- 当前四个 center：
  - `C_A2x_cosine_broad_to_recent_strong_24m_12m__W_r00516_o000135_d000804`
  - `C_A2x_cosine_broad_to_recent_strong_24m_12m__W_r00456_o000199_d000692`
  - `C_A2x_cosine_broad_to_recent_strong_24m_12m__W_r00636_o000103_d000692`
  - `C_A2x_cosine_broad_to_recent_strong_24m_12m__W_r00456_o000103_d001027`
- 当前局部扰动规则：
  - `total_scale_factors = [0.85, 1.0, 1.15]`
  - `transfer_delta = 0.01`
  - `step_scale = 1.5`

## ce_only 的解释

- `ce_only` 是主线里的诊断锚点
- 它不再是旧 `family survivor` 逻辑里的强制 gate
- 如果 `all_three` 稳定输给 `ce_only`，应解释为当前三头配比失败
- 只有 `all_three` 稳定赢过 `drop_*` 和 `ce_only`，才说明当前三头全开成立

## calibration 输出如何被读取

- `protocol_decide / winner_refine` 默认读取 `triple_combo_factor`
- `drop_rank` 读取 `opp_danger_combo_factor`
- `drop_opp` 读取 `rank_danger_combo_factor`
- `drop_danger` 读取 `rank_opp_combo_factor`
- 历史 `joint_combo_factor` 只作为 `opp_danger_combo_factor` 的兼容 alias

## 不该怎么用

- 不要用 `calibration` 单独宣布“哪一个头赢了”
- 不要把各类 `acc` 重新塞回排序键
- 不要把旧 `solo / pairwise / joint refine` 结构重新当当前主线
- 不要把当前 `winner_refine` 误读成任意上下文的自动 `top-k center`
