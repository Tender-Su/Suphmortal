# P1 统一选模口径

这份文档是当前项目里 `P1` 的唯一有效选模规范。任何脚本、自动摘要、人工总结或无上下文新 agent，只要要回答 `P1` winner 相关问题，都以这里为准。

## 适用范围

- `A2y internal-shape micro AB`
- `p1_protocol_decide_round`
- `p1_winner_refine_round`
- `p1_ablation_round`

历史 `p1_solo_round / p1_pairwise_round / p1_joint_refine_round` 仍可作为诊断证据，但已经不是当前主线实验结构。

## 核心规则

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
- `full_recent_loss` 在 `P1` 里只保留为总 loss / aux tax 诊断字段
- 自动摘要里的 `cmp_policy` 就是 `recent_policy_loss`
- 自动摘要里的 `full_loss(diag)` 就是 `full_recent_loss`

## 各轮职责

- `p1_calibration`
  - 负责 budget mapping 和组合耦合定标
  - 不能单独用来宣布 winner
- `p1_protocol_decide_round`
  - 负责在统一三头脚手架下尽早选出协议 winner
- `p1_winner_refine_round`
  - 负责只在 winner 协议内部细调三头全开配比
- `p1_ablation_round`
  - 负责比较 `all_three / drop_* / ce_only` 的边际贡献

## 当前默认的 `protocol_decide` 网格

- `total_budget_ratios = [0.09, 0.12]`
- `mixes = anchor / rank_lean / opp_lean / danger_lean`
- `opp_lean` 使用对称版 `0.38 / 0.31 / 0.31`

## `ce_only` 现在是什么意思

- `ce_only` 是主线里的诊断锚点，不再是旧 `family survivor` 逻辑里的强制 gate
- 如果 `all_three` 稳定输给 `ce_only`，应解释为当前三头配比失败
- 只有 `all_three` 稳定赢过 `drop_*` 和 `ce_only`，才能说明当前三头全开真的成立

## `calibration` 可以产出什么

- `opp_weight_per_budget_unit`
- `danger_weight_per_budget_unit`
- `rank_opp_combo_factor`
- `rank_danger_combo_factor`
- `opp_danger_combo_factor`
- `triple_combo_factor`

其中：

- `protocol_decide / winner_refine` 默认读取 `triple_combo_factor`
- `drop_rank` 读取 `opp_danger_combo_factor`
- `drop_opp` 读取 `rank_danger_combo_factor`
- `drop_danger` 读取 `rank_opp_combo_factor`
- 历史 `joint_combo_factor` 只作为 `opp_danger_combo_factor` 的兼容 alias

## 不该怎么用

- 不要用 `calibration` 单独宣布“哪一个头赢了”
- 不要把各类 `acc` 重新塞回排序键
- 不要把旧 `solo / pairwise / joint refine` 结构重新当当前主线

## 代码锚点

- `mortal/run_stage05_fidelity.py`
- `mortal/run_stage05_p1_only.py`
- `mortal/stage05_selection.py`
