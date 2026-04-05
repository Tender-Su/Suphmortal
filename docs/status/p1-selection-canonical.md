# P1 统一评估口径

这份文档是当前项目里 `P1` 的唯一有效评估规范。  
任何脚本、自动摘要、人工总结或无上下文新 agent，只要要回答 `P1` 的排序、第一梯队、或 downstream 入口，都以这里为准。

当前额外约束：

- `2026-04-05` 当前官方 `P1 winner` 已固定为 `anchor*1.0`
- 当前第一替补已固定为 `opp_lean*0.85`
- `winner_refine` 仍只负责 pre-formal 第一梯队；最终 winner 由 `formal triplet playoff -> formal_1v3` 决定

## 适用范围

- `A2y internal-shape micro AB`
- `p1_protocol_decide_round`
- `p1_winner_refine_round`
- `p1_ablation_round`

历史 `p1_solo_round / p1_pairwise_round / p1_joint_refine_round` 仍可作诊断证据，但已经不是当前主线结构。  
`p1_ablation_round` 仍使用这套口径，但它现在是 `backlog / manual only`，不是默认主线阶段。

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
  - 当前只负责给出 pre-formal 第一梯队与内部排序证据
  - 当前不单独产出官方 `P1 winner`；官方 winner 已由下游 `formal triplet playoff / formal_1v3` 收口
- `p1_ablation_round`
  - 负责比较 `all_three / drop_* / ce_only` 的边际贡献
  - 当前只作为手动 backlog 诊断，不再阻塞 downstream

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
- 当前四个 center 用结构别名写为：
  - `anchor`
  - `opp_lean`
  - `rank_lean`
  - `danger_lean`
- 当前局部扰动规则：
  - `total_scale_factors = [0.85, 1.0, 1.15]`
  - `transfer_delta = 0.01`
  - `step_scale = 1.5`
- 文档里的结构别名约定：
  - center 只写 `anchor / rank_lean / opp_lean / danger_lean`
  - 全头统一缩放写成 `*0.85 / *1.0 / *1.15`
  - center 内部再分配写成 `rank+/rank++/opp-/danger++` 这类局部符号
  - 原始 `W_r..._o..._d...` 只保留在 run artifact，不再写进当前 canonical 文档

## winner_refine 与 manual formal playoff 的边界

- `winner_refine` 当前只负责：
  - 给出 pre-formal 第一梯队
  - 保留内部排序证据
  - 当前历史内部 `top1` 是 `opp_lean*0.85`
- 当前新增的 `formal triplet playoff` 不会回写 `winner_refine` 内部排序
- 它只负责：
  - 给 downstream `formal_train / formal_1v3` 选择第一梯队候选集
  - 在 triplet child run 之间继续比较谁值得保留
  - 当前已固定官方 `P1 winner = anchor*1.0`
  - 当前已固定第一替补 `opp_lean*0.85`
- 当前固定 triplet 用结构别名写为：
  - `opp_lean*0.85`
  - `anchor*1.0`
  - `opp_lean(rank--/danger++)`
- 当前正式判据固定为：
  - `formal_1v3` 里 `avg_pt` 为主、`avg_rank` 为辅
  - 当前 active 位次分是 `90 / 45 / 0 / -135`
- `3210` 只保留为诊断对照，不覆盖当前 winner 决议
- 当前 triplet 的固定口径看：
  - `docs/status/supervised-formal-triplet-playoff-canonical.md`

## ce_only 的解释

- `ce_only` 是主线里的诊断锚点
- 它不再是旧 `family survivor` 逻辑里的强制 gate
- 如果 `all_three` 稳定输给 `ce_only`，应解释为当前三头配比失败
- 当前默认主线不会要求先跑 `ablation` 才继续 downstream
- 只有在手动执行 backlog `ablation` 时，才用 `all_three / drop_* / ce_only` 去补做边际贡献确认

## calibration 输出如何被读取

- `protocol_decide / winner_refine` 默认读取 `triple_combo_factor`
- `drop_rank` 在手动 backlog `ablation` 中读取 `opp_danger_combo_factor`
- `drop_opp` 在手动 backlog `ablation` 中读取 `rank_danger_combo_factor`
- `drop_danger` 在手动 backlog `ablation` 中读取 `rank_opp_combo_factor`
- 历史 `joint_combo_factor` 只作为 `opp_danger_combo_factor` 的兼容 alias

## 不该怎么用

- 不要用 `calibration` 单独宣布“哪一个头赢了”
- 不要把各类 `acc` 重新塞回排序键
- 不要把旧 `solo / pairwise / joint refine` 结构重新当当前主线
- 不要把当前 `winner_refine` 误读成任意上下文的自动 `top-k center`
- 不要把 `winner_refine` 当前内部 `top1` 直接写成官方 `P1 winner`
- 不要用 `3210` 的诊断口径覆盖当前正式 `formal_1v3` 决议
- 不要在当前 canonical 文档里硬编码原始 `W_r..._o..._d...` 名字
