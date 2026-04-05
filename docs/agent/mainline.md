# 当前主线与冻结默认

这份文档只保留仍然有效、会直接影响后续判断的稳定默认。  
实时停点看 `docs/agent/current-plan.md`。  
当前核对状态看 `docs/status/supervised-verified-status.md`。  
运行顺序看 `docs/agent/experiment-workflow.md`。  
当前 triplet playoff 口径看 `docs/status/supervised-formal-triplet-playoff-canonical.md`。

## 总目标

- 目标是最强麻将 AI，不是最省时间的麻将 AI
- 强度差异明确时，优先更强方案
- 只有强度接近时，才让效率和工程成本参与取舍

## 当前项目结构

- 项目当前只有两大阶段：
  - `监督学习阶段`
  - `强化学习阶段`
- `GRP` 仍然保留，但它现在视为监督学习阶段的前置辅助模型，不单独占一个项目阶段名
- 监督学习阶段内部仍保留：
  - `P0`
  - `P1`

## 代码与机器默认

- 当前开发分支：`main`
- 当前源码真源：台式机 `main` 工作树
- Git 同步入口：`docs/agent/code-sync.md`
- 主节点：`i5-13600KF + RTX 5070 Ti`
- 副节点：`i9-13900HX + RTX 4060 Laptop GPU 8GB + 32GB DDR5`

### 监督学习 loader 默认

- 台式机：
  - train：`4 / 10 / 3`
  - val：`8 / 5`
- 笔记本：
  - train：`4 / 10 / 4`
  - val：`7 / 5`

### 1v3 默认

- 台式机 `RTX 5070 Ti`：
  - `seed_count = 1024`
  - `shard_count = 4`
- 笔记本 `RTX 4060 Laptop GPU`：
  - `seed_count = 640`
  - `shard_count = 3`

## GRP 默认

- 默认主线：`384x3 fp32`
- 默认下游 checkpoint：`best_loss`
- `best_acc` 只作受控对照
- `latest` 只用于续训

## 监督学习阶段当前冻结默认

- 当前内部结构固定为：
  - `P0 -> P1 -> formal_train -> formal_1v3`
- `formal_train` 不再直接做 canonical checkpoint 落位
- `formal_train` 只保留：
  - `best_loss`
  - `best_acc`
  - `best_rank`
- `latest` 在进入 `formal_1v3` 前直接丢弃
- 当前正式发布结构固定为：
  - `formal_train -> checkpoint pack(best_loss / best_acc / best_rank) -> formal_1v3 -> canonical alias落位`
- 当前 `formal` 长度已经上调到旧口径的 `1.5x`
  - 基础 `phase_steps = 9000 / 6000 / 3000`
  - 默认 `formal_step_scale = 5.0`
  - 当前有效 `phase_a / phase_b / phase_c = 45000 / 30000 / 15000`
- 当前监督学习 canonical checkpoint 仍沿用历史文件名：
  - `./checkpoints/stage0_5_supervised.pth`

## P0 / P1 当前冻结默认

- `P0` 事实 `top3` 顺序固定为：
  - `C_A2y_cosine_broad_to_recent_strong_12m_6m`
  - `C_A2x_cosine_broad_to_recent_strong_24m_12m`
  - `C_A1x_cosine_broad_to_recent_mild_24m_12m`
- `P1` 当前唯一主线：
  - `calibration -> protocol_decide -> winner_refine`
- `P1 ablation` 当前定位：
  - `backlog / manual only`
  - 不作为监督学习阶段的默认 gate
- 当前 downstream 协议 winner：
  - `C_A2x_cosine_broad_to_recent_strong_24m_12m`
- 当前人工确认的 winner 点位：
  - `0.12 + A2x`
- 当前正式 winner：
  - `anchor*1.0`
- 当前第一替补：
  - `opp_lean*0.85`
- 当前瘦版 `calibration` 固定为：
  - `A2y-only + combo_only`
- 当前三类辅助头内部 shape 已冻结：
  - `rank = 18K_ROUND_ONLY`
  - `opp = HYBRID_GRAD`
  - `danger = 18K_STAT`

## P1 选模与搜索默认

- `P1` 排序与第一梯队解释只看：
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

### winner_refine 默认

- 当前只允许在 `A2x` 协议内部继续
- 当前冻结规则是：
  - `center_mode = top_ranked_keep`
  - `center_keep = 4`
- 当前四个 center：
  - `anchor`
  - `opp_lean`
  - `rank_lean`
  - `danger_lean`
- 当前局部细搜规则：
  - `total_scale_factors = [0.85, 1.0, 1.15]`
  - `transfer_delta = 0.01`
  - `step_scale = 1.5`
- 当前 `winner_refine` 只保留 pre-formal 第一梯队与内部排序证据
- 当前历史内部 `top1`：
  - `opp_lean*0.85`
- 当前正式发布的监督学习 winner：
  - `anchor*1.0`

## triplet playoff 当前冻结默认

- 当前固定 triplet：
  - `opp_lean*0.85`
  - `anchor*1.0`
  - `opp_lean(rank--/danger++)`
- 当前 triplet 解释边界看：
  - `docs/status/supervised-formal-triplet-playoff-canonical.md`
- 当前已经写死的结论是：
  - 当前 winner：`anchor*1.0`
  - 当前第一替补：`opp_lean*0.85`

## 强化学习阶段当前默认

- 当前只确定一条原则：
  - RL 的默认起点应是当前 canonical supervised winner
- 当前默认接力方式：
  - 先读 `[control].state_file = ./checkpoints/mortal.pth` 作为 RL 自身续跑文件
  - 若它不存在，再回退到 `[online].init_state_file = ./checkpoints/stage0_5_supervised.pth`
- 当前监督学习阶段不再继续 Oracle 路线
- RL 里的 Oracle 方案尚未敲定
- 所以当前没有新的 RL canonical workflow 写死在默认文档里
