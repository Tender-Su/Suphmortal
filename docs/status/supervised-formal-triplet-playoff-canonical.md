# 监督学习阶段 Formal Triplet Playoff 口径

这份文档只回答当前 `winner_refine` 收口之后，`3 candidate formal playoff` 如何把 pre-formal 第一梯队收口成当前固定 winner / 第一替补。

## 适用范围

- 当前来源 run：
  - `logs/stage05_fidelity/s05_fidelity_p1_top3_cali_slim_20260329_001413/`
- 当前只适用于：
  - `A2x winner_refine` 双 seed 已收口后的下游 `formal_train / formal_1v3`
- 这不是新的 `P1` selector
- 当前官方 `P1 winner` 已由这轮 playoff 固定为 `anchor*1.0`
- `P1` 统一评估口径看：
  - `docs/status/p1-selection-canonical.md`

## 命名口径

- center 只写：
  - `anchor / opp_lean / rank_lean / danger_lean`
- 全头统一缩放只写：
  - `*0.85 / *1.0 / *1.15`
- center 内部再分配只写：
  - `rank+/rank++/opp-/danger++` 这类结构符号
- 当前 canonical 文档不再手写原始 `W_r..._o..._d...`

## 当前 triplet

当前送进 `formal_train` 的 3 个候选固定为：

1. `opp_lean*0.85`
2. `anchor*1.0`
3. `opp_lean(rank--/danger++)`

## 2026-04-05 formal 结果（1.5x）

- 当前 coordinator run：
  - `logs/stage05_fidelity/s05_formal_triplet_20260405/`
- 当前 `3 / 3 child formal` 已全部完成
- 当前 `3 / 3 child formal` 的 `offline_checkpoint_winner` 都是：
  - `best_loss`
- 当前按主线现有 `LOSS_EPSILON + selection_tiebreak_key` 规则重排后，cross-run offline 顺序为：
  1. `opp_lean*0.85`
  2. `opp_lean(rank--/danger++)`
  3. `anchor*1.0`

当前三条结果可这样理解：

- `opp_lean*0.85`
  - `best_full_recent_loss = 0.480049`
  - `macro_action_acc = 0.847933`
  - 当前是 offline front-runner
- `opp_lean(rank--/danger++)`
  - `best_full_recent_loss = 0.480872`
  - 当前总 loss 与 front-runner 很接近
  - 当前 `opp / danger` 头最强，是最值得保留的 hedge challenger
- `anchor*1.0`
  - `best_full_recent_loss = 0.480868`
  - `rank_acc = 0.450905`
  - 当前 `rank_acc` 最强

## 2026-04-05 cross-run formal_1v3 结果

- 当前 playoff run：
  - `logs/stage05_fidelity/s05_formal_triplet_20260405_winner_playoff_1v3/`
- 当前正式判据：
  - `avg_pt` 为主
  - `avg_rank` 为辅
  - 当前 active 位次分：`90 / 45 / 0 / -135`
- 当前最终顺序：
  1. `anchor*1.0`
  2. `opp_lean*0.85`
  3. `opp_lean(rank--/danger++)`
- 当前固定结论：
  - 当前官方 `P1 winner = anchor*1.0`
  - 当前第一替补 = `opp_lean*0.85`
  - `opp_lean(rank--/danger++)` 不再作为当前保留候选

## 单个 formal 的本机耗时实测

- 当前 `formal` 基础 `phase_steps = 9000 / 6000 / 3000`
- dispatch 固定：
  - `formal_step_scale = 5.0`
- 当前有效训练长度：
  - `phase_a / phase_b / phase_c = 45000 / 30000 / 15000`
- `2026-04-05` triplet 实测 wall-clock：
  - 台式机单条约 `4.5 h`
  - 笔记本单条约 `11.2 h`

## 当前已经写死的 downstream 结论

- 当前监督学习 winner：
  - `anchor*1.0`
- 当前第一替补：
  - `opp_lean*0.85`
- 当前监督学习阶段不再继续 Oracle 路线
- 当前默认下一步不是重开 triplet，而是等待强化学习阶段方案收口
