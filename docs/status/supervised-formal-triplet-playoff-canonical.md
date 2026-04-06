# 监督学习阶段 Formal Triplet Playoff 口径

这份文档定义 pre-formal 第一梯队如何通过 formal triplet 与 cross-run `formal_1v3` 产生当前官方 winner。

## 适用范围

- source run：
  - `logs/sl_fidelity/sl_fidelity_p1_top3_cali_slim_20260329_001413/`
- downstream coordinator run：
  - `logs/sl_fidelity/sl_formal_triplet_20260405/`
- downstream playoff run：
  - `logs/sl_fidelity/sl_formal_triplet_20260405_winner_playoff_1v3/`

## 命名口径

- center：
  - `anchor / opp_lean / rank_lean / danger_lean`
- 全头统一缩放：
  - `*0.85 / *1.0 / *1.15`
- center 内部再分配：
  - `rank+/rank++/opp-/danger++`
- 文档统一使用结构别名

## 当前 triplet

当前送入 `formal_train` 的 `3` 个候选：

1. `opp_lean*0.85`
2. `anchor*1.0`
3. `opp_lean(rank--/danger++)`

## formal child run 结果

- `3 / 3 child formal` 已全部完成
- `3 / 3 child formal` 的 `offline_checkpoint_winner` 都是：
  - `best_loss`
- cross-run offline 顺序：
  1. `opp_lean*0.85`
  2. `opp_lean(rank--/danger++)`
  3. `anchor*1.0`

当前离线解释：

- `opp_lean*0.85`
  - `best_full_recent_loss = 0.480049`
  - offline front-runner
- `opp_lean(rank--/danger++)`
  - `best_full_recent_loss = 0.480872`
  - hedge challenger
- `anchor*1.0`
  - `best_full_recent_loss = 0.480868`
  - `rank_acc` 最强

## cross-run `formal_1v3` 结果

- 判据：
  - `avg_pt` 为主
  - `avg_rank` 为辅
  - 位次分：`90 / 45 / 0 / -135`
- 最终顺序：
  1. `anchor*1.0`
  2. `opp_lean*0.85`
  3. `opp_lean(rank--/danger++)`

## 当前官方结论

- supervised winner：
  - `anchor*1.0`
- 第一替补：
  - `opp_lean*0.85`

## 运行长度与耗时

- formal 长度：
  - `phase_a / phase_b / phase_c = 45000 / 30000 / 15000`
- `2026-04-05` triplet 实测 wall-clock：
  - 台式机单条约 `4.5 h`
  - 笔记本单条约 `11.2 h`
