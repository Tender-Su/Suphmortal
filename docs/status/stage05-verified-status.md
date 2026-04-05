# Stage 0.5 当前核对状态

这份文档只放“现在真实默认到哪里了”。它由人工维护，不让自动流程覆盖。

## 当前结论

- 核对日期：`2026-04-05`
- 当前主线阶段：`manual formal triplet + cross-run formal_1v3` 已收口
- 当前活跃主线 run：
  - `logs/stage05_fidelity/s05_fidelity_p1_top3_cali_slim_20260329_001413/`
- 当前 downstream coordinator run：
  - `logs/stage05_fidelity/s05_formal_triplet_20260405/`
- 当前 downstream playoff run：
  - `logs/stage05_fidelity/s05_formal_triplet_20260405_winner_playoff_1v3/`
- 当前已验证协议 winner：
  - `C_A2x_cosine_broad_to_recent_strong_24m_12m`
- 当前已验证 winner 点位：
  - `0.12 + A2x`
- 当前正式 `P1 winner`：
  - `anchor*1.0`
- 当前第一替补：
  - `opp_lean*0.85`
- 当前历史 pre-formal `top1`：
  - `opp_lean*0.85`
- 当前 triplet formal 已完成：
  - `3 / 3 child formal completed`
  - `3 / 3 child formal` 的 `offline_checkpoint_winner = best_loss`
- 当前 triplet cross-run `formal_1v3` 已完成
- 当前 triplet cross-run `formal_1v3` 顺序：
  1. `anchor*1.0`
  2. `opp_lean*0.85`
  3. `opp_lean(rank--/danger++)`
- 当前 triplet formal 的 cross-run offline 排序：
  1. `opp_lean*0.85`
  2. `opp_lean(rank--/danger++)`
  3. `anchor*1.0`
- 当前下一步只应进入：
  - 以 `anchor*1.0` 作为 downstream 默认 winner 继续
  - 保留 `opp_lean*0.85` 作为第一替补与争 `1` 型预备队

## 当前主线已经确认的事实

- `P0` 官方 `top3` 顺序：
  - `C_A2y_cosine_broad_to_recent_strong_12m_6m`
  - `C_A2x_cosine_broad_to_recent_strong_24m_12m`
  - `C_A1x_cosine_broad_to_recent_mild_24m_12m`
- 三类辅助头内部 shape 已冻结：
  - `rank = 18K_ROUND_ONLY`
  - `opp = HYBRID_GRAD`
  - `danger = 18K_STAT`
- `P1` 当前唯一主线结构：
  - `calibration -> protocol_decide -> winner_refine`
- `P1 ablation` 当前定位：
  - `backlog / manual only`
  - 不作为 downstream 默认 gate
- 当前 `winner_refine` 之后的固定下游结构：
  - `formal_train -> checkpoint pack(best_loss / best_acc / best_rank) -> formal_1v3 -> canonical alias落位`
- `P1` 当前唯一有效评估口径：
  - `docs/status/p1-selection-canonical.md`
- 当前 downstream manual `formal` 口径：
  - `docs/status/formal-triplet-playoff-canonical.md`

## 当前 protocol_decide 已收口

- 双 seed 已完成
- 当前结果为 `27 / 27` 全部有效
- 先前唯一失败臂 `A2x danger_lean 0.12` 已补跑成功
- 补跑前它被压到总榜第 `27`
- 补跑后它回到总榜第 `6`
- 协议 winner 没有变化，仍然是 `A2x`
- 当前被人工确认的 winner 点位仍然是 `0.12 + A2x`

## 启动前清理状态

- 当前主线 run 的启动前清理已完成
- 没有旧 `winner_refine` 产物残留需要恢复或复用
- 已删除退役的旧 `ambig` 状态字段：
  - `state.json`
  - `p1_protocol_decide_round.json`
  - `distributed/protocol_decide_dispatch/dispatch_state.json`
- 当前仍保留的有效 seed2 规则是：
  - `flip_or_gap @ 0.001`

## 2026-04-02 额外 probe 结论

- 额外 probe run：
  - `logs/stage05_fidelity/s05_fidelity_p1_a2x_budget018_probe3_20260402_195712/`
- 这轮只测了 `A2x @ 0.18` 的三臂：
  - `anchor_18`
  - `rank_lean_18`
  - `danger_lean_18`
- 这轮是 `seed1-only negative probe`
- 它不改变默认网格
- 它不改变当前主线 winner
- 它不恢复 `seed2`

## 当前 winner_refine 状态

- `2026-04-05` 已确认：
  - `seed1 = 36 / 36 completed`
  - `seed2 = 12 / 12 completed`
- 当前仍只在 `A2x` 内部继续
- 当前四个 center 用结构别名写为：
  - `anchor`
  - `opp_lean`
  - `rank_lean`
  - `danger_lean`
- 当前 pre-formal 第一梯队固定为：
  - `opp_lean*0.85`
  - `anchor*1.0`
  - `opp_lean(rank--/danger++)`
- 第一梯队外但仍重要的近邻挑战者：
  - `rank_lean*0.85`
  - `opp_lean*1.15`
- `danger_lean` family 当前没有 arm 进入第一梯队
- 不恢复 `0.18` probe 的 `seed2`
- 不复用任何旧预算口径留下的 `winner_refine` 残留
- 当前 downstream 不直接只送单一 front runner 进 formal，而是先进入：
  - `manual formal triplet playoff`
- `2026-04-05` 当前已确认这轮 triplet formal 的解释边界：
  - `opp_lean*0.85`
    - 当前 offline front-runner
    - `best_full_recent_loss = 0.480049`
    - `macro_action_acc = 0.847933`
  - `opp_lean(rank--/danger++)`
    - 当前最接近 front-runner 的 hedge challenger
    - `best_full_recent_loss = 0.480872`
    - 当前 `opp / danger` 头最强
  - `anchor*1.0`
    - 当前 `rank_acc` 最强的平衡基准
    - `best_full_recent_loss = 0.480868`
    - 但在现有 selection policy 的 cross-run offline 排序中列第 `3`
- `2026-04-05` 当前已确认这轮 triplet playoff 的正式收口结论：
  - `anchor*1.0`
    - 当前 `formal_1v3` 正式 winner
    - 当前按 `avg_pt` 为主、`avg_rank` 为辅的口径排第 `1`
    - 当前更像“少四位”的下游正式 winner
  - `opp_lean*0.85`
    - 当前 `winner_refine` 历史内部 `top1`
    - 当前 offline front-runner
    - 现在固定为第一替补与争 `1` 型预备队
  - `opp_lean(rank--/danger++)`
    - 当前 final playoff 第 `3`
    - 不再作为当前保留候选
- `ablation` 仅在需要额外边际贡献确认时，作为 backlog 手动补跑
- 原始 `W_r..._o..._d...` 名字只留在 run artifact，不再写进当前 canonical 文档
