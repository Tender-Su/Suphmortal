# Stage 0.5 当前核对状态

这份文档只放“现在真实默认到哪里了”。它由人工维护，不让自动流程覆盖。

## 当前结论

- 核对日期：`2026-04-04`
- 当前主线阶段：`P1 winner_refine` 进行中
- 当前活跃主线 run：
  - `logs/stage05_fidelity/s05_fidelity_p1_top3_cali_slim_20260329_001413/`
- 当前状态字段：
  - `running_p1_winner_refine`
- 当前已验证协议 winner：
  - `C_A2x_cosine_broad_to_recent_strong_24m_12m`
- 当前已验证 winner 点位：
  - `0.12 + A2x`
- 当前下一步只应进入：
  - `完成 A2x winner_refine`

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
- `P1` 当前唯一有效选模口径：
  - `docs/status/p1-selection-canonical.md`

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

- `2026-04-04` 已确认：
  - `seed1 = 36 / 36 completed`
  - `seed2 = 6 completed / 2 running / 4 pending`
- 当前仍只在 `A2x` 内部继续
- 不恢复 `0.18` probe 的 `seed2`
- 不复用任何旧预算口径留下的 `winner_refine` 残留
- `winner_refine` 收口后，默认直接把 front runner 当作当前 `P1 winner`
- `ablation` 仅在需要额外边际贡献确认时，作为 backlog 手动补跑
