# Stage 0.5 当前核对状态

这份文档只放“现在真实默认到哪里了”。它由人工维护，不让自动流程覆盖。

## 当前结论

- 核对日期：`2026-03-30`
- 当前主线阶段：`P1 protocol_decide` 已完成
- 当前活跃 run：
  - `logs/stage05_fidelity/s05_fidelity_p1_top3_cali_slim_20260329_001413/`
- 当前状态字段：
  - `stopped_after_p1_protocol_decide`
- 当前已验证协议 winner：
  - `C_A2x_cosine_broad_to_recent_strong_24m_12m`
- 当前下一步只应进入：
  - `A2x winner_refine`

## 当前已冻结的事实

- `P0` 官方 `top3` 顺序：
  - `C_A2y_cosine_broad_to_recent_strong_12m_6m`
  - `C_A2x_cosine_broad_to_recent_strong_24m_12m`
  - `C_A1x_cosine_broad_to_recent_mild_24m_12m`
- 三类辅助头内部 shape 已冻结：
  - `rank = 18K_ROUND_ONLY`
  - `opp = HYBRID_GRAD`
  - `danger = 18K_STAT`
- `P1` 当前唯一主线结构：
  - `calibration -> protocol_decide -> winner_refine -> ablation`
- `P1` 当前唯一有效选模口径：
  - `docs/status/p1-selection-canonical.md`

## 当前 P1 默认

### calibration

- 当前瘦版 `calibration` 代表协议固定为 `A2y`
- 它只用于 budget mapping 和 combo factor 定标
- 它不代表当前下游 protocol winner

### protocol_decide

- 当前默认 seed2 扩展规则固定为：
  - `ambiguity_mode = flip_or_gap`
  - `gap_threshold = 0.001`
- 历史旧 `ambig` 规则只允许作为旧 run 兼容字段读取

### winner_refine

- 当前默认只在 `A2x` 内部继续
- 当前默认不是自动 `top-k center`
- 当前冻结三中心：
  - `C_A2x_cosine_broad_to_recent_strong_24m_12m__B_r0046_o0037_d0037`
  - `C_A2x_cosine_broad_to_recent_strong_24m_12m__B_r0034_o0014_d0041`
  - `C_A2x_cosine_broad_to_recent_strong_24m_12m__B_r0052_o0025_d0043`
- 当前局部搜索规则：
  - `total_scale_factors = [0.85, 1.0, 1.15]`
  - `transfer_delta = 0.01`
  - `step_scale = 1.5`

## 当前这轮已经确认的结果

- 当前 `protocol_decide` 已收口
- 当前结果是 `27 / 27` 全部有效
- 先前唯一失败臂 `A2x danger_lean 0.12` 已补跑成功
- 补跑前它被压到总榜第 `27`
- 补跑后它回到总榜第 `6`
- 协议 winner 没有变化，仍然是 `A2x`

## 当前停点

- 当前停在 `protocol_decide` 收口点
- 不自动进入 `winner_refine`
- 用户确认方向后，只在 `A2x` 内部继续下一阶段

## 怎么用这份文档

- 问“项目现在真实停在哪一步”，看这里
- 问“P1 应该按什么口径解释 winner”，看 `p1-selection-canonical.md`
- 问“某个 run 当时写出了什么摘要”，看 `stage05-fidelity-results.md`
- 问“历史旧流程当时怎么跑”，去 `docs/archive/`
