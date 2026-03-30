# 当前默认续工作入口

这份文档只回答三件事：当前默认主线是什么、现在真实停在哪一步、下一步应该怎么做。

## 建议读取顺序

1. `AGENTS.md`
2. 本文档
3. `docs/status/stage05-verified-status.md`
4. `docs/status/p1-selection-canonical.md`
5. `docs/agent/mainline.md`

## 当前默认

- 总目标：优先做出最强麻将 AI；只有强度接近时才考虑效率。
- 当前代码主分支固定为：`main`
- 双机代码同步默认入口：`docs/agent/code-sync.md`
- `Stage 0.5 / P0` 的事实 `top3` 顺序固定为：
  - `C_A2y_cosine_broad_to_recent_strong_12m_6m`
  - `C_A2x_cosine_broad_to_recent_strong_24m_12m`
  - `C_A1x_cosine_broad_to_recent_mild_24m_12m`
- 当前 `P1` 唯一主线固定为：`calibration -> protocol_decide -> winner_refine -> ablation`
- 当前已验证 `protocol_decide` winner，也是当前下游默认协议：
  - `C_A2x_cosine_broad_to_recent_strong_24m_12m`
- 当前 `protocol_decide` 的默认 seed2 扩展规则固定为：
  - `ambiguity_mode = flip_or_gap`
  - `gap_threshold = 0.001`
- 当前 `winner_refine` 默认不是“自动取聚合榜前 k 名”，而是冻结三中心：
  - `C_A2x_cosine_broad_to_recent_strong_24m_12m__B_r0046_o0037_d0037`
  - `C_A2x_cosine_broad_to_recent_strong_24m_12m__B_r0034_o0014_d0041`
  - `C_A2x_cosine_broad_to_recent_strong_24m_12m__B_r0052_o0025_d0043`
- 当前 `winner_refine` 的局部扰动规则固定为：
  - `total_scale_factors = [0.85, 1.0, 1.15]`
  - `transfer_delta = 0.01`
- 当前 `winner_refine` 的训练长度固定为：
  - `step_scale = 1.5`
  - 单 arm 约等于 `phase_a / phase_b / phase_c = 9000 / 6000 / 3000`

## 当前真实停点

- 当前主线 run：
  - `logs/stage05_fidelity/s05_fidelity_p1_top3_cali_slim_20260329_001413/`
- 当前状态：
  - `stopped_after_p1_protocol_decide`
- 当前这轮 `protocol_decide` 已经收口：
  - 双 seed 已完成
  - 唯一失败 arm 已补跑成功
  - 当前结果是 `27 / 27` 全部有效
  - winner 仍然是 `A2x`
- 当前原则：停在 `protocol_decide` 收口点，不自动进入 `winner_refine`

## 下一步

- 如果继续主线，只允许在 `A2x` 协议内部启动 `winner_refine`
- 直接使用上面的冻结三中心，不再用自动 `top-k center`
- `winner_refine` 跑完后再次停下，人工确认方向，再决定是否进入 `ablation`

## 不该怎么做

- 不要把 `docs/status/stage05-fidelity-results.md` 当成当前默认入口；它只是 run snapshot
- 不要把 `winner_refine` 理解成“自动取协议内前 k 名”
- 不要再把历史旧 `ambig` 规则当成当前 `protocol_decide` 的默认 seed2 规则
- 不要把旧 `solo / pairwise / joint refine` 结构重新当作当前主线
- 不要在没有人工确认的情况下把多个阶段一键串到底
