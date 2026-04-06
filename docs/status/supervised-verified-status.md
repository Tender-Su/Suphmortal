# 监督学习阶段当前核对状态

这份文档只放人工确认后的当前真实结论。自动摘要如果和这里冲突，以这里为准。

## 当前结论

- 核对日期：`2026-04-06`
- 项目阶段位置：
  - 监督学习阶段完成
  - 强化学习阶段方案定义进行中
- 当前活跃主线 run：
  - `logs/sl_fidelity/sl_fidelity_p1_top3_cali_slim_20260329_001413/`
- 当前 downstream coordinator run：
  - `logs/sl_fidelity/sl_formal_triplet_20260405/`
- 当前 downstream playoff run：
  - `logs/sl_fidelity/sl_formal_triplet_20260405_winner_playoff_1v3/`
- 当前已验证协议 winner：
  - `C_A2x_cosine_broad_to_recent_strong_24m_12m`
- 当前已验证 winner 点位：
  - `0.12 + A2x`
- 当前正式 supervised winner：
  - `anchor*1.0`
- 当前第一替补：
  - `opp_lean*0.85`
- 当前 canonical supervised checkpoint：
  - `./checkpoints/sl_canonical.pth`

## 已冻结事实

- `P0` 官方 `top3` 顺序：
  - `C_A2y_cosine_broad_to_recent_strong_12m_6m`
  - `C_A2x_cosine_broad_to_recent_strong_24m_12m`
  - `C_A1x_cosine_broad_to_recent_mild_24m_12m`
- 三类辅助头内部 shape：
  - `rank = 18K_ROUND_ONLY`
  - `opp = HYBRID_GRAD`
  - `danger = 18K_STAT`
- `P1` 主线：
  - `calibration -> protocol_decide -> winner_refine`
- `P1 ablation`：
  - 手动诊断轮
- triplet child formal：
  - `3 / 3 completed`
  - `3 / 3 offline_checkpoint_winner = best_loss`
- triplet cross-run `formal_1v3` 最终顺序：
  1. `anchor*1.0`
  2. `opp_lean*0.85`
  3. `opp_lean(rank--/danger++)`

## 证据入口

- `P1` 唯一有效评估口径：
  - `docs/status/p1-selection-canonical.md`
- formal triplet / `formal_1v3` 证据：
  - `docs/status/supervised-formal-triplet-playoff-canonical.md`
- 自动生成的 run snapshot：
  - `docs/status/supervised-fidelity-results.md`
