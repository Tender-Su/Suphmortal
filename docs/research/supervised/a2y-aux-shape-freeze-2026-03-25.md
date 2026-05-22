# 2026-03-25 A2y 辅助头内部 shape 冻结备忘

> 原始记录用于冻结 `A2y` 线的辅助头内部 shape。  
> 当前监督学习主线以 `docs/status/supervised-mainline.md` 为准；本文只保留冻结结论和关键证据。

## 这次实验回答的问题

不是重新判断 `rank / opponent / danger` 是否进入主线，而是回答：

- 如果某个 family 继续进入后续 `P1 solo / pairwise / joint refine`，
- 它的内部 shape 应该用哪个默认，
- 从而后续只需要扫 family total weight。

权威 run：

- `logs/sl_fidelity/sl_fidelity_a2y_internal_mix_micro1s_20260324_224456/`
- corrected summary：`final_summary_policy_corrected.json`

辅助输入：

- `logs/aux_heuristic_audit_18k.md`
- `logs/aux_subhead_gradient_audit.md`

## 关键 metric 修正

第一版解读过度看了 `full_recent_loss`，这是错误的。

`P1` auxiliary 比较应使用：

- `comparison_recent_loss = recent_policy_loss`
- 然后按 `selection_tiebreak_key`
- 即 `selection_quality_score -> -recent_policy_loss -> -old_regression_policy_loss`

`full_recent_loss` 包含 auxiliary tax，只能做诊断，不能直接当 winner metric。

## micro-AB 设置

- protocol arm：`C_A2y_cosine_broad_to_recent_strong_12m_6m`
- seed：`20260312`
- reduced budget：`step_scale = 0.20`
- 共享 `CE-only` baseline
- 三个小 round：
  - `opp_internal_mix_round`
  - `rank_shape_round`
  - `danger_internal_mix_round`

这只是 shape pre-filter，不是完整 `P1 solo`。

## 冻结结果

| family | 冻结 shape | 参数 |
| --- | --- | --- |
| `rank` | `18K_ROUND_ONLY` | `south=1.59`, `all_last=1.617`, `gap=4000`, `bonus=0.0` |
| `opp` | `HYBRID_GRAD` | `shanten=0.8506568408`, `tenpai=1.1493431592` |
| `danger` | `18K_STAT` | `any=0.0904217947`, `value=0.8180402859`, `player=0.0915379194` |

## 关键证据

CE-only baseline：

- `policy_loss = 0.6319450849`

`rank`：

- `18K_ROUND_ONLY`: `policy -0.000839`, `action -0.000228`, `selection +0.000222`
- `18K_WIDE_GAP`: `policy +0.000202`, `action -0.000397`, `selection +0.000068`
- `CURRENT`: `policy +0.000194`, `action -0.000175`, `selection -0.000120`

解释：`18K_ROUND_ONLY` 是唯一明确改善 real comparison loss 的 rank shape。

`opp`：

- `HYBRID_GRAD`: `policy +0.001124`, `action +0.000114`, `selection +0.000163`
- `EQ_CURRENT`: `policy +0.000498`, `action -0.000388`, `selection -0.000998`
- `18K_STAT`: `policy +0.002613`, `action -0.000609`, `selection -0.001044`

解释：`HYBRID_GRAD` 在 policy gate 内，且 action / selection 质量最好。

`danger`：

- `18K_STAT`: `policy +0.000443`, `action +0.000257`, `selection +0.000310`
- `HYBRID_GRAD`: `policy +0.000899`, `action -0.000351`, `selection -0.000316`
- `CURRENT`: `policy +0.000905`, `action -0.000354`, `selection -0.000834`

解释：`18K_STAT` 在 action 和 selection 质量上最好。

## 冻结与开放边界

已冻结：

- `rank` internal shape。
- `opp` internal shape。
- `danger` internal shape。
- `P1` auxiliary winner 不能用 raw `full_recent_loss` 判。

仍开放：

- family total weights。
- family 是否进入最终长预算主线。
- pairwise / joint interaction。
- `danger_ramp_steps` 等稳定性 knob。

## 当前保留价值

- 后续若重开监督学习 auxiliary 搜索，默认不应重新扫这些 shape。
- 只在出现明确反证时，才重开内部 shape 轴。
- 当前监督学习官方 winner 已冻结；本文只作历史证据。
