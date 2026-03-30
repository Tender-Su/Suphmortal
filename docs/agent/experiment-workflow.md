# 当前实验流程

这份文档只回答“现在应该怎么跑”，不重复展开所有历史原因。

## 总体原则

- 每个阶段结束后都停下来，让人确认方向。
- 不使用默认一键跑到底的工作流。
- 新 run 必须用新输出目录，不复用旧目录解释新实验。
- run snapshot 只记录当次产物，不代替当前默认。

## Stage 0.5 当前流程

1. `P0`
   - 目标：从监督协议里选出值得进入 `P1` 的事实 `top3`
   - 当前已冻结，不需要为继续主线而重跑
2. `P1 calibration`
   - 目标：统一三头 budget 量纲，并估计组合耦合
   - 当前默认：`A2y-only + combo_only`
   - 这一步只能定标，不能直接宣布 winner
3. `P1 protocol_decide`
   - 目标：在统一三头脚手架下尽早选出协议 winner
   - 当前这轮已经完成，winner 已验证为 `A2x`
   - 当前 seed2 扩展规则固定为：
     - `ambiguity_mode = flip_or_gap`
     - `gap_threshold = 0.001`
4. `P1 winner_refine`
   - 目标：只在 winner 协议内部细调三头全开配比
   - 当前只允许在 `A2x` 内部启动
   - 当前默认不是自动 `top-k center`
   - 当前默认冻结三中心：
     - `C_A2x_cosine_broad_to_recent_strong_24m_12m__B_r0046_o0037_d0037`
     - `C_A2x_cosine_broad_to_recent_strong_24m_12m__B_r0034_o0014_d0041`
     - `C_A2x_cosine_broad_to_recent_strong_24m_12m__B_r0052_o0025_d0043`
   - 当前局部搜索规则：
     - `total_scale_factors = [0.85, 1.0, 1.15]`
     - `transfer_delta = 0.01`
     - `step_scale = 1.5`
5. `P1 ablation`
   - 目标：验证 `all_three / drop_* / ce_only` 的边际贡献
   - 也需要人工确认后再启动

## 当前人工停点

- `protocol_decide` 当前已经结束并停住
- 不自动进入 `winner_refine`
- `winner_refine` 结束后再次停
- `ablation` 结束后再决定是否写入更下游默认

## 当前运行纪律

- 默认不设置 `MORTAL_CPU_AFFINITY`
- 多 seed round 收口前，如果仍有 invalid arm 且补跑成本可接受，先补跑再宣布结果
- 台式机训练快路径默认：
  - train：`4 / 10 / 3`
  - val：`8 / 5`
- 笔记本独立 benchmark 默认：
  - train：`6 / 7 / 3`
  - val：`7 / 6`

## P1 结果怎么解释

- 是否入围、谁是 winner、谁只是诊断信息，一律按 `docs/status/p1-selection-canonical.md`
- `protocol_decide` 只回答“哪个协议继续往下走”
- `winner_refine` 只回答“winner 协议下三头该怎么配”
- `ablation` 只回答“三个头是否都还有边际贡献”

## 不再使用的旧流程

- 旧 `solo -> pairwise -> joint refine` 仍有诊断价值
- 但它已经不是当前主线，也不能覆盖当前 `P1` winner 的解释
