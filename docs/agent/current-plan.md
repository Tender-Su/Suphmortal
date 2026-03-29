# 当前默认续工入口

这份文档只回答三件事：现在的默认主线是什么、当前真实停在哪一步、下一步该做什么。细节分别下放到：

- `docs/agent/mainline.md`
- `docs/agent/experiment-workflow.md`
- `docs/status/stage05-verified-status.md`
- `docs/status/p1-selection-canonical.md`

## 建议读取顺序

1. `AGENTS.md`
2. 本文档
3. `docs/status/stage05-verified-status.md`
4. `docs/status/p1-selection-canonical.md`
5. `docs/agent/mainline.md`
6. 需要长背景时，再看 `docs/research/`

## 当前默认

- 总目标：优先做出最强麻将 AI；强度接近时再考虑效率。
- `Stage 0 / GRP` 默认：`384x3 fp32`，下游默认用 `best_loss`。
- `Stage 0.5` 当前官方 `P0 top3`：
  - `C_A2y_cosine_broad_to_recent_strong_12m_6m`
  - `C_A2x_cosine_broad_to_recent_strong_24m_12m`
  - `C_A1x_cosine_broad_to_recent_mild_24m_12m`
- `P1` 当前唯一主线：`calibration -> protocol_decide -> winner_refine -> ablation`
- 当前 `P1 calibration` 默认是 `A2y-only + combo_only`：
  `combo_only` 明确表示沿用 `2026-03-25` 那轮旧 `single-head cali` 数值，
  本轮只重跑 `pairwise / triple` 组合探针，不在当前 run 里重算纯单头量纲
- `P1` 唯一有效选模规范：`docs/status/p1-selection-canonical.md`
- `Stage 1` 默认方向：`Oracle Dropout Supervised Refinement`
- `Stage 2` 只有在新 `Stage 1` 稳定后才推进

## 计算资源分布

- 主节点仍是台式机：`i5-13600KF + RTX 5070 Ti`
- 另有一台可直接用于并行实验的副节点笔记本：`i9-13900HX + RTX 4060 Laptop 8GB + 32GB DDR5`
- 笔记本当前定位是“独立实验节点”，适合并行跑 `GRP`、`Stage 0.5` A/B、loader / validation benchmark、短程 `Stage 1` probe；默认不是跨机分布式训练
- 桌面端已可通过 SSH 直接进入笔记本 shell；代码同步的默认真源应是 `origin/main`，不要把笔记本上的旧拷贝默认当成最新源码
- 双机并行时必须使用不同 run 名称和不同输出目录，不能共写同一份 checkpoint / log
- 笔记本 `Stage 0.5` 当前可操作默认：
  - train：`6 / 7 / 3`
  - val：`7 / 6`
  - `7 / 5` 是非常接近的验证备选
- 这组笔记本默认来自 `2026-03-30` 的本地子集 benchmark；在笔记本尚未持有全量数据根之前，把它当成“当前操作默认”，不要写成对全量数据已经最终验收的永久结论

## 当前运行状态

- 当前在跑的主线 run：`logs/stage05_fidelity/s05_fidelity_p1_top3_cali_slim_20260329_001413/`
- 当前阶段：`running_p1_protocol_decide`
- 当前原则：只推进到 `protocol_decide`，结束后停下来人工确认；不要自动进入 `winner_refine`

## 下一步

- 持续观察 `protocol_decide`
- 等 `seed1` 或整轮结束后汇总协议 winner 候选
- 汇总时按 `docs/status/p1-selection-canonical.md` 解释结果
- 等用户确认后，才决定是否进入 `winner_refine`
- 并行任务侧，如果要立刻在笔记本起 `Stage 0.5`，优先用上面的 `6/7/3 + 7/6`
- 若后续把全量数据正式同步到笔记本，应再做一次 full-data 复核，再决定是否把这组值提升为长期冻结默认

## 不该做的事

- 不要把 `docs/status/stage05-fidelity-results.md` 当默认主线
- 不要把 `docs/archive/` 下的旧 `solo / pairwise / joint refine` 结构重新当成当前入口
- 不要在没有人工确认的情况下把多个阶段一键串到底
