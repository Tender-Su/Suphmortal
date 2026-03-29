# 当前主线与稳定默认

这份文档汇总当前仍然有效、且会影响后续实验判断的默认结论。它不是逐日日志，也不记录所有历史试错。

## 口径优先级

1. `AGENTS.md`
2. `docs/agent/current-plan.md`
3. `docs/status/stage05-verified-status.md`
4. `docs/status/p1-selection-canonical.md`
5. `docs/status/stage05-fidelity-results.md` 仅作 run 快照

## 总目标

- 目标是最强麻将 AI，不是最省时间的麻将 AI。
- 当两个方案强度接近时，优先更小更快。
- 当强度差异明确时，优先更强方案。

## 计算资源与同步默认

- 主节点：台式机 `i5-13600KF + RTX 5070 Ti`
- 副节点：笔记本 `i9-13900HX + RTX 4060 Laptop 8GB + 32GB DDR5`
- 笔记本当前是独立实验节点，不是已配置好的分布式训练成员；默认把它当成“另一台可并行跑完整实验脚手架的 Windows 机器”
- 跨机器源码同步默认以 `origin/main` 为准；如果笔记本本地目录只是旧拷贝，先同步再信任结果
- 双机并行纪律：不同机器必须使用不同 run 名称、不同输出目录、不同 checkpoint 文件，避免互相覆盖

## Stage 0

- 默认主线：`384x3 fp32`
- 默认下游 checkpoint：`best_loss`
- 受控对照：`best_acc`
- `latest` 只用于续训
- 仍值得继续探索的更大候选：`512x4`
- 研究入口：`docs/research/stage0/grp-experience.md`

## Stage 0.5

- 默认正式训练快路径：
  - train：`num_workers = 4`、`file_batch_size = 10`、`prefetch_factor = 3`
  - val：`val_file_batch_size = 8`、`val_prefetch_factor = 5`
- 上述 `4/10/3 + 8/5` 是台式机默认，不自动外推到笔记本；笔记本需要单独 benchmark 后再写入默认
- 当前笔记本的独立 benchmark 结论（`2026-03-30`，本地代表性子集）：
  - train：`6 / 7 / 3`
  - val：`7 / 6`
  - close fallback：`7 / 5`
- 该笔记本结论当前用于并行实验操作，不覆盖台式机默认，也不等同于“已用笔记本全量数据根完全验收”
- 当前 `P0` 官方入选 `top3`：
  - `C_A2y_cosine_broad_to_recent_strong_12m_6m`
  - `C_A2x_cosine_broad_to_recent_strong_24m_12m`
  - `C_A1x_cosine_broad_to_recent_mild_24m_12m`
- `B-side` 当前是资源冻结，不进入当前 `P1`
- `P0 / P2 / formal` 的默认主比较字段仍是 `full_recent_loss`

## P1 主线

- 当前结构固定为：`calibration -> protocol_decide -> winner_refine -> ablation`
- `calibration` 的职责是定标，不是宣告 winner
- 当前默认使用瘦版 `calibration`：
  - 代表协议固定为 `A2y`
  - 单头量纲直接沿用 `2026-03-25` post-shape `top3 calibration` 的旧数值
  - 这意味着当前 `combo_only` 不是“本轮重新估单头，只少跑几项”，而是复用旧 `single-head cali`
  - 当前 run 只补 `pairwise / triple combo factor`
- 当前冻结的三头内部 shape：
  - `rank = 18K_ROUND_ONLY`
    - `south_factor = 1.59`
    - `all_last_factor = 1.617`
    - `gap_focus_points = 4000`
    - `gap_close_bonus = 0.0`
  - `opp = HYBRID_GRAD`
    - `shanten = 0.8506568408`
    - `tenpai = 1.1493431592`
  - `danger = 18K_STAT`
    - `any = 0.0904217947`
    - `value = 0.8180402859`
    - `player = 0.0915379194`
- 当前 `protocol_decide` 默认网格：
  - `total_budget_ratios = [0.09, 0.12]`
  - `anchor = 0.43 / 0.21 / 0.36`
  - `rank_lean = 0.53 / 0.16 / 0.31`
  - `opp_lean = 0.38 / 0.31 / 0.31`
  - `danger_lean = 0.38 / 0.16 / 0.46`
- `P1` 选模、门槛和解释规则只看：`docs/status/p1-selection-canonical.md`

## 选择器当前默认

- `selection_quality_score = action_quality_score + 0.20 * scenario_quality_score`
- `recent_policy_loss` 门槛：`+ 0.003`
- `old_regression_policy_loss` 门槛：`+ 0.0035`
- `selection_tiebreak_key = selection_quality_score -> -recent_policy_loss -> -old_regression_policy_loss`
- 各类 `acc` 只保留为诊断字段
- 统计审计入口：`docs/research/stage05/selector-stat-audit.md`

## Stage 1 与 Stage 2

- `Stage 1` 默认主线：`Oracle Dropout Supervised Refinement`
- 它从 `Stage 0.5` winner 协议和配比继续，而不是回到旧 `GRP-AWR`
- `Stage 2` 在 `Stage 1` 稳定后再推进，不提前抢跑

## 当前活跃 run

- run 目录：`logs/stage05_fidelity/s05_fidelity_p1_top3_cali_slim_20260329_001413/`
- 当前状态：`running_p1_protocol_decide`
- 当前人工停点：`protocol_decide` 结束即停，等待确认

## 进一步阅读

- 当前真实进度：`docs/status/stage05-verified-status.md`
- 运行流程与停点：`docs/agent/experiment-workflow.md`
- 工程方法与有效经验：`docs/research/stage05/engineering-playbook.md`
- 笔记本 loader benchmark 记录：`docs/status/laptop-stage05-loader-benchmark-2026-03-30.md`
- 历史长入口：`docs/archive/agent/current-plan-legacy-2026-03-29.md`
