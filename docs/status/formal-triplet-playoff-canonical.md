# Stage 0.5 Formal Triplet Playoff 口径

这份文档只回答当前 `winner_refine` 收口之后，`3 candidate formal playoff` 如何把 pre-formal 第一梯队收口成当前固定 winner / 第一替补。

## 适用范围

- 当前来源 run：
  - `logs/stage05_fidelity/s05_fidelity_p1_top3_cali_slim_20260329_001413/`
- 当前只适用于：
  - `A2x winner_refine` 双 seed 已收口后的下游 `formal_train / formal_1v3`
- 这不是新的 `P1` selector
- 当前官方 `P1 winner` 已由这轮 playoff 固定为 `anchor*1.0`
- `P1` 统一评估口径看：
  - `docs/status/p1-selection-canonical.md`

## 命名口径

- center 只写：
  - `anchor / opp_lean / rank_lean / danger_lean`
- 全头统一缩放只写：
  - `*0.85 / *1.0 / *1.15`
- center 内部再分配只写：
  - `rank+/rank++/opp-/danger++` 这类结构符号
- 当前 canonical 文档不再手写原始 `W_r..._o..._d...`
- 原始 arm id 只保留在当前 source run 的 `winner_refine_round.json / state.json / dispatch_state.json`

## 角色边界

- `winner_refine` 当前只回答：
  - 哪些 arm 还在 pre-formal 第一梯队
  - 当前内部排序谁更像 front runner
- 当前内部 `top1` 是：
  - `opp_lean*0.85`
- 当前新增的 `formal triplet playoff` 只回答：
  - 哪些第一梯队候选值得进入更长预算 `formal_train`
  - 三个 child formal run 的 `formal_1v3` 结果里，哪一个应该成为当前正式 winner
  - 哪一个应保留为第一替补
- 它不会回写或重排 `winner_refine` 内部排序

## 当前 triplet

当前送进 `formal_train` 的 3 个候选固定为：

1. `opp_lean*0.85`
2. `anchor*1.0`
3. `opp_lean(rank--/danger++)`

对应解释：

- `opp_lean*0.85`
  - 当前 `winner_refine` 内部第 `1`
  - 是当前 score / selector 口径下最像 front runner 的点
  - 代表“`opp_lean` center 整体缩到 `0.85x` 后”的稳态版本
- `anchor*1.0`
  - 当前 `winner_refine` 内部第 `2`
  - 是当前 loss / old-regression 最稳的平衡点
  - 代表当前 `anchor` center 原位不动的基准 formal 候选
- `opp_lean(rank--/danger++)`
  - 当前 `winner_refine` 内部第 `4`
  - 是最值得补做长预算 formal 的 loss 向 challenger
  - 它检验的是：`opp_lean` center 保持主方向不变时，进一步减 `rank`、加 `danger` 会不会在长预算里翻盘

## 为什么不是别的点

- `rank_lean*0.85`
  - 当前 `winner_refine` 内部第 `3`
  - 场景分和 selector 不差，但整体厚度与 formal 稳定性证据仍略逊于当前 triplet
- `opp_lean*1.15`
  - 当前 `winner_refine` 内部第 `5`
  - `recent_policy_loss` 很强，但更像 aggressive 上沿点，不像当前最稳 formal 候选
- `danger_lean` family
  - 当前没有任何 arm 进入第一梯队

## 2026-04-05 formal 结果（1.5x）

- 当前 coordinator run：
  - `logs/stage05_fidelity/s05_formal_triplet_20260405/`
- 当前 `3 / 3 child formal` 已全部完成
- 当前 `3 / 3 child formal` 的 `offline_checkpoint_winner` 都是：
  - `best_loss`
- 当前这轮 triplet formal 本身不直接产出官方 `P1 winner`
- 它在进入 `formal_1v3` 之前的解释边界仍是：
  - 只给出进入 `formal_1v3` 前的 offline front-runner / hedge order
  - 最终 canonical winner 仍由后续 `formal_1v3` 决定

当前按主线现有 `LOSS_EPSILON + selection_tiebreak_key` 规则重排后，cross-run offline 顺序为：

1. `opp_lean*0.85`
2. `opp_lean(rank--/danger++)`
3. `anchor*1.0`

当前三条结果可这样理解：

- `opp_lean*0.85`
  - `best_full_recent_loss = 0.480049`
  - `macro_action_acc = 0.847933`
  - `rank_acc = 0.445781`
  - 当前是 offline front-runner
  - 当前 policy / action 侧最强
- `opp_lean(rank--/danger++)`
  - `best_full_recent_loss = 0.480872`
  - `macro_action_acc = 0.842595`
  - `rank_acc = 0.445557`
  - 当前总 loss 与 front-runner 很接近
  - 当前 `opp / danger` 头最强，是最值得保留的 hedge challenger
- `anchor*1.0`
  - `best_full_recent_loss = 0.480868`
  - `macro_action_acc = 0.843318`
  - `rank_acc = 0.450905`
  - 当前 `rank_acc` 最强
  - 但在现有 selection policy 的 cross-run offline 排序里仍列第 `3`

当时三条都继续进入了 `formal_1v3`；当前已经收口，不再停留在这一步。

## 2026-04-05 cross-run formal_1v3 结果

- 当前 playoff run：
  - `logs/stage05_fidelity/s05_formal_triplet_20260405_winner_playoff_1v3/`
- 当前正式判据：
  - `avg_pt` 为主
  - `avg_rank` 为辅
  - 当前 active 位次分：`90 / 45 / 0 / -135`
- 当前最终顺序：
  1. `anchor*1.0`
  2. `opp_lean*0.85`
  3. `opp_lean(rank--/danger++)`
- 当前固定结论：
  - 当前官方 `P1 winner = anchor*1.0`
  - 当前第一替补 = `opp_lean*0.85`
  - `opp_lean(rank--/danger++)` 不再作为当前保留候选
- 当前 `close_call` 仍然成立：
  - top2 仍在噪声带内
  - 但当前正式决议不再停留在“暂不写死”
- 当前 top2 的结构性解释：
  - `anchor*1.0` 不是更会冲第 `1`，而是更少第 `4`
  - 所以它在当前 `90 / 45 / 0 / -135` 口径下赢出了 `avg_pt`
  - `opp_lean*0.85` 仍然是当前最强的争 `1` 型预备队
- 如果按 `3210` 给 `1 / 2 / 3 / 4` 位打分：
  - top2 会翻成 `opp_lean*0.85 > anchor*1.0`
  - 但这只保留为诊断口径，不覆盖当前正式 winner 决议

## 单个 formal 的本机耗时实测

台式机当前 `formal_train` 默认已经上调到旧口径的 `1.5x`，实际理解为：

- `formal` 基础 `phase_steps = 9000 / 6000 / 3000`
- dispatch 仍固定：
  - `formal_step_scale = 5.0`
- 所以当前有效训练长度应按：
  - `phase_a / phase_b / phase_c = 45000 / 30000 / 15000`
- 这比旧的 `30000 / 20000 / 10000` 正好多 `1.5x`
- 每相位预算末尾都要跑：
  - `monitor validation`
  - `full_recent validation`
  - `old_regression validation`

`2026-04-05` 这次 triplet 的实测 wall-clock 是：

- 台式机：
  - `opp_lean(rank--/danger++)` 约 `4 h 33 min`
  - `anchor*1.0` 约 `4 h 32 min`
  - 所以当前台式机单条 `1.5x formal` 可按 `约 4.5 h` 理解
- 笔记本：
  - `opp_lean*0.85` 约 `11 h 10 min`
  - 所以当前 `4060 Laptop` 节点更适合承担并行分摊，但不应再拿台式机 wall-clock 直接外推

也就是说，这次确实按 `1.5x` 跑了，只是台式机实测 throughput 比之前保守估算更高。

## 如需重开 triplet

### 1. 先跑 triplet formal dispatch

双机并行入口：

```powershell
python mortal/run_stage05_formal_distributed.py dispatch `
  --run-name s05_formal_triplet_20260405 `
  --source-run-name s05_fidelity_p1_top3_cali_slim_20260329_001413 `
  --candidate-arm 'opp_lean*0.85' `
  --candidate-arm 'anchor*1.0' `
  --candidate-arm 'opp_lean(rank--/danger++)'
```

说明：

- `run_stage05_formal_distributed.py` 现在可以直接吃这些结构 alias
- 它会从 source run 的 `winner_refine_round` 里把 alias 解析回实际 arm id
- 当前 canonical 文档不再把原始 `W_r...` arm id 写死在命令示例里

状态查看：

```powershell
python mortal/run_stage05_formal_distributed.py status `
  --run-name s05_formal_triplet_20260405
```

这一步会生成 3 个 child run。

- child run 的真实目录后缀仍然沿用 source run 里的实际 arm id
- canonical 文档不再手工展开这些 raw 后缀
- 每个 child run 都会产出标准的：
  - `state.json`
  - `formal.result`
  - `best_loss / best_acc / best_rank` checkpoint pack

### 2. 再对每个 child run 跑现有 formal_1v3

`formal_train` 收口后，继续复用现有入口：

```powershell
python mortal/run_stage05_formal_1v3_distributed.py dispatch `
  --run-name <child_run_name>
```

当前推荐执行顺序是：

1. `opp_lean*0.85`
2. `opp_lean(rank--/danger++)`
3. `anchor*1.0`

但如果只是沿用当前主线结论，不需要再重开；当前默认直接使用 `anchor*1.0`。

### 3. 最后比较 triplet 的 formal_1v3 结果

当前 cross-run 解释口径固定为：

1. 先取每个 child run 自己的 `formal_1v3` 最终 winner
2. 再比较这 `3` 个 child run 的最终 `avg_pt`
3. 若 `avg_pt` 很接近，再用 `avg_rank` 辅助
4. 如果前两名仍落在显著噪声带里，再对这两条 child run 追加新 `seed_key` 的 `formal_1v3`

也就是说，当前 triplet playoff 的“1v3 决胜”是：

- 先在 run 内比较 `best_loss / best_acc / best_rank`
- 再在 run 间比较 `3` 条 child run 的最终 `formal_1v3` 统计

## 双机脚本的资产边界

`run_stage05_formal_distributed.py` 当前约定：

- 复用 `protocol_decide / winner_refine` 的 dispatch / pause / resume worker 框架
- 远端 task 只吃同步过去的 dispatch 状态，不再要求手工预放 source run 的完整 logs
- 远端 formal 完成后，会把 child run 与对应 `stage05_ab/<child_run_name>_formal` 产物拉回台式机
- child run 回传后仍是标准本地 run，可直接继续跑现有 `formal_1v3`

## 不要这样做

- 不要把 triplet playoff 解释成“`P1` 官方 selector 改成 top3 晋级”
- 不要把 `winner_refine` 当前内部 `top1` 直接写成官方 `P1 winner`
- 不要再把 `opp_lean*0.85` 写成当前 winner；它现在固定是第一替补
- 不要在当前 canonical 文档里继续硬编码 raw `W_r...` arm 名字
- 不要在没有 triplet formal 完整结果前，直接只送单一 front runner 进入 Stage 1
- 不要把 `docs/status/stage05-fidelity-results.md` 的单 run 自动快照当作当前 triplet playoff 的统一入口
