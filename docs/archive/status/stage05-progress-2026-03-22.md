# 2026-03-22 Stage 0.5 进展快照

> 历史快照说明：这份文档记录的是 `2026-03-22` 当时的阶段判断，不再等价于当前实时进度。最新续工口径以 `docs/agent/current-plan.md` 为准；人工核对后的当前状态以 `docs/status/stage05-verified-status.md` 为准；自动摘要以 `docs/status/stage05-fidelity-results.md` 为准；`P1` 的唯一有效选模规范以 `docs/status/p1-selection-canonical.md` 为准。
> 当前额外说明（`2026-04-04`）：文中出现的 `Stage 0.5 / P2` checkpoint 去重层已从当前主线删除；当前 `formal` 直接接在 `P1 winner_refine` 之后。

## 2026-03-23 补充

- 截至 `2026-03-23 22:42`，项目已经从“准备重跑新微预算 `P1 solo`”推进到“raw artifacts 已落地两轮 `P1 solo` 种子结果”
- 当前已出现的两份 `P1 solo` round 文件是：
  - `logs/stage05_fidelity/s05_fidelity_main/p1_solo_round__s20261817.json`
  - `logs/stage05_fidelity/s05_fidelity_main/p1_solo_round__s20262826.json`
- 当前还没有 `p1_pairwise`、`p1_joint`、`p2` 或 `formal` 目录，因此 `P1 solo` 之后的阶段尚未启动
- `docs/status/stage05-fidelity-results.md` 的自动摘要当前落后于原始产物；人工核对结论已单独移到 `docs/status/stage05-verified-status.md`

## 这份文档解决什么问题

这轮仓库改动同时涉及代码、测试、自动生成结果和人工总结，单看某一个文件很容易误判当前进度。这份文档专门把三件事拆开：

- 代码已经改到了哪一步
- 已经跑完并可作为证据的实验产物有哪些
- 当前流程真正停在什么位置，下一步该做什么

## 一句话结论

当前项目并不是在推进新的主干模型结构，而是在把 `Stage 0.5` 的选模、验证和 `P1` 辅助头搜索做成更可靠的实验系统。实际进度可以概括为：

- `P0` 已按新 selector 口径完成重验，并收敛出新的 `top4` 下游协议种子
- `P1 calibration` 已完成一轮可用校准，确认旧辅助权重搜索区间过重
- 新代码已经切到“微预算 + family-specific + `policy_quality` + protocol-local `ce_only` / `old_regression` 护栏”的 `P1` 搜索逻辑
- 但新的 `P1 solo / pairwise / joint refine` 结果还没有在当前运行目录里完整落地
- 因此 `P2` 和 `Stage 0.5 formal` 还不能视为在新口径下已经定型

## 本轮代码改动的核心含义

### 1. selector 已从“看 loss + 少量动作指标”升级成完整的三层口径

这轮最重要的变化不是训练目标，而是验证/选模目标：

- `action_quality_score` 不再直接线性加总原始 loss，而是改成“`loss / (1 + loss)` 饱和化 + 按样本数做置信度 shrinkage”
- 动作主分里补入了 `chi_exact_nll`，把“该不该吃”和“吃哪一种”拆开计分
- `scenario_quality_score` 从少量 `riichi late / south+ / threat` 切片扩展到三层：
  - `P0`：已有硬场景切片
  - `P1`：位次目标、上下家点差方向、威胁强度、对手状态条件切片
  - `P2`：`discard` 的关键场景 `nll` 与 `push_fold_core / push_fold_extreme`
- 主排序键改成 `selection_quality_score = action_quality_score + 0.20 * scenario_quality_score`

这说明当前主线已经不再接受“只有总体 loss 更低就算更强”的旧口径，而是明确把末盘、高压和微差场景纳入了第一层排序。

### 2. 验证链路已经为新 selector 补齐了缺失标签和统计

为了让新 selector 真正可用，训练/验证链路补了几项之前缺失的基础设施：

- 验证阶段即使没有打开 `opponent_state aux`，也可以统一发放 `opponent-state` 标签，用于公平计算条件切片
- 新增 `discard` 场景切片统计，支持 `push_fold_core / push_fold_extreme`
- 新增 `chi_exact` 指标导出
- `unpack_batch()` 改成按剩余列数动态解析，避免训练/验证标签列不一致时直接错位

这部分改动的本质是“让所有 arm 走同一条评估链路”，否则新的 `scenario_quality_score` 只是形式上存在，比较结果并不公平。

### 3. `P1` 已从粗糙扫权重切到“统一预算轴”搜索

这轮 `P1` 代码改动说明项目已经明确放弃“直接比较 rank/opp/danger 原始权重”的做法，改为：

- 先用 `P1 calibration` 估计三类辅助头对 trunk 的有效预算
- 再把 `rank / opp / danger` 映射到统一预算轴
- 再做 `solo -> pairwise -> joint refine`
- `solo` 晋级不再看全局 eligible，而是看“是否满足各自协议内的 `policy_quality` 门槛：`comparison_recent_loss = recent_policy_loss` 在 `group_best + 0.003` 内，且 `old_regression_policy_loss` 可用时再过 `group_best_old + 0.0035` 护栏，并且不能明显输给同协议 `ce_only`”

这一步很关键，因为它意味着当前团队判断已经从“哪类辅助头听起来更强”转到“先让比较口径公平，再决定谁值得保留”。

### 4. 项目开始显式处理“旧缓存与新 selector 不一致”问题

新增的 `revalidate_stage05_round.py` 和相关缓存版本逻辑，说明当前一个真实工程问题已经被确认：

- selector 公式一旦升级，旧 `arm_result.json` 里缓存的显式分数不能再直接信
- 必须带版本号重算 `scenario_quality_score / selection_quality_score`
- `P0/P1` 的缓存结果需要支持重验和重新排序

这代表 `Stage 0.5` 已经从“先跑实验再看”进入“必须保证历史结果可重放、可复核”的阶段。

## 当前已经落地并可视为证据的实验进展

### 1. `P0` 已完成新口径下的重验和 top4 收敛

当前运行目录 `logs/stage05_fidelity/s05_fidelity_main/state.json` 最可靠的已固化结论是：

- `P0 round0 / round1 / round2` 已经按新 selector 口径重排
- 当前 `top4` 下游协议种子为：
  - `C_A1x_cosine_broad_to_recent_mild_24m_12m`
  - `C_A3x_cosine_broad_to_recent_two_stage_24m_12m`
  - `C_B2z_cosine_recent_broad_recent_strong_6m_6m`
  - `C_B2x_cosine_recent_broad_recent_strong_24m_12m`

和更早文档相比，这里的变化不是简单换了排序，而是说明新的 selector 口径已经实际改变了 `P0` 的保留名单。

### 2. `P1 calibration` 已完成，但它还是“定标证据”，不是正式 winner

当前已完成的 `P1` 产物是：

- `logs/stage05_fidelity/s05_fidelity_main/p1_calibration.json`
- `logs/stage05_fidelity/s05_fidelity_main/p1_calibration__s20261716.json`

从这些产物可以确认：

- `rank_only` 仍是当前最稳的 calibration 锚点
- 旧 `opp=0.06 / danger=0.06` probe 下，`opp` 与 `danger` 的训练压力并不对等
- `danger` 在旧 probe 强度下比 `opp` 更不容易伤主目标
- `opp + danger` 不能按原始权重简单相加

因此这轮 calibration 的价值在于“定标”，不是在于直接宣布哪种辅助头赢了。

### 3. 已有 `P1 solo` 结果仍然是旧预算证据，只能用于缩搜索空间

当前目录里存在：

- `logs/stage05_ab/s05_fidelity_main_p1_solo_s20261817/revalidated_best_loss_final_round.json`

但这批结果对应的还是旧预算命名和旧扫法，例如：

- `r025 / r050 / r100 / r150`
- `o025 / o050 / o075 / o100`
- `d025 / d050 / d075 / d125`

这批结果的用途已经不是“直接选新主线 winner”，而是用来支持下面这个结论：

- 旧 `solo` 搜索区间整体过重
- 在旧最小预算下，`rank / opp / danger` 都会推高 `full_recent_loss`
- 其中 `danger` 伤害最小，`opp` 最差
- 所以代码里已经把下一轮搜索改成更轻、更不对称的微预算区间

换句话说，当前 `P1 solo` 的历史结果更像“反证数据”，证明旧搜索空间不该继续沿用。

## 当前代码状态与实验状态之间的错位

这是现在最容易看错的一点。

### 1. 代码已经前进到“新微预算搜索”，但运行状态文件还没有完整回写

当前代码里的 `P1` 默认搜索区间已经改成：

- `rank`: `0.03 / 0.06 / 0.10 / 0.15`
- `opp`: `0.03 / 0.06 / 0.10 / 0.15`
- `danger`: `0.05 / 0.10 / 0.20 / 0.30`

同时新增了：

- protocol-local `ce_only` survivor gate
- 多轮缓存重验
- 新 selector 版本控制

但是 `logs/stage05_fidelity/s05_fidelity_main/state.json` 在最近一次落盘时只固化到了 `P0 top4`，还没有把这套新 `P1` 逻辑的完整 round 结果写进去。

### 2. 当前 live run 仍在运行，但不能只看 `state.json`

当前 `run.lock.json` 记录的 Python 进程仍在运行，启动时间为：

- `2026-03-22 03:24:39`

这说明流程没有结束，但现在不能把 `state.json` 当成唯一进展来源。当前更可靠的判断方式是：

- `P0` 看 `state.json`
- `P1 calibration` 看 `p1_calibration*.json`
- `P1 solo` 的旧证据看 `revalidated_best_loss_final_round.json`
- 新微预算 `P1` 是否完成，要以后续 round summary 是否生成并回写为准

## 现阶段最准确的项目阶段判断

截至 `2026-03-22`，当前项目处在下面这个阶段：

1. `Stage 0.5 / P0` 已按新 selector 收敛出新的 `top4` 协议种子。
2. `Stage 0.5 / P1` 的比较口径、搜索空间和 survivor 规则已经完成代码升级。
3. 旧 `P1 solo` 结果已经足够证明“旧预算区间过重”，所以主线不应继续沿用旧扫法。
4. 新的 `P1 solo -> pairwise -> joint refine` 还没有以完整结果形式固定下来。
5. 因此 `P2` 去重、默认 checkpoint 选择和 `Stage 0.5 formal` 启动条件，目前都还不能视为在新口径下完成。

换成更直接的话说：项目现在已经跨过了“重写 selector 和评估基础设施”阶段，但还没跨过“用新口径把 `P1/P2` 真正跑完并定型”这道坎。

## 当时建议把什么当作接下来的主线

如果继续按当前分支推进，最合理的顺序是：

1. 用新微预算区间重跑 `P1 solo`
2. 按 protocol-local `ce_only` 基线筛掉明显不成立的 family
3. 只围绕幸存 family 做 `pairwise`
4. 再围绕幸存组合做 `joint refine`
5. 最后再进入 `P2` 去重和 `Stage 0.5 formal`

在这之前，不应把旧 `solo` winner、旧 `P1` 排序或旧 `P2` 结论当成当前主线结论。

## 这次检查额外确认的事项

- `mortal.test_stage05_selection`：通过
- `mortal.test_train_supervised`：通过
- `mortal.test_stage05_fidelity`：通过

说明这轮 selector、训练批解析、turn weighting 和 `P1` 搜索逻辑的基础单测目前是闭环的。
