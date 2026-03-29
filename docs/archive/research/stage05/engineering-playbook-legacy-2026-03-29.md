# Stage 0.5 训练经验记录

> 历史说明：本文保留大量 `2026-03-28` 之前的工程与实验记录。
> 凡是把 `P1` 主线写成 `SoloAuxGate -> Pairwise -> JointRefine` 的段落，都只能当历史设计与排错背景，不能当当前默认。
> 当前默认 P1 结构以 `docs/agent/current-plan.md` 与 `docs/status/p1-selection-canonical.md` 为准，即：
> `calibration -> protocol_decide -> winner_refine -> ablation`。

本文档只记录 `Stage 0.5` 的工程经验和排障结论，不重复 `AGENTS.md` 的通用规范，也不重复 `docs/agent/current-plan.md` 的长期路线。

## 当前状态说明

当前正式训练入口仍然是 `.\scripts\run_supervised.bat` 或 `python mortal/run_stage05_formal.py`。`docs/status/stage05-verified-status.md` 负责记录人工核对后的当前口径，`docs/status/stage05-fidelity-results.md` 保留自动生成的 `fidelity / P0 / P1` 摘要，本文负责背景、排障、方法和试错过程。`2026-03-24` 之后需要额外注意：本文中凡是把 `C_B3z / C_B2z / C_A3x / C_A1x` 写成当前官方 `P0 top4` 或默认主线种子的表述，都只能视为泄露修复前的历史记录；修复 `old_regression` 泄露并 clean rerun `P0` 后，当前官方 `P1 entry top3` 已更新为 `C_A2y_cosine_broad_to_recent_strong_12m_6m`、`C_A2x_cosine_broad_to_recent_strong_24m_12m`、`C_A1x_cosine_broad_to_recent_mild_24m_12m`，以 `docs/agent/current-plan.md` 与 `docs/status/stage05-verified-status.md` 为准。当前代码下的 `Stage 0.5` 协议需要以 `AB1234/P0` 的全局联合筛选和重跑后的 `AB1` 复核结果为准。工程运行参数当前默认快路径为：`batch_size = 1024`、`num_workers = 4`、`file_batch_size = 10`、`prefetch_factor = 3`；正式训练验证节奏仍采用 `val_every_steps = 20000` 与 `monitor_val_batches = 512`，并关闭 `full_val_every_checks` 与 `old_regression_every_checks`。

CPU affinity 也已从“默认绑 `p_cores`”改为“显式 opt-in”。现在正式训练和 A/B 如果没有手动设置 `MORTAL_CPU_AFFINITY`，就保持 Windows 默认调度；只有在明确做 `p_cores`、`all` 或 CPU 掩码对照实验时，才显式注入该环境变量。

## 2026-03-29 P1 主线调整过程

这次主线调整不是一次性拍板，而是沿着“旧结构回答不了真正问题”这条线逐步收缩出来的。

最早的 `P1` 设计是典型的分层筛选思路：先跑 `solo` 看三类辅助头各自是否有用，再把单头 survivor 送进 `pairwise`，最后围绕双头强点做 `joint refine`。这个结构的直觉很强，因为它对应“先看单项，再看两两组合，再看整体”的常见实验套路。

真正的问题是在结果解释上暴露出来的。`solo` 的价值是清楚的，它能回答“`rank / opp / danger` 各自单独开时是否有净收益”。但往下走到 `pairwise` 后，问题开始出现：它把单头 winner 直接相加，默认假设“单头最优点彼此兼容”，而这在当前任务上并不成立。实验中反复出现的现象是，`pairwise` 更像一个“组合是否过重/互相打架”的诊断轮，而不像一个能稳定代表三头全开最优配比的主决策轮。

这一步带来的关键认识是：项目真正要解决的问题，不是“哪一个单头最强”，而是两件更贴近最终目标的事：

1. 在 `top3 protocol` 里，哪条协议应该尽早胜出，避免无谓的并行前进。
2. 在 winner 协议内部，三头全开时应该如何分配总预算。

一旦把问题这样重写，旧的 `solo -> pairwise -> joint_refine` 就不再适合作为主线骨架了。于是主线改成了：

- `calibration`
- `protocol_decide`
- `winner_refine`
- `ablation`

这个四段结构的逻辑是：

- `calibration` 只负责定标，不宣布 winner
- `protocol_decide` 尽早在 `top3 protocol` 中选出 winner
- `winner_refine` 只在 winner 协议内部细调三头全开配比
- `ablation` 最后验证三个头是否都还保留边际价值

接下来又出现了第二层收缩。最初设想的新 `calibration` 版本仍然偏重，想把 `top3` 协议都重跑一遍，并且单头、双头、三头 probe 都重新测齐。这样做当然最稳，但问题在于它会重复测量已经有充分证据支持的东西。

重新检查 `2026-03-25` 的 `top3 calibration` 之后，结论很明确：

- 那轮 calibration 实际已经发生在内部 shape freeze 之后
- 它跑的就是新 shape，而不是旧 shape
- `A2y / A2x / A1x` 三条协议在单头量纲上几乎不波动
- 旧版能测到的 `opp + danger` 组合因子也几乎不波动

这说明“多协议 + 单头重跑”已经不是当前信息瓶颈。真正仍然缺失的，是新主线三头全开时还没被直接测过的组合耦合信息，也就是：

- `rank + opp`
- `rank + danger`
- `opp + danger`
- `rank + opp + danger`

于是 `calibration` 又做了第二次收缩，形成当前默认的瘦版：

- 代表协议固定为 `A2y`
- 单头量纲直接沿用 `2026-03-25 post-shape calibration`
- 本轮只补 `pairwise / triple combo factor`

换句话说，这次主线调整背后的思路不是“把流程做复杂”，而是不断把“还缺什么信息”与“已经稳定知道什么信息”拆开：

- 旧三段式回答不了“三头全开应该怎么配比”，所以被替换
- 新版重 `calibration` 会重复验证已经稳定的单头量纲，所以被收缩
- 最终留下的当前主线，只保留那些对“训练最强麻将 AI”这个目标仍然提供新增信息的环节

因此，当前默认主线可以理解为一条被多轮实验和复盘不断削减后的最小有效链路，而不是一条事先设计好的固定模板。

## 与新 Stage 1 的衔接更新（2026-03-17）

当前 `Stage 0.5` 的职责已经从历史上服务旧 `Oracle AWR / GRP-AWR` 流水线，转为“为新 `Stage 1` 选协议、选种子、准备辅助监督接口”。后续默认主线不是回到旧 `GRP-AWR`，而是进入合并后的 `Oracle Dropout Supervised Refinement`。这条新主线默认从 `P0 round2` 的 `top4` 协议种子启动：`C_B3z_cosine_recent_broad_recent_two_stage_6m_6m`、`C_B2z_cosine_recent_broad_recent_strong_6m_6m`、`C_A3x_cosine_broad_to_recent_two_stage_24m_12m` 和 `C_A1x_cosine_broad_to_recent_mild_24m_12m`。这四个种子的作用不是直接宣布 `Stage 0.5` 唯一 winner，而是保证新 `Stage 1` 一开始就同时覆盖两条课程线和两类近期加权风格。

新 `Stage 1` 的辅助监督也不再只围绕 `rank aux` 与 `opponent-state aux` 两条线展开，而是升级为 `rank aux + opponent-state aux + danger aux` 的联合预算设计。其中 `opponent-state aux` 继续负责逼模型学会理解别家向听/听牌进度和手切摸切信号；`danger aux` 则进一步把这种状态理解落实到动作级别，直接监督“当前可打出的每一张牌是否会即时放铳、最大 ron 点数损失有多大、会放给哪一家”。这条 `danger aux` 在实现上默认一次性包含三路输出，而不是拆成三轮串行开发；训练稳定性通过分头归一化、内部加权、阶段式启用和类不平衡处理解决。

此外，长期价值建模也不再默认继续压在 `GRP` 的样本加权链路上。当前更合理的方向是：保留 `GRP` 作为独立全局名次/价值模型和对照基线，同时单独立项 `Oracle Value Predictor`，让它成为后续 `Stage 1` 长期价值理解与 `Stage 2 critic` 初始化的候选来源。也就是说，`Stage 0.5` 现在更像“协议筛选 + 种子筛选 + 接口准备”阶段，而不是历史 `GRP-AWR` 流水线的前置附庸。

对应地，旧 `Oracle AWR` 已经从当前仓库主线移除，不再保留脚本与代码路径。后续如果确实需要回看这类历史方案，只通过 `git` 历史复盘；当前主线的全部算力与工程复杂度都应集中到合并后的 `Stage 0.5 -> 新 Stage 1` 路线上，而不是继续维护传统 `AWR / GRP-AWR` 分支。

## 当前代码下的 `AB1` 重跑结论（2026-03-16）

旧文档中曾记录过一次 `AB1` 结果，结论是 `plateau > cosine > phasewise`。这条结论现在已经不能继续引用，因为在当前代码下重跑同一问题后，结果稳定反转，而且不是单次偶然。重跑时仍然固定 `curriculum = broad_to_recent`、`weight = two_stage`、`window = 24m_12m`，只比较 `scheduler`，并分别使用 `20260312 / 20260413 / 20260514` 三个 seed。三次结果全部是 `B_cosine` 胜出：`seed=20260312` 时 `B_cosine loss=0.604391`、`A_plateau loss=0.625892`、`C_phasewise loss=0.642336`；`seed=20260413` 时 `B_cosine loss=0.610014`、`A_plateau loss=0.628657`、`C_phasewise loss=0.647641`；`seed=20260514` 时 `B_cosine loss=0.609496`、`A_plateau loss=0.624212`、`C_phasewise loss=0.643931`。这说明在当前代码和当前口径下，`AB1` 的有效结论已经变成 **`cosine > plateau > phasewise`**。

更重要的是，这个结果和当前 `P0` 干净 `round1` 的同一 slice 完全一致。`C_A3x_cosine_broad_to_recent_two_stage_24m_12m` 在 `P0 round1` 中的 `full_recent_loss = 0.603870`，而同 slice 的 `P_A3x_plateau_broad_to_recent_two_stage_24m_12m` 为 `0.623538`。因此，当前不是 `AB1` 和 `P0` 互相打架，而是“旧 `AB1` 历史记录”与“当前代码下的新结果”打架。后续所有关于 scheduler 的讨论，都应以当前代码下的重跑结果为准，而不再以旧文档里的 plateau 结论为准。

## A/B 实验摘要

`Stage 0.5` 的 A/B 不是在试一个孤立超参数，而是在回答一个更实际的问题：监督预训练怎样做，才能真的帮到当前环境下的后续训练。为此，实验比较了三类调度器、两类课程顺序、三类近期加权、三类时间窗口，以及多种 checkpoint 口径；课程和加权、课程和窗口还额外做了联合实验，用来检查交互效应。这里需要把候选的意思说清楚。`plateau` 是根据验证结果调学习率，`cosine` 是全程平滑退火，`phasewise` 则是在现有三段训练上使用 `phase_a = cosine`、`phase_b = cosine`、`phase_c = plateau`。`broad_to_recent` 的顺序是先广覆盖再收向近期，`recent_broad_recent` 则是先贴近近期、再补覆盖、最后再回近期。近期加权也不是抽象的“温和”“强烈”，而是具体比例：`mild` 在 `phase_a` 使用 `0.40:0.30:0.30` 的 `recent:mid:early` 配比，在 `phase_b` 使用 `0.75:0.25` 的 `recent:replay` 配比，在 `phase_c` 使用 `0.90:0.10`；`strong` 分别是 `0.60:0.25:0.15`、`0.90:0.10`、`0.98:0.02`；`two_stage` 分别是 `0.50:0.30:0.20`、`0.85:0.15`、`0.95:0.05`。时间窗口里的两个数字则分别表示 `phase_b` 和 `phase_c` 使用的近期窗口长度，例如 `12m_6m` 就是 `phase_b` 用最近 12 个月，`phase_c` 用最近 6 个月。

更具体地说，本阶段实际落地的是 6 组主 A/B 和 1 组可行性检查。`AB1` 是调度器对照，在固定 `curriculum = broad_to_recent`、`weight = two_stage`、`window = 24m_12m` 的前提下比较 `plateau`、`cosine` 和 `phasewise`，目的是先确定学习率调度该走“验证驱动”还是“预先设定轨迹”。`AB2` 是课程顺序对照，在固定 `scheduler = AB1 winner`、`weight = two_stage`、`window = 24m_12m` 的前提下比较 `broad_to_recent` 与 `recent_broad_recent`，核心问题是“先学全时期共性，再向近期收束”是否优于“先贴近期，再补覆盖，再回近期”。`AB3` 是近期加权强度对照，在固定 `scheduler` 和 `curriculum` 的前提下比较 `mild`、`strong` 和 `two_stage`，它回答的是近期漂移该压多重、压得多早。`AB4` 是时间窗口对照，在固定 `scheduler`、`curriculum` 和 `weight` 的前提下比较 `24m_12m`、`12m_6m` 与 `6m_6m`，检验近期数据窗口应保留多宽，才能既跟上时代又不至于样本过窄。

单因素实验之后又做了两组联合实验。`AB23 joint` 把课程顺序和近期加权放进同一张对照表，6 个候选分别是 `A1/A2/A3 = broad_to_recent × {mild,strong,two_stage}`，以及 `B1/B2/B3 = recent_broad_recent × {mild,strong,two_stage}`；这里的目的不是重新做一遍 `AB2` 和 `AB3`，而是检查“课程 winner 是否依赖于加权方案”，“加权 winner 是否依赖于课程顺序”。`AB234 joint` 则进一步把时间窗口也并入，形成 `2 × 3 × 2 = 12` 个候选：其中 `A/B` 表示两类课程，`1/2/3` 表示三类近期加权，`x/y` 分别表示 `24m_12m / 12m_6m`。历史上曾出现过阶段性候选 `A1y_broad_to_recent_mild_12m_6m`，它的名字来自 `A = broad_to_recent`、`1 = mild`、`y = 12m_6m`；但这个结果只属于旧代码状态下“固定 `scheduler` 后”的联合 winner，不能再视为当前全局主线结论。

后续复盘时我们确认，`AB1` 与 `AB234` 之间也是紧耦合的，不能再把“先做 `AB1` 得到 scheduler winner，再在这个 winner 上做 `AB234`”直接当成全局搜索。原因很直接：`scheduler` 决定模型以什么节奏适应数据分布，而 `curriculum / weight / window` 决定模型看到的数据分布本身；二者同时变化时，优劣可能翻转。因此新增了 `AB1234 joint` 作为最高优先级实验，它把四个维度放进同一张表中联合筛选，总规模是 `3 × 2 × 3 × 3 = 54` 个候选。命名方式也相应扩展为 `S_CWx` 形式：`P/C/W` 分别表示 `plateau / cosine / phasewise`，`A/B` 仍表示两类课程，`1/2/3` 仍表示三类近期加权，`x/y/z` 分别表示 `24m_12m / 12m_6m / 6m_6m`。例如 `P_A1y` 表示 `plateau + broad_to_recent + mild + 12m_6m`。只有在 `AB1234 joint` 中胜出的组合，才有资格被称为当前 `Stage 0.5` 的全局主线候选。

除训练协议本身外，还做了两组与选模相关的对照。`AB5` 不是训练实验，而是质量信号可行性检查：脚本会抽样日志文件名中的房间码，判断数据里是否存在足够稳定、可用于分层或加权的对局质量信号。如果日志层面根本拿不到稳定质量标签，这条路线就暂时阻塞，不能硬做。`AB6` 是 checkpoint 口径对照，它不比较两套训练协议，而是在同一条训练曲线内部比较 `best_loss`、`best_acc`、`best_rank` 和 `latest` 四类候选，解决“正式下游默认该接哪一个 checkpoint”这个工程问题。也正因为有 `AB6`，当前文档才会明确规定：主线默认接 `best_loss`，`best_acc` 只作为受控对照候选。

实验得到的几个关键信息很明确。第一，课程顺序、加权和时间窗口之间确实有交互，单因素 screening 得到的 winner 不能直接当主线结论。第二，早期多臂实验存在公平性问题，而且这个问题来自最开始一种很顺手但不适合严格对照的实现方式：不同 arm 各自生成自己的评估子集，phase 的随机种子也按课程顺序中的位置派生。前者会让不同 arm 落在难度不同的验证子集上，后者会把“课程差异”和“随机性差异”混在一起。后续修正就是两步：同一轮实验先固定 `eval split`，再让所有 arm 共用；phase 种子改成按固定的 `phase_name` 偏移量派生，而不是按顺序位置派生。修正这些问题并干净重跑以后，部分早期 winner 被推翻，说明实验方法本身会改变结果。第三，旧的 `AB1` 调度器结论只应视为历史证据，不能直接覆盖当前代码下的联合搜索结果；是否仍然成立，必须通过当前代码下的重跑复核。

最后，选模方式也必须一起修正。单纯的 `loss-first` 会偏向交叉熵略低但动作质量未必更好的模型，而把动作 `acc` 按先后顺序做字典序比较，则会让前面几个指标权重近似无限大，后面的指标基本失效。这在 `Stage 0.5` 里尤其危险，因为立直、吃碰杠、和牌这些动作大部分样本都很容易判断，真正拉开水平的是少量边界局面；如果只看普通 `acc`，这些微差会被大量容易负例淹没。当前代码已经改成三层口径：第一层是按阶段固定的主门槛，`P0 / P2 / formal` 使用 `full_recent_loss <= best + 0.003`，`P1` 则单独冻结到 `policy_quality`，也就是 `comparison_recent_loss = recent_policy_loss`，并在 `old_regression_policy_loss` 可用时再加 `0.0035` 护栏；第二层把动作侧指标折算成显式的 `action_quality_score`；第三层再用 `scenario_quality_score` 承接末盘、高压和微差场景。新的 `action_quality_score` 仍然围绕 `discard_nll`、`riichi_decision_balanced_bce`、`agari_decision_balanced_bce`、`chi_decision_balanced_bce`、`pon_decision_balanced_bce` 和 `kan_decision_balanced_bce` 这些原子动作指标展开，但现在补入了 `chi_exact_nll`，把“该不该吃”和“吃哪一种”拆开计分；对应权重改为 `0.45 / 0.18 / 0.18 / 0.04 / 0.03 / 0.07 / 0.05`。更关键的是，主分已经不再对原始 loss 直接做线性求和，而是先对每项应用 `loss / (1 + loss)` 的饱和化变换，再乘基于样本数的置信度 shrinkage。这样做的目的很直接：避免 `agari` 这类量纲较大的单项天然绑架总分，也避免稀有动作、稀有切片仅凭极少样本的偶然波动挤掉更稳的候选。这里仍然不用 `macro_action_acc`、`call_decision_acc`、`pass_decision_acc` 和 `ryukyoku_decision_acc` 做主分：前两者与原子动作口径重叠，`pass` 本质上已经体现在各个动作决策的负例里，`ryukyoku` 对牌力区分的贡献也远小于舍牌、立直和和牌。

在这个基础上，当前实现又额外加入了第三层“难场景比较”口径，用来承接立直麻将里最关键的微差决策。具体做法是：除了总体 `action_quality_score` 之外，还会计算一个 `scenario_quality_score`，并把主排序升级为 `selection_quality_score = action_quality_score + 0.20 * scenario_quality_score`。这里的关键变化有三层：第一，`scenario` 仍然直接进入第一排序层，但权重已经从过去偏重的 `0.5` 收敛到 `0.20`，不再压过动作主分，但也不至于弱到几乎失声；第二，`selection_tiebreak_key` 不再重复写 `scenario_quality_score -> action_quality_score`，也不再把各类单项 `acc` 当成尾部裁判，排序键已经收敛到 `selection_quality_score -> -recent_policy_loss -> -old_regression_policy_loss`；第三，各类 `acc` 仍然保留在摘要里，但角色变成诊断字段，而不是主判胜字段。这套分数已经不再只看早期那几个 `riichi late / south+ / all-last / threat / close-gap` 切片，而是按三层逐步扩展。第一层 `P0` 先把已经存在于验证链路里的关键切片直接纳入，包括 `riichi` 在 `late / south+ / all-last / threat / calm / role / rank / gap_close_4k` 下的表现，以及 `agari / kan` 在 `late / south+ / all-last / threat / gap_close_4k` 下的表现。第二层 `P1` 把原来没有真正进入选模口径的方向性与位次目标信息纳入：`up_gap_points / down_gap_points` 分离后的切片、`all_last_target_keep_first / chase_first / keep_above_fourth / escape_fourth` 这类南4位次目标切片、`single_threat / multi_threat` 威胁强度切片，以及 `opp_any/multi_tenpai`、`opp_any/multi_near_tenpai` 这类对手状态条件切片。第三层 `P2` 则把最贴近实战上限的舍牌质量直接纳入 `scenario_score`：不仅比较 `discard` 在 `all-last / threat / micro-gap / opponent-state` 下的专项 `nll`，还单独加入 `push_fold_core / push_fold_extreme` 两组场景分数，用来衡量末盘与高压场景里的弃和押退质量。和动作主分一样，这些切片现在也都走“饱和化 loss + 样本数 shrinkage”的计分方式，避免小样本切片在 selector 里获得不成比例的话语权。

这次扩展还有一个工程上的关键点：为了让 `CE-only`、`rank`、`danger`、`opponent_state` 这些 arm 在 `scenario_quality_score` 上可公平比较，验证链路现在会统一携带 `opponent-state` 标签，而不是只有打开 `opponent_state aux` 时才生成这部分标签。否则同一轮 A/B 里有的 arm 能算“对手状态条件切片”，有的 arm 完全没有这类指标，选模口径会天然失真。也正因为有了这条统一口径，当前的 `scenario_quality_score` 才真正开始覆盖“南4位次目标”“多家威胁强度”和“对手是否已接近听牌”这些过去容易漏掉、但对牌力上限非常关键的难点。

围绕“如何让模型真正利用对手的手切/摸切信号”这个问题，当前补了两条动作外的辅助监督线。第一条是 `OpponentStateAuxNet`：现有输入里其实已经显式编码了对手河的 `is_tedashi` 信息，但模型未必会主动把这些信号转化成对进张、牌速和听牌概率的判断。因此我们在不改主干结构的前提下，用全日志可见的隐藏真值监督主干去预测三家对手当前的 `shanten bucket = {0, 1, 2, 3+}` 和 `tenpai flag`。它的目标不是直接替代策略头，而是强迫编码器把河里“哪些牌是手切、哪些牌是摸切”这类线索变成对手状态表征。第二条是 `danger aux`：同样不改主干，只是在动作头旁边额外监督“当前每张可打牌是否会即时放铳、最大 ron 点数损失有多大、会放给哪一家”。`opponent_state` 更偏状态理解，`danger` 更偏把这种理解落实到当前舍牌风险。

这里的 `opponent_state_weight` 与 `danger_weight` 指的分别是这两条辅助监督在线总 loss 中的总权重。当前 `Stage 0.5` 的训练目标更准确地写成：`L = L_policy + w_rank(sample) * L_rank + opponent_state_weight * (opponent_shanten_weight * L_opp_shanten + opponent_tenpai_weight * L_opp_tenpai) + danger_weight * (danger_any_weight * L_danger_any + danger_value_weight * L_danger_value + danger_player_weight * L_danger_player)`。其中 `w_rank(sample)` 不再是全局常数，而是一个按场景动态变化的样本级权重；`opponent_state_weight` 决定模型在训练时有多大程度被要求去“看懂别人”；`danger_weight` 决定模型会不会真的把这种理解转化成对当前弃牌危险度的内部表示；而 `opponent_*` 与 `danger_*` 这些内部系数，再决定各自辅助线内部不同子任务的相对权重。当前默认配置把 `opponent_state_weight` 和 `danger_weight` 都设为 `0.0`，目的是先完成工程接入和指标体系重构，再通过受控 A/B 决定它们是否值得进入正式主线。

对名次辅助本身，当前也不再沿用“所有样本统一乘一个固定权重”的旧设计。原因是最终顺位标签天然是一个粗粒度、长时距信号：它适合让模型具备基本的顺位意识，但不应该在东场早巡就以和南场末盘同样的力度去约束每一个动作。现在 `Stage 0.5` 已经改成“低基线权重 × 阶段因子 × 相邻位差因子”的口径：默认 `base_weight = 0.03`，若样本处于 `south` 则乘 `south_factor = 1.59`，若处于 `all-last` 则再乘 `all_last_factor = 1.617`；除此以外，还会根据与相邻位的最近点差做连续加权，默认关注窗口 `gap_focus_points = 4000`，当前冻结默认不再额外加入 gap bonus，也就是 `gap_close_bonus = 0.0`，最后再用 `max_weight = 0.10` 做上限。这样一来，名次辅助仍然保留，但它的作用更接近“末盘顺位意识提醒”，而不是一个在整局所有样本上都占过大比重的粗暴全局监督。

### 辅助头权重的来源与证据等级

这里把当前实现里几类“权重”拆开记清楚，避免把 `启发式模板`、`统计支持的 turn weighting` 和 `P1 calibration` 产物混成一类。

**1. `rank` 的样本权重**

当前代码里的 `rank` 实际权重更准确地写成：

`w_rank(sample) = base_weight × turn_bucket_weight × south_factor? × all_last_factor? × gap_factor`

其中：

- `south_factor = 1.59`、`all_last_factor = 1.617`、`gap_focus_points = 4000`、`gap_close_bonus = 0.0` 是 `2026-03-25` 起冻结的 **rank 形状模板**，来源是“18k 统计缩网格 + `A2y` 主线微型 AB + `P1 solo` 真实选模口径复核”的组合结论。旧的 `1.4 / 1.8 / 4000 / 1.5` 仍保留为有麻将语义的历史启发式基线，但不再是当前默认。
- `turn_bucket_weight` 则是后来补进来的另一层权重，默认 `early / mid / late = 1.00 / 1.05 / 1.15`，这一层才是统计支持的。对应证据记录在 `docs/research/stage05/p1-aux-adjustment-2026-03-22.md`：使用 `1080` 局、`707,930` 个监督状态的本地样本，加上公开手速资料，把巡目分成 `0-4 / 5-11 / 12+`。其中 `rank_match_rate = 0.4719 / 0.4676 / 0.4643`，说明 `rank` 在线程内始终相关，但随巡目变化并没有像 `opp` / `danger` 那么陡，所以最后只给了一个很轻的后巡上调，而不是大幅重权。
- `base_weight = 0.03` 与 `max_weight = 0.10` 当前也属于工程侧的保守约束，不是 calibration 产物。它们的作用是保证 `rank` 仍然是轻辅助，而不是重新变回一个会压住 `policy CE` 的粗暴主损失。

**2. `opponent_state` 的内部组合方式**

代码里的 `opponent_state` 先分别对三家对手计算两类监督：

- `shanten bucket`：每家一个 `4` 分类 CE
- `tenpai flag`：每家一个 `2` 分类 CE

然后先对三家分别求平均，再按下面的式子合起来：

`L_opp_raw = opponent_shanten_weight × mean_3p(L_shanten) + opponent_tenpai_weight × mean_3p(L_tenpai)`

当前冻结默认是：

- `opponent_shanten_weight = 0.8506568408`
- `opponent_tenpai_weight = 1.1493431592`

也就是说，**按系数语义看默认已经轻微偏向 `tenpai`**。这里刻意保留了旧 `1:1` 方案“系数和为 `2.0`”的尺度语义，因此不是直接把归一化比值 `0.4253 / 0.5747` 塞进配置，而是使用同一比值按和为 `2.0` 重新缩放后的版本。这个冻结默认来自 `A2y` 主线下按 `P1 solo` 真实选模口径复核后的 `HYBRID_GRAD` winner；旧的 `1:1` 现在只作为历史对照。

`opponent_state` 外层还有两类权重：

- **全局权重 `opponent_state_weight`**
  - 默认主配置里是 `0.0`，表示默认并不打开这条辅助线。
  - `P1` 搜索里如果需要打开它，非零权重不是拍脑袋给的，而是来自 `P1 calibration`：`opponent_state_weight = budget_ratio × opp_weight_per_budget_unit`。历史上这里曾记录过 `2026-03-20` 那轮更早期的 `0.064`，但当前默认已改为沿用 `2026-03-25 post-shape calibration` 的单头映射，见 `docs/agent/current-plan.md`。
- **巡目权重 `opponent_turn_weighting`**
  - 默认是 `0.20 / 1.00 / 1.60`。
  - 这一层确实来自统计，不是 calibration。证据同样在 `docs/research/stage05/p1-aux-adjustment-2026-03-22.md`：本地样本里 `opp_any_tenpai_rate = 0.0342 / 0.4078 / 0.8303`、`opp_any_near_rate = 0.2772 / 0.8721 / 0.9817`，说明“看懂别人”在早巡信息量有限，中后巡才快速变得关键，因此默认配置有明显的后巡抬升。

**3. `danger` 的内部组合方式**

代码里的 `danger` 有三路子任务：

- `danger_any`
  - 当前每张可打牌是否会即时放铳
- `danger_value`
  - 如果会放铳，最大 ron 点数损失有多大
- `danger_player`
  - 会放给哪一家

对应的原始混合式是：

`L_danger_raw = danger_any_weight × L_any + danger_value_weight × L_value + danger_player_weight × L_player`

当前冻结默认配置写成：

- `danger_any_weight = 0.0904217947`
- `danger_value_weight = 0.8180402859`
- `danger_player_weight = 0.0915379194`

代码里还会把这三个系数再归一化到和为 `1`。这意味着它的真实语义已经改成：**主信号是“放铳价值损失”，`any` 与 `player` 只保留必要的结构约束权重。** 这组默认值来自 `18k` 统计结果在 `A2y` 主线下经微型 AB 复核后的 `18K_STAT` winner；旧的 `0.45 / 0.35 / 0.20` 现在只保留为历史启发式对照。

`danger` 外层同样有三类权重需要分开看：

- **全局权重 `danger_weight`**
  - 默认主配置里也是 `0.0`，表示默认关闭。
  - `P1` 搜索里如果需要打开它，非零权重同样来自 `P1 calibration`：`danger_weight = budget_ratio × danger_weight_per_budget_unit`。历史上这里曾记录过 `2026-03-20` 的 `0.144`，但当前默认已改为沿用 `2026-03-25 post-shape calibration` 的单头映射，见 `docs/agent/current-plan.md`。
- **巡目权重 `danger_turn_weighting`**
  - 默认是 `0.05 / 1.00 / 2.50`。
  - 这一层也来自统计，不是 calibration。对应证据同样写在 `docs/research/stage05/p1-aux-adjustment-2026-03-22.md`：本地样本里 `danger_state_has_any_rate = 0.0100 / 0.1650 / 0.4117`，`danger_positive_discard_rate_given_valid = 0.0014 / 0.0291 / 0.0909`。也就是说，早巡“立即放铳风险”几乎接近零，因此早巡默认大幅压低，后巡显著抬高。
- **`danger_ramp_steps`**
  - 这不是另一种“统计权重”，而是一个训练稳定性开关。实现上它会把整条 `danger` 辅助线的外层系数从 `0` 线性升到目标 `danger_weight`：`ramp = min(step / danger_ramp_steps, 1.0)`。
  - 目的很简单：`danger` 标签天然更稀疏、更偏后巡、也更容易在训练早期给 trunk 施加尖锐梯度，所以可以先让主干和主策略头站稳，再逐步把 `danger` 压上去。
  - 因而 `danger_ramp_steps` 的来源是工程稳定性考量，不是统计脚本，也不是 calibration。当前主配置默认是 `0`；`P1` fidelity 搜索里为了排除“刚开 danger 时前几百步过猛”的噪声，候选覆盖默认使用了 `1000` 步线性 ramp。

总结起来：

- `rank` 的 `south / all-last / gap` 模板：当前默认已冻结为 `18K_ROUND_ONLY`，来源是 `18k` 统计缩网格后再经 `A2y` 微型 AB 与 `P1 solo` 选模口径复核。
- `rank / opp / danger` 的 `turn weighting`：来自 `2026-03-22` 那轮公开资料 + 本地样本统计。
- `opp / danger` 的搜索用全局非零 head weight：来自 `P1 calibration` 的统一预算映射。当前正式默认是沿用 `2026-03-25` post-shape single-head baseline，再在代表协议 `A2y` 上补跑 `combo-only calibration`。
- `opp` 与 `danger` 的内部配比已在 `2026-03-25` 冻结：`opp = HYBRID_GRAD`、`danger = 18K_STAT`；它们不再是当前主线里的开放形状轴，后续默认只再搜索总权重。
- `danger_ramp_steps`：纯工程稳定性旋钮，不是统计量，也不是“旧 calibration”的一部分。

## 待做 A/B 执行表

本节直接定义 `Stage 0.5` 后续 A/B 的执行协议，不依赖前文上下文。目标只有一个：在固定工程环境下，先选出最优监督预训练协议，再选出最合适的辅助监督预算、checkpoint 口径和下游交接方式。后面如果有人只看这一节，也应当能直接照表执行。

### 固定前提

- 所有当前默认 `Stage 0.5` A/B 都使用同一套基础数据管线：`batch_size = 1024`、`num_workers = 4`、`file_batch_size = 10`、`prefetch_factor = 3`
- 所有 scheduler 都统一使用 warmup 起手；warmup 是固定前提，不是搜索轴
- 同一轮多臂实验必须共用同一套 `monitor_recent / full_recent / old_regression` 评估切分
- phase 随机种子必须按固定 `phase_name` 偏移派生，不能按课程顺序位置派生
- 每个 arm 都必须使用全新输出目录，不能复用旧日志目录
- 选模规则按阶段拆开固定：
  - `P0 / P2 / formal checkpoint`：先按 `full_recent_loss <= best_loss + 0.003` 进入 eligible 集，再按 `selection_tiebreak_key` 排序，顺序为 `selection_quality_score -> -recent_loss -> -old_regression_loss`
  - `P1 family / solo / pairwise / joint refine`：不再使用 `full_recent_loss` 做主门槛；唯一有效口径固定为 `docs/status/p1-selection-canonical.md`，即用 `comparison_recent_loss = recent_policy_loss` 过 `policy_quality` 门槛，再按同一 `selection_tiebreak_key` 比较；`full_recent_loss` 只保留为 aux tax / 总 loss 诊断字段
  - 任何出现 `NaN`、缺失关键验证指标、训练提前崩溃、checkpoint 不完整的 arm 直接淘汰

### 总览

| 层级 | 项目 | 目标 | 依赖关系 |
| --- | --- | --- | --- |
| P0 | `AB1234 joint` | 选出主训练协议 | 无；这是最先执行的联合搜索 |
| P1 | `CE-only -> SoloAuxGate -> Pairwise -> JointRefine` | 在统一有效预算轴上筛选 `rank / opponent-state / danger` 组合 | 依赖 `P0` 产出的 `top4` 协议种子 |
| P2 | `selector weight`、`AB6 checkpoint` | 固化选模口径与默认 checkpoint | 依赖 `P0/P1` 的稳定训练结果 |
| P3 | `Stage 1 transfer` | 检查监督阶段 winner 是否真能转化为下游强度 | 依赖 `P2` 之前的所有结论 |

### P0：`AB1234 joint`

`P0` 是主协议联合搜索。调度器、课程顺序、近期加权和时间窗口四个轴同时变化，不能拆开顺序做。实际候选总数为 `3 × 2 × 3 × 3 = 54` 个 arm，其中 warmup 对所有 arm 都固定开启，不进入命名。

**命名规则**

- 完整格式：`<S>_<C><W><T>_<scheduler>_<curriculum>_<weight>_<window>`
- `S`：scheduler 前缀，`P = plateau`，`C = cosine`，`W = phasewise`
- `C`：课程前缀，`A = broad_to_recent`，`B = recent_broad_recent`
- `W`：近期加权前缀，`1 = mild`，`2 = strong`，`3 = two_stage`
- `T`：时间窗口前缀，`x = 24m_12m`，`y = 12m_6m`，`z = 6m_6m`

**profile 含义**

- `phase_a / phase_b / phase_c` 分别表示：广覆盖预训练阶段、近期适应阶段、最终近期收束阶段
- `plateau`：`phase_a / phase_b / phase_c` 全部使用 `ReduceLROnPlateau`
- `cosine`：`phase_a / phase_b / phase_c` 全部使用 `warmup + cosine`
- `phasewise`：`phase_a = warmup + cosine`，`phase_b = warmup + cosine`，`phase_c = warmup + plateau`
- 因此当前执行表里没有单独再列一个 `warm+cos+plateau` arm；在现有三阶段实现里，这个含义已经被 `phasewise` 完整覆盖
- `broad_to_recent`：阶段顺序固定为 `phase_a -> phase_b -> phase_c`
- `recent_broad_recent`：阶段顺序固定为 `phase_b -> phase_a -> phase_c`
- `mild`：`phase_a` 的 `recent:mid:early = 0.40:0.30:0.30`，`phase_b` 的 `recent:replay = 0.75:0.25`，`phase_c` 的 `recent:replay = 0.90:0.10`
- `strong`：`phase_a` 的 `recent:mid:early = 0.60:0.25:0.15`，`phase_b` 的 `recent:replay = 0.90:0.10`，`phase_c` 的 `recent:replay = 0.98:0.02`
- `two_stage`：`phase_a` 的 `recent:mid:early = 0.50:0.30:0.20`，`phase_b` 的 `recent:replay = 0.85:0.15`，`phase_c` 的 `recent:replay = 0.95:0.05`
- `24m_12m`：`phase_b` 使用最近 `24` 个月，`phase_c` 使用最近 `12` 个月
- `12m_6m`：`phase_b` 使用最近 `12` 个月，`phase_c` 使用最近 `6` 个月
- `6m_6m`：`phase_b` 与 `phase_c` 都只使用最近 `6` 个月

**实际候选名**

- `P_A1x_plateau_broad_to_recent_mild_24m_12m`
- `P_A1y_plateau_broad_to_recent_mild_12m_6m`
- `P_A1z_plateau_broad_to_recent_mild_6m_6m`
- `P_A2x_plateau_broad_to_recent_strong_24m_12m`
- `P_A2y_plateau_broad_to_recent_strong_12m_6m`
- `P_A2z_plateau_broad_to_recent_strong_6m_6m`
- `P_A3x_plateau_broad_to_recent_two_stage_24m_12m`
- `P_A3y_plateau_broad_to_recent_two_stage_12m_6m`
- `P_A3z_plateau_broad_to_recent_two_stage_6m_6m`
- `P_B1x_plateau_recent_broad_recent_mild_24m_12m`
- `P_B1y_plateau_recent_broad_recent_mild_12m_6m`
- `P_B1z_plateau_recent_broad_recent_mild_6m_6m`
- `P_B2x_plateau_recent_broad_recent_strong_24m_12m`
- `P_B2y_plateau_recent_broad_recent_strong_12m_6m`
- `P_B2z_plateau_recent_broad_recent_strong_6m_6m`
- `P_B3x_plateau_recent_broad_recent_two_stage_24m_12m`
- `P_B3y_plateau_recent_broad_recent_two_stage_12m_6m`
- `P_B3z_plateau_recent_broad_recent_two_stage_6m_6m`
- `C_A1x_cosine_broad_to_recent_mild_24m_12m`
- `C_A1y_cosine_broad_to_recent_mild_12m_6m`
- `C_A1z_cosine_broad_to_recent_mild_6m_6m`
- `C_A2x_cosine_broad_to_recent_strong_24m_12m`
- `C_A2y_cosine_broad_to_recent_strong_12m_6m`
- `C_A2z_cosine_broad_to_recent_strong_6m_6m`
- `C_A3x_cosine_broad_to_recent_two_stage_24m_12m`
- `C_A3y_cosine_broad_to_recent_two_stage_12m_6m`
- `C_A3z_cosine_broad_to_recent_two_stage_6m_6m`
- `C_B1x_cosine_recent_broad_recent_mild_24m_12m`
- `C_B1y_cosine_recent_broad_recent_mild_12m_6m`
- `C_B1z_cosine_recent_broad_recent_mild_6m_6m`
- `C_B2x_cosine_recent_broad_recent_strong_24m_12m`
- `C_B2y_cosine_recent_broad_recent_strong_12m_6m`
- `C_B2z_cosine_recent_broad_recent_strong_6m_6m`
- `C_B3x_cosine_recent_broad_recent_two_stage_24m_12m`
- `C_B3y_cosine_recent_broad_recent_two_stage_12m_6m`
- `C_B3z_cosine_recent_broad_recent_two_stage_6m_6m`
- `W_A1x_phasewise_broad_to_recent_mild_24m_12m`
- `W_A1y_phasewise_broad_to_recent_mild_12m_6m`
- `W_A1z_phasewise_broad_to_recent_mild_6m_6m`
- `W_A2x_phasewise_broad_to_recent_strong_24m_12m`
- `W_A2y_phasewise_broad_to_recent_strong_12m_6m`
- `W_A2z_phasewise_broad_to_recent_strong_6m_6m`
- `W_A3x_phasewise_broad_to_recent_two_stage_24m_12m`
- `W_A3y_phasewise_broad_to_recent_two_stage_12m_6m`
- `W_A3z_phasewise_broad_to_recent_two_stage_6m_6m`
- `W_B1x_phasewise_recent_broad_recent_mild_24m_12m`
- `W_B1y_phasewise_recent_broad_recent_mild_12m_6m`
- `W_B1z_phasewise_recent_broad_recent_mild_6m_6m`
- `W_B2x_phasewise_recent_broad_recent_strong_24m_12m`
- `W_B2y_phasewise_recent_broad_recent_strong_12m_6m`
- `W_B2z_phasewise_recent_broad_recent_strong_6m_6m`
- `W_B3x_phasewise_recent_broad_recent_two_stage_24m_12m`
- `W_B3y_phasewise_recent_broad_recent_two_stage_12m_6m`
- `W_B3z_phasewise_recent_broad_recent_two_stage_6m_6m`

**执行清单**

1. `Round 0`：54 个 arm 全部运行，`step_scale = 0.5`
2. `Round 0` 结束后，用下面的统一排序键取 `top 18`
   - `valid_flag`：有效 run 记为 `1`，无效 run 记为 `0`
   - `eligible_flag`：`full_recent_loss <= best_loss + 0.003` 记为 `1`，否则为 `0`
   - `priority`：`selection_tiebreak_key(...)`
   - 最终排序键：`(valid_flag, eligible_flag, priority, -full_recent_loss)`
3. `Round 1`：`top 18` 运行，`step_scale = 1.0`，保留 `top 6`
4. `Round 2`：`top 6` 运行，`step_scale = 2.0`，做深复验，并从这里产出后续真正使用的 `top4` 协议种子
5. `Round 3`：`top 2` 运行，`step_scale = 4.0`，只用于决出 `round3 winner / runner-up`
6. 每轮都写出独立 `summary.json`，不得覆盖上一轮结果
7. 当前口径下，真正送入后续 `P1 / Stage 1` 的不是旧的 `top3`，而是 `P0 round2` 的 `top4` 协议种子
8. 当前官方 `top4` 为：
   - `C_B3z_cosine_recent_broad_recent_two_stage_6m_6m`
   - `C_B2z_cosine_recent_broad_recent_strong_6m_6m`
   - `C_A3x_cosine_broad_to_recent_two_stage_24m_12m`
   - `C_A1x_cosine_broad_to_recent_mild_24m_12m`

### P1：统一预算口径的 `CE-only -> SoloAuxGate -> Pairwise -> JointRefine`

`P1` 的目标仍然不是重选主协议，而是在 `P0 round2` 的 `top4` 协议种子上，为新 `Stage 1` 选出最合适的辅助监督组合。当前实现不再直接把 `rank_scale`、`opponent_state_weight`、`danger_weight` 这三个原始参数并排暴力搜索，而是先把它们映射到同一个“有效辅助预算”坐标系，再做分层筛选。

之所以必须这么改，是因为这三个参数的语义本来就不一致。`rank` 不是一个普通常数权重，而是 `rank_scale × w_rank(sample)`：其中 `w_rank(sample)` 已经内含 `south / all-last / 点差接近` 等场景模板，因此 `rank_scale = 1.0` 的含义是“整套 rank 模板原样启用”，并不等于“辅助权重 1”。相反，`opponent_state_weight` 和 `danger_weight` 是直接乘在对应辅助损失上的全局常数。如果直接拿这三个原始数值比较，会把“参数写法不同”误当成“强度不同”，这是不公平的。

因此，新的 `P1` 先做一个短的 `calibration` 阶段，把三条辅助线统一映射到“名义上相同的有效预算”。这里的“统一”现在分成两层：

- `loss-based calibration`：继续看三条辅助各自真实进入总 loss 的加权贡献
- `grad-based calibration`：额外看三条辅助对共享表示 `phi` 的梯度 RMS，直接衡量它们究竟给 trunk 施加了多大训练压力
- 当前默认不是只信其中一条，而是把两者先各自映射成 `weight per budget unit`，再取 `geomean hybrid`
- 这样做的目的不是让 calibration 更复杂，而是避免出现“表面 loss 很公平，但某条辅助通过 head 结构或任务尺度差异，实际上对 trunk 施压更大/更小”的假公平

训练目标仍然写作：

- `L = L_policy + rank_scale * w_rank(sample) * L_rank + opponent_state_weight * L_opp + danger_weight * L_danger`

但真正参与搜索和比较的，不再是这三个原始系数，而是它们各自对应的 `budget_ratio`。这里的 `budget_ratio = 1.0` 表示“与当前 `rank` 基线大致同量级的有效辅助预算”；`0.5 / 1.5 / 2.0` 表示这个统一预算轴上的更轻或更重配置。

**固定的 `rank` 形状模板**

- `south_factor = 1.59`
- `all_last_factor = 1.617`
- `gap_focus_points = 4000`
- `gap_close_bonus = 0.0`
- `base_weight` 与 `max_weight` 随 `rank_scale` 同步缩放，但模板形状本身不作为搜索轴

**Calibration：把三条辅助映射到同一预算轴**

- 对 `P0 round2 top4` 的每条主协议，各跑四个短 probe：
  - `rank_only = (rank_scale=1.0, opp=0, danger=0)`
  - `opp_probe = (rank_scale=1.0, opp=0.06, danger=0)`
  - `danger_probe = (rank_scale=1.0, opp=0, danger=0.06)`
  - `both_probe = (rank_scale=1.0, opp=0.06, danger=0.06)`
- 这一步不负责选 winner，只负责测量三条辅助的量纲
- 当前实现记录两组核心量：
- `loss` 轴：
  - `rank_effective_base = median(rank_aux_weight_mean × rank_aux_raw_loss)`
  - `opp_effective_per_unit = median(opponent_aux_loss / 0.06)`
  - `danger_effective_per_unit = median(danger_aux_loss / 0.06)`
  - `joint_combo_factor = median((opp+danger)_effective / (opp_effective + danger_effective))`
- `grad` 轴：
  - `rank_grad_effective_base = median(rank_phi_grad_rms)`
  - `opp_grad_effective_per_unit = median(opponent_phi_grad_rms / 0.06)`
  - `danger_grad_effective_per_unit = median(danger_phi_grad_rms / 0.06)`
  - `joint_combo_factor_grad = median(opp_danger_phi_combo_factor)`
- 其中 `phi_grad_rms` 指各辅助损失对共享表示 `phi` 的梯度均方根；它不看 head 自己学得多快，而只看 trunk 真实感受到的监督强度
- 同时还会记录 `aux ↔ policy`、`opp ↔ danger` 的梯度余弦，主要用于解释“为什么某条辅助单独有益、组合后却开始互相稀释或冲突”
- 随后把统一预算轴扩成：`budget_ratio ∈ {0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5}`。`SoloAuxGate` 不再用同一组粗糙挡位硬套三类辅助，而是按 family 选取相同数量的**有效**采样点
- 映射方式是：
  - `rank`：`rank_scale = budget_ratio`
  - `opp`：`opponent_state_weight = budget_ratio × opp_weight_per_budget_unit`
  - `danger`：`danger_weight = budget_ratio × danger_weight_per_budget_unit`
- 这里的 `opp_weight_per_budget_unit / danger_weight_per_budget_unit` 不再是单条 `loss` 轴直接给出的数字，而是：
  - 先算 `loss-based weight per budget unit`
  - 再算 `grad-based weight per budget unit`
  - 最后取两者的 `geomean hybrid`
- 如果同时开启 `opp + danger`，就再乘一次基于 `both_probe` 的 `loss/grad combo factor hybrid` 组合补偿，避免把两条部分重叠的辅助简单当成独立旋钮
- 如果 calibration 失效，就退回保守默认映射，而不是让整轮实验直接失效

`2026-03-20` 这轮 `P1 calibration` 是历史阶段结果，不再是当前默认单头映射来源。它当时按旧阶段实现完整跑完，得到 `mapping_mode = hybrid_loss_grad_geomean`，`rank_effective_base = 0.068604`，`opp_weight_per_budget_unit = 0.064`，`danger_weight_per_budget_unit = 0.144`，`joint_combo_factor = 0.884`。当前默认已切换到沿用 `2026-03-25 post-shape calibration` 的单头映射；这里保留此段仅作历史背景。

更重要的是，这轮 probe 在四条 `P0 round2 top4` 主协议上方向一致：`rank_only` 总是最轻，`danger_probe` 明显重于 `rank_only` 但显著轻于 `opp_probe`，`both_probe` 则稳定最重。这说明两件事。第一，旧 `opp=0.06 / danger=0.06` 的对照只能回答“高权重下谁更容易伤主目标”，不能直接回答“谁更强”；因为它们根本不在同一个有效预算轴上。第二，`opp + danger` 的确存在重叠，后续联用时必须乘 `combo factor` 做折扣，不能把两个单辅助赢家的原始权重直接相加。

**Step 1：`CE-only` 与 `SoloAuxGate`**

- 每条主协议先明确跑一个 `CE-only` 锚点
- 然后分别只开一个辅助：
  - `CE + rank`
  - `CE + opponent_state`
  - `CE + danger`
- `SoloAuxGate` 的公平版执行网格按 family 裁剪为：
  - `rank`：`{0.25, 0.5, 1.0, 1.5}`
  - `opponent_state`：`{0.25, 0.5, 0.75, 1.0}`
  - `danger`：`{0.25, 0.5, 0.75, 1.25}`
- 这样设计的原因不是主观偏爱某条辅助，而是为了让三类辅助都拥有相同数量的**有效**预算点：
  - `rank` 当前更需要补低预算段，检查“弱 rank”是否比“强 rank”更稳
  - `opponent_state` 在 calibration 中已经证明 `0.06` 接近 `1.0 budget unit`，因此高预算尾部的信息增益低于低预算细化
  - `danger` 由于 `danger_weight_per_budget_unit = 0.144` 且全局上限是 `0.18`，临界点恰好在 `budget_ratio = 1.25`；再往上只会产生 clamp 重复点
- 在当前这轮 calibration 结果下，`SoloAuxGate` 的正式落地映射固定为：
  - `rank`：`rank_scale = budget_ratio`
  - `opponent_state`：`opp_weight = 0.064 × budget_ratio`
  - `danger`：`danger_weight = 0.144 × budget_ratio`
- 这一步的职责不是直接决出最终组合，而是回答三个问题：
  - `rank` 单独开时有没有净收益
  - `opponent_state` 单独开时有没有净收益
  - `danger` 单独开时有没有净收益
- 每条主协议最终只保留：
  - 一个 `CE-only`
  - 每个辅助家族各自最强、且通过当前选模门槛的单辅助 survivor

**Step 2：`Pairwise`**

- `Pairwise` 只围绕 `SoloAuxGate` 的幸存者构造，不再重新开全空间
- 候选组合只有三类：
  - `rank + opponent_state`
  - `rank + danger`
  - `opponent_state + danger`
- 这一步的目标不是“调最优比例”，而是先确认互补性：哪些辅助单独有帮助，但组合后反而互相稀释；哪些辅助虽然单看一般，但组合后能形成净增益
- `Pairwise` 轮会把 `CE-only`、单辅助幸存者和双辅助组合一起重排，确保最终送进下一轮的是“在同一预算与同一多 seed 口径下”真正更强的中心点

**Step 3：`JointRefine`**

- `JointRefine` 只围绕 `Pairwise` 每协议 `top2` 中心点做局部细化
- 当前细化步长统一改成 `Δbudget_ratio = 0.25`
- 细化不是再回到原始 `rank_scale / opp_weight / danger_weight` 轴，而是继续在统一预算轴附近做局部立方体扰动
- 这样做的好处是：局部细调仍然保持三个辅助的“预算语义”一致，不会出现 `rank` 调 `0.1`、`opp/danger` 调 `0.005` 这种本质上不在同一量纲上的微调

**最终产出**

- `JointRefine` 结束后，每条主协议保留一个 `P1 winner`
- 最后再做一次跨协议 `final_compare`
- `P1` 的最终结论不再是“哪组原始权重数字好看”，而是：
  - 哪些辅助单独值得保留
  - 哪些辅助组合存在净增益
  - 在统一预算口径下，最终哪套 `rank / opponent_state / danger` 组合最稳

这个改动的关键价值，不是让 `P1` 变得更复杂，而是让比较口径终于公平：我们比较的是三条辅助真正给 trunk 施加了多大的监督预算，而不是比较三种语义不同的参数写法。

- `calibration` 本身的职责是定量纲，不是直接判 winner，因此它对多 seed 的依赖低于后面的 winner 轮次
- 当前默认做法是：先用单 seed + 多 batch probe 拿到 `loss/grad` 两条轴的稳健中位数；如果不同协议间的定标值波动异常大，或 `loss` 轴与 `grad` 轴给出的辅助排序明显冲突，再把 calibration 升级为 `3-seed` 复验
- 对 `Round 1` 的每协议 `top 3` 做更长预算复验；`Round 2` 默认使用比前两轮更高的多 seed 强度，专门承担最终稳定性确认的职责
- 每条主协议最终只保留 `1` 个 `P1 winner`
- 最后再做一次 `P1 final_compare`，只在这四个“每协议 winner”之间比较

**为什么旧三段式被删除**

- 旧 `P1a/P1b/P1c` 的根本问题不是实现麻烦，而是统计口径不公平
- 当 `rank_aux_weight_mean` 本身只有一个较小的动态均值时，先定 `total_budget` 再拆比例，等价于默认假设“新辅助主要通过替换 rank 生效”
- 这会系统性低估“保住 rank，再额外叠加 `opp / danger`”的真实上限
- 因此当前代码与文档都不再保留旧 `P1a/P1b/P1c` 方案作为主线

### P2：选模与 checkpoint

`P2` 不负责改变训练轨迹，只负责把 `P0/P1` 已经跑出来的轨迹用更清晰的口径选出来。

**`selector weight` 候选名**

- `S0_default = 0.45 / 0.18 / 0.18 / 0.07 / 0.07 / 0.05`
- `S1_riichi_heavy = 0.40 / 0.24 / 0.18 / 0.06 / 0.07 / 0.05`
- `S2_call_heavy = 0.40 / 0.18 / 0.16 / 0.10 / 0.10 / 0.06`
- `S3_discard_heavy = 0.52 / 0.16 / 0.16 / 0.06 / 0.06 / 0.04`

**`selector weight` 执行清单**

1. 冻结 `P0/P1` 的候选 run 池，不允许边跑边改 selector
2. 用 `S0/S1/S2/S3` 对同一批 run 离线重打分
3. 去重：如果两个 selector 选中了同一个 checkpoint，只保留一个
4. 最终保留至多 3 个不同 checkpoint，送入 `P3`

**`AB6 checkpoint` 候选名**

- `K1_best_loss`
- `K2_best_acc`
- `K3_best_rank`
- `K4_latest`

**`AB6 checkpoint` 执行清单**

1. 在每条入围训练曲线上都比较 `K1/K2/K3/K4`
2. 仍按当前固定 selector 规则选每条曲线的默认 checkpoint
3. 不允许跨训练曲线直接比较 `K1/K2/K3/K4`；先在曲线内决出默认 checkpoint，再进入 `P3`

### P3：`Stage 1 transfer`

`P3` 的目的不是再看监督指标，而是验证监督阶段 winner 是否真的能转化为新 `Stage 1`（`Oracle Dropout Supervised Refinement`）的下游强度。

**候选名**

- `T1_best_loss`
- `T2_best_acc`
- `T3_best_rank`
- 如有必要，再补 `T4_alt_selector`

**执行清单**

1. 只使用 `P2` 产出的去重后候选
2. 所有 `Stage 1` 对照必须使用同一份 `GRP`、同一份 Oracle 配置、同一份 `oracle-dropout supervised refinement` 训练预算；历史 `Oracle AWR` 只作为 `git` 里的旧方案背景，不再是当前仓库可直接启动的分支
3. 先做小预算迁移对照，保留 `top 2`
4. 再做长预算迁移对照，最终 winner 才能进入正式主线

### 最终执行顺序

1. 跑 `P0`，从 54 臂中产出 `round3 winner / runner-up`，并从 `round2` 产出当前官方 `top4` 协议种子
2. 先对这 `top4` 条主协议跑 `P1 calibration`
3. 在统一有效预算轴上执行 `P1 SoloAuxGate`，得到每协议的 `CE-only + 单辅助 survivor`
4. 基于单辅助 survivor 执行 `P1 Pairwise`
5. 围绕每协议 `top2` 中心做 `P1 JointRefine`
6. 冻结所有候选 run，执行 `P2`
7. 把 `P2` 去重后的候选送去 `P3`

### 保真版计划表

下表是当前准备执行的 `Stage 0.5` 保真版计划。它不是方向性草案，而是可直接照着跑的排程表。所有耗时都按同一口径估算：单卡串行、沿用当前 `AB runner` 默认训练池与验证预算、训练吞吐大致保持在你机器上已经见过的 `5~6 batch/s` 区间。这里给的是区间，不是单点，因为不同 arm 的验证触发次数和早停行为会有波动。

| 阶段 | 轮次 | arm 数 | 每轮预算 | 淘汰规则 | 产出 | 估计耗时 |
| --- | --- | --- | --- | --- | --- | --- |
| `P0` | `Round 0` | `54` | `step_scale = 0.5` | 按固定 selector 取 `top 18` | 54 臂首轮 `summary.json` | `4.5 ~ 6 h` |
| `P0` | `Round 1` | `18` | `step_scale = 1.0` | 取 `top 6` | 18 臂复验 `summary.json` | `2.5 ~ 3.5 h` |
| `P0` | `Round 2` | `6` | `step_scale = 2.0` | 深复验并确定 `top4` 协议种子 | 6 臂深复验 `summary.json` | `1.5 ~ 2.5 h` |
| `P0` | `Round 3` | `2` | `step_scale = 4.0` | 仅决出 `winner / runner-up` | 最终主协议排序 | `1.5 ~ 2.0 h` |
| `P1` | `Calibration` | `4 × 4 = 16` | `step_scale = 0.5` | 不淘汰主协议；只做统一预算定标 | calibration 报告 | `2 ~ 4 h` |
| `P1` | `SoloAuxGate` | `4 × (1 + 4 + 4 + 4) = 52` | `step_scale = 0.5` | 每条主协议保留 `CE-only` 与单辅助 survivor | 单辅助结论 | `4.5 ~ 7 h` |
| `P1` | `Pairwise` | `最多 4 × (1 + 3 + 3) = 28` | `step_scale = 1.5` | 每条主协议筛出 `top2` 联调中心 | 双辅助结论 | `5 ~ 9 h` |
| `P1` | `JointRefine` | `约 4 × (4 + 2 × 11) = 104` | `step_scale = 3.0` | 每条主协议各取 `P1 winner` | 最终 `P1` 候选 | `14 ~ 24 h` |
| `P2` | 离线 | `最多 4` 条训练曲线 | 无重训 | `selector weight` 去重后保留至多 `4` 个 checkpoint | `P3` 输入候选 | `1 ~ 3 h` |
| `P3` | `Shortlist` | `2 ~ 4` | 小预算 `Stage 1` 迁移 | 取 `top 2` | 下游 shortlist | `12 ~ 18 h` |
| `P3` | `Final` | `2` | 长预算 `Stage 1` 迁移 | 选最终 handoff winner | `Stage 1` 默认初始化 checkpoint | `30 ~ 52 h` |

**阶段汇总**

| 阶段 | 估计耗时 |
| --- | --- |
| `P0` 合计 | `10 ~ 14 h` |
| `P1` 合计 | `25.5 ~ 44 h` |
| `P2` 合计 | `1 ~ 3 h` |
| `P3` 合计 | `42 ~ 70 h` |
| 总计 | `105 ~ 163 h` |

**执行备注**

- `P0` 的下游入口已经不是旧的 `winner / runner-up / third`，而是 `round2 top4`；`round3` 只负责给主协议排序
- `P1 calibration` 只负责把三条辅助映射到统一预算轴，不直接决定 winner
- `P1` 的四条主协议必须彼此独立比较，不允许把不同主协议下的 `solo / pairwise / refine` winner 混成一张表
- `P2` 的主要时间不在训练，而在整理曲线、去重 checkpoint 和确认 selector 结果
- `P3` 的区间最大，因为它最受 `Stage 1` shortlist 预算和是否出现明显早期分层影响
- 如果中途出现“同轮 winner 与 runner-up 极接近”的情况，优先加一轮复验，不要跳过

## 核心结论

### 1. 时间漂移是真问题

- `2009-2026` 的超长时间跨度不能用随机切分来评估主线模型
- `Stage 0.5` 必须采用时间切分和课程学习
- 当前已验证更合理的口径是“宽时间训练，再向近期收束”

### 2. Screening 和 Formal 必须彻底分离

- A/B 筛选需要高频、小预算验证
- 正式训练需要低频、低打断验证
- 两者共用同一套默认参数，会直接把训练速度和判断口径一起搞乱

### 3. 重要重启必须使用全新输出目录

- 旧日志和孤儿子进程会污染判断
- 只杀父进程不够，必须检查子进程
- 重启正式训练时不要复用旧 run 目录

### 4. Windows 下优先保住快路径

- `1455` 共享文件映射错误确实存在
- 但不能为了“绝对稳定”把训练主链路整体降级
- 验证可以保守，训练主链路应尽量保留已验证吞吐更好的配置
- `2026-03-19` 对 `P1 calibration` 做了两次顺序验证探针，口径与当前 `opponent` 路径一致，结果分别记录在 `logs/validation_mem_probe_p1_opp_monitor_recent_seq.json` 与 `logs/validation_mem_probe_p1_opp_full_recent_seq.json`
- 这两次探针的峰值系统内存分别约为 `14.686 GB` 与 `14.911 GB`，峰值占用都约 `46%`，对应吞吐约 `2.985 batch/s` 与 `3.203 batch/s`
- 因此，单独跑验证时并没有出现“物理内存打满”；此前 `1455` 更像是长寿命训练进程叠加 Windows shared mapping / pagefile 压力，而不是简单的 RAM 不够
- 结论是：`1455` 不应直接触发整套验证降级，更不应反推训练参数一定过大；先保留 `train = 10/3`、`val = 8/5` 的默认口径，再针对触发场景单独重试或最小化修补

### 4.1 当前已经验证通过的稳定修法（2026-03-19）

- 当前这轮修法之后，`P1 calibration` 的干净重跑已经连续通过多次 `monitor validation` 与 `full validation`，没有再复现 `1455`、`shared file mapping`、`worker exited unexpectedly` 或 `_queue.Empty`
- 关键经验不是“把验证改成更慢的安全模式”，而是 **保留快路径，并把训练/验证之间的资源生命周期管理好**
- 当前稳定链路的核心做法有四条：
  - 训练保持 `file_batch_size = 10`、`prefetch_factor = 3`，验证保持 `val_file_batch_size = 8`、`val_prefetch_factor = 5`；不要因为一次 Windows 资源错误就把验证降成单进程 safemode
  - 验证阶段把 `1455`、`Couldn't open shared file mapping`、`paging file is too small`、`DataLoader worker exited unexpectedly`、`_queue.Empty` 统一视为 **瞬时资源错误**，按同一参数无限重试，而不是改配置口径
  - 每次验证都必须在 `finally` 中显式关闭 batch iterator / worker，随后 `gc.collect()`；在 CUDA 下再补一次 `torch.cuda.empty_cache()`，防止验证对象长时间悬挂
  - 当训练因 `max_steps` 或额外验证点即将切入验证时，必须先释放当前训练 loader，再进入验证；也就是先断开训练侧的长寿命 worker 与 shared mapping，再拉起验证侧 loader
- 实践上，真正有效的修法是最后一条：**先释放训练 loader，再做预算边界验证**。单纯调小 `val` 配置、单纯顺序验证探针、单纯降 worker，都不足以从根上解决这个问题
- 当前代码里，验证链路已经落成以下默认行为：
  - 验证 worker 非 persistent，训练 worker 继续保持 persistent
  - 验证循环退出时统一回收 iterator/worker
  - 预算边界验证前先 `release_train_loader()`
  - 遇到瞬时 loader/resource 错误时保持 `8/5` 原配置重试
- 这套链路的意义是：既保住了当前已经验证过的训练吞吐，也保住了验证的并发与速度；后续如果再看到 `1455`，默认思路应先检查“是否有训练 loader 没释放、是否有残留 worker/孤儿进程、是否有新代码绕开了统一的回收逻辑”，而不是第一时间降配置
- 顺序验证探针仍然有价值，但它的职责只剩下“定位峰值内存和验证资源边界”；它不能替代当前主线验证链路，更不能反客为主成为默认模式

### 5. 不要使用错误的 `numpy collate` 绕路

- 这类方案会显著拉低吞吐，严重时还会带来额外内存问题
- 当前正式方案应保留 PyTorch 默认快路径

### 6. Windows `spawn` 下，worker 相关对象必须可 pickle

- `collate_fn` 或 worker 逻辑不能写在局部作用域里
- 否则会出现 worker 启动异常、吞吐不稳定、表面看像随机崩溃

### 7. 回退代码后要检查“残留变量”和“残留日志”

- 只回退主逻辑，不回退旧变量引用，容易出现假故障
- 回退后至少做一次全局搜索和基础语法检查

### 8. A/B 必须先保证公平，再谈 winner

- 不同 arm 必须共享同一套评估切分
- phase 随机源应按 `phase_name` 派生，而不是按课程顺序派生
- 早期不满足这两条的 winner 只能视为 provisional

### 9. 选模不能只看 `loss`

- 纯 `loss-first` 会偏向“CE 略低但动作质量未必更好”的模型
- 当前更合理的做法是：
  - `P0 / P2 / formal` 先看 `full_recent_loss`
  - `P1` 先看 `comparison_recent_loss = recent_policy_loss`
  - 只在 `0.003` 容忍带内再比较 `selection_tiebreak_key`
  - `P1` 的 `old_regression_policy_loss` 再额外用 `0.0035` 作护栏

### 10. `macro_action_acc` 还不够细

- 仅看整体动作准确率，无法表达关键决策的质量
- 当前需要同时跟踪：
  - `discard_top1_acc`
  - `riichi_decision_acc`
  - `chi_decision_acc`
  - `pon_decision_acc`
  - `kan_decision_acc`
  - `agari_decision_acc`
  - `ryukyoku_decision_acc`
  - `call_decision_acc`
  - `pass_decision_acc`

### 11. 正式训练的验证必须稀疏

- 正式训练的目标是持续推进主训练，不是频繁打断自己
- 高密度验证更适合 screening，不适合 full-scale formal run

### 12. 近期高频实现错误清单（禁止重犯）

- **不要把选模口径误当训练损失。** 当前 `action_quality_score` 与 `scenario_quality_score` 都是验证/选模指标，不参与反向传播。修改这两套口径，默认动作应是“重验、重排 winner”，而不是立刻重训整轮；只有当新指标真正接入 `total_loss` 时，才需要把旧训练结果视为失效。
- **selector 读到原始分项时，必须重算总分，不能盲信缓存里的旧 `action_quality_score / scenario_quality_score`。** 否则一旦总分公式升级，历史 `arm_result.json` 里缓存的旧显式分数就会把新口径直接短路，出现“代码已经改了、排序却完全没变”的假象。
- **`discard slice` 必须按完整动作分布计分，不能先裁成 37 张弃牌再重归一化。** 如果样本同时存在合法 `riichi`，而实现又把 `riichi` 概率质量直接丢掉，那么 `discard_*_nll / top1_acc / top3_acc` 会在最关键的立直准备场景里系统性偏乐观，进而污染 `push_fold_*` 与整体 `scenario_quality_score`。
- **样本数必须跟着指标一起导出，否则 selector 无法做置信度 shrinkage。** 只导出均值型 `bce/nll/acc` 而不导出 `count / pos_count / neg_count`，会让稀有动作和稀有场景切片在排序时拿到与大样本指标同等的话语权，噪声会被系统性放大。
- **对手状态标签的发放要区分训练热路径和验证口径。** 训练阶段如果 `opponent_state_weight == 0`，就不应让 `GameplayLoader` 额外生成并搬运对手状态标签；否则会在默认 `CE-only / danger-only` 运行里白白拖慢吞吐。相反，验证阶段如果 `scenario_score` 依赖 `opponent-state` 条件切片，就必须统一给所有 arm 发这批标签，否则不同 arm 的指标定义都不一致，A/B 不公平。
- **`unpack_batch()` 不能假设 batch 列数永久固定。** 一旦训练和验证对标签列的需求不同，或者不同辅助头的开关不同，batch 结构就会发生变化。正确做法是按“基础列 + 剩余列数”动态解析，并对不合法形状直接报错；不要把“当前某个配置刚好能跑”误当成通用正确。
- **只要某个选模指标依赖隐藏标签，所有被比较的 arm 都必须走同一条验证链路。** 不能出现“只有打开某个辅助头的 arm 才有某类场景指标，其余 arm 记为缺失再混排”的情况；这不是模型强弱差异，而是评估输入不一致。
- **Windows 下的验证资源错误，优先检查资源生命周期，不要先降 safemode。** 已验证有效的修法是：预算边界验证前先 `release_train_loader()`，验证 `finally` 中显式关闭 iterator/worker，保留 `train = 10/3`、`val = 8/5` 原配置并无限重试。不要因为一次 `1455 / shared mapping / worker exited unexpectedly` 就把整套链路降级成单进程慢模式。
- **`run.lock` 不能只看 PID。** Windows 会复用 PID，旧 run 被强杀后留下的锁文件，如果只凭 PID 判断活性，会把新 run 错判成“已有实例在跑”。锁校验至少要同时检查 `pid + process_start_time`。
- **partial rerun 不能盲复用旧状态和旧 cache。** 只要 `P0/P1` 的 seed、候选集、eval split、stop flag 或 round signature 有变化，就必须重新验证缓存是否仍匹配；否则很容易在“看起来重跑了”的前提下，实际沿用旧 winner、旧 shortlist 或旧 formal 结果。
- **多 seed 轮次不能容忍“半残 arm”继续晋级。** `2-seed` 轮次里只跑通 `1/2` 个 seed，不足以说明 arm 稳定；winner 选择与后续晋级都必须过滤掉 `valid = False` 或 seed 不完整的候选。

## 操作守则

- 正式训练入口只用 `scripts/run_supervised.bat`（内部调用 `run_stage05_formal.py`）
- 单轮协议 A/B 只用 `run_stage05_ab.py`
- 自动串联保真流程只用 `run_stage05_fidelity.py`
- `run_stage05_fidelity.py` 现在带同名 `run` 锁；同一个 `--run-name` 不允许重复启动。若需要强制停止，先用 `scripts\stop_stage05_fidelity.bat <run_name>` 杀整棵进程树，再重新启动
- 没有硬证据时，不要随意改动已验证的 `4 / 10 / 3` 训练数据管线参数
- 当前验证默认就是 `8/5 + 无限重试`；不要再回退到 safemode，也不要把顺序探针误当成正式默认链路
- 如果后续又修改验证逻辑，必须保留三条安全带：验证 `finally` 回收 iterator、预算边界前 `release_train_loader()`、资源错误按原参数重试
- 不要复用重要实验输出目录
- 不要把 screening 的验证节奏带进 formal
- 宣布新 winner 之前，先确认公平性修复和干净重跑都已完成
## P1 staged multiseed（2026-03-20 追加）

`P1 calibration` 仍然保持单 seed；它的职责是统一 `rank / opponent_state / danger` 的有效预算量纲，不直接判 winner。真正的 winner 轮次改为 staged multiseed，以减少对明显劣势 arm 的重复训练，同时保持最终决策对 seed 波动有防护。

`SoloAuxGate` 的默认执行口径现在是：

- `seed1`：52 个单辅助候选全量跑完
- `seed2 probe`：每条主协议只补 `CE-only + rank top2 + opp top2 + danger top2`
- 对每条主协议单独判断是否需要全量补 `seed2`：如果 probe 后发生 `winner flip`，或者 `top1-top2 comparison_recent_loss gap <= max(0.003, 2 × probe seed noise)`，则说明仅靠 probe 还不足以确认 winner；如果 `old_regression_policy_loss` 护栏也已接近 `0.0035` 边界，同样应补齐这条主协议剩余候选的 `seed2`
- 如果上述条件都不触发，则直接用 probe 池作为该主协议的最终单辅助决策池，不再给明显落后的候选补第二个 seed

`Pairwise` 采用相同思想，但 probe 池改成每条主协议的 `top4`。这样做的原因很直接：`Pairwise` 的候选空间本来就比 `SoloAuxGate` 小，`top4` 已经足以覆盖 `CE-only`、单辅助 survivor 和最有希望的双辅助组合；只有在 probe 明确提示该协议内部仍然存在翻盘风险时，才值得补齐该协议的全量 `seed2`。

`JointRefine` 暂时不跟进这条 staged 策略，仍然保守使用固定多 seed。原因不是它不重要，而是它已经处在候选空间收缩后的末端，额外保留完整多 seed 更有利于稳定挑出最终 `P1 winner`。等 `Solo/Pairwise` 的 staged 结果积累足够后，再决定是否把同样的机制继续推广到 `JointRefine`。
