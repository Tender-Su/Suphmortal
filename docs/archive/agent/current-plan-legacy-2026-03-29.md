# 最强麻将 AI 当前计划

> Archive note (`2026-04-04`)：
> 本文档只保留 `2026-03-29` 当时的阶段判断，不能当当前主线入口。
> 其中 `Stage 0.5 / P2` checkpoint 去重层已从当前主线删除；当前 `formal` 直接接在 `P1 winner_refine` 之后，不读取任何 `P2` 输出。
> 当前默认以 `docs/agent/current-plan.md`、`docs/agent/mainline.md`、`docs/status/stage05-verified-status.md`、`docs/status/p1-selection-canonical.md` 为准。

本文档是 agent 的续工入口，只保留当前主线、已证实结论和待验证项，不再重复 `AGENTS.md` 的基础规范。

## 文档分工

- `AGENTS.md`：工程规范、架构常量、硬件假设、强约束
- `README.md`：项目简介和快速启动
- `docs/README.md`：给你看的文档总索引
- `docs/status/stage05-verified-status.md`：人工核对后的当前运行状态
- `docs/research/stage0/grp-experience.md`：`Stage 0` 的架构、精度与训练配置实验记录
- `docs/research/stage05/engineering-playbook.md`：`Stage 0.5` 排障经验
- `docs/status/stage05-fidelity-results.md`：`Stage 0.5 fidelity / P0 / P1` 的自动生成短结果
- 本文档：当前最强方案、已定主线、待做 A/B

## 一、已证实 / 主线默认

### 1. 总目标

- 首要目标是在当前机器上做出尽可能强的麻将 AI
- 当两个方案强度接近时，优先更小更快
- 当强度差异明确时，优先更强方案

### 2. GRP（Stage 0）主线

- 当前最强实用方案：`384x3 fp32`
- 默认下游使用：`best_loss`
- 备选对照：`best_acc`
- `latest` 只用于断点续训，不用于默认下游
- 效率优先 fallback：`256x3`
- `512x4` 是当前最高优先级的大模型继续探索候选
- 详细实验记录见 `docs/research/stage0/grp-experience.md`

### 3. Stage 0.5（监督预训练）主线

- 旧的 `A1y_broad_to_recent_mild_12m_6m` 只应视为历史条件 winner，不能再作为当前主线结论
- 当前 `Stage 0.5` 全局主线未定，必须以当前代码下的 `AB1234/P0` 与重跑 `AB1` 复核结果为准
- 当前代码下的 `AB1` 三个 seed 复核结果一致为 `cosine` 胜出：
  - `seed=20260312`：`B_cosine loss=0.604391`，`A_plateau loss=0.625892`
  - `seed=20260413`：`B_cosine loss=0.610014`，`A_plateau loss=0.628657`
  - `seed=20260514`：`B_cosine loss=0.609496`，`A_plateau loss=0.624212`
- 因此，旧的 “`AB1: plateau > cosine`” 结论已经失效；后续 `Stage 0.5` 不应再把 `plateau` 视为默认 scheduler
- `2026-03-24` 基于泄露修复后的 `P0` clean rerun，当前 `Stage 0.5` 已经得到足够强的正向入选结论：可直接进入下一轮 `P1` 的 `top3` 协议种子为：
  - `C_A2y_cosine_broad_to_recent_strong_12m_6m`
  - `C_A2x_cosine_broad_to_recent_strong_24m_12m`
  - `C_A1x_cosine_broad_to_recent_mild_24m_12m`
- 这次 `P0` 的当前主口径来自 `logs/stage05_fidelity/s05_fidelity_p0_cosine18_direct_r2top8_20260324_0230/` 与对应 `logs/stage05_ab/..._p0_r2/*/arm_result.json`，不再以历史 `s05_fidelity_main` 的泄露版 shortlist 为准
- 泄露前实际进入下游口径的种子池是 `A1x / A3x / B2z / B2x`；clean rerun 后只有 `A1x` 保留，`A2y / A2x` 新上位，说明泄露确实改变了 `P0` shortlist
- 这条结论的强项是“正向入选”而不是“穷尽淘汰”：当前已经足以确认上述 3 条值得进入 `P1`
- 基于效率优先的项目决策，`B-side` 协议在当前阶段正式冻结，不再继续补跑或进入 `P1`
- 这里的 “冻结 `B-side`” 是资源分配决策，不是数学意义上对全部未补跑 B 臂的完全证伪；后续如需翻案，必须以新的专门 clean rerun 为准
- `A2y` 是当前最强的 recent-facing winner：已完成 `round2` 中 `full_recent_loss / action_quality_score / selection_quality_score` 都是第 1，`old_regression_loss` 第 2
- `A2x` 是最强的 broad-strong hedge：`round0` selector 排名第 1，`round2` 的 `action_quality_score / selection_quality_score` 均第 2，且 `24m_12m` 比更窄窗口更稳
- `A1x` 是最强的 mild/robust hedge：`round2 rank_acc` 第 1，`old_regression` 也稳定，适合作为不那么激进的下游种子
- 配置层面的当前解释是：
  - `cosine` 仍是 scheduler 主线，这一点由 clean `AB1` 重跑单独确认
  - `broad_to_recent` 最终包揽 `top3`，说明先学宽分布再收束到最近分布，比 `recent_broad_recent` 更能在真 holdout 下站住
  - `strong` 在合适课程线上明显提高 recent/action side 表现，但 `mild` 仍保留更稳健的 hedge 价值
  - `12m_6m` 是当前最强的近期对齐 sweet spot，`24m_12m` 则提供更稳的长期缓冲；`6m_6m` 更容易把 recent loss 压低，但稳健性不一定最好
  - `two_stage` 和 `recent_broad_recent` 在泄露前显得过强，但 clean rerun 没能进入当前 `top3`，这与“泄露会放大 replay-heavy / repeat-exposure 协议优势”的主观逻辑一致
- 泄露前 raw run 与 clean rerun 的分值量纲不应做跨期绝对横比；跨期比较应主要看名单变化、同轮内排序和同口径下的指标结构
- 历史 `s05_fidelity_main` 已推进到 `P1 solo`；但它的 `P0` seed pool 建立在泄露版口径上，不能继续作为当前 `P0` 选种依据
- 当前工程默认参数仍为：
  - `batch_size = 1024`
  - `num_workers = 4`
  - `file_batch_size = 10`
  - `prefetch_factor = 3`
  - `val_every_steps = 20000`
  - `monitor_val_batches = 512`
  - `full_val_every_checks = 0`
  - `old_regression_every_checks = 0`
- `P0 / P2 / formal` 选模原则：
  - 先按各自的主比较字段过滤；当前 `P0 / P2 / formal` 固定使用 `full_recent_loss`，容忍带为 `loss epsilon = 0.003`
  - 再按 `selection_tiebreak_key` 比较候选，当前主排序为 `selection_quality_score -> -recent_policy_loss -> -old_regression_policy_loss`
  - `action_quality_score` 现在不再直接对原始 loss 做线性求和；每项先做 `loss / (1 + loss)` 饱和化，再乘样本数置信度 shrinkage，避免 `agari` 等量纲较大的单项绑架总分
  - 当前默认动作主分项为 `discard_nll / riichi_bal_bce / agari_bal_bce / chi_bal_bce / chi_exact_nll / pon_bal_bce / kan_bal_bce`，权重分别为 `0.45 / 0.18 / 0.18 / 0.04 / 0.03 / 0.07 / 0.05`
  - `selection_quality_score = action_quality_score + 0.20 * scenario_quality_score`
  - `scenario_quality_score` 已扩展为三层：
    - `P0`：`riichi / agari / kan` 在 `late / south+ / all-last / threat / calm / role / rank / close-gap`
    - `P1`：`up_gap/down_gap` 分离、`all_last_target_{keep_first,chase_first,keep_above_fourth,escape_fourth}`、`single/multi threat`、`opp_any/multi_tenpai|near_tenpai`
    - `P2`：`discard` 在 `all-last / threat / micro-gap / opponent-state` 下的专项 `nll`，以及 `push_fold_core / push_fold_extreme`
  - `old_regression` 只作为护栏
- `P1` 的单辅助 / family / pairwise / joint refine 不再沿用上面的泛化 `loss-first` 简写口径：
  - 唯一有效规范固定为 `docs/status/p1-selection-canonical.md`
  - 主门槛字段固定为 `comparison_recent_loss = recent_policy_loss`
  - `eligible` 必须按 `protocol_arm` 分组判断，而不是跨协议混排
  - `full_recent_loss` 在 `P1` 里只保留为 aux tax / 总 loss 诊断字段
- 三类辅助重新定位：
  - `policy CE` 是唯一默认主损失；`rank / opponent_state / danger` 都不再预设主次或默认保留资格
  - `danger aux`：直接监督当前可打牌候选的即时放铳风险，标签为 `是否放铳 / 最大放铳点数 / 放给哪家`；它是动作最近因、噪声最低的局部信号，因此适合作为“直接并入 policy loss 的第一候选辅助项”
  - `opponent_state aux`：预测三家对手当前 `shanten bucket / tenpai flag`；它更像对手隐状态建模与表征约束，目标是逼迫模型真正利用手切 / 摸切等河牌信号，与 `danger` 既互补也部分冗余
  - `rank aux`：目标上最接近长期胜负，但当前实现仍是把粗粒度位次信号直接压回单步动作，并额外做 `south / all-last / 点差` 加权；因此它不再被视为默认无害项，应作为待证实分支，或后续升级为 `Oracle Value Predictor`
  - 旧 `P1 calibration` 里 `o=0.06 / d=0.06` 的 probe 只能说明“在当前高权重下，`danger` 比同强度 `opponent_state` 更不容易伤主目标”，不等于 `danger` 在长期目标上天然高于 `rank`
- 当前最高优先级实验：`AB1234 joint`
  - 把 `scheduler / curriculum / weight / window` 放进同一张联合表
  - 候选集使用 `3 × 2 × 3 × 3 = 54` 个组合
  - 所有调度器统一使用 warmup 起手；warmup 不是搜索轴
  - 三类调度器为 `plateau / cosine / phasewise`
  - 其中 `phasewise` 在当前三阶段实现里已经等价于 `warmup + cosine + plateau`
  - 不再把 `AB1` 与 `AB234` 人为拆开后再拼接成主线
- 详细实验记录见 `docs/research/stage05/engineering-playbook.md`

### 4. Stage 0.5 + Stage 1 联合主线

- 旧 `Stage 1 = Oracle AWR + GRP hand-level advantage` 不再作为默认主线
- 原因不是 `Oracle Dropout` 本身有问题，而是“用 `GRP` 产生的整局/整小局粗粒度顺位变化去指数加权每一步动作”噪声过大：
  - 同一小局内所有动作共享同一 `advantage`，无法区分局中好手与恶手
  - 晚局/末盘会把运气型结果过度放大，防守好但结果差的样本被错误压低
  - 该信号更像“局结果归因”，不像“动作质量归因”
- 新默认主线改为：**合并 `Stage 0.5` 与旧 `Stage 1`，做 `Oracle Dropout Supervised Refinement`**
- 新主线的固定骨架：
  - 从 `Stage 0.5 / P0` clean rerun 得到的 `top3` 协议种子初始化：`A2y / A2x` 负责主攻，`A1x` 负责稳健 hedge
  - 维持人类动作监督 `policy CE` 作为主损失，其他项全部视为辅助项
  - 保留 `Oracle Dropout`，但不再使用旧 `GRP-AWR` 样本权重
  - 三类辅助不再按“谁更高贵”排序，而是按“与当前动作的因果距离、噪声、与最终目标的对齐程度”重新评估
  - `P1` 以 `CE-only` 为唯一锚点，不再预设 `rank gate` 为第一层必经路线
- `2026-03-28` 起，`P1` 主线实验结构改为：`calibration -> protocol_decide -> winner_refine -> ablation`
- 重构原因：
  - 历史 `solo` 只验证单头是否各自有用，不能决定“三头全开时怎么配比”
  - 历史 `pairwise` 把单头 winner 直接相加，容易把组合推到过重区间，因此只能当诊断证据，不能继续当主线决策轮
  - 当前主线目标改为：尽早在 `top3 protocol` 中选出 winner，然后只在 winner 协议内部搜索三头全开配比
- `2026-03-29` 又对这条新主线做了一次收缩，不改方向，只删冗余：
  - 保留 `calibration -> protocol_decide -> winner_refine -> ablation` 这个四段结构
  - 但把 `calibration` 从“`top3` 协议都重跑 + 单头/双头/三头全探”收缩成“`A2y` 代表协议 + 只补组合耦合”
  - 收缩理由不是省事，而是证据已经足够说明：`2026-03-25` 的 post-shape `top3 calibration` 已经在新内部 shape 下证明单头量纲跨协议几乎不变，继续重跑这些单头项不会增加多少信息
  - 因此当前 `P1 calibration` 的真正职责被进一步聚焦为：复用已验证单头量纲，只补 `rank_opp / rank_danger / opp_danger / triple` 四类组合耦合因子

### 4.1 P1 主线调整过程

- 第一阶段的想法是让 `rank / opp / danger` 先各自单独前进：先跑 `solo`，再把单头 survivor 送进 `pairwise`，最后做 `joint refine`。这个结构的优点是容易理解，但它隐含了一个前提：单头 winner 的直接相加，能代表三头全开的最优配比。
- 跑到中途后，这个前提被证伪了。`solo` 确实能回答“每个头单独开有没有净收益”，但 `pairwise` 经常把组合推到过重区间，导致它更像“诊断某两头会不会打架”，而不像“主线 winner 判决轮”。这一步说明旧三段式在统计口径上不再适合作为正式主线。
- 第二阶段的调整，是把主问题重新定义成两个更贴近目标的问题：第一，三条 `top3 protocol` 谁应该尽早胜出；第二，winner 协议内部三头全开时的最佳配比是什么。于是主线改成 `calibration -> protocol_decide -> winner_refine -> ablation`，不再让多个协议和多个辅助家族一起无限并行前进。
- 第三阶段的调整，是把 `calibration` 也重新定位。它不再承担任何 winner 判决职责，只做量纲和耦合定标。先用 `loss contribution + shared-phi gradient RMS` 把单头映射到统一预算轴，再测 `pairwise / triple combo factor`，让后面的三头全开搜索建立在可解释的预算语义之上。
- 第四阶段的调整，就是这次 `2026-03-29` 的收缩。检查旧日志后确认：`2026-03-25` 那轮 `top3 calibration` 实际已经在新内部 shape 冻结之后跑过，而且 `A2y / A2x / A1x` 三条协议的单头量纲和旧 `opp+dang` 组合因子几乎完全稳定。因此“多协议 + 单头重跑”不再是当前信息瓶颈，真正缺失的是三头全开主线尚未直接测过的组合耦合。基于这个判断，当前正式默认收缩为 `A2y-only + combo-only` 的瘦版 `calibration`。
- 这条演化链路的核心思想不是把流程做复杂，而是不断把“当前真正缺的信息”从“已经重复验证过的信息”里剥离出来。现在的主线因此有一个很明确的分工：`calibration` 只负责定标，`protocol_decide` 尽早选协议 winner，`winner_refine` 只在 winner 协议里细调三头配比，`ablation` 最后验证三个头是否真的都有边际价值。
- 冻结后的三类内部 shape 继续沿用 `2026-03-25` 的 `A2y` internal-shape micro AB 结论：
  - `rank = south_factor 1.59 / all_last_factor 1.617 / gap_focus_points 4000 / gap_close_bonus 0.0`（`18K_ROUND_ONLY`）
  - `opp = shanten 0.8506568408 / tenpai 1.1493431592`（`HYBRID_GRAD`，保留旧 `1:1` 方案的总系数和为 `2.0` 语义）
  - `danger = 0.0904217947 / 0.8180402859 / 0.0915379194`（`18K_STAT`）
- `2026-03-25` 这轮 `A2y` micro AB 也修正了口径：`P1` 的单辅助 winner 判定必须按 `policy_quality` 做，也就是用 `comparison_recent_loss = recent_policy_loss` 过门槛，再按 `selection_tiebreak_key` 比较；当前排序键已经收敛到 `selection_quality_score -> -recent_policy_loss -> -old_regression_policy_loss`，各类 `acc` 只保留为诊断字段；`full_recent_loss` 只保留为 aux tax 诊断字段，不再作为这类实验的主判胜口径
- 上面这条 `policy_quality` 规则现在已经冻结为项目内唯一有效的 `P1` 口径；任何脚本、文档、人工总结或自动摘要若与它冲突，一律以 `docs/status/p1-selection-canonical.md` 与 `mortal/run_stage05_fidelity.py` 为准
- `2026-03-26` 新增了 `mortal/analyze_selection_heuristics.py` 的 selector 审计：在跨 `2009-2026` 的 `3240` 文件样本和现有多 seed `P1 solo` 结果上，确认 `policy_loss_epsilon = 0.003` 可以视为统计支持；`old_regression_policy_loss_epsilon` 与主门槛解绑，单独固定为 `0.0035`
- 同一天又做了针对新 selector 语义的 `scenario_factor` 细搜：在正式搜索带 `0.0-0.25` 内，先看 pairwise 稳定性、再用 aggregate winner 一致率做次级筛选；最优点落在 `0.199`，因此把运行时默认值四舍五入到 `0.20`
- `P1 calibration` 口径进一步升级为两层：
  - 第一层仍是 `loss contribution + shared-phi gradient RMS` 双定标，用两者的 `geomean hybrid` 生成 `opp/danger weight per budget unit`
  - 第二层新增组合耦合定标：显式产出 `rank_opp / rank_danger / opp_danger / triple` 四类 `combo_factor`
  - 当前正式默认已收缩为瘦版 `cali`：只在代表协议 `A2y` 上补跑 `rank_opp_probe / rank_danger_probe / opp_danger_probe / triple_probe`
  - `rank / opp / danger` 的单头量纲不再每次重跑，而是沿用 `2026-03-25` 那轮 post-shape `top3 calibration` 已验证过的单头映射
  - 这次收缩的理由已经确认：`2026-03-25` 那轮旧版 `top3 calibration` 实际就是在新内部 shape 冻结之后跑的，而且跨 `A2y / A2x / A1x` 三条协议得到的单头量纲与旧 `opp+danger` 组合因子都几乎不波动，因此“多协议重跑单头”不再是当前信息瓶颈
  - 当前真正缺失的不是“单头映射是否稳定”，而是“新主线三头全开时 `rank+opp / rank+danger / opp+danger / triple` 的耦合因子是多少”；所以现在把 `cali` 收缩成 `A2y-only + combo-only`，本质上是删掉已经有证据重复稳定的部分，只补真正缺的维度
  - 新主线读取规则是：`protocol_decide / winner_refine` 读 `triple_combo_factor`，`drop_rank` 读 `opp_danger_combo_factor`，`drop_opp` 读 `rank_danger_combo_factor`，`drop_danger` 读 `rank_opp_combo_factor`
  - 历史字段 `joint_combo_factor` 继续保留，但现在只作为 `opp_danger_combo_factor` 的 legacy alias，不能再被无上下文地理解成“三头全开”的正式定标值
  - 这样做的目的不是从 calibration 直接宣布 winner，而是让后续三头全开主线在进入 `protocol_decide` 之前，先把“单头量纲”和“组合过重/过轻”两件事都定清楚
  - 新增 `danger aux`，直接监督“当前候选出牌的即时放铳风险”，并与 `opponent_state aux` 共用辅助预算池
  - `danger aux` 不拆成漫长串行开发；第一版直接一次性实现 `binary 放铳 / value 点数损失 / by-player 放给哪家` 三路标签与多头输出
  - `danger aux` 的训练稳定性优先通过归一化、分头加权、阶段式启用和必要的 focal / class-balance 处理解决，而不是把三种标签拆成三轮开发
  - `Oracle Value Predictor` 作为独立价值模型立项，用来承接长期价值建模职责；如果后续它能稳定优于当前 `rank aux` 形态，则由它接管这部分高层战略信号，而不是继续把粗粒度 `rank aux` 直接并入每一步动作损失
- 新主线下，`GRP` 的定位下调为：
  - 全局顺位/价值建模候选
  - 下游评估或奖励塑形候选
  - 可做弱辅助头候选
  - **不再默认直接充当 `Stage 1` 的主样本加权器**
- 如果后续还要做离线强化学习，前提应改成：
  - 先有更细粒度、动作更近因的 value/critic 信号
  - 再重启 `AWR/PPO` 类训练
  - 如需比较旧 `AWR / GRP-AWR`，只通过 `git` 历史或临时复现实验，不在当前仓库保留常驻分支

### 4.1 旧 `Oracle AWR` 处理结论

- 旧 `Oracle AWR / GRP-AWR` 已从当前代码仓库移除；如需追溯历史方案，只通过 `git` 历史复盘
- 当前主线相关训练、A/B、入口与文档都不再依赖传统 `AWR`

### 5. 工程纪律

- 正式训练和 A/B 必须分离入口
- 重要重启必须使用全新输出目录
- 不要复用旧日志目录判断新训练
- CPU affinity 现在改为显式开启；默认保持 `MORTAL_CPU_AFFINITY` 未设置，不再自动绑到 `p_cores`
- 不要随意动已验证的 `4 / 10 / 3` 训练数据管线参数
- 不要引入已证明会拖慢吞吐的 `numpy collate` 绕路

## 二、当前主线执行顺序

### 1. Stage 0

- 使用 `384x3 fp32` 训练并维护 `best_loss / best_acc / latest`

### 2. Stage 0.5

- 先完成当前 `P0/P1/P2` 所需的监督协议筛选
- 保留至少以下 checkpoint：
  - `latest`
  - `best_loss`
  - `best_acc`
  - `best_rank`

### 3. Stage 1

- 将旧 `Stage 1` 改造为“`Oracle Dropout Supervised Refinement`”
- 默认从 `Stage 0.5 top-k` 协议中选 `top3` 种子启动：`A2y / A2x` 提供当前最强主攻线，`A1x` 提供稳健 hedge
- 默认比较顺序：
  - `visible-only CE` 基线
  - `oracle-dropout CE + protocol_decide winner`
  - `oracle-dropout CE + winner_refine front runner`
  - `oracle-dropout CE + ablation confirmed winner`
  - `Oracle Value Predictor` 先独立训练，再决定是否进入后续 `Stage 1` / `Stage 2` 迁移链路
- 新 `Stage 1` 默认选模仍以 `recent loss + action/scenario quality` 组合口径为准
- 旧 `Oracle AWR` 不再作为当前工程结构的一部分；后续只讨论现有主线与新 value/critic 方案

### 4. Stage 2

- 在新的 `Stage 1` 稳定后，推进 PPO 在线自博弈
- 奖励塑形和对战口径通过单独 A/B 再定

### 5. 后续实验执行树（冻结版）

- 当前后续实验按五个连续区块推进：`A = Stage 0.5 收尾`，`B = 新 Stage 1 训练器落地`，`C = 新 Stage 1 配方筛选`，`D = Stage 1 迁移定型`，`E = Oracle Value Predictor / Stage 2`
- 五个区块允许少量工程并行，但不允许在统计上互相污染；任何下游区块都不能反过来改写上游区块的比较口径

**区块 A：Stage 0.5 收尾**

- 固定泄露修复后的 `P0 top3` 为当前下游协议种子池：`A2y / A2x / A1x`
- `P0 round3` 只负责给主协议排序，不再阻塞后续主线；真正阻塞后续的是 `P1 / P2`
- 继续完成：
  - `P1`：改为 `CE-only 锚点 -> protocol_decide -> winner_refine -> ablation`
  - `P1 protocol_decide`：三头同时开启；所有协议用同一套小总预算 + 配比模板，先尽早选出协议 winner
  - `P1 winner_refine`：只在 winner 协议内继续搜索三头全开配比，不再让三个协议共同前进
  - `P1 ablation`：固定 winner 配比后，比较 `all_three / drop_rank / drop_opp / drop_danger / ce_only`，验证三个头是否都还有边际价值
  - 当前已在跑的 `calibration` 只作为 `opp/danger@0.06` 高权重排雷，不直接产生正式 `P1 winner`
  - `P2`：`best_loss / best_acc / best_rank / selector`
- `区块 A` 完成门槛：
  - 已得到单一 `P1` 协议 winner，并在该协议内部冻结一版三头全开 front runner
  - 已完成 `P2` 去重，形成 `2 ~ 4` 个可送入下游的 checkpoint 候选
  - `Stage 0.5` 的默认选模口径被冻结，不再边跑边改

**区块 B：新 Stage 1 训练器落地**

- 在 `区块 A` 运行期间，允许并行完成新的 `Oracle Dropout Supervised Refinement` 训练器
- 这一步只做工程落地，不做大规模结论宣告；目标是把后续实验入口准备好
- `区块 B` 的工程实现必须遵守四条硬约束：
  - **共享 core + 薄入口**：不得再复制一份大号训练脚本；`Stage 0.5` 与新 `Stage 1` 必须共用同一套训练核心，只允许保留各自的薄入口脚本
  - **配置语义隔离**：新 `Stage 1` 必须使用独立的配置段与独立输出路径，不能继续复用 `[supervised]` 的语义去承载 `oracle dropout / normal export / Stage 1` 专属逻辑
  - **visible-only 验证主导**：即使训练过程中使用 `oracle dropout`，验证、early stopping、LR 调度与默认选模也必须以 `visible-only` 路径为准；oracle 侧指标只允许作为诊断，不允许反客为主
  - **双形态 checkpoint**：训练态必须保留可恢复的 oracle checkpoint；同时必须导出可直接下游使用的 `visible-only / normal` checkpoint，两者职责不得混淆
- 新训练器必须满足：
  - 以 `policy CE` 为主损失
  - 支持 `rank aux + opponent_state aux + danger aux`
  - 支持 `oracle dropout`
  - 明确禁止旧 `GRP hand-level advantage` 样本加权链路进入主线
  - 支持从 `Stage 0.5` 的 `best_loss / best_acc / best_rank` 加载初始化
- `区块 B` 完成门槛：
  - 新训练入口可独立启动
  - 同一配置下可稳定恢复、保存、验证
  - 可导出 `visible-only` 形态 checkpoint

**区块 C：新 Stage 1 配方筛选**

- `区块 C` 的目标不是马上找最终最强 seed，而是先确定新 `Stage 1` 的默认损失骨架和 `oracle dropout` 日程
- 为避免组合爆炸，先固定一条代表性强种子做配方筛选；如果只能选一条，当前默认优先使用 `C_A2y_cosine_broad_to_recent_strong_12m_6m`
- 默认比较顺序固定为：
  - `S1-A = visible-only CE`
  - `S1-B = oracle-dropout CE + protocol_decide winner`
  - `S1-C = oracle-dropout CE + winner_refine front runner`
  - `S1-D = oracle-dropout CE + ablation confirmed winner`
- `oracle dropout` 曲线单独做小规模配方筛选，只比较少数几条代表曲线：
  - `G0 = no dropout`
  - `G1 = linear 1 -> 0`
  - `G2 = smooth / cosine-like 1 -> 0`
- `区块 C` 完成门槛：
  - 得到一套默认 `Stage 1` 损失骨架
  - 得到一套默认 `oracle dropout gamma` 日程
  - 明确 `danger aux` 是否进入主线第一版

**区块 D：Stage 1 迁移定型**

- 在 `区块 C` 固定默认配方后，再把它迁移到 `P2` 保留下来的 `2 ~ 4` 个 `Stage 0.5` 候选 checkpoint 上
- 这一阶段才回答“哪个 `Stage 0.5` checkpoint 最能转化成下游强度”
- 这一步对应 `Stage 0.5` 文档中的 `P3`
- 默认执行顺序：
  - 先做小预算迁移筛选，保留 `top 2`
  - 再做长预算迁移对照，决出最终 `Stage 1` 默认初始化
- `区块 D` 完成门槛：
  - 得到新的 `Stage 1` 默认初始化 checkpoint
  - 在统一预算下，明确新主线相对历史旧方案的增益或退化

**区块 E：Oracle Value Predictor / Stage 2**

- `Oracle Value Predictor` 当前只作为独立立项，不得阻塞 `A/B/C/D`
- 它的职责是长期价值建模与后续 `Stage 2 critic` 初始化候选，而不是当前 `Stage 1` 的主损失组成部分
- 只有在 `区块 D` 完成后，才允许进入：
  - `Oracle Value Predictor vs GRP` 对照
  - `Stage 2` reward table 对照
  - `Stage 2 critic` 初始化对照

**并行与禁止事项**

- 允许并行：`区块 A` 的 `P1/P2` 与 `区块 B` 的工程开发
- 不允许并行下结论：`区块 C/D/E` 必须等上游默认口径冻结后再宣告 winner
- 不允许把 `Oracle Value Predictor` 与 `Stage 2 reward table` 混入当前 `Stage 1` 默认配方筛选
- 不允许在 `P1/P2` 还没冻结时，提前宣布某个 `Stage 0.5` checkpoint 是最终主线赢家
- 不允许因为局部离线 loss 好看，就绕过 `区块 D` 直接宣告下游最强

**当前立即动作**

- 立即动作 1：先跑瘦版 `P1 calibration`，沿用 `2026-03-25` 已验证的单头映射，只补 `pairwise / triple combo factor`，再启动 `protocol_decide`
- 立即动作 2：`protocol_decide` 默认保持三头全开；不再先做主决策用的单头淘汰
- 立即动作 3：待 `protocol_decide` 选出协议 winner 后，只在 winner 协议内部进入 `winner_refine -> ablation`
- 立即动作 4：并行落地新的 `Stage 1` 训练器与配置模板
- 立即动作 5：待 `P1 / P2` 冻结后，再在单一代表性种子上做 `区块 C`
- 立即动作 6：只有 `区块 C` 定型后，才进入 `区块 D/E`

## 三、待 A/B 验证

这些项目仍值得做，但不应混入当前正式主线。

### A. 长训练敏感项

这些结论可能受到训练长度影响，短预算 A/B 只能给方向，不能直接当最终定论：

- `Stage 0`：`384x3` vs `512x4 / 384x4 / 512x3` 的长预算复验
- `Stage 0.5`：`AB1234 joint` 的 54 臂长预算复验与 top-k 二次复验
- `Stage 1`：`P1 calibration` 现在不再只看 `loss` 量纲；它会同时记录 `rank / opp / danger` 对共享表示 `phi` 的梯度 RMS、两两梯度夹角，并据此生成 `pairwise / triple combo factor`，避免“loss 看似公平、trunk 实际受力不公平”
- `Stage 1`：主决策轮不再使用 `SoloAuxGate / Pairwise / JointRefine` 三级串联；历史结果只保留为诊断和缩范围证据
- `Stage 1`：`rank` 若胜出，只能说明“当前实现下存在净收益”；`rank` 若失利，也只说明“当前实现与当前权重接法不成立”，不等于长期价值信号本身无用
- `Stage 1`：`P1` 当前主线固定为“冻结内部 shape + 三头全开 + 只搜索总预算/内部配比”；`rank / opp / danger` 都默认在搜索空间里，不先互相关闭
- `Stage 1`：`protocol_decide` 必须在统一三头脚手架下比较协议；协议 winner 选出后，后续只沿单一 winner 协议继续前进
- `Stage 1`：`winner_refine` 只做小步长三头配比微调；旧 `Δbudget_ratio = 0.25` 的 `JointRefine` 口径已废弃
- `Stage 1`：`P1` 的主线 winner 选择（`protocol_decide / winner_refine / ablation / final_compare`）继续使用多 seed 聚合；单 seed 只作排错或方向判断，不直接宣告 winner
- `Stage 1`：`P1 calibration` 本身是“量纲定标”而不是“winner 判决”；默认先依赖单 seed + 多 batch 中位数探针，只有当 `loss/grad` 两条轴给出的排序冲突、或不同协议之间定标结果波动过大时，才把 calibration 升级到和 winner 相同的 3-seed 级别
- `Stage 1`：`phase-wise auxiliary schedule` 暂不直接进主线；只有在固定 `P1` 静态权重基线后，通过受控 A/B 同时打赢静态方案和小预算 `Stage 1 transfer`，才允许升级为默认主线
- `Stage 0.5`：当前默认执行方案为“保真版”，详见 `docs/research/stage05/engineering-playbook.md`
- action-side 综合评分权重
- `best_loss` / `best_acc` / `best_rank` 对下游 `Stage 1` 的迁移差异
- 新 `Stage 1`：`oracle-dropout` 是否应使用 `P0` 的 `top3` 全部复筛，还是先在单一 backbone 上筛损失设计
- 新 `Stage 1`：`danger aux` 三路输出的总权重与是否分阶段启用（内部配比已冻结为 `18K_STAT`）
- 新 `Stage 1`：`rank / opponent_state / danger` 的联合预算与 selector 口径是否需要同步改成更偏 `policy loss` 的主门槛
- 新 `Stage 1`：`oracle dropout gamma` 曲线与 phase 长度

### B. 下游强度项

这些需要看最终对局强度，而不是只看监督指标：

- `Stage 0.5 best_loss` 与 `best_acc` 对 `Stage 1` 的影响
- `GRP best_loss` 与 `best_acc` 对 `Stage 1/2` 的影响
- 新 `Stage 1` 相比历史旧方案对 `Stage 2` 初始化强度的影响
- `Oracle Value Predictor` 相比 `GRP` 在 `Stage 1` 长期价值建模与 `Stage 2 critic` 初始化上的收益
- `Stage 2` 奖励表的实际对局收益

### C. 奖励表项

需要在 `Stage 2` 做较重的对照：

- `2,1,0,-3`
- `3,2,1,0`
- 基于雀魂 / 天凤顺位价值再归一化的版本

当前判断：

- 如果目标是稳定优化“争一”和惩罚四位，`2,1,0,-3` 更像主线候选
- 如果目标是更平滑的顺位学习信号，`3,2,1,0` 值得作为对照
- 最终必须由自博弈和实战评测决定

## 四、各阶段完成标准

### Stage 0 完成标准

- `best_loss` 稳定收敛
- 大模型相对更小模型没有明显下游回报再提升时，停止继续放大

### Stage 0.5 完成标准

- `recent loss` 进入平台区
- action-side 指标不再持续改善
- 没有明显旧分布灾难性回退
- 获得可用于 `Stage 1` 的 `best_loss / best_acc / best_rank`

### Stage 1 完成标准

- `oracle-dropout` 退出到 `gamma=0` 后仍能稳定提升
- action-side 指标与难场景指标同步改善，而不是只压低总体 `loss`
- 留在主线里的辅助项（`rank / danger / opponent_state`）已经在长预算下证明净收益；无收益项已被剔除
- 导出可直接供 `Stage 2` 使用的 normal 形态 checkpoint

### Stage 2 完成标准

- 对战评估相对基线有稳定提升
- 奖励表和 checkpoint 口径经过对照验证

## 五、实验记录要求

- 正式训练输出目录必须全新
- A/B 结果必须写入对应 `summary.json`
- 关键阶段切换前做一次本地 git 保存
- 宣布新默认方案前，先确认：
  - 公平性口径正确
  - 至少有一次干净重跑
  - 不是由旧日志或旧进程误导

## 六、当前口径摘要

如果现在要继续主线训练，默认做法就是：

1. `Stage 0` 使用 `384x3 fp32`
2. `Stage 0.5` 先完成 `P1` 的 `CE-only -> protocol_decide -> winner_refine -> ablation` 与 `P2`，冻结辅助预算和 checkpoint 口径，再进入下游
3. 新默认主线直接进入合并后的 `Oracle Dropout Supervised Refinement`
4. `Stage 0.5 / P0` clean rerun 的 `top3` 直接作为新 `Stage 1` 的首批协议种子池；真正进入 `Stage 1` 迁移的是 `P2` 去重后的候选 checkpoint
5. `Stage 1` 先筛配方，再筛 seed；配方筛选顺序固定为 `CE-only -> protocol winner -> winner-only 配比 refine -> ablation winner -> oracle dropout 细化`
6. `Oracle Value Predictor` 与 `Stage 2 reward table` 都是后置实验，不阻塞当前 `Stage 0.5 -> Stage 1` 主线
## P1 staged multiseed（2026-03-20 追加）

- `P1 calibration` 继续保持 `single-seed`；当前默认是 `A2y-only + combo-only`，职责是复用已验证单头量纲并补充组合耦合，不直接判 winner
- `P1 protocol_decide` 默认改为 `progressive_probe_then_expand`
  - `seed1`：全量候选都跑
  - `seed2 probe`：只跑每条主协议的 `top4`
  - 只有 probe 证明该主协议仍然不稳定时，才补齐该主协议全量 `seed2`
  - 当前默认网格已收紧为 `total_budget_ratios = [0.09, 0.12]`
  - 当前默认 mix 为：
    - `anchor = 0.43 / 0.21 / 0.36`
    - `rank_lean = 0.53 / 0.16 / 0.31`
    - `opp_lean = 0.38 / 0.31 / 0.31`
    - `danger_lean = 0.38 / 0.16 / 0.46`
  - 收紧理由：`protocol_decide` 只负责尽快选协议 winner，不负责把三头预算上沿一次性铺满；旧上沿预算留给 `winner_refine`
- `P1 winner_refine` 当前默认直接使用固定双 seed；等新的 `protocol_decide` 结论稳定后，再考虑进一步 staged 化
- `P1 ablation` 当前默认也使用固定双 seed；它的职责是验证边际贡献，不是再开新一轮全空间搜索
