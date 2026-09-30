# MahjongAI 项目完整技术说明与专家审阅请求

> 版本：2026-09-05。对象：熟悉监督学习、强化学习和统计评测，但不必熟悉立直麻将的研究者或工程专家。
>
> 本文是当前实现的技术快照和审阅材料。包含源码事实、有效配置、已复核实验、实现缺陷、未决假设与建议实验；不是宣布训练方案已经通过资格认证。正文可独立阅读，源码和机器可读附件用于进一步核验。

## 0. 给审阅专家的说明

我们希望训练一个四人立直麻将 AI，最终目标是可复现、可推广的实际牌力。当前以人类牌谱监督学习建立可见信息策略，再训练独立的 Oracle critic，计划用真实回报、value / GAE 和在线策略优化继续提升。项目已有完整的 Rust 环境、Python 训练系统、数据处理、模型选择、模拟评测和推理接口。

我们最需要的不是确认现有设计“看起来合理”，而是请您判断：**目标函数是否正确；数据和时间语义是否自洽；Oracle 提供的信息是否真的改善 actor 的学习；异步采样和 PPO 的组合是否有明确数学依据；现有实验是否足以支持结论；下一笔算力应投在哪里。**

此前若干方案由不同阶段的模型辅助设计，实现和文档有历史层叠。我们不要求维护旧方案的面子，也不希望仅因为复杂就全部推倒。请把可以直接证明的实现问题、需要实验判断的方法选择、目前无法回答的问题分开。对于“应该换成某算法”的建议，希望同时得到最小对照实验、预期改善机制、失败信号和成本估计。

### 0.1 请优先回答的六个问题

1. 应当优化平均顺位、正式 `avg_pt`，还是其他效用？现有训练奖励与正式评测奖励不同，怎样统一最合理？
2. 如何定义麻将中的 RL 时间步？当前 Oracle native state fold 会改变折扣回报，同一状态可能因采样方式不同而得到不同标签；应如何修复及迁移现有模型？
3. 人类牌谱中的部分隐藏信息加随机牌山补全，适合作为什么 critic 输入？怎样界定其因果含义、噪声、偏差及接入资格？
4. 独立 `all_players` Oracle critic 是否值得继续？什么证据能说明它改善了 p0 的 advantage，而不仅改善了四个输出的平均 MSE？
5. 应先建立多新鲜的 PPO 参照？当前行为版本重建、V-trace advantage 与 PPO ratio 的组合应怎样推导、简化和验证？
6. 如何在有限算力下设置 SL 选择、critic 资格和正式 `1v3` 的统计协议，避免反复查看同一验证集后选出偶然 winner？

### 0.2 证据等级与阅读约定

| 标记 | 含义 | 不应推导出的结论 |
| --- | --- | --- |
| 实现事实 | 本次对照当前 Python / Rust 源码核验 | 不自动表示运行中的二进制或所有旧实验都采用这版实现 |
| 有效配置 | 已读取指定 run 配置或 checkpoint 内部配置 | 不表示基础 TOML 中同名字段一定在该阶段被消费 |
| 复核证据 | 原始产物、CPU 反例、checkpoint 检查或配对重算支持 | 小样本诊断不自动成为最终牌力结论 |
| 历史证据 | 有日期、有来源的旧训练或评测记录 | 不当作当前机器状态或新候选已通过的门槛 |
| 工程推断 | 由实现和数学关系推导出的影响 | 不冒充已测量的总体误差或因果效应大小 |
| 建议 / 待决 | 拟实施的实验、目标选择或替代设计 | 不表示已经启动、修复或获得收益 |

本文中的 `true` Oracle 是现有接口名称，特指当前解码器生成的隐藏输入；对人类牌谱，它不等于已知完整真实牌山。`p0` 指相对当前样本视角的玩家本人。`step` 默认是对应模块的计数单位，涉及训练步、动作步和牌局步时另行注明。

### 0.3 快照、实施范围与复现身份

核验时 Git HEAD 为 `94d942f1745dd23751965d27ffa7c9d93ce82331`。工作树存在大量既有修改，因此 **commit 不是完整源码身份**。本次单独保存了主要文件 SHA-256、选定 checkpoint 的结构和配置，以及新的标签反例。

本次撰写期间只新增说明和独立 CPU 探针，没有改动训练热路径、替换原生扩展、覆盖 checkpoint、启动新 RL 或打开 sealed test。此前同日的独立审计已经完成少量在线契约修复；具体范围见第 18 节。

源码事实和有效配置是主要依据。当前状态由 [状态索引](../README.md#当前状态) 维护；此前审计见 [SL / RL 独立审计](sl-rl-audit-2026-09-05.md)。本文不逐次追加训练进度。

### 0.4 内容导航

- [1. 任务与麻将背景](#1-任务与麻将背景)
- [2. 整体方案及当前阶段](#2-整体方案及当前阶段)
- [3. 数据、切分、采样与恢复](#3-数据切分采样与恢复)
- [4. 状态表示与动作接口](#4-状态表示与动作接口)
- [5. 可见信息模型与辅助头](#5-可见信息模型与辅助头)
- [6. GRP：全局结果预测](#6-grp全局结果预测)
- [7. SL：监督学习目标和训练流程](#7-sl监督学习目标和训练流程)
- [8. SL 课程、选择与现有候选](#8-sl-课程选择与现有候选)
- [9. Oracle critic：输入、结构与梯度路径](#9-oracle-critic输入结构与梯度路径)
- [10. Oracle 标签与奖励的精确定义](#10-oracle-标签与奖励的精确定义)
- [11. 新发现：native fold 改变了折扣时钟](#11-新发现native-fold-改变了折扣时钟)
- [12. Oracle 预训练的有效配置和自适应控制](#12-oracle-预训练的有效配置和自适应控制)
- [13. 在线 RL 系统与行为策略](#13-在线-rl-系统与行为策略)
- [14. GAE、V-trace 与实际 PPO 目标](#14-gaev-trace-与实际-ppo-目标)
- [15. C/D/E 与其他可选机制](#15-cde-与其他可选机制)
- [16. 正式评测、统计单位与模型发布](#16-正式评测统计单位与模型发布)
- [17. 已有结果及其证据边界](#17-已有结果及其证据边界)
- [18. 工程、运行身份与已实施修复](#18-工程运行身份与已实施修复)
- [19. 风险清单与建议实验顺序](#19-风险清单与建议实验顺序)
- [20. 请求专家逐项审阅的问题](#20-请求专家逐项审阅的问题)
- [21. 讨论流程与意见交付模板](#21-讨论流程与意见交付模板)
- [22. 符号与术语表](#22-符号与术语表)
- [23. 源码、证据和复核入口](#23-源码证据和复核入口)
- [24. 外部文献与本项目的关系](#24-外部文献与本项目的关系)

## 1. 任务与麻将背景

### 1.1 游戏单位与决策

项目处理四人立直麻将。四名玩家轮流摸牌、打牌，并可对其他玩家的打牌作出吃、碰、杠、荣和或放弃等响应。一局牌称为 `kyoku`；一场通常由东场、南场及连庄等组成，称为半庄 `hanchan`。**一局和一场不是同一个统计单位。** 最终顺位通常由整场结束时四人的分数决定。

学习系统在玩家有可表达决策时提取状态和动作标签。某些事件只更新环境，不对应一个训练样本；一次杠决策还可能拆成“是否杠”和“杠哪一种牌”。因此不能将 JSON 事件数、打牌数、神经网络调用数和 RL 轨迹长度视为相同。

| 麻将术语 | 对机器学习问题的影响 |
| --- | --- |
| 手牌 | 当前玩家知道自己的牌，通常不知道其他三人的完整手牌 |
| 牌河 | 已打出的公开牌，包含历史行为信息 |
| 副露 | 吃、碰、明杠等公开组合，会影响合法动作和手牌结构 |
| 向听数 | 距离听牌还需要多少次有效改良；可作为工程特征或辅助标签 |
| 听牌 / 和牌 | 已有等待完成牌型的机会 / 成功完成牌型并结算 |
| 放铳 | 打出的牌让对手荣和，需要支付点数；风险具有明显尾部 |
| 立直 | 满足条件后支付供托并声明听牌，后续行动受约束 |
| 宝牌、赤牌、里宝牌 | 影响得分；部分信息公开，部分仅在特定条件下揭示 |
| 振听 | 某些情况下不能荣和，必须由规则引擎正确处理 |
| 本场、供托、连庄 | 使局数和分数演化不固定，影响终局目标的信用分配 |
| all-last | 接近整场结束的关键阶段，当前 SL 对部分辅助任务增加权重 |

本文不将所有平台的麻将规则当作完全相同。终止条件、排名同分处理、流局、多家和牌、赤牌和计分细节由实际引擎及评测配置决定；跨平台比较必须冻结这些语义。

### 1.2 数学问题

可以把它视为具有部分可观测性、四个相互影响策略和随机环境的序贯博弈。对玩家 i，完整事件历史为 h，部署可用特征为 x=f(h_i)，合法动作集合为 M(x)。actor 学习：

```text
πθ(a | x, M),    a ∈ M
```

对手策略、牌山随机性和玩家此前行为共同决定后续状态与终局结果。当前 actor 是前馈网络，历史被编码进固定维度特征；它不是读取完整历史序列的递归策略，也没有被证明得到一个充分的信念状态。

训练 critic 时可以额外使用模拟器或牌谱还原的隐藏信息 z，但部署 actor 仍只能使用 x。因此核心问题不只是 critic 预测误差，还包括 `V(x,z)` 是否与当前策略、目标和 advantage 估计一致。

### 1.3 “最强”必须落到效用函数

当前存在两个不同的效用：

| 用途 | 一位 / 二位 / 三位 / 四位 |
| --- | --- |
| 已核验训练 `env.pts` | 6 / 4 / 2 / 0 |
| 训练中心化值 | 3 / 1 / -1 / -3 |
| 正式 `avg_pt` | 90 / 45 / 0 / -135 |
| 正式 pt 除以 45 | 2 / 1 / 0 / -3 |

等距位次奖励倾向于改善平均顺位；正式 pt 对四位处罚更重。这两组值不是正比例加常数关系。例如确定第三与一半第二、一半第四，在训练等距奖励下同分，在正式 pt 下分别为 0 和 -45。

**待专家与项目负责人决定：** 最终主效用究竟是什么？建议若正式发布仍以 `avg_pt` 为准，就把 `[2,1,0,-3]` 作为统一效用候选。它会改变 Oracle 标签及价值尺度，不能只改评测脚本或把旧 critic 输出统一乘一个常数。该建议尚未实施。

## 2. 整体方案及当前阶段

### 2.1 系统链路

```text
人类牌谱 / 模拟日志
    │
    ├─ Rust 规则还原、合法动作、可见特征、部分隐藏信息
    │
    ├─ GRP：局级前缀 → 最终四人排名排列概率
    │             └─ 可供历史奖励预测链路使用
    │
    ├─ SL：可见状态 → 人类动作 + 辅助标签
    │             └─ canonical / S70 / Long-ABC 候选
    │
    └─ Oracle critic 预训：可见 + 隐藏 → 四人的真实 return
                  └─ 离线选择与资格确认

冻结的 SL actor + 合格 critic + 指定对手分布
    → 模拟采样 client → 参数 / 牌谱 server → RL trainer
    → value / GAE 或可选 V-trace → PPO 风格更新
    → 独立正式 1v3 → 发布模型

部署：可见状态 → actor → 合法动作；Oracle 不进入部署 actor
```

图中的最后几步是完整系统能力与拟推进路线。**当前运行重点是独立 Oracle critic 预训练，并不是已有一个经确认强于 SL 的长期 RL 主线。**

### 2.2 组件职责

| 组件 | 输入 | 输出 / 作用 | 当前边界 |
| --- | --- | --- | --- |
| `libriichi` | MJAI 风格事件、规则状态、seed | 环境、特征、合法动作、数据标签 | Rust / PyO3，原生扩展版本影响语义 |
| 可见 actor | 1012×34 可见特征 | 46 动作概率 | 当前 v4 / GN / categorical |
| SL 辅助头 | 同一个可见表示 | 顺位、对手状态、危险度 | 通过监督任务改善表示，不能当作终局牌力证明 |
| GRP | 每局开头的 7 维信息序列 | 24 种完整排名排列的分布 | 与 Oracle 真实回报标签独立 |
| Oracle critic | 可见 1012×34 + 隐藏 217×34 | 相对视角四人的价值 | 当前 dual tower、精确零和、MSE |
| online trainer | 轨迹、行为版本、actor / critic | PPO 风格策略和 value 更新 | 能力存在；收益待正式验证 |
| `1v3` | 一个 challenger、三个指定对手、固定牌山 | 四座轮换的完整半庄结果 | 最终强度判据 |
| 推理集成 | 对局事件、可见状态 | 合法动作、协议响应 | 接口兼容与牌力必须分别验证 |

### 2.3 当前模型身份

| 对象 | 已核验身份 | 当前可下的判断 |
| --- | --- | --- |
| canonical | `mortal/checkpoints/sl_canonical.pth`，内部 step 15000 | 当前发布基准；step 是该阶段计数，不是全部历史训练量 |
| S70 | `best_action_score.pth`，内部 step 390000 | 后续 SL 候选，也是当前 Oracle 初始化来源 |
| Long-ABC | 后续持续课程候选 | 远端当前状态未直连确认；不凭旧快照宣称训练完成 |
| Oracle adaptive best | 当前 run 的 `adaptive_best.pth`，本次读取内部 step 150000 | 某个选择规则保存的 best；不等于最新训练进度或已经具备 actor 接入资格 |
| 旧 C/D/E RL 候选 | 多组有完整日志的历史实验 | 重算未确立正式优于 canonical |

Oracle 当前 run 标识为 `s70_broad_to_recent_strong24m12m_adaptive_sf200_wd0_20260901_r1`，case 是 `phases/phase_a/sf_lr200_wd000`。不同 `best_dev`、`best_primary`、`adaptive_best` 可能对应不同准则和时间，不合并成一个含混的“最新最好模型”。

## 3. 数据、切分、采样与恢复

### 3.1 原始数据到张量

人类牌谱提供事件序列、已知手牌和最终结果等；模拟日志来自固定版本环境与策略。Rust loader 重放事件，逐个视角维护状态，生成可见 obs、合法动作 mask、行为 action、所在局 `at_kyoku`、必要辅助标签，以及 Oracle 隐藏特征。Python 负责组织文件池、批次、目标、shuffle、worker 分配和训练。

一个半庄通常包含多个相邻状态，也可以生成多个玩家视角。它们共享牌山、终局结果和大量历史，因此不是独立样本。训练可按状态做损失，但验证误差和显著性不能把所有状态当作独立游戏。

当前课程 cache manifest 提供了下面的真实构建统计。它们是源文件和 cache chunk 的数量，不等于重新按内容去重后的唯一半庄数、独立玩家数或实际完成的训练曝光次数。这些后者仍需补数据卡。

| Oracle source bucket | 当前划分月份 | 源文件数 | cache chunks |
| --- | --- | ---: | ---: |
| early | 早于 202112 | 1,785,942 | 111,622 |
| mid | 202112–202211 | 170,341 | 10,647 |
| old_regression | 202212–202311 | 165,313 | 10,333 |
| recent_older12 | 202312–202411 | 184,362 | 11,523 |
| recent_12 | 202412–202511 | 178,083 | 11,131 |
| 有效源文件合计 | 不含排除的一份无效文件 | 2,484,041 | 155,256 |

manifest 记录原始源文件 2,484,042 份，排除 1 份后得到上述有效数量；每个 chunk 最多打包 16 份日志，cache 总计 11,507,515,593 bytes，约 10.72 GiB。old_regression bucket 不进入当前课程训练池。每阶段加权池长度为 144,923 个 chunk 引用，其中可能重复引用同一 chunk；不等于每阶段拥有同样数量的唯一游戏。

dev=16,194 源文件，sealed test=12,188 源文件；这里只读 manifest 的数量和指纹，没有读取 test index 或 test 内容。旧分布 quick eval 取 64 个 chunk，不应将其误记为 64 个独立游戏。源牌谱的完整来源质量分层、原始玩家身份与全流程使用史尚未重新核验，本文不据文件数量推断全部样本都来自某个固定强度等级。

### 3.2 时间窗口：不同阶段有不同含义

SL 课程构造器当前定义：

| 名称 | 时间范围 | 作用 |
| --- | --- | --- |
| early | 200901–202012 | 早期广域数据 |
| mid | 202101–202212 | 中期数据 |
| recent_24 | 202301–202412 | 课程中的最近 24 月训练窗口 |
| recent_12 | 202401–202412 | 课程中的最近 12 月训练窗口 |
| recent_6 | 202407–202412 | 可选最近 6 月训练窗口 |
| full_recent | 202501–202512 | SL 外部分布验证 |
| monitor_recent | 202601 | SL 反复监控和选择 |
| old_regression | 202201–202212 | 历史分布诊断；可能和训练时期重合，不自动是独立 holdout |

这里的“recent”相对于该课程的固定截止点，不是每次运行时自动取最近月份。代码窗口是构造器事实；某个旧 checkpoint 实际用了哪些文件，还必须恢复当时 index 和 manifest。

当前 Oracle 另有时间隔离设计：训练到 202511，dev 为 202512，sealed test 为 202601；课程在广域池与 recent_24m / recent_12m 之间加权。不能把上表 SL 的具体训练月份直接套用到 Oracle。

### 3.3 阶段内 holdout 不等于全流程独立

Oracle 初始化来自 S70。S70 所属 SL 流程使用的 full_recent / monitor 时间范围可能与 Oracle dev / test 重叠。即使 Oracle trainer 从未打开 test，也不自动说明这些数据从未参与上游 actor 或特征的选择。

需要恢复 S70 **实际使用过** 的文件 ID、内容哈希、评测索引及选择记录，再将样本标记为：训练使用、上游选择使用、Oracle 调参使用、全流程未使用。现有证据支持存在需要审计的重叠风险，不足以直接宣布某个 checkpoint 发生了训练集泄漏。

当前 curriculum design 明确把人类 sealed test 标记为 closed，把旧 actor replay `sid0 / sid1` 标记为已消费且不得重复作为新确认集。本文和新探针均没有打开 sealed test 内容。

### 3.4 文件身份与去重

Python 数据模块用确定性哈希生成 source game ID。SL 的 `stable_source_game_id` 使用带 personalization 的 BLAKE2b；Oracle 也由 source name 得到确定性 ID。**确定性文件名 ID 不等于内容去重 ID。** 同一牌谱被复制、改名或转换缓存路径后，若没有稳定的原始身份映射，可能被当作新游戏。

建议数据卡至少保存：原始游戏 ID、内容哈希、来源时期、格式版本、是否增强、缓存成员名、玩家视角、split 归属和上游使用历史。应以原始游戏为切分单位，在生成四人视角、状态和增强样本之前完成隔离。

### 3.5 状态采样与数据增强

SL 保存配置中启用数据增强；事件层对牌进行变换后重新编码。当前 `Tile::augment` 交换万子与筒子，保留索子、字牌和赤牌属性，并非全部花色置换群的穷举。合法动作和标签必须与事件变换一致，赤牌和吃牌方向尤其需要回归验证。

当前 Oracle 明确 `enable_augmentation=false`、`augmented_first=false`。它依靠 native hash state fold 控制每次流过数据时生成的状态数量：训练 fold_count=64，验证 fold_count=128。fold 并非将每个原始时间步平均划分成固定 64 或 128 步的间隔；实际由哈希选择，保留数随机波动，部分玩家视角可能没有样本。

这项性能优化在 feature 编码前筛状态，存在第 11 节的标签语义问题。不能把 state fold 当作完全不影响监督目标的纯性能参数。

### 3.6 worker、流式遍历与恢复

Oracle loader 先对文件列表做全局确定性 shuffle，再按 worker 分片。stream pass 会改变 shuffle seed 与训练 fold index；worker 内另有 buffer shuffle。这样设计是为了避免每个 worker 独立随机打散后产生重复或遗漏，并使长流训练可恢复。

配置中的 `num_epochs=1` 描述单次迭代行为，外层可以不断重建流并推进 pass；它不表示整个训练只看一遍数据。训练 step 是 optimizer 更新计数，必须结合 batch、保留比例、实际文件曝光次数解释。

恢复保存 safe progress token，包含 worker、stream pass、文件偏移等。为了不漏掉尚在预取或 buffer 中的样本，偏移可以回退到安全的文件 batch 边界。因此：

- 模型、优化器、scaler、scheduler 可以恢复到保存点；
- 数据可以从保守位置恢复，但可能重复一部分在途样本；
- Rust 隐藏补全 RNG、异步 worker 调度和预取进一步影响逐位复现；
- “有 cursor”不等于“中断前后每一个更新完全一致”。

对于严格配对实验，需要显式冻结数据序列与输入，或测量恢复带来的轨迹差异；不能单凭 checkpoint 能 load 就称为 exact 对照。

## 4. 状态表示与动作接口

### 4.1 固定张量契约

| 项目 | 当前值 |
| --- | --- |
| 模型 / 特征版本 | v4，`MAX_VERSION=4` |
| 可见观测 | `[B,1012,34]` |
| Oracle 额外观测 | `[B,217,34]` |
| 牌种轴 | 34 种基础牌类型 |
| 动作空间 | `ACTION_SPACE=46` |
| GRP 每局输入 | `GRP_SIZE=7` |

34 维轴是牌类型，不是 34 个时间步。特征通道携带牌数、布尔状态、历史位置、标量广播和工程计算结果。赤五在观测中有单独编码，在动作中有额外 ID。特征通道数变化必须同时更新 Rust 编码、Python 网络和 checkpoint 兼容性。

### 4.2 可见信息的主要家族

`libriichi/src/state/obs_repr.rs` 维护精确顺序和版本分支。v4 包括：

| 家族 | 信息来源与作用 |
| --- | --- |
| 自己手牌 | 各牌计数、赤牌信息；当前可合法观察 |
| 分数和顺位 | 四人分数、相对顺位与局势 |
| 场况 | 场风、自风、庄家、局数、本场、供托、剩余牌等 |
| 公开宝牌信息 | 已揭示宝牌指示牌及相关编码 |
| 牌河历史 | 按位置编码弃牌、摸切 / 手切、立直相关信息及摘要 |
| 副露和杠 | 四家公开组合、赤牌和杠的状态 |
| 可见牌统计 | 已公开牌与剩余可能张数等 |
| 自己手牌分析 | 向听、等待、振听及合法决策上下文 |
| 特定近期行为摘要 | 最近手切、立直宣言牌等 |
| 单人手牌计算表 | required tiles、未来若干步听牌 / 和牌概率和 EV 特征 |

表格解释语义家族，不伪造一份未经逐通道核对的 offset 表。改变编码时应从源码生成 schema，并用固定牌谱做逐通道 diff；手工复制一千多个通道的说明很容易再次过期。

### 4.3 `search=false` 仍有工程计算特征

v4 编码器中存在单人手牌计算表相关的 123 通道区块，包含 required-tile 映射、不同未来步数的听牌 / 和牌概率和 EV 等。这些计算成为神经网络输入，独立于推理期可选的动作重排搜索。

因此当前系统不是完全从原始牌面端到端学习。训练和评测关闭 `search` 指关闭额外的搜索 / 重排机制，不能写成“输入完全没有规则搜索和 EV 计算”。这也提出一个重要归因问题：当前强度有多少来自学习到的表示，有多少来自工程特征？需要在同样训练预算和评测条件下单独消融，而不是推理时突然清零后直接比较。

### 4.4 46 个动作的精确分组

| ID | 通常语义 | 必须注意 |
| --- | --- | --- |
| 0–33 | 34 种普通牌的打牌选择 | 在选杠牌等二级上下文中也可能表示牌种选择 |
| 34–36 | 赤五的打牌选择 | 不能与普通五随意合并 |
| 37 | 立直 | 与随后打牌事件的拆分由 loader 决定 |
| 38 / 39 / 40 | 吃：低 / 中 / 高位置 | 同一张被吃牌对应不同组合 |
| 41 | 碰 | 消耗牌和赤牌细节由环境处理 |
| 42 | 杠的高层决策 | 可能有后续牌种选择，不能只按 ID 统计动作类型 |
| 43 | 和牌 | 荣和或自摸由上下文决定 |
| 44 | 九种九牌等受合法条件约束的流局声明 | 不是所有自动流局的训练动作 |
| 45 | 通过 / 放弃响应 | 必须区分真实选择与被别人更高优先级行动中断 |

所有策略概率先使用合法动作 mask。非法动作 logits 置为负无穷，再 softmax。训练不能允许非法标签混入；评测动作类别分析也要考虑上下文，否则可能把杠牌选择错误记为弃牌质量。

### 4.5 pass、多家和牌与时间步

Rust loader 不把所有“没有最终执行动作”的机会都当作 pass。其他玩家的高优先级动作可能使当前响应没有实际被接受。代码专门处理真实 pass 和多家和牌等情况。这是行为概率和 off-policy 修正正确性的前提：存入轨迹的 action 必须对应当时真正定义的决策。

旧数据结构仍有 `apply_gamma` 标记；现代完整轨迹 `value / GAE` 路径按其实际保留的动作序列递推，不能沿用旧注释推断所有奖励只在打牌时折扣。完整 action-scope 与采样时钟需要一并进入训练契约。

### 4.6 Oracle 217 通道

| 内容 | 通道计算 | 合计 |
| --- | --- | ---: |
| 三名对手的手牌和状态 | 每人：手牌计数 4 + 赤牌 3 + 向听 one-hot 7 + 向听标量 1 + 等待 1 + 振听 1 | 51 |
| 剩余活牌山 | 69 个位置，每个牌种 / 赤牌两个通道 | 138 |
| 岭上牌 | 4 个位置 × 2 | 8 |
| 宝牌指示牌 | 5 个位置 × 2 | 10 |
| 里宝牌指示牌 | 5 个位置 × 2 | 10 |
| 总计 | 51 + 138 + 8 + 10 + 10 | 217 |

模型确实看到了远比 actor 多的信息。人类日志中有些部分能根据已记录事件还原，有些需要补全；不同来源不能一概称为完全观测的真实环境状态。

## 5. 可见信息模型与辅助头

### 5.1 主干

当前 SL actor 使用 v4、192 channels、40 个 pre-activation ResBlock，显式使用 GN。`Brain.__init__` 的通用默认是 BN，但这不是当前训练的有效选择。

```text
x: [B,1012,34]
  → Conv1d(1012→192, kernel=3, padding=1, bias=False)
  → 40 × pre-activation residual block
  → GN + Mish
  → Conv1d(192→32, kernel=3, padding=1)
  → Mish
  → flatten: 32×34 = 1088
  → Linear(1088→1024)
  → Mish
φ: [B,1024]
```

每个残差块包含 GN、Mish、两层 3 宽卷积以及 SE 风格的通道注意力。当前 GN 使用 32 groups，eps=1e-3。注意力对牌种轴做平均池化和最大池化，共享降维 / 升维 MLP，192→12→192，合并后 sigmoid，再调制残差分支。

1D 卷积在牌种轴上共享局部滤波器，因此引入了牌种排列的归纳偏置；它并不天然严格等变于所有花色置换。不同花色边界与字牌的语义也不同。这是可讨论的架构选择，不是已证实的缺陷。

### 5.2 策略头

```text
φ: 1024 → Linear 256 → tanh → Linear 46 → legal mask → softmax
```

主路径是 `CategoricalPolicy`。代码中还保留 DQN / dueling 类及兼容接口，有些变量仍命名为 `dqn`；不能据此把当前算法描述为 Q-learning。

### 5.3 辅助头

| 头 | 输出 | 监督信息 | 是否通过 SL 反传到可见主干 |
| --- | --- | --- | --- |
| rank aux | 4 类排名 logits | 真实最终排名及局势权重 | 是 |
| opponent state | 三人 × 4 类向听 + 三人 × 2 类听牌 | 从完整牌谱还原的对手真实状态 | 是 |
| danger any | 37 种弃牌的放铳风险 logits | 可结算的真实危险标签 | 是 |
| danger value | 37 种弃牌的损失尺度预测 | 放铳点数，经 log1p 压缩 | 是 |
| danger player | 37 种弃牌 × 三名对手 | 对各对手的放铳标签 | 是 |
| tile / furo / hand-value regret | 其他可选辅助预测 | 工程反事实 / 近似标签 | 当前 S70 / canonical 的 SL `forward_loss` 没有训练这些头 |

privileged labels 用于训练可见表示，不等于把对手真实手牌直接送入 actor。但如果编码器、缓存或推理入口错误拼入 hidden tensor，就会形成信息泄漏，因此还需要接口和真实 smoke 保证边界。

### 5.4 参数规模

本次读取 canonical 与 S70 的 state dict，得到相同结构：

| 组件 | 张量元素数 |
| --- | ---: |
| visible Brain | 10,787,456 |
| policy head | 274,222 |
| rank aux | 4,096 |
| opponent aux | 18,432 |
| danger aux | 189,440 |
| actor 主干 + policy | 11,061,678 |
| 含上述 SL 辅助头 | 11,273,646 |

这些是已保存模型张量的元素数。不能从 checkpoint 文件字节数直接推断参数量，因为文件还可能包含优化器、scaler 和训练元数据。

## 6. GRP：全局结果预测

### 6.1 输入和预测对象

GRP 以整场中各局开头的信息序列预测最终四人的联合排名。每一步为 7 维：

```text
[grand_kyoku, honba, kyotaku, score_0, score_1, score_2, score_3]
```

分数按数据编码缩放，例如 25000 点表示为 2.5。GRP 不读取完整手牌、牌河或每一次动作。训练可从一场中构造多个局级前缀，它们共同指向同一个最终排名排列；这些前缀必须留在同一个 split。

四个人的完整排名有 `4! = 24` 种。GRP 对 24 类做分类，再通过固定映射得到每个玩家的一至四位边际概率。相比四个互相独立的分类器，这种方式在表示层面保证联合排名可行：每个玩家有一个名次，每个名次恰好一个玩家。

### 6.2 当前结构和训练

```text
7 维序列 → GRU(hidden=384, layers=3)
  → 三层最终 hidden 拼接：1152
  → Linear 1152→1152 → ReLU → Linear 1152→24
```

有效配置使用 float32。通用模型构造器的旧默认容量和 dtype 不代表当前配方。训练使用 AdamW，学习率由配置设置为 2e-5；`ReduceLROnPlateau` 的 factor=0.5、patience=2、threshold=0.0005、cooldown=1、min_lr=1e-6。当前配置还有 label smoothing=0.1、batch=1024、min_epochs=8、max_epochs=30、early-stop patience=8、最少两次 LR reduction 等约束。

代码中 GRP AdamW 构造没有像 SL 那样显式传入所有 betas / weight decay；这些参数需随 PyTorch 版本和 optimizer state 记录。不能仅凭主配置存在一个同名 optimizer 字段，就认为所有训练入口都消费它。

### 6.3 奖励预测链路

设 GRP 给玩家 i 在局 k 开始的最终名次分布为 q(k,i,r)，排名效用为 u(r)，则预测效用：

```text
F(k,i) = Σ_r q(k,i,r) u(r)
局间预测奖励 = F(k+1,i) - F(k,i)
```

历史 reward calculator 可在末端追加真实结果编码，并可使用平滑后的最终分布；label smoothing=0.1、四分类时，正确名次概率为 0.925，其余各 0.025。这样做能生成较密的局间信号，但引入了预测误差和目标变换。

**当前 Oracle `score_rank_mc` 分支直接从真实分数顺位和终局结果计算标签，并不调用 GRP 来生成监督标签。** 源码变量 `grp_feature` 在这条路径只是承载局级元数据，其名字不等于“使用 GRP 神经网络预测”。

### 6.4 checkpoint 与审阅问题

`best_loss` 默认供下游使用；`best_acc` 仅用于受控对照；`latest` 用于续训。GRP 的准确率 / loss 进步不自动说明 downstream actor 更强。

需要专家判断：24 类联合表示是否合适；局级输入是否足够校准；多前缀训练是否过度加权长半庄；label smoothing 怎样影响 reward calibration；在当前真实 reward / Oracle 路线下，是否还有必要优先扩大 GRP，而不是投入目标和验证修复。

## 7. SL：监督学习目标和训练流程

### 7.1 主任务是行为克隆

对每个有效状态 x、人类动作 a 和合法集合 M：

```text
L_policy = mean[-log πθ(a | x,M)]
L_SL = L_policy + L_rank + L_opponent + L_danger
       + w_distill × L_search_distill（仅启用时）
```

当前 `forward_loss` 中策略交叉熵是样本平均，不按最终顺位、回报绝对值或优势加权。TOML 中继承的 `awr_beta`、`awr_clip` 等字段不表示这个 SL 入口在做 AWR。辅助任务的 context weighting 也没有自动变成策略标签权重。

行为克隆学到的是牌谱行为分布，不是每一步最优动作。数据质量、时代差异、玩家风格、多个近似等价动作和错误示范都影响其极限。较低 NLL 可以来自更好地拟合人类，也可以偏向高频简单决策；因此必须与切片诊断和正式牌力共同判断。

### 7.2 顺位辅助损失

当前样本权重按基础权重、巡目、南场、all-last 和可选分差因子组合，并受上限约束：

```text
w_rank(x) = clip(base × f_turn × f_south × f_all_last × f_gap, max=w_max)
L_rank = mean[w_rank(x) × CE(rank_head(φ(x)), final_rank)]
```

canonical / S70 保存配方中的关键值：

| 参数 | 数值 |
| --- | ---: |
| base | 0.001548 |
| south_factor | 1.59 |
| all_last_factor | 1.617 |
| max_weight | 0.00516 |
| gap_focus_points | 4000 |
| gap_close_bonus | 0，当前该额外加权不生效 |
| 巡目：early / mid / late | 1.0 / 1.05 / 1.15 |
| 巡目边界 | early≤4，late≥12 |

这里预测的是结果相关辅助变量，不是以 actor 的动作 Q 值替代人类标签。损失权重很小不代表梯度一定很小，实际取决于各头输出、样本数量、归一化与共享主干梯度。

### 7.3 对手状态辅助损失

输出三名对手的向听分类和听牌分类。每个状态先对三名对手的分类损失做平均，然后组合和按巡目加权：

```text
L_opponent = w_opp × mean[f_turn(x) ×
    (w_shanten × mean_j CE(shanten_j)
     + w_tenpai × mean_j CE(tenpai_j))]
```

当前 `w_opp=0.00135`；`w_shanten≈0.85065684`、`w_tenpai≈1.14934316`；early / mid / late 权重为 0.2 / 1.0 / 1.6，边界同上。

向听和听牌具有相关性，标签又来自隐藏状态。需要判断它们是否改善 actor 的信念表示，还是形成冗余梯度；应报告听牌基率、各巡目校准、对不同公开信息条件的误差，以及移除该任务后的正式牌力。

### 7.4 危险度辅助损失

危险头针对 37 种弃牌表示，结合 `danger_valid` 与当前合法弃牌 mask。无效上下文不应被当作安全负样本。内部有三部分：

1. any：是否对任意对手放铳；
2. player：对三名对手分别是否放铳；
3. value：正例上的损失点数，先截断并压缩再回归。

any / player 使用逐状态的平衡 BCE：分别求该状态内正例均值和负例均值，两类都存在时各占一半；只有一类存在时使用存在的类别；没有 eligible 项则贡献 0。它并不是数据集全局 class weight。

value 的目标为：

```text
y_value = log(1 + clip(points, 0, 96000)) / log(1 + 96000)
L_value_pos = SmoothL1(sigmoid(value_logit), y_value)
```

仅正例参与该项；没有正例的状态该项为零。三项混合权重归一化后约为：any 0.09042179、value 0.81804029、player 0.09153792。外层权重为 0.00804，前 1000 optimizer steps 线性 ramp；focal gamma=0，当前没有额外 focal 调制。危险任务的巡目权重为 0.05 / 1.0 / 2.5。

需要注意这套归一化改变了训练分布：一个只有少量 eligible 牌的状态和一个有很多 eligible 牌的状态，可以拥有相近的状态级权重。正例回归还受到“可构造真实放铳标签”的条件限制，不能把它简单理解为所有状态的真实风险期望。

### 7.5 搜索蒸馏与 regret 的实际状态

SL 存在可选搜索蒸馏路径：planner 生成概率，筛选 active / hard 状态及 teacher gap，teacher probabilities detach 后对 student 做交叉熵。当前 canonical / S70 主线没有把它当作默认启用机制。

保存配置中还可能出现 tile efficiency、furo regret、hand value regret 的非零字段，但本次核验的 SL checkpoint 没有这些头，SL `forward_loss` 也没有对应训练项。不能按配置键存在与否统计“已启用的辅助任务数量”。后述在线可选 regret 路径还使用 `phi.detach()`，不会通过这些头训练 actor 主干。

### 7.6 优化器与有效配置

SL 使用 AdamW，将 Conv / Linear 的权重列入 decay 组，其他参数不做同样的 weight decay。当前保存配置的 weight_decay=0.1、betas=(0.9,0.999)、eps=1e-8；`max_grad_norm=0` 表示未启用该梯度裁剪阈值。

peak learning rate 先读取 `[supervised].lr`，缺省才读取通用 optimizer scheduler 的 peak。scheduler 的总步数也有阶段内覆盖。审阅时应读调用路径与 checkpoint 有效值，不能只看基础配置里某个醒目的 `max_steps`。

| 已保存配置 | canonical | S70 |
| --- | ---: | ---: |
| batch | 1024 | 1024 |
| seed | 20263312 | 20263312 |
| train workers / file batch / prefetch | 4 / 10 / 3 | 4 / 10 / 4 |
| val file batch / prefetch | 8 / 5 | 7 / 5 |
| monitor 间隔 | 20000 steps | 10000 steps |
| monitor batches | 512 | 1024 |
| 阶段最大步数 / cosine horizon | 15000 | 420000 |
| 本次 checkpoint 内部 step | 15000 | 390000 |
| warmup | 1000 steps | 1000 steps |
| warmup 初值 / cosine 最终值 | 1e-8 / 1e-5 | 1e-8 / 1e-5 |
| 有效 peak LR（两者无 SL lr 覆盖） | 3e-4 | 1e-4 |
| Oracle actor | 关闭 | 关闭 |

这张表描述保存配置，并不把两个阶段解释为只差训练时长的严格配对实验。初始权重、数据曝光、阶段切换和选择记录也可能不同。完整配置见附件；这里已核对两者均无 `[supervised].lr` 覆盖，peak 来自通用 scheduler。

### 7.7 SL 指标应如何理解

| 指标 | 回答的问题 | 不能回答的问题 |
| --- | --- | --- |
| policy loss / NLL | 人类动作的平均负对数概率是否下降 | AI 是否更能赢得整场 |
| action accuracy | top-1 与示范动作的一致率 | 多个近似等价动作的实际差距 |
| 动作 / 场景切片 | 吃碰杠、立直、进攻防守、关键局势是否变化 | 单独切片是否足以替代总效用 |
| auxiliary loss | 特定辅助任务是否拟合改善 | 总 loss 更低是否意味着 policy 更好 |
| `full_loss` | 主任务和辅助任务共同成本 | 不同辅助权重方案的可比 policy 排名 |
| formal `1v3` | 在固定对手和规则上的实际效用差 | 对任意对手、任意平台都更强 |

当前方案之间比较 policy 应使用 `comparison_recent_loss` / `recent_policy_loss` 等一致口径；不同辅助权重的 full loss 只能解释辅助任务成本，不能直接选出更强 actor。

## 8. SL 课程、选择与现有候选

### 8.1 课程构造

课程支持 broad→recent 的 A/B/C 阶段，也保留其他顺序作为实验选项。strong 权重组合为：A 阶段 recent 60%、mid 25%、early 15%；B 阶段 recent 90%、replay 10%；C 阶段 recent 98%、replay 2%。不同 window profile 决定 recent 是 24/12 月、12/6 月或 6/6 月。

这些百分比是构造训练文件池的目标权重，不等于每次 optimizer 更新中严格固定的状态比例。不同文件包含不同数量的决策，同一文件还可能按 pool expansion 重复，实际曝光需从采样记录量化。这里的 replay 是历史人类数据混入，与在线 RL 的过期策略 replay 不同。

主张是先学广域基本决策，再适应近期高相关分布，同时留少量早期数据减轻遗忘。**这个叙述是动机，不是已经隔离证明的因果机制。** 若要证明课程优于静态混合，应匹配初始化、总样本曝光、优化步数、LR 预算和最终选择协议。

### 8.2 scheduler 与自适应课程

构造器支持全 cosine、全 plateau、A/B cosine 加 C plateau 等安排，也有 full_dynamic 模式。自适应 SL 以 policy loss 为 primary，action accuracy 和 old-regression policy loss 为次指标，并保存有限的候选 portfolio。

当前共享 adaptive controller 的统计和 guardrail 问题见第 12 节；不能假定名称里有 guardrail 就表示选 best 时对所有退化做了否决。SL 包装层对某些“降 LR 后仍未产生新改善又继续降 LR”的循环设有停止逻辑，但这不取代统计功效设计。

### 8.3 离线 selector

`run_sl_ab.py` 中的候选选择先找最小 full-recent policy loss，将 loss 在 epsilon 范围内的候选纳入 eligible，再用动作质量 / 场景质量等 key 排序。它是一个可复现的离线筛选启发式。

问题在于：epsilon 和复合 action score 的统计尺度、与正式 pt 的相关性、不同动作类别的权重，以及多轮试验反复使用同一验证集时的选择偏差，需要独立校准。不能把函数返回值叫作 winner 就认为已经通过牌力确认。

### 8.4 发布层与候选层

canonical 保留发布身份；S70 和后续 Long-ABC 是候选。已有 canonical 对第一替补的巨大场数比较也很接近，不能写成 canonical 的优势已经确定。另一方面，S70 的保存 NLL 更好也不能直接覆盖 canonical。

建议将候选数量限制在少量冻结 finalist：先通过同一输入、同一 evaluator 的离线复核，再在新牌山上筛选，最后用独立确认集决定是否发布。保留 no-update / canonical 对照，使未观察到显著增益时有明确的默认退回对象。

## 9. Oracle critic：输入、结构与梯度路径

### 9.1 设计动机

麻将的随机性很强，可见状态下预测回报的方差可能很大。训练期若知道对手手牌和牌山等额外信息，critic 有可能更好地解释结果，把更低噪声的 advantage 提供给只看可见信息的 actor。

但“更多信息让回归更准”不自动推出“policy gradient 更正确”。完整历史是否保留、隐藏变量是否因动作而选择性揭示、bootstrap 的条件期望、数据行为策略与当前策略是否一致，都影响该推论。本文采用独立 Oracle critic 路线，仍请求专家检验其理论条件和实证价值。

### 9.2 当前 dual tower

两个主干分别编码可见和隐藏特征：

```text
φv = visible_encoder(x)       # 1012→192×40→1024
φo = oracle_encoder(z)        # 217 →192×40→1024
φ  = φv + residual_mlp(LayerNorm(concat(φv,φo)))
raw_value = Linear(Mish(Linear(φ, 256)), 4)
V_i = raw_value_i - mean_j(raw_value_j)
```

residual MLP 输入 2048，hidden 1024，输出 1024。末层在初始化时置零，因此初始融合表示等于 visible 分支输出。前向中加入隐藏塔并不表示初始化第一步它就已有非零贡献；最后融合权重变为非零后，梯度才开始正常流向较早的隐藏分支。这是初始化设计的直接后果。

可见分支从 S70 迁移；Oracle tower 使用 visible-transfer，第一层做 hand-aligned 初始化，以适配不同的输入通道。当前 `train_scope=all`，并非一直冻结整个 visible tower。

### 9.3 `all_players` 的含义

四个输出按当前样本玩家的相对顺序排列：

```text
absolute player ids = [(player_id + offset) mod 4 for offset in 0..3]
p0 = 自己；p1 / p2 / p3 = 依次循环的其他玩家
```

训练对四人的真实目标进行回归；在线 actor 使用 p0 的 advantage，但其他输出提供联合局势约束和训练信号。当前不把 `all_players` 简化为只训练 p0。是否给 p0 更大固定输出权重，需要预声明 primary 与尾部 guard 并做配对实验，不能按样本实际 outcome 的大小加权来迎合主指标。

四人使用中心化排名效用，真实四人回报在一致时钟下和为零。网络直接减去输出均值，保证精确零和。这个约束减少不必要自由度，但 **零和误差为零是结构性质，不是预测质量证据**。

### 9.4 独立参数与 actor 的关系

预训练时只训练 Oracle critic。接入在线 RL 时，独立 `oracle_brain` 及 value head 不与 actor 共享参数；value 回归损失不会沿 critic 网络直接更新 actor。actor 通过 advantage 改变策略，critic 则在自己的参数空间拟合价值。

这与“Oracle actor 先看隐藏信息，再逐步撤掉”是不同方案。仓库保留 Oracle guiding / dropout / ramp 等历史能力，但当前路线没有默认给部署 actor 开启隐藏输入。也不能把 sidecar critic 的良好回归当作 Oracle actor 的成功蒸馏。

### 9.5 参数量与替代结构

本次 Oracle adaptive best：`oracle_brain` 24,268,864 个张量元素，`value_net` 263,428，共 24,532,292。相比一个约 11M 的部署 actor，训练侧 critic 的计算和内存负担更高。

代码还支持单塔、其他融合方式和 HLGauss 等分布式 value head。当前有效方案是 `dual_tower + residual_mlp + MSE + exact_zero_sum`，不是把所有可选架构同时组合。

若专家建议 HLGauss、分位数或其他 distributional critic，需要同时定义支持范围、尾部截断、预测均值、actor 使用的 advantage 及比较准则；“分布预测更丰富”不能替代与当前 MSE 基线的同目标对照。

### 9.6 隐藏输入的真实性与随机补全

`invisible.rs` 在不信任 simulator seed 时，根据记录还原已知摸牌、宝牌等，再把未知剩余牌随机填入牌山。当前随机补全使用 `filler.shuffle(&mut rng())`。固定文件列表、fold 和 Python seed 不保证这一步固定。

人类日志的 hidden input 同时包含真实记录信息与补全信息。它可能是有用的辅助状态，但需要明确它对应怎样的条件分布。不能把事后记录到的部分未来牌序与完全已知、预先固定的原始牌山混为一谈。

`trust_seed=true` 只可在拥有正确 seed 且生成器版本、事件与重建完全一致的模拟日志上使用。重复解码变得确定不等于其重建正确。对人类日志直接打开这个开关不能补回原本未知的信息。

## 10. Oracle 标签与奖励的精确定义

### 10.1 排名效用和当前顺位

令原始排名效用为 u=[6,4,2,0]，中心化 `u_c=u-mean(u)=[3,1,-1,-3]`。局 k 开始时，用四人分数排序得到各自的当前名次；同分按绝对 player ID 升序确定。令：

```text
Φ(k,i) = u_c[rank_of_player_i_at_start_of_kyoku_k]
U(i)   = u_c[final_rank_of_player_i]
```

`score_rank` 的含义是“根据分数确定顺位，再取排名效用”，**不是原始点数差，也不是分数除以某个比例后的奖励**。

### 10.2 三种可用标签

| 模式 | 定义 | 统计含义 |
| --- | --- | --- |
| `terminal_rank` | 对该视角状态使用 U(i) | 终局中心化排名效用 |
| `rank_delta` | U(i)−Φ(k,i) | 从当前局初顺位到终局的效用变化 |
| `score_rank_mc` | 在动作序列上对局间顺位变化做 discounted return | 当前运行模式；依赖奖励落点和动作时钟 |

MC 标签来自真实结果，包含未来人类行为和对手行为。因此其条件期望是相关行为分布下的回报，不自动等于当前 AI actor 的 `V^π`。

### 10.3 局间 reward

```text
r_kyoku(k,i) = Φ(k+1,i) - Φ(k,i)   （非最后一局）
r_kyoku(K-1,i) = U(i) - Φ(K-1,i)  （最后一局）
```

局间 reward 被展开到该视角的决策序列。一般步骤 reward=0；当下一保留步骤进入后续局时，当前步骤承接这段局间 reward；最后保留步骤承接剩余到终局的 reward。若中间缺少完整局，`expand_kyoku_rewards_to_steps` 会将跳过局的 reward 求和，避免直接丢失结算。

### 10.4 MC 递推与边界

```text
G_T = 0
G_t = r_t + γ G_(t+1),  t = T-1,...,0
```

当前 `discount_gamma=0.999`。终止是整个轨迹 / 半庄末端；并不是每个 kyoku 结束都把未来回报清零。四人目标按同一视角的 `at_kyoku` 序列展开，再旋转为 p0…p3。

在 gamma=1、奖励无遗漏、目标完全一致时，局间差分可望远镜相消为 `U−Φ(current)`。gamma<1 时，中间顺位改善和回落发生的时间会进入目标；它不再仅由终局顺位和起始顺位决定。

### 10.5 一个不依赖随机牌山的例子

某玩家局初效用先为 3，中途降到 -1，最后升到 1。两次差分为 -4 和 +2。若第二次差分比第一次晚 Δ 个动作，前者时点上的 return 为：

```text
gamma=1:      -4 + 2 = -2
gamma<1:     -4 + gamma^Δ × 2
```

即使终局不变，Δ 不同也会改变标签。这个例子同时说明：改变动作计时单位、删除中间动作或把 Δ 压缩成 1，都可能改变目标。

### 10.6 奖励设计需要共同决定的部分

如果希望严格优化未折扣终局 pt，可比较终局 reward / gamma=1；如果希望使用折扣和 shaping，应明确基础终局奖励与势函数，并推导与 `gamma` 相匹配的变换及终止边界。不能只对现有局间差分乘一个 gamma 就宣布理论上完全等价。

无论最终选择什么，训练签名都应包含：效用向量、中心化方式、排序同分规则、reward source、return mode、discount gamma、决策时钟、跳步语义、终止 / 截断处理、target view 和信息来源。

## 11. 新发现：native fold 改变了折扣时钟

### 11.1 实现位置与因果链

本次为完整方案说明核验数据路径时发现并复现了一个新的契约问题：

```text
Rust Gameplay::add_entry
  → 按 native sample fold 判断是否保留
  → 不保留则提前 return
  → 只给保留状态追加 obs / action / at_kyoku

Python OracleTerminalValueDataset.populate_buffer
  → 取这个已变稀疏的 at_kyoku
  → oracle_step_value_targets(..., gamma=0.999)
  → discounted_returns 在稀疏序列上每次乘一次 gamma
```

Python 后处理 fold 分支则先在完整序列上生成 targets，再索引保留状态。因此 native fold 和 Python fold 不只是性能不同，可能给相同状态产生不同的标签。训练 native fold=64、验证=128；在线完整轨迹 GAE 又使用另一个保留密度。

### 11.2 合成控制实验

构造三局、每局 128 个决策的完整轨迹，固定局初分数和最终排名，无隐藏输入随机性。对同一初始状态，比较“完整轨迹先算标签再采样”和“先每隔若干状态采样再算标签”：

| gamma | 保留间隔 | 保留状态 | 完整标签 p0 | 先采样标签 p0 | 全部保留状态 / 四输出最大差 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.999 | 1 | 384 | -4.674370 | -4.674370 | 0 |
| 0.999 | 64 | 6 | -4.674370 | -5.982026 | 1.307656 |
| 0.999 | 128 | 3 | -4.674370 | -5.994002 | 1.319633 |
| 1.0 | 64 / 128 | 6 / 3 | -6 | -6 | 0 |

这是控制时钟的合成反例，固定间隔采样不冒充实际 native hash 采样，也不估计生产数据的平均偏差。

### 11.3 使用当前 runtime overlay 的真实 dev 反例

进一步使用一份此前输入审计已用过的人类 dev 牌谱，加载当前课程脚本指定的 `libriichi_native_fold_capacity_v2` 原生扩展。分别完整解码与 native hash fold，再按同一 player 下的“可见 obs 字节 + action + at_kyoku”逐状态匹配。所有保留状态都能在完整序列找到唯一对应。

| 视角 / 原始位置 | fold | 完整序列状态数 | fold 后状态数 | 完整标签：自己 | fold 后标签：自己 |
| --- | ---: | ---: | ---: | ---: | ---: |
| player 0 / index 42 | 64 | 147 | 1 | -0.207370 | 0 |
| player 0 / index 42 | 128 | 147 | 1 | -0.207370 | 0 |
| player 1 / index 81 | 64 | 154 | 2 | -3.774863 | -3.998000 |
| player 1 / index 81 | 128 | 154 | 1 | -3.774863 | -4.000000 |

gamma=0.999 时，该牌谱已匹配状态中四输出最大绝对差为 **0.345931**。gamma=1 时，相同测试中的差全部为 0。探针只使用 CPU，没有更新模型或访问 sealed test。

该反例还有一个实用后果：不仅连续回归值变化，`exact_zero / nonzero` 等切片成员也会变化，阈值尾部切片同样可能受影响。不能把修复前后的 guardrail 数值当作在完全相同标签分布上测得。

### 11.4 原生版本身份

本机普通 Python 环境中默认导入的扩展不含 `set_sample_fold`；当前课程通过 runtime overlay 使用另一份扩展。第一次直接运行探针因此报接口缺失，按课程脚本所指定的 overlay 加载后完成反例验证。没有重新编译或覆盖已安装扩展。

实际使用的 `libriichi.cp312-win_amd64.pyd` SHA-256：

```text
19dd5df4ce467b97737809fb4081033ff5c772b6bf867356ff1f24bc42cdd6bb
```

这说明仅记录 Python 环境名或仓库 commit 不足以复现训练。扩展路径、哈希和接口能力必须一起记录。

### 11.5 能下的结论与不能下的结论

**已确认：** native fold 在生成 MC 标签之前压缩决策序列；对相同实际状态，gamma<1 时可改变 return。只核对 reward 向量、gamma 数字和输出 shape，不足以保证离线与在线目标一致。

**尚未测量：** 当前全数据分布的标签偏差、best 排名改变比例、现有 critic 的实际价值损失、修复后 actor 是否更强。单局最大差不是这些量的估计。

**也不能反推：** 此前“完整轨迹 MC 与 lambda=1 GAE 对拍通过”无效。它仍验证了相同完整序列上的递推；只是没有覆盖 native fold 先删状态的生产路径。

### 11.6 建议的修复选择

| 选择 | 做法 | 需要验证 |
| --- | --- | --- |
| A：完整轻量轨迹先生成目标 | 保留所有决策的轻量时钟 / 局索引，先算完整 return，再只对被采样状态生成或保留昂贵特征 | native / Python fold、不同 fold_count 对同一状态的标签一致；性能与内存可接受 |
| B：显式持续时间 | 输出原始决策 index / duration，正确累积中间折扣 reward，并在跨步 bootstrap 使用 gamma^Δ | 对每种跨局、末端、无样本局与多次结算情况对拍完整递推 |
| C：改成 gamma=1 或终局目标 | 重新定义希望优化的效用，再统一所有路径 | 这是目标实验和模型迁移，不能冒充不改变语义的性能修复 |

仅把 bootstrap 写成 `gamma^Δ`，却仍无折扣地把中间所有 reward 相加，并不能普遍恢复原完整目标。建议先选择 A 作为最清晰的正确性参照，再评估 B 的优化必要性。

本次没有热修复活跃 loader。应在新源码快照中完成契约验证，给标签定义升级版本，并在同一份新验证集上重新评估旧 anchor / best / latest；不要沿用旧 best loss 作为修复后的可比基线。

复验产物：[合成反例](../../logs/expert_review/20260905/fold_target_probe.json)、[原生真实牌谱反例](../../logs/expert_review/20260905/native_fold_target_probe.json)。

## 12. Oracle 预训练的有效配置和自适应控制

### 12.1 当前损失

使用四人真实回报、MSE 和精确零和输出。等权时：

```text
L_critic = mean_states[mean_i (V_i - G_i)^2]
```

代码支持固定四输出权重及其 schedule，但当前配方不应被解释为按实际 outcome 大小进行样本加权。输出权重改变的是多任务容量分配，需与样本选择、primary 选择分开。

当前 exact_zero_sum=true，因此 `zero_sum_weight=0`；不额外依赖惩罚项近似满足零和。离线 primary 关注 p0 的 MSE，整体 all_players MSE 和多个 p0 切片作为其他指标。

### 12.2 完整关键配置表

| 类别 | 参数 | 当前 case |
| --- | --- | --- |
| 架构 | version / norm / channels / blocks | 4 / GN / 192 / 40 |
| 架构 | critic / fusion | dual_tower / residual_mlp |
| 架构 | fusion_hidden / value_hidden | 1024 / 256 |
| 训练范围 | train_scope | all |
| 标签 | target_mode / return_mode | all_players / score_rank_mc |
| 标签 | pts / gamma | [6,4,2,0] / 0.999 |
| 输出约束 | exact_zero_sum / penalty weight | true / 0 |
| 优化器 | 类型 / lr / weight decay | Schedule-Free AdamW / 0.0002 / 0 |
| 优化器 | betas / warmup | (0.9,0.999) / 2000 |
| 优化器 | r / weight_lr_power / foreach | 0 / 2 / true |
| 分组 LR | visible / Oracle / fusion / value scale | 均为 1 |
| scheduler | type | optimizer，交给 Schedule-Free |
| 训练批次 | batch / workers / file_batch / prefetch | 640 / 2 / 6 / 2 |
| 验证 loader | workers / file_batch / prefetch | 0 / 8 / 5；workers=0 时不构成多进程预取 |
| 采样 | train fold / val fold | 64 / 128，native_hash |
| 随机种子 | split / data shuffle / fold | 20260416 |
| 增强 | enable_augmentation / augmented_first | false / false |
| 流式 | reserve_ratio / 单次 num_epochs | 0 / 1，外层可继续 pass |
| 精度 | train AMP / eval AMP | true / false |
| scaler | initial scale / growth interval | 2048 / 1000000 |
| 后端 | cuDNN benchmark / TF32 / compile | true / true / false |
| 监控 | val_every / val_batches | 10000 steps / 256 |
| 保存 | save_every / log_every | 10000 / 100 steps |
| 输入依赖监测 | eval_input_modes / dependency_val_every | true 模式 / 0 |
| sealed test | final_test_enabled | false |
| 资源上限 | max_steps | 5000000，非最优训练长度结论 |

test_batches=1024 只是配置中的可用上限，本次没有执行 sealed test。常规 monitor 只跑 `true` 不代表已经自动完成 `true / zero / shuffled` 的资格测试。

### 12.3 Schedule-Free 的参数点

Schedule-Free 优化器区分训练时参数点与用于评估的参数点。需要调用 optimizer 的 train / eval 切换，而不仅是 `model.eval()`。当前预训保存与验证路径已检查这种切换，没有证据表明该 run 把训练参数点误当成最终评估参数点导出。

续训必须保存优化器完整状态；只拿评估权重重新初始化 optimizer 是新训练分支，不是无缝续训。官方实现还要求关注与 BN 的交互；当前用 GN，仍应记录 optimizer 版本及切换顺序。[Schedule-Free 官方说明](https://github.com/facebookresearch/schedule_free)

### 12.4 当前课程

| 阶段 | 目标文件池权重 | 控制方式 |
| --- | --- | --- |
| A | early 15%、mid 25%、recent_24m 60% | 无固定阶段长度，以 paired gate 决定是否换阶段 |
| B | recent_24m 90%、replay 10% | 同一自适应规则 |
| C | recent_12m 98%、replay 2% | 自适应加分级 LR，最终停止 |

每 10k steps monitor，每 50k 做 paired gate。phase C 的 LR levels 为 2e-4、1e-4、5e-5、2.5e-5、1e-5。每层都需要连续两个 futile gate 才继续降 LR 或停止。

dev 的 game ID 取模进一步区分 monitor remainder=0 与 formal-dev remainders=1、2、3。该划分有助于控制复用，但不自动消除前述上游模型选择重叠。

### 12.5 当前统计量

`paired_cluster_summary` 按游戏对齐，要求相同 game ID 和样本数。设每局损失差之和为 D_g、状态数为 n_g，N 为总状态数：

```text
Δ = Σ_g D_g / N
SE_cluster = sqrt[ G/(G-1) × Σ_g(D_g - Δ n_g)^2 / N^2 ]
CI = Δ ± 1.96 × SE_cluster
```

它还报告每游戏等权平均差。主差值本身仍是状态加权；cluster SE 处理组内相关，不等于把目标改成每局等权。长半庄和状态多的视角仍更影响状态平均 MSE。

只有一个 cluster 时实现的 SE 为 0，不能据此将极小样本的退化或改善解释为高置信结论。正式决策应另有最小独立游戏数与有效样本检查。

### 12.6 门槛与 guardrail 的实际行为

当前 primary 为 `primary_loss`，lower is better，meaningful_delta=0.0002，primary noninferiority margin=0.0002，confidence_z=1.96，required_futile_gates=2。

其他配置门槛包括 all_players loss 0.0002、p0 MAE 0.0001，以及 exact_zero / nonzero / abs_ge_2 / abs_ge_4 loss 各 0.0005。这些值记录了现有控制器的比较尺度，**不能全部叫作已经执行的硬性非劣约束**。

当前实现一旦 primary 达到明确改善，就更新 best；没有在此之前要求所有 guardrail 通过。guardrail 主要用于判断在 primary 未明显进步时，是否有补偿足以继续该阶段。此前已构造 primary 1→0.99、tail 1→2 的反例，仍返回 `update_best`。

建议把两类指标分开：硬约束用于否决候选，补偿指标用于决定是否值得继续训练。每项明确方向、非劣 margin、缺样本策略、是否要求所有通过，并在代码中形成可执行的决策表。

### 12.7 监测功效与停止含义

历史核验中，某次 primary delta 的区间约为 [-0.009715,+0.016517]，而 meaningful_delta 只有 0.0002。区间半宽约是目标效应的 65 倍。此时“仍可能改善”会长期成立，反复 observe 不能证明训练仍有效，也不能证明已经无效。

应先修复输入和标签，再用 A/A 与固定候选的配对方差估计可检测效应；独立游戏数量优先于只给相同游戏增加很多相关状态。若预算内仍无法判断，应按预声明资源上限给出未决，而不是无限延训或临时放宽门槛。

## 13. 在线 RL 系统与行为策略

### 13.1 三个进程的责任

| 进程 | 主要责任 | 关键状态 |
| --- | --- | --- |
| server | 分发策略参数，接收完整对局日志，向 trainer 转交一批文件 | param_version、submission ID、buffer / drain |
| client | 拉取参数，在指定对手分布下生成对局并上传 | 本次使用的参数版本、运行时设置、seed、对手 |
| trainer | 从日志重建轨迹、计算 target / advantage、更新并发布参数 | actor、old snapshot、critic、optimizer、历史行为快照 |

这里的 replay 主要通过牌谱文件重新解码，不是一个仅存 `(s,a,r,s')` 的固定形状数组。server 在文件名中加入 `pv{version}_sid{submission}_...`，使行为参数版本可追踪。转交文件和保留旧策略快照是两个不同机制，不能把它们合并称为一个 replay buffer 超参数。

基础配置 `server.capacity=1600` 的单位是日志文件数量，`force_sequential=false` 时不必等到凑满才 drain。`online.history_window=50` 用于 client 展示最近 session 的移动统计，**不是保留 50 个行为策略版本**。

### 13.2 采样不是四名玩家同时更新同一网络

当前 `TrainPlayer` 使用 `OneVsThree`：一名 trainee 对三名 baseline。每个 session 从配置的 baseline pool 中选择一个 engine，再用它控制三个对手，进行换座对局。pool 支持 champion、anchor、history 等权重；也支持只有一个固定 baseline。

这与四个位置都由最新策略自主更新的纯 self-play 不同。说“自博弈”时应给出真实对手分布，尤其要记录 pool 是否会在 session 间重载、成员权重和内容哈希。对手变化会改变 critic 的回报分布，不能仅修正 trainee 的行为概率就认为所有环境非平稳性消失。

### 13.3 实际动作采样

`MortalEngine` 在 categorical 模式下输出 masked softmax 概率 q。设 explore_rate=e：

```text
以概率 1-e 选择 argmax(q)
以概率 e 从 Categorical(q) 采样
```

当前典型训练配置 e=1，因此行为分布就是 q；正式 greedy 推理 e=0。若将 e 改成中间值，行为概率应为：

```text
μ(a|x) = (1-e) × 1[a=argmax(q)] + e × q(a|x)
```

这不是普通 epsilon-greedy 的“均匀随机动作”。若启用搜索重排，q 还会被 planner 改写。仅保存裸网络权重而在 trainer 重算 `softmax(logits)`，未必能重建真正的 μ；搜索、混合探索、mask、AMP、Oracle runtime 等都应进入行为签名，最好在采样时记录实际 log-prob。

### 13.4 三个容易混淆的策略

| 符号 | 含义 |
| --- | --- |
| μ | 实际生成该轨迹的行为策略，可能是较早的发布版本 |
| ν | trainer 保存的 old actor snapshot，按配置周期更新 |
| πθ | 当前正在优化的 actor |

代码先用 old snapshot 计算 old log-prob；开启 replay IS 且对应行为版本可追踪时，再用该版本快照重建行为概率并替换分母。对这样的样本，后面的 ratio 实际是 `πθ/μ`，不是始终 `πθ/ν`。

如果版本未知或已被淘汰，而且 `drop_untracked_samples=false`，代码仍可能保留样本并使用 fallback 的 old probability。此时不能宣称每一个 ratio 都拥有真实行为分母。基础配置和 C/D/E 构造器都存在 drop=false 的设置。

### 13.5 版本缓存与陈旧度

基础配置保留最多 8 个策略版本；C/D/E 公共构造默认 16。历史实验可能覆盖这些值。版本差和真实 KL 不是同一个量：小学习率的多个版本可能很近，一次剧烈更新也可能很远。

最低限度应记录：已追踪样本比例、未知版本比例、每版本样本数、采样到学习的延迟、policy ratio 分布、ESS、固定状态上的 KL、每条轨迹实际使用的目标计算模式。当前有部分相关机制，但尚未完成所有分布和端到端概率的资格验证。

### 13.6 初始化和续训优先级

在线启动优先尝试 `[control].state_file`，再到 `[online].init_state_file`，再到监督阶段的 best_loss / best 路径。一个陈旧 control checkpoint 可以使“我指定了新 SL 初始化”的直觉失效。正式 challenger 路径又是 `[1v3.challenger].state_file`，不等于 control 路径。

每次实验应输出最终解析的 actor、critic、opponent 路径及哈希，明确是 continuation、仅权重 warm start，还是新 optimizer 分支。实验名称中的 C/D/E 或 s5000 不能替代 checkpoint 内部 metadata。

## 14. GAE、V-trace 与实际 PPO 目标

### 14.1 先组完整轨迹，再生成标签

完整轨迹路径先读取动作序列、局索引和局级奖励，用当前 critic 批量预测 value，再递推优势和 value target，最后拆成训练 minibatch。它不会把随机打散 minibatch 的相邻行当作时间相邻状态。

当前 base policy 配置 `gae_gamma=0.999`、`gae_lambda=0.95`。代码支持其他模式；每次实验仍需记录有效配置和数据路径。所谓“完整”指该 loader 定义的全部有效决策序列，不是所有 MJAI 事件。

### 14.2 GAE 的递推

对某一玩家输出，终局后 value 取 0：

```text
δ_t = r_t + γ V_(t+1) - V_t
A_t = δ_t + γ λ A_(t+1)
y_t = A_t + V_t
```

四个玩家目标分别递推；actor 使用 p0 的 A。`λ=1` 且完整终止轨迹、相同 reward 和时钟时，`y_t` 应与 MC return 对齐。此前真实 98 步轨迹的 CPU 对拍最大差约 5.96e-7；第 11 节说明该测试仍需补 native fold 生产路径。

λ<1 会依赖 critic 的 bootstrap 质量，在减少方差的同时引入近似和分布偏差。Oracle 人类 MC loss 下降不保证该 bootstrap 对当前在线策略准确。

### 14.3 advantage 标准化与 value 标签

GAE 模式下，policy 使用 minibatch 内标准化 advantage；value target 保持原尺度。当前已修复单样本 `std()` 导致 NaN 的边界：单样本 advantage 标准化为 0，value 标签不被改为 0。

多样本仍使用原有样本标准差口径。标准化发生在过滤后的训练批次，样本组合改变可能影响 actor 的相对权重和符号；小 batch、强过滤和稀少尾部样本时尤其需要检查。原始 advantage 与标准化 advantage 应分别记录，不能只看一个平均数。

### 14.4 当前 PPO 风格 surrogate

设 a 是记录动作，`b` 是实际用于分母的 old / behavior policy，标准化 advantage 为 A：

```text
r = exp(log πθ(a|x) - log b(a|x))
r_clip = clamp(r, 1-ε, 1+ε)
ρ = min(r, rho_cap)   （rho_cap>0 时；否则 ρ=r）
m = min(ρ A, r_clip A)
s = max(m, dual_clip A)  if A<0 else m
L_policy = -mean[s + α H(πθ)]
```

base clip_ratio=0.2、dual_clip=3。C/D/E 可以覆盖 clip，公共构造关闭了额外 importance cap 和 raw-logit gate。不能只引用标准 PPO 公式而省略当前 dual clip、cap、分母替换、advantage 归一化及更新门控。

PPO clipping 限制特定样本上的 surrogate 行为，不是策略累计 KL 的硬上界，也不保证多个 minibatch、多个旧版本上的整体更新温和。建议用固定状态集和当前采样分布分别测 KL、clip fraction、动作变更和梯度范数。[PPO 原论文](https://arxiv.org/abs/1707.06347)

### 14.5 raw-logit gate 的可选路径

若启用正阈值 T，代码在原始 advantage 为正且选中动作 raw logit≥T，或原始 advantage 为负且 raw logit≤−T 时，将该样本 surrogate 贡献置零。base 配置有 T=2；C/D/E 公共构造 T=0，关闭。

这是额外启发式。softmax 对所有 logits 加同一常数不变，raw-logit gate 却可能变化，因此它不只依赖策略分布。若未来使用，应提供设计依据并做不变性和梯度对照，不能把它当作标准 PPO 必备步骤。

### 14.6 entropy、actor 更新时钟与辅助项

代码支持固定 / 动态 entropy weight；动态路径按目标 entropy 与当前 entropy 的差更新 log α，并限制其范围。base 有 entropy weight=0.001、target=1、adjust_rate=0.0001 等历史设置。C/D/E 默认把相应调节和 floor 设为 0，之后可按实验覆盖。

critic warmup 期间可以停 actor；也可以只按指定 update_interval / phase 更新 policy。独立 actor LR clock 用于避免 critic-only 步数提前消耗 actor 的 LR schedule。记录总 steps 时必须同时报告 actor 实际更新次数和数据曝光量。

在线代码还能加入 rank / opponent / danger 辅助项，但 C/D/E 公共配置将它们关闭，以减少排因变量。regret 头读取 detached φ，因此默认只能训练这些预测头；没有理由把它们的 loss 下降当作 actor 表示改善。

### 14.7 value 损失

在线 value 的目标是前述 GAE / V-trace target。普通 MSE 路径可加 `zero_sum_weight × (Σ_i V_i)^2`，再乘 value loss 权重加入总训练目标。base value weight=0.05；具体 C/D/E case 可不同。当前新 Oracle 的精确零和结构与旧 checkpoint 的惩罚式零和并非同一架构，加载和比较必须核对。

独立 critic 时，降低 value weight 主要改变 critic 优化强度，不通过共享主干直接降低 actor 的 value 梯度，因为两者没有这条共享梯度路径；它仍会间接改变之后的 advantage。

### 14.8 可选 V-trace

当相应 clip 为正且 online / GAE / value 条件满足时，代码可以按轨迹开启 V-trace；auto 模式还检查发布版本差，默认阈值为 2。不是所有轨迹和历史 run 都启用。

先计算当前 target policy 与行为分母的 ratio，截断为 ρ̄ 和 c̄：

```text
δ_t^V = ρ̄_t [r_t + γ V_(t+1) - V_t]
v_t = V_t + δ_t^V + γ c̄_t [v_(t+1) - V_(t+1)]
A_t^PG = ρ̄_t [r_t + γ v_(t+1) - V_t]
```

末端 bootstrap 为 0。递推按每个玩家的奖励分别执行；该视角轨迹的同一组行为 ratio 被用于四输出的 target 修正。其含义应是修正该受控 actor 对整场回报分布的影响，而不能在解释时误写成三名对手各自的行为 ratio。

### 14.9 当前组合尚未完成的数学审计

`A_t^PG` 已包含一层截断 importance ratio，而随后 actor surrogate 又乘 `π_now/behavior`，并做标准化、clipping、dual clipping。因此整体不能仅凭 V-trace value 递推正确就称为标准 IMPALA。

也不能简单声称最终权重恒等于 rho²：两次 target / update 的策略时点、截断和 advantage 标准化会改变关系。正确审计单位应是从采样分布到最终 actor gradient 的整条链。

建议先构建 fresh-data PPO 参照，令实际行为 log-prob 可直接核验；再选择推导清楚的 replay PPO 或 IMPALA 分支。针对小型可解析序贯问题，比较解析梯度、枚举期望和实现自动微分，而不只测试 value recursion。[IMPALA 原论文](https://arxiv.org/abs/1802.01561)

### 14.10 部分可观测性下的 Oracle baseline

Oracle critic 使用更多信息可能降低方差，但 state-only / history-state critic 在部分可观测策略下具有不同理论条件。当前 actor 的输入是历史压缩特征，critic 使用同一可见表示再加隐藏信息；它既不是完整历史输入，也不能未经证明当作充分状态。

请专家判断哪些条件能支持我们当前估计器，哪些误差来自函数逼近、bootstrap、行为分布、隐藏补全或历史压缩。有关非对称 actor-critic 在部分可观测条件下的偏差问题，可参考 [Baisero 与 Amato 的原论文](https://arxiv.org/abs/2105.11674)。该文提示审计方向，并不直接判定本实现有某个已量化的偏差。

## 15. C/D/E 与其他可选机制

### 15.1 C/D/E 的实际区别

| 分支 | critic 初始化 | actor 前的 critic warmup | 想回答的问题 |
| --- | --- | --- | --- |
| C | 不加载离线预训 critic | 新 run 必须有正 warmup | 仅当前在线分布的 critic 准备是否足够 |
| D | 加载指定预训 critic | 0 | 预训练 critic 能否直接支持 actor 更新 |
| E | 加载指定预训 critic | 新 run 必须有正 warmup | 预训练后在在线分布再校准是否更稳 |

warmup 的有效终点会考虑 resume_steps。分支字母不说明训练目标、对手、实际初始权重、LR、clip 或具体 warmup 长度；必须读取每个 case。

共同构造默认启用 score_rank、GAE、Oracle critic、all action scope 与行为版本追踪；关闭 actor Oracle、search、search distill、expected reward、主要辅助头和内部 test_play。正式评测在单独流程进行。

### 15.2 可选机制矩阵

| 机制 | 代码存在 | 当前方案中的身份 |
| --- | --- | --- |
| visible categorical actor | 是 | 部署主线 |
| SL rank / opponent / danger | 是 | canonical / S70 保存模型中的实际辅助任务 |
| 独立 dual tower Oracle | 是 | 当前预训练主线 |
| Oracle actor ramp / guiding | 是 | 历史与可选分支，不是当前 actor 默认 |
| GRP 预测奖励 | 是 | 其他奖励路径；不是当前 Oracle 标签 |
| 推理搜索 / 重排 | 是 | 默认训练关闭，需单独 A/B |
| v4 单人手牌工程特征 | 是 | 即使 search=false 仍属于当前输入 |
| V-trace | 是 | 受配置和版本门控制；组合梯度待审计 |
| HLGauss / 其他融合 | 是 | 可选架构，当前 MSE / residual_mlp 没有启用 |
| KL budget / early-stop 完整协议 | 可加入诊断和控制 | 本文建议，不能称现有完整主线已经具备 |
| 自动 guardrail 否决 best | 当前控制器不满足该语义 | 待修复 |
| 固定生产 Oracle 验证输入 / 标签 | 独立探针可固定 | 活跃生产 loader 尚未统一修复 |

### 15.3 对复杂度的态度

不建议在目标、输入、标签时钟和 baseline 尚未固定时，同时增加 actor ramp、更多辅助头、reward shaping、replay 和复杂搜索。这样即使分数改善，也很难知道哪个机制起作用。

这不意味着复杂方法没有价值。我们希望每个新增机制有明确假设、最小对照、可观察的中间指标和正式结果；无收益的机制可以删除或保留在研究分支，减少主线维护负担。

## 16. 正式评测、统计单位与模型发布

### 16.1 `1v3` 是什么

一名 challenger 对三名指定 baseline。每个 seed set 做四次座位轮换，因此 500 个 seed set 对应 2000 个半庄。完整评测应冻结双方 checkpoint、模型结构、规则引擎、原生扩展、推理模式、搜索、精度、seed_key、seed 范围和对手 pool。

固定同一 seed 不能保证两个 actor 行动后永远处在相同状态；策略分歧会改变牌局演化。配对的作用是共享环境随机源与换座设计以降低方差，而不是制造完全相同的后续事件。

### 16.2 主要结果

正式主指标是 `avg_pt`，同时报告平均顺位、一至四位率、和牌 / 放铳等诊断项。单次和牌率上升或短 `test_play` 漂亮不自动代表整个半庄效用更好。

当前硬件默认可解析为台式机 seed_count=1024、shards=4，即 4096 局；笔记本 seed_count=640、shards=3，即 2560 局。这只是机器入口默认，正式实验样本量由协议决定。

### 16.3 配对 bootstrap

对候选 A 和参照 B，先按完全匹配的 seed key / seed / 四座日志计算结果差。bootstrap 的抽样单位是四局组成的 seed set，而不是把每局或每个动作全当作独立样本。

当前严格配对工具会拒绝缺失、重复、座位不全或 key 不一致。仍需另外核验模型、规则、对手和推理 provenance；文件配对成功不能替代这些条件。

### 16.4 多次选择与新确认集

对几十个 checkpoint 反复看同一组 2000 局，每个各自给一个名义 95% CI，并不能保证最后选中的 best 具有 95% 的可信增益。应区分：

1. 训练监控：用于发现错误和产生候选；
2. screening：控制候选数的比较；
3. confirmation：finalist 冻结后的一次独立确认；
4. 后续外部分布：评估泛化与对手适应。

如果中途查看确认集后又调整超参数，这组牌山已成为选择数据，应另建新的确认集。对测试预算、最大候选数、停止规则和效应门槛做预声明，比事后给最好的那条曲线加一个误差条更重要。

### 16.5 功效和预算

所需样本量取决于配对差方差和最小有意义 pt 增益。应先用未消费的新 pilot 或明确诊断集估计方差，再给预算；不能把某个常用 2000 / 20000 / 40000 数字当作普遍充分。

此前建议的 16k screening、64k confirmation 只是预算讨论起点。可以通过固定候选数、共享对手、配对设计和更好的控制方差降低成本，但不能靠选择已经表现最好的 seed 来省预算。

### 16.6 A/A 必须先成立

同一 actor、同一对手、同一引擎和完全相同推理条件的 A/A，应解释其结果是否逐位一致；若不一致，要定位随机采样、batch 数值、tie-breaking、搜索、运行版本或对手变化。否则 A/B 的细小差异可能混入运行差异。

GPU / AMP 条件下 logits 的极小变化可能在 near-tie 动作上分叉，最终扩大成整局差异。可先用 CPU / FP32 / 固定批次做诊断，再制定正式运行条件；这属于定位不确定性，而非要求所有大规模训练都放弃 GPU。

### 16.7 跨对手与黑盒比较

强于一个固定 baseline 不等于对所有对手都更强，尤其麻将可能存在明显风格交互。建议确认集包括冻结强基线与少量独立风格，但主指标与权重应预先规定。

若与外部模型比较，可采用固定中立 MJAI arena、四座轮换、双方本地推理且不交换权重，仅传事件 / 动作，并记录 timeout、断线和非法动作规则。第三方 ranked queue 的偶遇排名只能作外部观察，不能冒充指定对手的正式 1v3。

## 17. 已有结果及其证据边界

### 17.1 SL

| 证据 | 数值 / 事实 | 合理解释 |
| --- | --- | --- |
| canonical 对 `opp_lean*0.85` 历史 playoff | 各 39,936 局；只差 0.087891 pt；combined stderr 0.598451；close-call=true | 保留 canonical 发布身份，但优势未确立 |
| S70 保存 full-recent NLL | 0.447225；canonical 保存值 0.476685 | 有较好的离线信号；历史输入指纹未完全匹配，不作严格配对结论 |
| Long-ABC 本地远端快照 | 记录 OOM 与恢复回退 | 工程可靠性需确认；不是方法已收敛或失败的证据 |

### 17.2 旧 C/D/E 正式结果重算

37 个目录中，33 个保留完整原始日志，含 canonical，共 66,000 个半庄。32 个可复核 RL 候选对 canonical 的名义配对 95% CI 下界均未为正。

结论是：**未证明可稳定提升 SL**，而不是证明所有 RL 路线都没有价值。这组结果本身有重复查看同一 seed set 的多重选择问题；即使某个单项区间为正，也仍需要新的 confirmation。

E 9k 与 E 10k policy-stop 的 actor 主干和 policy 逐张量相同，却出现不同的评测结果。这要求优先审计运行身份和推理条件，不能把这些分数差全归因于 actor 更新或 critic 带来的策略改善。

### 17.3 Oracle 具有预测信号，但不等于接入成功

同日较早快照中，best 与后续 monitor 的总体 MSE 约为 3.277892→3.268630，零预测 baseline MSE 约 5.154053，相关性约 0.605、解释方差约 0.366。该模型显然不是完全随机输出。

这些值属于当时输入和标签定义，受到隐藏补全及本次发现的 fold 时钟问题影响。不能拿其第六位小数证明新的 best，也不能直接和修复后标签上的 loss 相减。

### 17.4 冻结输入的 Oracle 依赖诊断

此前探针从旧 canonical 模拟对局无放回抽 32 个 seed set × 四座=128 局，每局 4 状态，共 512 状态。固定可见、隐藏和标签快照，对同一模型比较 true / zero / shuffled，bootstrap 按 seed set 聚类。

| MSE 差：true 减对照，负值较好 | 点估计 | 名义 95% CI |
| --- | ---: | --- |
| all_players / zero | -0.09912 | [-0.17644,-0.02515] |
| all_players / shuffled | -0.12370 | [-0.19375,-0.05905] |
| p0 / zero | -0.02581 | [-0.09535,+0.04543] |
| p0 / shuffled | -0.05579 | [-0.12439,+0.01164] |

两次重复的完整三模式 JSON 哈希一致，证明这份冻结诊断可复验。总体四输出存在信息受益，p0 区间仍跨零。样本来自旧 canonical 分布、隐藏信息含冻结补全，不能替代当前 S70 的正式 actor 分布资格测试。

zero / shuffled 也可能形成训练分布外输入。它们用于诊断依赖，不是完美隔离 Oracle 因果价值；建议加入合理条件下的多次固定补全、同等训练预算的 visible-only critic，以及反事实梯度对照。

### 17.5 输入 A/A 和本次新反例

同一真实人类 dev 日志重复解码，可见 obs 相同，147/147 个隐藏状态变化；同一模型抽查 16 状态，p0 预测最大变化一轮达到 0.314929。原因来自 loader 随机补全，CPU 下即可出现。

本次另证实在同一真实可见状态上，native fold 改变 MC 标签；两者是独立问题。固定 hidden input 不能修复 target clock，修复 target clock 也不能自动固定 hidden input。

### 17.6 当前还没有什么证据

- 没有证明 S70 / 最新 Long-ABC 在匹配的新正式协议上优于 canonical。
- 没有证明当前 Oracle 的 p0 advantage 在新 actor 分布下优于合格 visible critic。
- 没有证明当前混合 PPO / replay / V-trace 的完整梯度是希望优化的估计器。
- 没有完成全流程从未消费过的 sealed test 独立性认证。
- 没有用新的多训练种子确认集证明 RL 增益可重复。
- 没有测出本次 fold 标签问题在整个数据集上的平均影响。

这些空白是待办证据，不是对应方法一定失败的结论。

## 18. 工程、运行身份与已实施修复

### 18.1 源码组织

| 目录 | 职责 |
| --- | --- |
| `libriichi/` | 规则、状态、特征、数据解码、arena 和 PyO3 |
| `mortal/core/` | 模型、优化调度、checkpoint、公共协议和自适应控制 |
| `mortal/data/` | 牌谱组织、reward、Oracle target、轨迹 |
| `mortal/supervised/` | GRP、SL 训练和课程 / 选择 |
| `mortal/online/` | server / client、在线更新、Oracle 预训和实验配置 |
| `mortal/eval/` | 推理、对手、1v3、严格配对统计 |
| `mortal/research/` | 独立审计和诊断入口 |
| `scripts/` | Windows / 课程 / 资源与文档入口 |
| `docs/` | 状态、接手、研究报告与历史证据 |

包模式执行使用 `python -m mortal...`；配置可由 `MORTAL_CFG` 覆盖。包路径、进程当前目录与配置解析后的产物目录均需记录，避免看似相同命令启动不同实验。

### 18.2 双机

台式机 main 工作树是源码真源；笔记本是独立实验 runner。默认不共享梯度、replay 或 checkpoint。run name、输出目录和模型产物带机器区分，源码以冻结快照 / overlay 同步。

已记录台式机为 i5-13600KF + RTX 5070 Ti；笔记本为 i9-13900HX + RTX 4060 Laptop 8 GB + 32 GB RAM。笔记本当前进度未直连确认，旧快照显示 RAM / VRAM 压力与 OOM 后回退约 6000 steps。不能把 OOM 等同于算法不稳定，也不能在活跃训练期间随意重装原生扩展。

### 18.3 checkpoint 应记录的完整身份

推荐至少保存：模型各组件、optimizer、scaler、scheduler、内部 steps 与 actor 更新次数、配置解析结果、reward / target 签名、数据 index 和来源哈希、游标语义、RNG 状态、代码 / 扩展哈希、Python / Torch / CUDA / optimizer 包版本、父 checkpoint 和切换理由。

这是一份审阅要求清单，现有 checkpoint 已保存其中许多项，但不能声称每一个旧产物都齐全。特别是旧配置的路径、未消费字段和不同 best 选择器需要统一解释。

### 18.4 已经落地的同日审计修复

| 修改 | 已验证行为 | 限制 |
| --- | --- | --- |
| reward 签名校验 | 拒绝中心化排名效用不一致的 Oracle 接入和静默 resume | 还需加入本次揭示的时钟 / fold 语义 |
| unknown replay 过滤边界 | 开启 drop 时，整批未知会跳过，-1 不再被错误保留 | drop=false 的旧 run 不因此自动受保护 |
| 单样本 advantage 标准化 | 不再因无偏 std 得到 NaN，value 标签保留 | 不解决整体 actor estimator 理论问题 |
| 严格 paired `1v3` 工具 | 检查完整四座、key、缺失、重复和精确配对 | 不自动校正多重选择或环境版本差异 |
| 独立 CPU 审计入口 | 可冻结输入、重复诊断、采集源码与 checkpoint 身份 | 未替换活跃 Oracle loader |

此前审计跑过 335 项相关 CPU 测试并做真实牌谱 smoke。这个数字属于此前审计的明确范围，不表示本文又运行了完整训练系统、GPU 长训或新 1v3。

### 18.5 推理与集成边界

部署只需要可见 actor 路径，但宿主必须匹配模型版本、GN / BN、categorical / DQN 头、特征通道、合法动作、Python 原生扩展 ABI 和依赖。一次独立 MJAI smoke 证明协议和模型可以协作，不等于整个 MahjongCopilot 或第三方客户端已经完整集成并回归。

外部客户端成绩也受队列对手、规则和网络故障影响。模型加载成功、合法动作通过、平台对局能完成、正式牌力达标是四个不同验收层次。

### 18.6 通信与数据边界

现有 `mortal/core/common.py` TCP / pickle 通信面向可信本机或受控环境。若未来用于外部黑盒对战，应使用有明确 schema 的事件 / 动作协议，独立处理认证、超时与重放；不应把训练内部 pickle 接口直接当作开放服务。

交给专家的文档与附件不需要原始玩家身份、真实机器地址、访问凭据或模型权重。可以先提供脱敏配置、目标反例、统计聚合和必要源码位置；需要原始数据复验时再按明确范围提供。

## 19. 风险清单与建议实验顺序

### 19.1 优先级的含义

P0 表示会阻断可信比较、目标一致性或接入资格的问题；P1 表示对学习稳定性和统计推断有实质影响；P2 表示在基础契约成立后值得做的性能 / 架构研究。这不是按修复代码行数排序。

| ID | 优先级 | 问题 | 当前置信度 | 下一份所需证据 |
| --- | --- | --- | --- | --- |
| R01 | P0 | native fold 改变同状态 MC 标签 | 高，源码 + 合成 + 实际 dev / 原生扩展反例 | 全目标对拍、影响分布、修复后候选重评 |
| R02 | P0 | Oracle 验证隐藏输入不固定 | 高，重复解码和固定模型实测 | 冻结输入 / 标签后的全量 A/A |
| R03 | P0 | 训练与正式 pt 效用不同 | 高，配置和代数反例 | 明确主效用与迁移协议 |
| R04 | P0 | guardrail 不能否决 best | 高，可执行控制器反例 | 硬约束决策表和完整边界测试 |
| R05 | P0 | 同 actor 评测不同、旧 winner 未确认 | 权重相同与分数差异已确认；具体原因未定 | 完整运行 provenance 与严格 A/A |
| R06 | P1 | Oracle sealed test 与上游 SL 选择潜在重叠 | 风险已定位，实际文件使用史未恢复 | 游戏内容 ID 与全流程使用账本 |
| R07 | P1 | V-trace advantage + PPO ratio 的完整估计器不清楚 | 组合事实高；收益 / 偏差大小未定 | 解析梯度对拍和明确算法分支 |
| R08 | P1 | 行为分母可能不是实际采样概率 | 条件性风险；e=1、无搜索时较简单 | rollout log-prob 与重建值逐样本对拍 |
| R09 | P1 | 低功效、反复 gate 与 best 选择偏差 | 现有 CI / 历史支持 | 固定输入上的方差、功效与 sequential 协议 |
| R10 | P1 | human MC critic 与当前 actor 分布不匹配 | 分布差异确定，实际后果待测 | 新 actor 轨迹上的 p0 校准与 advantage 诊断 |
| R11 | P1 | 进程恢复可能重复数据或回滚更新 | 恢复设计与历史 OOM 支持 | 中断恢复的曝光 / 状态一致性测试 |
| R12 | P2 | 辅助任务、工程 EV 特征与主干容量的真实贡献 | 未隔离 | 同预算分层消融与正式确认 |
| R13 | P2 | 对手单一、风格交互与泛化 | 方法风险，未量化 | 预声明多对手小矩阵 |
| R14 | P2 | 更大 GRP、distributional critic 或新骨干是否值得 | 当前无充分比较 | 先给最小 pilot 和停止规则 |

### 19.2 第一阶段：建立可信的测量对象

**实验 E01：目标时钟契约。** 固定若干正常、多次连庄、跳过局、杠选择和末端轨迹，先建立全决策 return 参照，再比较 native fold / Python fold / 不同 fold_count / worker / event cache。相同原始状态目标必须一致。比较对象包括 reward、target、p0…p3 旋转和 tail slice 成员。

**通过条件：** 在选定数值容差下逐状态一致；随机采样只改变状态是否出现，不改变出现状态的语义。gamma=1 和 gamma<1 都覆盖，不能仅用 gamma=1 的望远镜相消掩盖时钟错误。

**实验 E02：验证输入 A/A。** 冻结可见特征、合法 mask、hidden completion、目标、来源 ID、时钟版本；同一 checkpoint 连续评估。再用多个预声明 completion 版本报告补全方差。

**通过条件：** 同一快照完整结果一致；变化只能来自显式变更的 completion 或模型；不能用旧随机 best metrics 与新快照直接配对。

**实验 E03：formal evaluator A/A。** 选择一个固定 actor / opponent，小样本四座轮换，记录完整网络 / 扩展 / 精度 / batch 配置。先在可解释条件下复现，再使用正式 GPU 配置量化差异。

**通过条件：** 相同策略的差异能定位并被协议控制；完整 seed set、日志与输出可重算。E01–E03 只解决可信比较，不以“赢更多”作为正确性验收。

### 19.3 第二阶段：冻结效用与底座

**决策 D01：正式目标。** 项目负责人明确最终效用、最小有意义增益、允许的四位风险退化、gamma 与奖励落点。推荐以正式 pt 对齐为候选，但由专家解释其策略含义。

**实验 E04：SL finalist。** canonical、S70 和少量 Long-ABC finalist 在同一固定输入上重测 policy NLL 和切片，再用新 screening 与独立 confirmation 比较正式 pt。模型哈希在确认前冻结，保留 canonical。

**通过条件：** 预声明 primary 和全部硬 guard 通过；区间跨零时可以未决。不要因候选训练更久或离线 NLL 更低自动发布。

### 19.4 第三阶段：critic 资格

**实验 E05：同目标、同数据的 critic 比较。** 最终 actor 冻结，在指定对手池采集新的资格轨迹。比较至少：零预测基线、合格 visible-only critic、当前 Oracle anchor、修复语义后训练 / 校准的 Oracle。所有 candidate 使用相同输入和真实回报，保留 all_players。

**主要看：** p0 MSE / MAE / bias、分位校准、exact_zero / nonzero / abs_ge_2 / abs_ge_4、all_players、不同座位和局势、true / zero / shuffled / 固定补全。标签绝对值切片仅用于评测，不能按其大小选择训练样本或临时改变总体权重。

**进一步看：** 在冻结策略和轨迹上，估计 advantage 的尺度、符号、方差及与 MC / 高质量参照的相关性。更低 value MSE 只有在它改善 actor 可用信号时才体现主要价值。

**通过条件：** p0 primary 和每个预声明硬 guard 同时满足；没有必要要求四个输出每个指标都严格变好，但允许非劣幅度必须事先定义。offline finalist 确定后再打开其合格的 sealed test；先审计全流程独立性。

### 19.5 第四阶段：最小在线 PPO 参照

**实验 E06：fresh-data actor-critic。** 从同一 actor、相同 optimizer / scaler / 数据起点分支，固定对手；明确行为 log-prob，限制 stale 数据，关闭可选搜索、Oracle actor、regret、复杂 reward 混合和 V-trace。比较 no-update actor、visible critic 和合格 Oracle critic。

短窗用于暴露数值、采样、梯度和 KL 异常。原有 500→1500→3000 等 step gate 可作为诊断节奏，但绝不能把这些步数本身当作正式牌力资格；小 `test_play` 只用于诊断。

**主要看：** actor 真正更新次数、new/behavior KL、ratio / clip fraction、entropy、critic 校准、gradient norm、关键场景动作变化，以及独立正式 pt。至少多个训练 seed；此前建议至少 3 个是估计训练波动的起点，不保证任何给定功效。

**通过条件：** 契约稳定，且经过预声明选择后在新牌山确认 actor 提升；Oracle 分支的增益需相对 visible critic 而不仅相对无学习。

### 19.6 第五阶段：逐项增加复杂度

| 实验 | 保持不变 | 唯一主要变化 | 成功标准 |
| --- | --- | --- | --- |
| E07：replay / 陈旧度 | actor、critic、reward、对手和总样本预算 | 允许的行为陈旧度 / reuse | 吞吐提升且正式增益保持；ratio / ESS 可解释 |
| E08：V-trace 或异步算法 | 同样的数据和初始状态 | 一条推导明确的 estimator | 解析梯度通过，多个训练种子下有稳定作用 |
| E09：辅助任务 | 同一 SL / RL 主目标与预算 | 一组辅助任务及梯度路径 | 改善 policy 或正式 pt，而非只改善自身 loss |
| E10：工程特征 | 同一数据和训练预算 | 单人计算特征的使用方式 | 从头或充分重训后的强度 / 成本比较 |
| E11：分布式 value | 同目标、同数据、同骨干 | value head / loss | p0、尾部、advantage 与牌力一起改善 |
| E12：对手多样性 | 更新预算和评价协议 | 冻结对手池及权重 | 对预声明对手分布泛化且无主要退化 |

不要把所有维度做一次大网格后只报告最好的一格。先规定最大候选数和淘汰规则，再把算力集中在能改变方向判断的实验。

### 19.7 明确停止或回退条件

- 输入 / 标签 A/A 失败：暂停比较模型好坏，先修复测量路径。
- 行为 log-prob 无法匹配：不把相应 replay 当作已正确修正的数据。
- primary 改善但硬 guard 越界：保留原 anchor，报告 trade-off 待决。
- 统计功效不足：报告可检测范围和未决，不把零附近点估计写成等价。
- 多训练 seed 方向不一致：先解释训练方差，不只增加同一 checkpoint 的评测局数。
- 确认集被用于调参：将其降级为选择集，重新建立未消费确认集。
- 算法分支没有可解释收益：从主线移除或降级为研究选项，保留复验材料。

## 20. 请求专家逐项审阅的问题

以下问题不要求每项都给结论。请对不能确定的问题说明缺少什么证据；若您认为优先级应调整，请直接改排。建议以问题 ID 关联源码、反例和实验。

### 20.1 任务、目标与奖励

| ID | 具体审阅问题 |
| --- | --- |
| Q01 | 若最终看正式 pt，应如何权衡平均顺位、四位率、原始点数和对手泛化？一个主效用是否足够？ |
| Q02 | `[3,1,-1,-3]` 与 `[2,1,0,-3]` 的差异会诱导哪些进攻 / 防守策略变化？应怎样设计低成本反例或 policy probe？ |
| Q03 | 整场任务应否使用 gamma=1？如果保留折扣，麻将的自然时间单位是决策、摸牌、局还是显式持续时间？ |
| Q04 | 当前 `score_rank` 局间差分是否提供有用 shaping？如何构造与主效用一致、终止正确的替代 reward？ |
| Q05 | 初始同分按 seat ID 排序是否引入可避免的 baseline / shaping 偏置？怎样保持与实际排名规则一致？ |
| Q06 | 更改 reward 后，旧 critic 应重新预训练、部分校准还是保留某些表示？哪些 checkpoint 状态可以安全复用？ |

### 20.2 数据与标签

| ID | 具体审阅问题 |
| --- | --- |
| Q07 | 人类语料需要哪些质量分层、玩家选择和去重信息，才能评估行为克隆上限与偏差？ |
| Q08 | 以游戏为 split、以状态为 loss 单位会如何加权长半庄、多决策玩家和常见局势？应采用什么目标总体？ |
| Q09 | 如何重建 SL→Oracle→RL 全流程的数据使用账本，区分上游选择泄漏与直接训练泄漏？ |
| Q10 | 时间 holdout 与玩家 holdout 应如何组合，以评估跨时代、跨风格和当前 AI 分布泛化？ |
| Q11 | native fold 的修复应首选完整轻量轨迹还是 duration 形式？需要哪些最小边界测试？ |
| Q12 | 目前 exact_zero / tail slice 会随标签时钟变化，修复后应怎样重新设置阈值并保持历史解释？ |
| Q13 | 训练时随机隐藏补全是否等价于合理的条件期望采样？有哪些已知信息约束必须满足？ |
| Q14 | 同一原始牌谱的增强、四视角和缓存副本，应如何共享 game ID、cluster 和 split？ |

### 20.3 表示、主干与 SL

| ID | 具体审阅问题 |
| --- | --- |
| Q15 | 1012×34 工程表示是否保留了决定策略所需的历史？哪些缺失最值得先做可控验证？ |
| Q16 | 192×40 的卷积主干与 GN / 通道注意力是否合理？相比增加容量，是否更应改善历史表示或牌种归纳偏置？ |
| Q17 | 对同一个可见状态存在多个合理人类动作，单标签 CE 的局限应如何诊断？需要软标签还是仅更好评估？ |
| Q18 | rank / opponent / danger 三类辅助任务是否提供互补信号？怎样测梯度冲突和实际 policy 收益？ |
| Q19 | 危险度的逐状态正负平衡与正例点数回归，会不会扭曲概率校准或尾部风险？ |
| Q20 | 巡目、南场、all-last 权重的最佳验证方式是什么？如何避免只在被人工加权的切片上看起来更好？ |
| Q21 | broad→recent 课程相对固定混合的必要性如何证明？训练样本曝光与 LR 应怎样配平？ |
| Q22 | selector 的 loss epsilon 与 action score 应怎样校准？可否用更简单的规则达到同样或更可靠选择？ |

### 20.4 GRP

| ID | 具体审阅问题 |
| --- | --- |
| Q23 | 24 排列联合分类是否比四人边际或其他结构更适合，如何评价 calibration 而非仅 top-1？ |
| Q24 | 同一半庄多个前缀的训练权重是否合理？如何处理近终局的简单样本占比？ |
| Q25 | label smoothing 及末端平滑会如何影响 GRP reward 的尺度和偏差？ |
| Q26 | 当前直接真实 return 的 Oracle 路线下，GRP 最有价值的下游用途是什么，是否应降低训练优先级？ |

### 20.5 Oracle critic

| ID | 具体审阅问题 |
| --- | --- |
| Q27 | 该 critic 输入对应 state、history-state 还是混合的事后补全信息？哪些理论保证适用，哪些不适用？ |
| Q28 | dual tower + zero-init residual fusion 会不会延迟隐藏分支学习？应监测哪些梯度与表示量？ |
| Q29 | `all_players` 对 p0 的帮助应如何隔离？p0 固定输出权重的试验应设哪些 guard？ |
| Q30 | 精确零和是否总是匹配 target，包括 kyotaku、跳局、玩家视角和截断？未来改变奖励时如何检查？ |
| Q31 | 总体 MSE 改善与 actor advantage 改善的关系如何实测？哪些校准和偏差指标最有决策价值？ |
| Q32 | human MC 预训再在当前 AI 分布校准，是否是合适顺序？怎样控制分布漂移和对手更新？ |
| Q33 | true / zero / shuffled 的 OOD 问题怎样处理？何种条件随机化或 visible-only 对照更有说服力？ |
| Q34 | 如果 p0 没显著改善但 all_players 改善，应继续训练、改权重还是停在 anchor？需要什么功效？ |
| Q35 | MSE、Huber、HLGauss、分位数等目标中，哪个最值得最先比较，为什么？ |
| Q36 | Schedule-Free 的实际收益怎样与固定 LR / cosine 匹配比较，避免不同训练预算和参数点混淆？ |

### 20.6 在线 RL 与 off-policy

| ID | 具体审阅问题 |
| --- | --- |
| Q37 | 最小可信基线应是严格 on-policy PPO、近 on-policy PPO，还是独立 IMPALA？选择依据是什么？ |
| Q38 | μ、ν、π 三个策略在当前代码中的职责是否正确？哪些分母必须来自 rollout 原始 log-prob？ |
| Q39 | 当前 V-trace `pg_advantage` 再进入 PPO surrogate 应如何分析？是否存在合理的目标解释或应拆分算法？ |
| Q40 | 固定对手与变化对手下，单个 trainee 的 importance correction 分别能修正哪些分布变化？ |
| Q41 | 四人 value 使用同一 trainee 行为 ratio 是否对应我们想估计的联合回报？需要怎样的单元环境证明？ |
| Q42 | minibatch advantage 标准化、dual clipping、cap 和 raw-logit gate 各自应保留、删除还是独立测试？ |
| Q43 | 如何设置 KL budget、更新次数和 stale-data gate，使步长真正可控，而不是只看 clip_ratio？ |
| Q44 | critic-only warmup、独立 actor LR clock 和固定 critic 对照应该如何组合？ |
| Q45 | 应否在 RL 中继续 SL 辅助任务或加入 reference-policy 约束？怎样区分稳定作用与限制上限？ |
| Q46 | 稀疏负尾部样本的 advantage 方差如何降低，同时不通过结果大小加权改变训练总体？ |

### 20.7 评测、统计与算力

| ID | 具体审阅问题 |
| --- | --- |
| Q47 | 四座 seed set bootstrap 是否足以处理当前相关结构？对多对手、重复训练 seed 应如何分层？ |
| Q48 | 自适应 controller 的 state-weighted estimand 与 game-balanced 指标应如何选择？ |
| Q49 | hard guard 与补偿指标如何定义非劣 margin，缺少 tail 样本时怎样裁决？ |
| Q50 | 重复 monitor / best 选择应采用怎样的 sequential 或确认集策略，成本与收益是什么？ |
| Q51 | 多训练 seed 与更多评测牌山如何分配预算，才能区分训练不稳定和评测噪声？ |
| Q52 | 什么样的 A/A 差异应视为运行错误，什么样的数值非确定性可以被正式协议接受？ |
| Q53 | 如果所有 CI 都跨零，怎样给出有用的等价界、可检测范围和下一步，而不强行选 winner？ |
| Q54 | 哪三个实验最可能改变您对当前方案的判断？如果只能使用现有双机一周，优先级怎样排？ |

### 20.8 工程与整体决策

| ID | 具体审阅问题 |
| --- | --- |
| Q55 | 哪些运行身份和 target-contract 字段应成为启动时的硬检查，缺失时应如何迁移旧 checkpoint？ |
| Q56 | 如何在不重放遗漏数据的前提下恢复 worker / prefetch，而清楚说明可能重复的样本范围？ |
| Q57 | 哪些历史路径应从主线移除，以降低配置和梯度路径的歧义？ |
| Q58 | 模型推理、原生 ABI、协议、formal strength 四层验收应分别有哪些最小 smoke？ |
| Q59 | 若发现一个优先级最高的数学或数据错误，您建议保留哪些已训练表示、废弃哪些结果比较？ |
| Q60 | 在修复契约后，您倾向保留当前 SL→独立 Oracle→RL 总体路线，还是改变路线？请给出能推翻该建议的实验。 |

## 21. 讨论流程与意见交付模板

### 21.1 建议的首次讨论顺序

若有 90 分钟：前 15 分钟讲任务、效用和当前阶段；20 分钟讲 SL / 表示与数据；25 分钟讲 Oracle 信息、标签和新 fold 反例；20 分钟讲在线 estimator 与现有评测；最后 10 分钟确定前三个行动项与尚需材料。

若有更长时间，建议另开一次专门的源码审阅，逐函数走 `add_entry → populate_buffer → target → GAE / V-trace → train_batch`，另一次专门讨论统计协议。不要在第一次会议里同时争论所有 auxiliary 权重和架构细节。

### 21.2 希望收到的意见格式

| 字段 | 请填写的内容 |
| --- | --- |
| 问题 ID / 标题 | 关联 Q / R 编号，或新增问题 |
| 判断 | 已确认错误 / 高风险假设 / 合理但未证明 / 可保留 / 信息不足 |
| 证据 | 公式、源码、反例、原始论文或需要补采的数据 |
| 影响范围 | 哪个阶段、哪些 checkpoint、训练目标还是仅验证比较 |
| 置信度 | 以及什么新证据会改变判断 |
| 建议修复 | 最小清晰方案；是否改变语义与是否需要重训 |
| 对照实验 | baseline、唯一变量、数据、训练种子、样本量或功效计算 |
| 验收标准 | primary、全部 hard guard、失败 / 停止条件 |
| 成本与优先级 | 工程时间、算力和依赖顺序 |
| 未决事项 | 需要项目负责人选择的效用或取舍 |

### 21.3 可直接随文发送的请求

> 希望您从任务建模、数据与目标定义、SL、Oracle critic、在线策略梯度和统计评测六个层面审阅这份实现。我们愿意修复确定的问题，也愿意在证据充分时改变总体路线。请特别关注第 11 节已经复现的标签时钟问题，以及第 14 节中行为策略、V-trace advantage 和 PPO surrogate 的关系。对于不确定的方向，请优先提出最小可证伪实验，而不是只给通用调参建议。我们最希望得到一份按优先级排序、能明确验收的改进方案。

### 21.4 第二轮应补的材料

专家若需要深入复验，优先补齐实际语料数据卡与去重账本、S70 实际索引和选择史、固定 target clock 的逐步 trace、完整 rollout 行为概率、严格 A/A 的运行记录、p0 / 四人校准图，以及多训练 seed 的小型对照。不要先把整个 checkpoint / logs 目录无结构地交给对方。

## 22. 符号与术语表

| 符号 / 名称 | 本文定义 |
| --- | --- |
| h / x / z | 完整或玩家历史 / 部署可见编码 / 训练额外隐藏编码 |
| M | 当前合法动作集合或 mask |
| φ | actor 或 critic 的 1024 维中间表示，需注明来自哪个网络 |
| μ / ν / πθ | 实际行为策略 / old snapshot / 正在优化策略 |
| k / t / step | 局索引 / 该视角决策索引 / 对应 trainer 的更新计数 |
| u / Φ / U | 名次效用向量 / 局初当前名次效用 / 终局名次效用 |
| r / G | 单步奖励 / 完整 discounted return |
| V / A / y | critic 预测 / advantage / value target |
| gamma / lambda | 折扣因子 / GAE trace 参数 |
| p0…p3 | 当前玩家视角的相对四输出，p0 是自己 |
| all_players | 对四个玩家输出和目标共同建模 |
| score_rank | 从原始分数的排序得到名次效用；不是 raw score delta |
| GRP | Global Reward Prediction，局级历史到联合终局排名概率 |
| Oracle critic | 仅训练期额外看隐藏信息的价值估计器 |
| Oracle actor | 策略本身看到隐藏信息的另一类路径；当前默认关闭 |
| canonical | 已发布 / 受控参照身份，不等于数学上已证明最优 |
| best / latest | 特定选择规则的保存模型 / 续训保存点；不能互换 |
| state fold | 按确定性规则保留部分决策，当前 native 路径会影响标签时钟 |
| replay | SL 可指历史数据混入；RL 指较旧行为策略产生的数据再使用 |
| seed set | 同一 seed 设计下的四座轮换半庄组 |
| primary / hard guard | 主要改进指标 / 必须满足的非劣约束 |
| monitor / confirmation / sealed test | 反复查看的监控 / finalist 独立确认 / 按协议封存的测试 |
| A/A / A/B | 相同模型的运行对照 / 不同候选的匹配比较 |

## 23. 源码、证据和复核入口

### 23.1 核心源码索引

下面链接指向仓库相对路径。脱离仓库阅读时，正文足以解释主要方案；需要逐行审计时，使用同一快照的源码，不能默认未来 main 仍与本文相同。

| 内容 | 源码 / 关键符号 |
| --- | --- |
| 模型与 GRP / actor / Oracle / heads | [model.py](../../mortal/core/model.py)：`Brain`、`CategoricalPolicy`、`OracleDualTowerBrain`、value 与 aux 类 |
| 特征与动作常数 | [consts.rs](../../libriichi/src/consts.rs)、[obs_repr.rs](../../libriichi/src/state/obs_repr.rs) |
| 事件到样本、native fold | [gameplay.rs](../../libriichi/src/dataset/gameplay.rs)：`add_entry`、`set_sample_fold` |
| 隐藏牌山还原与补全 | [invisible.rs](../../libriichi/src/dataset/invisible.rs) |
| GRP 数据与训练 | [grp.rs](../../libriichi/src/dataset/grp.rs)、[train_grp.py](../../mortal/supervised/train_grp.py) |
| SL 损失、优化、验证和保存 | [train_supervised.py](../../mortal/supervised/train_supervised.py)：`forward_loss`、相关 metrics / weights |
| SL 课程与选择 | [run_sl_ab.py](../../mortal/supervised/run_sl_ab.py)：`WINDOWS`、`WEIGHT_PROFILES`、`select_winner_by_policy` |
| 数据、行为轨迹、value 视角 | [dataloader.py](../../mortal/data/dataloader.py) |
| Oracle 真实目标 | [oracle_value.py](../../mortal/data/oracle_value.py)：`oracle_step_value_targets`、`populate_buffer` |
| GRP 奖励路径 | [reward_calculator.py](../../mortal/data/reward_calculator.py) |
| Oracle 训练 / 资格 / checkpoint | [pretrain_oracle_critic.py](../../mortal/online/pretrain_oracle_critic.py) |
| 自适应统计与决策 | [adaptive_curriculum.py](../../mortal/core/adaptive_curriculum.py)：`paired_cluster_summary`、`observe_adaptive_curriculum` |
| 当前 Oracle 外层课程 | [课程脚本](../../scripts/run_oracle_critic_adaptive_curriculum.py)：runtime overlay、phase 与 LR |
| 在线系统 | [server.py](../../mortal/online/server.py)、[client.py](../../mortal/online/client.py) |
| GAE / V-trace / PPO | [train_online.py](../../mortal/online/train_online.py)：`compute_gae_advantages_from_step_rewards`、`compute_vtrace_targets_from_step_rewards`、`train_batch` |
| C/D/E 配方 | [oracle_cde_configs.py](../../mortal/online/oracle_cde_configs.py) |
| 实际动作分布 / 对手池 | [engine.py](../../mortal/eval/engine.py)、[player.py](../../mortal/eval/player.py) |
| 正式 1v3 / 严格统计 | [one_vs_three.py](../../mortal/eval/one_vs_three.py)、[paired_1v3.py](../../mortal/eval/paired_1v3.py) |

### 23.2 机器可读证据

| 材料 | 位置与含义 |
| --- | --- |
| 本文配置 / 结构快照 | [metadata.json](../../logs/expert_review/20260905/metadata.json)：已脱敏路径、checkpoint 内部 steps 与组件规模；其中 base 不是所有阶段的当前运行配置 |
| 本文源码指纹 | [source_fingerprints.json](../../logs/expert_review/20260905/source_fingerprints.json) |
| Oracle cache 构建统计 | [cache_summary.json](../../logs/expert_review/20260905/cache_summary.json)：边界、源文件 / chunk 数量和 manifest 指纹；不含 test 内容 |
| 合成 fold 反例 | [probe_fold_targets.py](../../logs/expert_review/20260905/probe_fold_targets.py)、[结果](../../logs/expert_review/20260905/fold_target_probe.json) |
| 当前扩展真实 dev 反例 | [probe_native_fold_targets.py](../../logs/expert_review/20260905/probe_native_fold_targets.py)、[结果](../../logs/expert_review/20260905/native_fold_target_probe.json) |
| 同日既有审计 | [完整报告](sl-rl-audit-2026-09-05.md)，包含每项既有证据的原始链接 |
| 原始数据和模型 | 本地受控产物；未包含在便携审阅包中 |

本地 `logs/` 一般不随 Git 分发。缺少附件时应注明无法在该机器复验，不能凭文档数字重新“生成”一份原始证据。便携审阅包只附实际已产生、经过脱敏的选定材料。

### 23.3 本次新反例的重跑方式

在仓库根目录、当前 Python 环境下运行；真实 dev 探针依赖此前输入审计 JSON 所记录的同一 dev 文件，以及指定 runtime overlay。它不需要 GPU，也不读取 sealed test：

```powershell
$reviewPython = 'C:\ProgramData\anaconda3\envs\mortal\python.exe'
& $reviewPython logs/expert_review/20260905/probe_fold_targets.py
& $reviewPython logs/expert_review/20260905/probe_native_fold_targets.py
```

脚本默认更新其同目录结果文件。若要保留本次快照，应复制脚本及依赖说明到新的证据目录后修改 output，或先另存原始结果；不要把重复运行的新时间点当作同一份不变快照。

### 23.4 本文的维护方式

本报告按日期冻结，作为专家讨论基准；当前动作和结论仍回写对应状态页。若改变效用、时钟、算法或数据切分，应创建新版本并明确差异，避免把旧实验数值悄悄替换成新语义。

脱敏配置、源码指纹和反例脚本作为附件独立保存。主状态页保持短小，不复制本文全部公式和超参数；这样既能从头完整讲解，也能避免所有文档同时维护多份运行默认。

## 24. 外部文献与本项目的关系

以下仅引用原始论文或官方实现。文献提供方法背景和需要检查的条件；本地实现是否正确、是否更强，仍由契约与实验决定。

| 文献 | 与本项目直接相关的内容 | 适用边界 |
| --- | --- | --- |
| [Suphx: Mastering Mahjong with Deep Reinforcement Learning](https://arxiv.org/abs/2003.13590) | 麻将中的 global reward prediction、oracle guiding、run-time policy adaptation | 本项目借鉴相关问题意识；当前独立 Oracle critic 不是对该论文完整系统的严格复现 |
| [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) | clipped surrogate 与多 minibatch 策略更新 | 本项目另有 replay 分母、dual clip 和门控，必须审计额外组合 |
| [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438) | 用价值估计构造优势的偏差 / 方差取舍 | 不自动解决隐藏输入、时钟和行为策略错配 |
| [IMPALA](https://arxiv.org/abs/1802.01561) | 异步 actor / learner 与 V-trace 修正 | 递推形式相似不代表本项目完整 PPO + V-trace 目标等同于 IMPALA |
| [Unbiased Asymmetric Reinforcement Learning under Partial Observability](https://arxiv.org/abs/2105.11674) | 部分可观测条件下非对称 critic 的理论边界 | 需要映射到本项目历史压缩与隐藏补全，不能机械套结论 |
| [The Road Less Scheduled](https://arxiv.org/abs/2405.15682) | Schedule-Free 优化与迭代平均 | 不证明本项目 lr / wd / 训练长度最优 |
| [Schedule-Free 官方实现](https://github.com/facebookresearch/schedule_free) | optimizer train / eval 模式、使用约束 | 本地版本和 checkpoint 状态仍需冻结 |

关于第 10 节势函数差分的说明，本文给出的折扣展开是对本项目公式的直接推导。任何“保持最优策略不变”的进一步主张，都应明确原始目标、状态、折扣、终止与 shaping 条件后再由专家确认。
