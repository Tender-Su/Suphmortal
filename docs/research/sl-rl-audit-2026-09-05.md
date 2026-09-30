# SL / RL 独立审计：2026-09-05

本文影响当前主线。结论已同步到 `docs/status/`；旧记录的原始数值保留，但其中“winner”“明显改善”“路线已被否定”等强弱判断须按本文限定。审计目标是最终模型强度，不以延长训练、增加机制或提高吞吐作为成功标准。

**结论：SL 和独立 Oracle critic 预训练都有可保留的基础；旧 RL 的收益结论、当前 Oracle 的微小改进判断和部分自动决策规则，证据不足。应先修复验证与奖励契约，再决定模型和长训练方向。没有证据支持现在把整个 SL / RL 重写，也没有证据支持照旧把 E 9k 当作已经证明更强的起点。**

| 对象 | 本次判断 | 证据强度 |
| --- | --- | --- |
| SL canonical / anchor | 保留现有发布身份；对第一替补的强度优势未确立 | 原始 formal 汇总可复核，差距极小且触发 close-call |
| S70 / 后续 SL 长训 | 有较好的离线学习信号，应作为正式候选；不凭 NLL 自动发布 | 本地 checkpoint 可读；远端当前运行未能直连复核 |
| 历史 D/E RL | 有探索价值，尚未证明可稳定提升 SL | 重算 66,000 局；32 个可复核 RL 候选的名义配对 95% CI 均未给出正下界 |
| 当前 Oracle critic | 学到了信息，`all_players` 和真实 return 标签值得保留；尚未完成 actor 接入资格确认 | 严格权重加载、真实牌谱契约 smoke、历史模拟局诊断通过；p0 受益仍不确定 |
| Oracle 固定 monitor | 文件固定，隐藏输入未固定；微小增益和 phase-best 排名需重验 | 同一实际 dev 牌谱重复解码，147/147 个状态的 Oracle 输入发生变化 |
| 自适应控制器 | “护栏”不能否决 best；低功效监测会长期保持 observe | 可执行反例和现有 gate 历史均支持 |

## 1. 审计范围与证据边界

以工作树、checkpoint 内部字段和原始对局为依据，旧文档仅用于定位要复核的主张。采集时 HEAD 为 `94d942f1745dd23751965d27ffa7c9d93ce82331`，工作树本来就有大量修改；只记 Git commit 不能代表本次审计源码，已另存相关文件副本和 SHA-256。主状态快照时间为 **2026-09-05 04:18（北京时间）**，后续 CPU 实测另有独立产物。

- 阅读 SL selector、数据窗口、训练/续训、自适应控制器、Oracle 数据与标签、PPO/GAE/replay IS/V-trace、1v3 和关键现行文档。
- 检查 canonical、S70、D 5k、E 9k、E 10k policy-stop checkpoint 的内部 steps、配置和逐张量组件哈希。
- 盘点 37 个 C/D/E 评测目录；其中 33 个保有完整原始日志，包括 canonical 对照，合计 66,000 局。
- 使用实际牌谱和当前 step 150000 的 `adaptive_best.pth` 做 CPU 加载、输入复现性及标签契约实测；未进行新的自博弈训练或 GPU 评测。
- 运行 335 项相关 CPU 回归测试，覆盖在线训练、数据语义、Oracle 预训、自适应、SL selector/训练、scheduler 和配对评测。
- 台式机 Oracle 预训练和 RiichiLab 客户端仍在运行。没有改动活跃训练依赖的 loader、模型、预训练或自适应模块，没有覆盖 checkpoint，也未打开 sealed test 数据。
- SSH 配置解析到笔记本，但两次只读连接均被远端关闭；因此笔记本结论限于已有的 2026-09-04 23:43 / 2026-09-05 03:34 本地快照，不冒充实时状态。

集中证据目录：[logs/sl_rl_audit/20260905](../../logs/sl_rl_audit/20260905/)。核心文件为 `evidence_before.json`、`cde_paired_vs_canonical.json`、`contract_smoke.json`、`oracle_input_repeatability.json`、`oracle_selfplay_probe.json` 和冻结的 `oracle_probe_features.pth`。

收尾核查见 [verification_after.json](../../logs/sl_rl_audit/20260905/verification_after.json)：记录实际运行的 `.pyd` 哈希、受保护进程及活跃源码未变化的检查。初始快照的 `libriichi` 字段仅记录包入口 `__init__.py`，原生扩展指纹以收尾文件为准；`collector_smoke/` 另验证了更新后的采证脚本能完整执行。

## 2. P0：Oracle 验证输入并没有固定

### 2.1 可复现的根因

`libriichi/src/dataset/invisible.rs` 在未使用可信模拟 seed 时，从日志还原已记录的摸牌、宝牌等，然后用 `filler.shuffle(&mut rng())` 随机补全剩余牌。`GameplayLoader` 的默认参数是 `trust_seed=false`；当前 `OracleTerminalValueDataset` 和在线 `FileDatasetsIter.iter_game_trajectories` 都没有覆盖它。当前预训每次验证重新建立 loader，因此固定文件、game ID 和 state fold 并不能固定这些隐藏输入。

在当前实际 dev index 的第一份人类牌谱上，用同一模型连续解码三次：

| 输入 | 可见 obs | Oracle obs | 抽查 16 个状态的 p0 预测最大变化 |
| --- | --- | --- | --- |
| 人类 dev，默认 loader | 完全相同 | 每次 147/147 个状态发生变化 | 两次相对首遍分别为 0.314929、0.048926 |
| 原生模拟局，默认 loader | 完全相同 | 每次 98/98 个状态发生变化 | 0.094004、0.073106 |
| 同一原生局，`trust_seed=true` 诊断 | 完全相同 | 完全相同 | 两次均为 0 |

这不是 GPU 非确定性：测试使用 CPU，同一个已加载的模型，变化来自重新生成的输入。详见 [输入复现实测](../../logs/sl_rl_audit/20260905/oracle_input_repeatability.json) 和 [可重跑脚本](../../mortal/research/audit_sl_rl_oracle_inputs.py)。

**影响：** 当前 Oracle 的 paired gate 比较还混入了随机补全变化，不能把细小 loss 差异全归因于权重更新；“每次相同验证输入”的表述不成立。这不证明所有旧改进都是假的，也不直接证明整套 CI 必然失效：随机补全可能使估计增加方差，而 best 选择和反复比较还叠加选择偏差。需要实际量化 A/A 噪声，不能凭单局的最大预测变化推算整个 dev 的 MSE 偏差。

### 2.2 应实施的修复

第一步固定验证特征及标签，记录源文件、采样位置、补全版本、特征 SHA-256，使不更新模型的 A/A 结果一致；本次小型 Oracle 诊断已采用冻结特征快照。第二步在人类日志上用多个预先固定的补全版本做稳健性确认，避免把一次随机填充误当成真实完整牌山。不能给人类日志直接开启 `trust_seed=true`；原生模拟日志也必须先核验引擎版本和 seed 重建与记录相符。

冻结后，512 个状态、三个输入模式的完整诊断又运行了一次，两份结果 JSON 的 SHA-256 完全相同，见 [冻结输入复验](../../logs/sl_rl_audit/20260905/oracle_frozen_input_verification.json)。这验证了本次诊断的修复效果，生产训练的验证 loader 尚未替换。

生产修复应在安全 checkpoint 后，以新 source fingerprint / 运行版本接入固定验证快照或显式可复现的补全 RNG；然后在**同一份新验证输入**上同时重算旧 best、latest 和 no-update anchor，重新建立比较基准。不要沿用旧 best 的随机输入统计，和修复后的新统计直接做 paired delta。

此处还涉及数据含义：人类日志的 Oracle 输入包含已记录隐藏信息和未知牌山的随机补全，并非全量真实 Oracle。训练期随机补全可以作为需要验证的建模选择；验证期可复现和“信息实际已知多少”的标注则是确定需要补齐的契约。没有修改运行中 Rust/Python loader，遵循 [活跃训练边界](../agent/code-health.md)。

## 3. P0：RL 的奖励目标与正式评价目标不同

当前 Oracle 配置、旧 C/D/E 和对应 checkpoint 的 `env.pts=[6,4,2,0]`，中心化后为 `[3,1,-1,-3]`。正式 `1v3` 的 `avg_pt` 使用 `[90,45,0,-135]`，除以 45 为 `[2,1,0,-3]`。二者不是正比例加常数关系。

这会实质改变策略偏好。例如“必定第三”和“一半第二、一半第四”，等距位次奖励完全同分；正式 pt 下，前者是 0，后者是 -45。模型即使优化了训练奖励，也可能未改善正式 pt。

**建议优先把正式 pt 确立为共同训练目标，使用数值温和的 `[2,1,0,-3]`，继续保留 `all_players`、真实 `score_rank` / return-to-go。** 若最终目标其实是平均顺位，应反过来正式更改主指标。这里需要用户决定优化什么；不能用“最强模型”含混替代效用函数。

奖励变更必须联动 Oracle 标签、value 的尺度和校准、GAE、在线 reward 以及续训签名。旧 critic 不能只因输出 shape 一样就直接接新奖励；也不存在把四个 value 输出统一乘一个数就完成转换的办法。本次已加入奖励契约检查，避免之后静默混用。

另有一个应独立验证的目标偏差：当前奖励是局间位次势能的普通差分，随后按决策步使用 `gamma=0.999`。当 gamma 小于 1，折扣和不再只由终局位次决定，还包含中途位次变化及其时机；这可直接由折扣求和展开看出。若希望严格对应未折扣终局 pt，应比较 `gamma=1` 的完整终局目标，或在明确终局基础奖励后采用匹配折扣的 potential shaping，并正确处理跳步和终止边界。不能只给现有差分乘一个 gamma 就宣称目标对齐。该方向是目标设计实验，不是本次热修复；有关变换保持策略的条件见 [Ng、Harada、Russell 原论文](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf)。

## 4. P0：旧 RL winner 的统计证据不足

### 4.1 从原始日志重新计算

37 个评测目录反复使用同一组 500 个 seed、四次换座，共 2000 局；`seed_key=2026052350`。重新按完整 `(seed, seed_key)` 匹配候选与 canonical，每组四座位一起 bootstrap。使用 10,000 次重采样，保留每个对局的来源、座位和位次。

下表 delta 为“候选减 canonical”，单位为正式 pt；canonical 的点估计为 -0.81。

| 候选 | 自身 avg_pt | 配对 delta | 名义 95% CI |
| --- | ---: | ---: | --- |
| D 5k / oldstart / clip 0.1 / value 0.02 | +0.7650 | +1.5750 | [-2.3850, +5.4450] |
| E 10k / policy stop | +0.8775 | +1.6875 | [-2.4525, +5.8050] |
| D 20k / cache16 | -0.4725 | +0.3375 | [-3.8475, +4.5000] |
| C 5k | -1.5525 | -0.7425 | [-3.7806, +2.3631] |
| D 20k / V-trace probe | -2.4300 | -1.6200 | [-5.5350, +2.3175] |
| D 20k / fix_old_policy_entropy 长窗 | -10.5300 | -9.7200 | [-13.9275, -5.4450] |
| D 5k / scheduler guard 前的坏版本 | -7.8075 | -6.9975 | [-11.3856, -2.4975] |

32 个保有完整日志的 RL 候选，没有一个得到正的 CI 下界；其中两个坏版本的负信号较明确。**不能据此证明所有 RL 都无效或等价**，但无法支持旧文档中细小差距已经决定最佳 clip、value 权重或稳定强弱顺序的说法。

E 9k 的 +1.3725 在 worker 汇总中仍可核对，但该目录以及另外三个 E 评测目录没有原始逐局日志，无法重建原评测的 paired CI。没有给它虚构区间。

更完整结果见 [配对统计 JSON](../../logs/sl_rl_audit/20260905/cde_paired_vs_canonical.json)。这些区间只描述固定模型在这批牌山上的差异，**没有校正反复挑 checkpoint/超参的选择偏差，也没有包含不同训练随机种子的方差**。历史配置指向相同 baseline 路径和推理选项，其中八份额外显式写了 `oracle_input_mode=zero`；历史对手文件是否始终同字节仍缺完整不可变哈希链。因此它们是有边界的回溯分析，不替代新确认实验。训练 seed 起点与评测的 10000 起点不同，不能仅凭 seed_key 一样指控训练牌山泄漏。

### 4.2 E 9k 与 policy-stop 是同一个部署 actor

逐张量检查得到两份 checkpoint 的 `mortal` 哈希完全相同，`policy_net` 哈希也完全相同：

- actor：`c88a8b8718012fc9cc5e46805860870667d48a7c3608f2ed75c7b9f7354fd53b`
- policy：`ab3fe46ef13ce330cce81ec29c586416fa6c2c496bc5e7b2d169765e5aec4e66`

Oracle/value 的哈希不同，与只更新 critic 一致。因此不能用 +1.3725 对 +0.8775 宣称两个 actor 的学习强度不同；相同权重在同一部署语义下是同一策略函数。评测运行的数值/批次/配置/产物来源差异尚未定位，须用 A/A 复现排查。本节与 Oracle loader 随机补牌是不同问题：visible-only 部署的 E actor 不应被直接归因于 Oracle 验证噪声。

### 4.3 以后 2000 局能决定什么

2000 局保留为粗筛和严重退化诊断，不能在同一批牌山上反复选最优再称作独立确认。以本次 D 5k 的 paired SE 约 2.0 pt 粗估，若真实增益只有 1 pt，要在双侧 5% 水平获得约 80% 检出功效，需要约 **15,688 个 seed set，即每臂约 62,752 局**；0.5 pt 则约 251,000 局/臂。公式为 `N_new = 500 × ((1.96 + 0.842) × SE_500 / delta)^2`。

这是依赖本次方差、固定模型的规划近似，不是所有后续实验都必须花同样预算，也未计多重比较。可以先用 16k 局筛选、只让少数 finalist 进入独立 64k 局确认；到预算仍跨零，就记录未决，不强行选 winner。多训练种子、区间和选择偏差的重要性可参考 [Agarwal 等，统计可靠的深度 RL 评估](https://arxiv.org/abs/2108.13264)。

## 5. SL：保留路线，降低对旧精细排名的信任

### 5.1 旧正式决赛是 close-call

2026-04-05 playoff 原始汇总：

| 候选 | 局数 | avg_pt | 日志记录的 stderr | avg_rank |
| --- | ---: | ---: | ---: | ---: |
| anchor × 1.0 | 39,936 | -0.775240 | 0.422654 | 2.509215 |
| opp_lean × 0.85 | 39,936 | -0.863131 | 0.423682 | 2.508213 |

pt 差仅 0.087891，汇总中的 combined stderr 为 0.598451，且 `close_call.triggered=true`。这里引用旧 stderr 仅说明量级，不把它当作已重建的精确 paired CI。保留 anchor 的发布身份合理，但“已经证明它更强”不成立，顺位点估计甚至反向。没有理由仅凭这次审计反过来发布替补。

### 5.2 S70 的离线改善有信息，但不是正式胜率

checkpoint 自带 full-recent policy loss：canonical 0.476685，S70 0.447225；old-regression policy loss 分别 0.516943、0.492205。它们支持把 S70 视为重要候选；由于需要补齐历史数据/验证配置指纹，不能把不同 checkpoint 保存的指标直接视为严格配对实验。

现有 S70 文档还报告了 64k 对局提升，但本次未重建该历史比较的完整逐局证据链，因此不把文档中的强度主张升级为本次已验证结果。外部公开权重的少量 2000 局对照也不足以证明普遍领先；RiichiLab 排位不是指定对手的 formal 1v3。

### 5.3 selector 和长训的实际改进

`sl_selection.py` 的动作权重、众多场景权重及 `SELECTION_SCENARIO_FACTOR=0.20` 是可解释的工程 proxy，不是由正式强度证据逐项确证的常数。场景重叠、罕见类样本数和反复调权都会影响选择。保留 `comparison_recent_loss / recent_policy_loss` 作为策略比较口径，`full_loss` 继续只诊断 auxiliary tax，不以它单独选强模型。

下一轮只保留少量预声明候选：policy-loss 最优、action-score 最优、满足全部退化约束的 adaptive-best；相同 actor 哈希去重后，再与现有 canonical / S70 作正式对照。不要重新开几十组辅助头权重搜索。

对于当前 B/C 课程，最有辨别力的对照是从同一 A best 分出“保持 A 分布的小步续训”和“进入 B/C”，固定起点、优化器、训练预算与确认集。这样才能区分新近数据课程的收益和单纯多训的收益。最终是否扩大网络、取消辅助头或更换架构，本次没有足够证据，均不直接实施。

## 6. Oracle：预测能力存在，自动决策与上线资格分开判断

### 6.1 当前数据和训练并非全无依据

当前 run 为 `s70_broad_to_recent_strong24m12m_adaptive_sf200_wd0_20260901_r1`，S70 初始化、dual tower、GN、`all_players`、真实 `score_rank_mc`、gamma 0.999、MSE、精确零和输出。Schedule-Free AdamW 使用 lr 0.0002、wd 0、warmup 2000；保存路径检查了 optimizer eval mode，没有发现把训练参数点误当验证参数点导出的证据。该检查符合 [Schedule-Free 官方使用说明](https://github.com/facebookresearch/schedule_free)。

主快照的 best 为 step 150000，latest monitor 为 1200000。总体 MSE 3.277892 → 3.268630，对应零预测 baseline MSE 5.154053，相关性约 0.605、解释方差约 0.366。因此它有预测能力，不能继续沿用“Oracle 分支还完全随机”的旧诊断。精确零和由网络结构强制，不能单独拿来证明质量。

真实牌谱契约实测：strict load 成功，输出 shape `[4,4]`，有限数值；98 步轨迹上离线 MC 与在线 `GAE lambda=1` 最大差异 `5.96e-7`。这证明当前标签和接入契约可工作，不证明 PPO 学习收益。

### 6.2 小型模拟局输入对照

本次从旧 canonical 对局按 seed 无放回抽取 32 组 × 四座位 = 128 局，每局均匀选 4 个状态，总计 512 个状态；不按结果大小挑选或加权。使用当前 best critic，在冻结特征快照上比较原始 Oracle 输入、置零和错配输入；错配限制在同一 seed set 的不同座位对局内，bootstrap 仍按 seed set 聚类。

结果见 [Oracle probe](../../logs/sl_rl_audit/20260905/oracle_selfplay_probe.json)。总体 `all_players` MSE 的输入依赖改善较明确，p0 的名义区间仍跨零。这里的“true”沿用代码命名，包含记录信息与冻结的一次未知牌山补全，不等于已知完整真牌山。该小样本来自旧 canonical 的模拟分布，只能作为探索性诊断，不能替代 S70 当前策略分布上的正式资格确认。

| 配对 MSE 差（true 减对照，负值较好） | 点估计 | 按 32 个 seed set 的 95% CI |
| --- | ---: | --- |
| all_players / zero | -0.09912 | [-0.17644, -0.02515] |
| all_players / shuffled | -0.12370 | [-0.19375, -0.05905] |
| p0 / zero | -0.02581 | [-0.09535, +0.04543] |
| p0 / shuffled | -0.05579 | [-0.12439, +0.01164] |

### 6.3 自适应的两个独立缺陷

**护栏不是否决条件。** 在 `observe_adaptive_curriculum` 中，primary 一旦显著改善就直接 `update_best`；护栏只在后续决定是否有补偿可继续训练。本次反例：primary 从 1.00 降到 0.99，而 tail 从 1.00 恶化到 2.00，32 个 cluster 完全一致，返回仍是 `update_best`。它与“所有 guardrail 通过才选优”的口径不一致。修复应显式区分“不得退化的约束”和“允许延长阶段的次指标”，每项给出非劣门限、CI 方向和缺失样本处理。

**低功效会让 observe 长期持续。** 1200000 对 150000 的 primary delta 为 +0.003401，CI 为 [-0.009715,+0.016517]；meaningful_delta 只有 0.0002，约为区间半宽的 1/65。200k 至 1200k 共 21 次 gate 都因为“仍可能改善”继续观察。固定验证集被反复查看，不等于增加了 21 份独立验证证据。

这也不能简单写成“后面没有学习”：总体 loss 和 p0 MAE 有小改善；与此同时 exact-zero MSE 恶化 +0.015962，名义 CI [0.001910,0.030014]。这些判断还受第 2 节输入随机性影响。**不应因为 latest 训练更久就覆盖 150k best，也不能因 best 长期没变就宣告该阶段绝对无用。** 应先固定输入做 no-update A/A，再扩充独立验证游戏/合理增加状态覆盖，随后决定可检测改善门限与未决时的资源上限。

### 6.4 接入 actor 前还缺什么

在人类 holdout 上 MSE 较低，不能保证它在当前 AI 策略分布上估计正确。建议在最终 SL actor 冻结时采集独立模拟轨迹，用与正式目标一致的真实 return，对 `all_players`、p0、非零及尾部切片、校准和 `true / zero / shuffled` 做确认；保留 no-update anchor。接着仅让 critic 在该分布上校准，再进入小步 actor 更新。资格门不通过时，不用 GRP 标签顶替真实结果，也不把输出简化成只训练 p0。

## 7. V-trace 和 PPO 的边界需要澄清

当前 `compute_vtrace_targets_from_step_rewards` 的 value 递推可以对应 IMPALA 的 V-trace；但返回的 `pg_advantages` 已包含截断的 `pi_target / mu`，随后 `train_batch` 又使用 `pi_now / mu` 进入 PPO surrogate。也就是两层比率共同参与 actor 更新；经过标准化和 clipping 后，不能笼统说总权重永远等于 rho 平方。

标准 IMPALA 的 actor gradient 使用一次显式截断比率，见 [原论文](https://proceedings.mlr.press/v80/espeholt18a/espeholt18a.pdf)；PPO 则以采样时策略和对应优势构造 clipped surrogate，见 [PPO 原论文](https://arxiv.org/abs/1707.06347)。当前组合不能仅凭递推函数正确就称整条训练目标已被验证为标准 IMPALA；本次也不据此断言所有 PPO/V-trace 组合都错误。

因此旧一次负 probe 只能说明该实现和配置的结果偏弱，不能否定 V-trace 算法本身。较稳妥的路线是先建立干净的近 on-policy PPO 参照，单独验证新鲜数据、行为 log-prob、目标策略、actor 更新幅度，再将 replay 或明确推导的异步 PPO / IMPALA 分支分别加入。新增分支应具备小型可解析轨迹上的完整 actor-gradient 测试，不能只测 value recursion。本次没有热改算法或直接启动替代长训。

PPO clipping 也不是累计策略变化的硬上界。应增加基于同一固定状态集的 action KL、clip fraction、行为版本覆盖和重要场景动作变化，再据实验选择 KL budget / early stop；不能只看 entropy 或一个最大 importance ratio，就把所有低分归因于 actor drift。

## 8. 数据隔离和远端可靠性

直接训练/验证的时间隔离有明确设计：当前 SL broad 到 202412，full-recent 是 202501–202512，monitor 是 202601；Oracle train 到 202511，dev 是 202512，test 是 202601，另有旧分布回归集。

但全流程独立性尚未证明：Oracle 初始化来自经过 SL 选择的 S70，而 SL 的 full-recent / monitor 与 Oracle dev / test 月份重叠。即使 Oracle trainer 没打开 test，也不能自动声称该 test 从未参与上游模型选择。需要恢复 **S70 实际用过的** index/文件哈希与选择记录，区分“只对 Oracle 调参封存”和“对整个训练流水线从未使用”。当前代码窗口只是风险证据，不足以断言 S70 历史训练发生了直接数据泄漏。本次没有打开 sealed test 内容。

笔记本的 03:34 本地快照记录 phase C 在 157890 左右 OOM，随后从 151890 恢复，至少回滚 6000 steps；当时系统可用 RAM 约 0.47 GiB / 31.7 GiB，GPU 使用约 7.3 / 8 GiB，workers=4。不能把这种工作集压力只解释成模型训练不稳定。建议安全 checkpoint 后先缩减 loader workers/prefetch 或驻留工作集、提高合理保存频率并保留 exact 状态，不先放宽 guard 或扩大 batch。没有因这次审计擅自重启远端任务。

## 9. 已经落实的修改

| 修改 | 复现与行为 | 边界 |
| --- | --- | --- |
| 奖励签名检查 | 修复前同 shape 但不同 rank 奖励的 Oracle checkpoint 会被接受；现在检查中心化 `env.pts`，并纳入在线 resume 签名 | 允许不改变中心化奖励的平移；尺度/效用变化需新实验，缺失来源时不猜测 |
| replay 过滤 | `drop_untracked_samples` 过去在整批都不 tracked 时反而不丢，`-1` 也被算作保留；现在未知/淘汰版本不保留，整批不可信则跳过 | 旧关键 run 通常未打开此开关，不能把它当作所有历史退化的根因 |
| 单样本 advantage 标准化 | 修复前无偏 `std()` 对单样本产生 NaN；现在单样本标准化 advantage 为 0，value 标签原样保留 | 多样本的既有数值口径保持 |
| 严格配对评测工具 | 校验完整 seed_key、四座位、缺失/重复和精确配对，按四局 seed set bootstrap | 不自动校正多重选择，也不替代对手/规则/推理 provenance 核验 |
| 审计/诊断工具 | 源码和 checkpoint 证据采集、真实契约 smoke、原始输入 A/A、冻结特征的 Oracle 输入对照 | 均为独立研究入口，没有接入活跃训练热路径 |

修改入口：[train_online.py](../../mortal/online/train_online.py)、[paired_1v3.py](../../mortal/eval/paired_1v3.py)、[审计回归测试](../../mortal/tests/test_online_audit_regressions.py)、[配对评测测试](../../mortal/tests/test_paired_1v3.py)。修复前反例保留于 `bugs_before.json`。335 项相关测试通过，另有真实数据 CPU smoke；这证明本次修改满足所测契约，不代表完整 GPU 训练系统已被重新资格认证。

机器可读 [测试结果](../../logs/sl_rl_audit/20260905/tests.json) 记录 335 tests、0 failure、0 error、0 skip；[本次 online 修改差异](../../logs/sl_rl_audit/20260905/train_online_changes.diff) 相对审计开始时的源码生成，避免把用户此前已有修改归入本次修复。

## 10. 可落地的下一轮决策

| 顺序 | 工作与建议 | 可验收结果 | 需要决定的部分 |
| --- | --- | --- | --- |
| 1 | 安全 checkpoint 后修复 Oracle 验证输入可复现性；重算 150k、latest、no-update anchor | 不更新模型的 A/A 同输入同输出；所有候选使用同一新快照/补全版本 | 不需要决定科学方向；需要安排活跃 run 的切换边界 |
| 2 | 将正式 pt 与训练效用对齐，推荐 `[2,1,0,-3]` | 预训/在线/评测的目标契约一致；旧奖励与新奖励不会静默 resume | 最终优化正式 pt 还是平均顺位；gamma/终局 reward 的对照协议 |
| 3 | 恢复固定 actor、对手、引擎、推理配置的 1v3 A/A | 同 actor 在固定运行条件下可复现；差异有可定位日志 | 仅在 A/A 排清后分配正式新牌山预算 |
| 4 | 在 canonical、S70、SL phase C 少量 finalist 中选择新底座 | fresh screen 与独立 confirmation 分离，完整四座位日志和区间 | 可接受的最小增益与预算；16k 筛选 / 64k 确认是本次建议而非已经通过的门 |
| 5 | Oracle 在最终 actor 的模拟分布上资格确认，保留 all_players 和 no-update | p0 主指标及每个非劣护栏同时达标；有置零/错配和校准对照 | 护栏允许退化幅度；不能为选出新模型临时放宽 |
| 6 | 从同一 actor + 完整状态分出最小 PPO 参照与合格 Oracle 分支，先限制数据陈旧度 | 小窗暴露数值/契约错误，候选在新牌山确认真实 pt 增益 | 至少 3 个训练种子估计训练波动；之后再决定 replay / V-trace / KL 目标 |

我会优先把算力投向这套验证顺序，而不是继续对已经被反复看的 2000 局做 clip/value 权重网格搜索。新实验应冻结一个 primary，明确最小有意义效应和每条 guardrail，预声明最大候选数、checkpoint 选择规则、确认集和停止规则。监控集用于选择；确认集只在 finalist 冻结后使用。误差条跨零时允许“未决”；任何新增策略都必须与未更新的 SL/actor anchor 同场比较。

## 11. 复核入口

以下命令在仓库根目录执行。Python 解释器为 `C:\ProgramData\anaconda3\envs\mortal\python.exe`；重新采证请换新 output 目录，避免覆盖本次快照。

```powershell
$auditPython = 'C:\ProgramData\anaconda3\envs\mortal\python.exe'
& $auditPython -m mortal.research.audit_sl_rl_evidence --output-dir logs/sl_rl_audit/NEW_SNAPSHOT

& $auditPython -m mortal.eval.paired_1v3 `
  --candidate-log-dir logs/oracle_cde/dual_D_s5000_value_w002_clip010_cache16_oldstart_seed2026052350_1v3 `
  --reference-log-dir logs/oracle_cde/sl_canonical_seed2026052350_1v3 `
  --candidate-name mortal --reference-name sl_canonical `
  --bootstrap-replicates 10000 --output logs/sl_rl_audit/NEW_SNAPSHOT/d5k_pair.json

& $auditPython -m mortal.research.audit_sl_rl_contracts --help
& $auditPython -m mortal.research.audit_sl_rl_oracle_inputs --help
& $auditPython -m mortal.research.audit_sl_rl_oracle_probe --help
```

Oracle probe 的 `--feature-snapshot` 用于第一次固化输入，后续以同一快照重算；复现时同时核验 checkpoint、快照和原始日志哈希。完整结果可追溯到机器可读证据，不依赖本报告文字才能复算。
