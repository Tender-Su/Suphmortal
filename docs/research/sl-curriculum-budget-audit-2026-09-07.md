# SL 阶段预算、近期数据与 S70 对战审计

> 核验：2026-09-07 · 范围：笔记本已完成 Long-ABC 的真实配置、索引、日志、checkpoint 交接与本机独立 1v3。辅助损失继承防护已修复；课程和学习率改法尚待受控训练验证。用户后续明确机制优先，具体推荐以 [课程机制方案](sl-curriculum-mechanism-proposal-2026-09-07.md) 为准，手工预算仅作退路。

## 判断

用户关于 A 占用过多时间、近期数据适应机会不足的怀疑有依据。核心是机会成本：A 先开始，而控制器从未比较“继续 A”与“现在转 B/C”的收益。三阶段没有共享步数额度，不消除这种顺序偏置。A 的很长核心日程、尾部耐心、后续低学习率、固定小训练池、辅助目标变化共同构成了问题，不能只归因于阶段长短。

现有结果没有证明模型的学习能力被 A 永久耗尽，也没有证明“近期数据已经充分利用”。下一步应先修正续训的可比性，再用固定预算对照；直接把旧 C 多跑几百万步，证据不足。

## 实际训练量与数据

| 阶段 | 本阶段新增 batch steps | 成功 optimizer updates | 占三阶段已完成轨迹 | 固定训练索引中的独立游戏 |
| --- | ---: | ---: | ---: | ---: |
| A | 3,800,000 | 3,798,575 | 89.41% | 863,572 |
| B | 200,000 | 199,927 | 4.71% | 120,000 |
| C | 250,000 | 249,909 | 5.88% | 80,000 |

C 最后 checkpoint 的 `steps=300000`，但从 B best 的 `steps=50000 / optimizer_steps=49985` 接续，因此本阶段实际新增 25 万步。这里统计逻辑轨迹；崩溃后重算、已放弃的旁支会增加物理消耗，不能把表格当作全部 GPU 工作账单。优化器计数需要忠实于所恢复的 checkpoint，累计计算费用应另设不随 best 回滚的账本。

A 的核心 cosine 日程本来就有 252 万 optimizer steps，之后才进入尾部。它最终到 380 万步；B 初始化选择 A 的 288 万步 checkpoint。A 后续 92 万步没有进入最终继承的权重轨迹，仍然消耗了计算。这些探索不能全部叫作无用，但它们有明确机会成本。

| 阶段 | 时间分布按索引条目计 | 近期独立游戏 | 解释 |
| --- | --- | ---: | --- |
| A | 60% 为 2023–2024；25% 为 2021；15% 为 2009–2020 | 353,201（2023–2024） | A 已包含全部当前近期窗口；加权索引共 2,153,721 条，最多重复 4 次 |
| B | 90% 为 2023–2024；10% 为 replay | 108,000（2023–2024） | 固定抽样池，近期独立局数约为 A 近期池的 30.6% |
| C | 98% 为 2024；2% 为 replay | 78,400（2024） | 固定抽样池，约为 A 索引中 2024 独立游戏的 42.5% |

这是文件采样比例，不等于模型实际消费的决策状态比例；各局长度不同，旧 loader 也没有精确逐样本续跑账本。B 已进入第二个 epoch，C 已进入第三个 epoch；步数较少不能直接推出每局学习次数较少。

当前“近期训练数据”最晚到 2024；2025 和 2026-01 用于验证。更近期的牌谱可能更接近目标对手与策略分布，但年份本身不能证明每个动作更优。不能把现有验证数据直接移入训练后仍沿用原有独立性声明。

原始证据：[逐阶段统计和摘要](../../logs/sl_curriculum_audit_20260907/curriculum_evidence.json)、[统计脚本](../../logs/sl_curriculum_audit_20260907/analyze_training.py)、[B/C 原始 summary](../../logs/sl_curriculum_audit_20260907/remote_evidence/bc_summary.json)、[C 交接记录](../../logs/sl_curriculum_audit_20260907/remote_evidence/phase_c/adaptive_phase_handoff.json)。

![阶段预算、数据覆盖和验证曲线](../../logs/sl_curriculum_audit_20260907/curriculum_evidence.png)

曲线来自日志中舍入后的数值。A 与 B/C 的完整验证集不同；B/C 期间也曾修复验证代码。因此图用于定位现象，不构成跨阶段、跨版本的严格因果比较。

## 具体问题

1. **耐心衡量的是当前 LR 档的改善，缺少转向后续数据的收益比较。** [旧 convergence 控制器](../../mortal/supervised/convergence.py) 在每次降 LR 后把本档 best 重新设为无穷大。A 在 289 万步附近达到 smoothed NLL 0.440306；之后 2.5e-6 与 1e-6 两档仍有 7 次本档“improved”，其数值都没超过这一全局水平，却会重启本档耐心。它不是数学上无限运行，但没有按未来 B/C 的机会成本停止。更长的 A 也不能全归因于耐心，其中 252 万步是预设核心日程。
2. **B/C 的适应强度和数据覆盖同时受限。** B 从 A 的 5e-6 只回升到 1e-5，C 沿用 1e-5 后降到 5e-6；A 主训练峰值是 1e-4。B/C 固定池 12 万/8 万来自 [AB 入口的 screening 默认池](../../mortal/supervised/run_sl_ab.py)，多跑 epoch 不会自动补齐遗漏的近期游戏。低 LR 不一定错误，缩小数据池也不自动证明过拟合；但这组设置无法排除“适应不足或固定池泛化不足”。
3. **A→B 实际更换了辅助训练目标。** rank_aux base 从 0.001548 到 0.03，max 从 0.00516 到 0.1，均约 19.38 倍；opponent weight 从 0.00135 到 0，danger 从开启且 weight=0.00804 到关闭。真实 B 日志也打印了这些值。即使只比较 policy loss，共享 backbone 的梯度也受到不同辅助目标影响。不能将结果解释成纯粹的数据课程实验。
4. **A 与 B 的比较输入没有锁定。** 512 局 full-recent 只有 2 局重合，256 局 old-regression 没有交集；monitor 的 12,188 局索引相同。B/C 的两组完整验证索引一致。A 的 0.440622 与 B 的 0.443979 因此不能直接相减来宣称 B 变差；bootstrap 标记 `eval_split_digests_match=false` 也已被原始索引验证。
5. **停止只证明这条训练配方没有获得要求的改善。** B 在 15/20 万步连续 futile 后转 C，C 在低 LR 档未改善后停止；C best 实际仍是继承的 B best。后续加载完整的权重通过 1v3 可以判断当前候选强度，但无法独自识别“早转 B/C”是否更优。

## 已实施的确定性修复

[SL bootstrap 入口](../../mortal/supervised/run_sl_ab.py) 现在默认继承 checkpoint 内的 `aux`、`supervised.aux` 与 `supervised.rank_aux`，并在不可变 bootstrap 记录中写明源配置、调用时配置、最终生效配置及是否改变目标。课程交接还保留辅助 ramp 时钟；直接恢复检查辅助系数和目标设置。需要做辅助目标消融时，显式选择 `--adaptive-bootstrap-auxiliary-policy current`；默认是 `inherit`。缺少源 config 时不能假称完成继承。补充实现与验证见 [机制方案](sl-curriculum-mechanism-proposal-2026-09-07.md#已修复的辅助目标契约)。

这项修复保留新阶段的数据、LR 和 optimizer reset 选择，也不改历史 checkpoint、旧 run manifest 或运行中的源码快照。它防止无意替换辅助目标，不等于已经验证继承旧配方必然更强。旧 run 仍由其冻结源码解释；新主树行为应使用独立 run。

验证：补充修复后 125 项相关测试通过，包含 bootstrap 两阶段传递、恢复/交接、辅助时钟及现有 adaptive 护栏；另用本次取回的真实 A/B 配置重放，确认继承 A 辅助目标，同时保留 B 数据/LR 设置。见 [测试记录](../../logs/sl_curriculum_audit_20260907/repair_tests_v2.log)、[真实配置回归](../../logs/sl_curriculum_audit_20260907/auxiliary_recipe_regression_v2.json) 和 [新增回归](../../mortal/tests/test_sl_bootstrap_recipe.py)。

## 可落地的下一步

**先处理当前模型的适应问题，再证明早转段是否值得。** 不需要为此立刻从零重训 A。

- 固定一个 A anchor、辅助配方、验证游戏和推理契约。每臂都先评估未更新的 step 0；A 与 B/C 都在这组输入上重算。
- 用一个 2×2 小对照拆开主要变量：固定小池 / 近期全量或轮换覆盖，分别搭配 peak LR 1e-5 / 2.5e-5。2.5e-5 是待验证候选，不是已证明最优值；保持原 replay 比例，保持相同有效更新预算、warmup 和 optimizer 初始化方式。先检验近期拟合、旧域护栏与策略 drift，再进入正式 1v3。
- 优先从同一权重比较 A/B/C 的后续目标域收益，让三种数据竞争计算机会。即使 A 仍在改善，B/C 若更有帮助也应获得机会；未决时对称扩大比较或增加验证精度，不能只续长 A。具体实验与短视防护见 [机制方案](sl-curriculum-mechanism-proposal-2026-09-07.md#推荐实施路径)。手工分配阶段预算仅在机制无法可靠工作时作为退路。
- 把本档 warmup/恢复与全局改善分开。保留最短适应机会，延长预算则要求相对全局候选的有效改善；记录被延长的总步数，避免每次降 LR 都重新获得完整尾部额度。
- 在适应方案确定后，再从不同 A 时点比较早转与晚转；各臂按**端到端相同总计算预算**比较。仅在已经训练到 288 万步的 A 上成功续训，不能证明提前停止 A 会更好。

支持这一实验方向的外部证据来自持续预训练研究：目标域继续训练可能带来收益，重新升高 LR、再衰减并加入 replay 是已有实证路线。但论文对象是语言模型，不能替麻将网络决定最佳 LR、年份比例或阶段预算。[Gururangan 等，ACL 2020](https://aclanthology.org/2020.acl-main.740/)、[Gupta 等，2023](https://arxiv.org/abs/2308.04014)、[Ibrahim 等，2024](https://arxiv.org/abs/2403.08763)。

本次没有启动以上长训对照，也没有凭单次历史曲线直接更改 LR 或把 2025 验证数据并入训练。

## SL 对三家 S70 的独立 1v3

已把真实 `phase_b_adaptive_best.pth` 取回本机，文件 SHA-256 与远端一致，actor SHA-256 为 `b15dd238de18e95ef367e751e9e20876306bfb1dd9b01f5ff06ee5026e08af67`，与先前 shortlist 的 Phase C 50k 相同。S70 文件与既有正式输入一致。评测使用历史候选原权重，未把本次辅助继承修复混入权重。

固定协议：离线 finalist 对三家 S70 2,000 局；S70 对三家 S70 另做相同 seeds 的 2,000 局。每个 seed 完成四座轮换，以整个四座 seed set 为 bootstrap 单位；主指标为 pt 差值及 95% CI。关闭 search、exploration、AMP、compile 和 agari guard。该比较用于回答本次直接对战问题，不替代笔记本已部署的 canonical 16k 筛选 / 独立 64k 发布确认。

A/A 两臂各 8 局的完整事件流检查已通过。两臂各 2,000 局现已完成：SL 平均 pt **+0.7425**，平均顺位 **2.4905**，一至四位为 **490 / 532 / 485 / 493**；S70 自对战平均 pt 0、平均顺位 2.5。500 个配对四座 seed sets 的 pt 差值 95% bootstrap CI 为 **[-2.1825, +3.6675]**，顺位差值 CI 为 **[-0.0475, +0.0285]**。因此目前不能确认 SL 强于 S70；没有发布或替换模型。运行采用冻结的独立源码，allocator 限制为显存的 12.5%，torch/rayon 各 2 线程。

产物：[完整配对结果](../../logs/sl_curriculum_audit_20260907/s70_1v3_2000/comparison.json)、[冻结协议](../../logs/sl_curriculum_audit_20260907/s70_1v3_2000/protocol.json)、[A/A](../../logs/sl_curriculum_audit_20260907/s70_1v3_2000/aa_decision.json)、[运行状态](../../logs/sl_curriculum_audit_20260907/s70_1v3_2000/apex_supervisor_status.json)、[执行入口](../../logs/sl_curriculum_audit_20260907/run_s70_comparison.py)。本 CI 条件于已锁定 finalist 和这些 seeds，不包含训练种子不确定性或候选搜索校正。
