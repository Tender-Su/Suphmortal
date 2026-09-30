# SL / RL 审计修复实施：2026-09-07

本文承接用户已批准的 [9 月 5 日审计](sl-rl-audit-2026-09-05.md)，影响 SL 选择、Oracle 预训练、在线 RL 与正式评测。实现缺陷已修复并落到台式机主工作树；修复版 Oracle 已开始新奖励校准，笔记本独立确认流水线已部署。**工程通过不等于牌力提高：新 canonical、Oracle 的 actor 接入资格和 RL 收益均尚未确认。**

## 1. 修复与验收对应

| 审计问题 | 实施后的行为 | 可复核证据 |
| --- | --- | --- |
| 验证隐藏输入每次变化 | 原生补全按 canonical events 的哈希和显式 seed 使用 ChaCha8；验证 seed 固定，训练按 pass 确定性变化；验证文件内容、顺序、补全和 native 摘要锁定 | 真实牌谱固定 seed A/A 完全一致；改 seed 会改变隐藏输入 |
| native fold 压缩折扣时钟 | 保留完整轻量决策时钟及选中 indices；完整轨迹计算 all_players return 后取样；旧 ABI 缺元数据时拒绝 folded 训练 | 601 状态，16/64 folds 覆盖全集；gamma 1、0.999、0.95 最大标签差均为 0 |
| primary 改善绕过 guard | 所有 guard 的配对区间均满足预声明非劣门限才更新 best；缺失或样本不足拒绝选优 | primary 1→0.9 而 tail 1→1.2 的反例被否决；旧规则可复现为误选 |
| 长期 observe 无预算 | 新状态 schema 明确 `inconclusive`；未决有次数上限，runner 不自动晋级；step 0 先建立未更新 anchor | 未决上限/后续调用/跨阶段终止回归通过；生产新分支最多 40k |
| 训练效用与正式 pt 不一致 | 新模板与新分支 `[2,1,0,-3]`、gamma 1；奖励/目标时钟/actor 版本/架构进入恢复契约；旧目标只可显式另开分支 | 旧、新 rank utility 不会静默混用；旧产物保持不变 |
| V-trace actor 再叠 PPO ratio | PPO clipped surrogate 与显式 V-trace policy gradient 分开；后者直接使用一次已校正 advantage，拒绝旧 hybrid 配置 | 可解析单步轨迹的完整 actor gradient 对拍；PPO clipping 梯度检查 |
| 行为版本及 actor drift | 在形成轨迹目标前过滤未知、过旧和未来版本；单样本归一化有限；记录近似 KL/clip fraction，越界 batch 不更新 | 版本覆盖、整批无效、NaN ratio、A/A drift 与历史 audit regressions |
| 离线 critic 接不进实际在线结构 | 在线构造器显式传 residual fusion / head 维度与 GN；严格校验 architecture、all_players、score_rank、gamma 和 reward | 实际新 checkpoint 通过在线 strict load；4 个真实状态预测逐位相同，824 个梯度 tensor 有限 |
| 重复筛选及弱正式证据 | 最多 3 个不同 actor，独立筛选与确认 seed key、固定四座/对手/推理；先通过完整事件流 A/A，再 16k 筛选、单 finalist 64k 确认 | 冻结协议、原始日志、按 seed set bootstrap；CI 跨零保持未决，不自动发布 |
| 中断和来源混合 | 完整 chunk 写 completion 记录；中断 chunk 隔离保留并整组重跑，重用前检查事件流和源码指纹；wheel 指纹解析到实际 `.pyd` | 完成/续跑/源码变化反例测试；双机测试与实际 native 摘要 |
| sealed test / 上游隔离不明 | test 入口要求冻结 finalist checkpoint 哈希；默认禁 final test；跨阶段 ledger 按原始游戏 ID 检查索引交集，未知祖先保持未决 | 仅读取 indexes 建账，未打开 sealed test 内容；不伪造全流程 holdout |

实现入口：[数据标签](../../mortal/data/oracle_value.py)、[原生 loader](../../libriichi/src/dataset/gameplay.rs)、[自适应选择](../../mortal/core/adaptive_curriculum.py)、[actor 目标](../../mortal/online/policy_objective.py)、[在线训练](../../mortal/online/train_online.py)、[正式确认](../../mortal/eval/confirmation_protocol.py)、[数据台账](../../mortal/data/split_ledger.py)。

这里的 KL 是训练 batch 的采样估计，并非全状态 KL 硬上界；0.02 / clip fraction 0.5 是保守起始阈值，仍需在固定状态和重要场景上评估策略变化。算法分离测试证明所测梯度契约，不能据此宣称 V-trace 的最终收益或所有异步训练细节已经资格认证。

## 2. 验证结果

| 验证层 | 结果 |
| --- | --- |
| 台式机 Python 全量相关 discovery | **789 tests，0 failure，0 error**；覆盖主树当前修复的独立源码副本 |
| 笔记本 Python discovery | **789 tests，0 failure，0 error**；本机环境和原生扩展运行，未复制台式机二进制 |
| Rust release / clippy | 独立 release 构建和 clippy 通过；未在活跃环境执行 maturin develop |
| 原生真实数据契约 | 601 状态；所有 folds 合并等于全集，visible/hidden/target 同索引对齐，固定补全可复验 |
| Oracle 真实训练 smoke | 新奖励 CPU 训练 2 步后续跑到 3 步；恢复契约通过；是数值与保存/加载检查，不是质量评测 |
| 在线 critic 实际接入 | residual dual tower 24,532,292 参数；与离线构造的 4 个真实状态输出逐位相同，零和残差 ≤2.39e-7，824 梯度 tensor 有限 |
| 原始正式 A/A | 台式机两臂各 8 局，完整事件流哈希相同；笔记本 GPU/native 新协议也已通过 8+8 局完整事件流 A/A，11:52 开始正式筛选 |

证据根目录为 `logs/sl_rl_fixes/20260905_desktop_r1`。关键附件：[Python 结果](../../logs/sl_rl_fixes/20260905_desktop_r1/python_suite_main_final.log)、[原生输入对拍](../../logs/sl_rl_fixes/20260905_desktop_r1/native_contract_verification_final.json)、[在线接入](../../logs/sl_rl_fixes/20260905_desktop_r1/online_critic_integration.json)、[旧 A/A](../../logs/sl_rl_fixes/20260905_desktop_r1/formal_protocol/aa_decision.json)。

笔记本最初出现一个测试假失败：测试只等 PowerShell 启动 2 秒就终止，冷启动尚未生成日志。已改为等待实际启动边界、禁 profile，然后检验 stale 文件不会触发退出；重新全量通过。原失败日志保留，没有通过跳过断言制造绿色结果。

笔记本原始附件已同步本地：[789 项测试](../../logs/sl_rl_fixes/20260905_desktop_r1/remote_verified/repair_tests_verified.log)、[新 A/A 判定](../../logs/sl_rl_fixes/20260905_desktop_r1/remote_verified/aa_decision.json)、[冻结确认协议](../../logs/sl_rl_fixes/20260905_desktop_r1/remote_verified/protocol.json)。协议指纹 `a493eea4bfb0f91d32c82e4f04698fdb1cf0ee0cb18f26eb45830a8ac62d91fa` 已重算匹配；reference / S70 文件哈希与台式机原件相同。

## 3. 旧 Oracle 结论重新约束

使用固定新补全与修正后的完整标签时钟，32 个 dev 游戏、1340 状态上重算 150k anchor、同 anchor 重复和当时的 latest 1419948。anchor 两次 p0 MSE 完全相同，均为 **2.96423769**；latest 为 **2.96972942**。

latest 减 anchor 的 p0 MSE 为 **+0.00549181**，按游戏聚类的 95% CI 为 **[-0.04227834, +0.05326197]**，仍未决。不能因训练更久选 latest，也不能据此断言后续训练完全无效。这次重算沿用旧 reward / gamma 以重审旧结论，不能与新正式 pt 分支的 loss 直接横比。见 [原始汇总](../../logs/sl_rl_fixes/20260905_desktop_r1/oracle_rebase_summary.json)。

## 4. 双机实际切换

台式机旧 Oracle 已安全暂停在 Phase C **170446**，确认训练树退出后退役专属 supervisor。完整 optimizer、scaler、scheduler、data progress、模型和旧配置保留；[退休记录](../../logs/sl_rl_fixes/20260905_desktop_r1/legacy_pause_20260907.json) 可复核。没有覆盖活跃 checkpoint、安装环境 `.pyd`，没有中断 RiichiLab 客户端。

新 run 为 `logs/oracle_critic_formal/sl_rl_repair_formal_pt_desktop_20260907_r1`。它从冻结的旧 Phase A 150k **仅继承权重**，新 reward、新 optimizer / scaler / scheduler / data cursor、新验证基线，明确非 exact resume。训练源码冻结在 `logs/sl_rl_fixes/20260905_desktop_r1/source`，Git `9a4de1f5b01671da29e7b8b905e46f2f2a79c764`，317 项文件摘要与 config/anchor 摘要均通过校验。

新 run 于 11:36 完成 step 0 baseline：3186 游戏、64676 状态，p0 MSE **2.628318**、all_players MSE **2.635991**。随后已保存 **1000 step** 的更新 checkpoint：824 个模型 tensor 全部有限、824 组 optimizer state 已建立，scheduler=1000，data progress 保留，adaptive best 仍为 0。见 [首次更新 checkpoint 核验](../../logs/oracle_critic_formal/sl_rl_repair_formal_pt_desktop_20260907_r1/first_updated_checkpoint_verified.json)。该 baseline 是新目标起点，不能解释为相对旧目标的模型收益。实时进度见 [新运行状态](../../logs/oracle_critic_formal/sl_rl_repair_formal_pt_desktop_20260907_r1/apex_supervisor_status.json)。保留 Apex 暂停，验证中途暂停不会使用不完整指标。

笔记本原 Phase C 于 9 月 6 日 11:10 正常完成 step 300000、optimizer_steps 299894；没有重新启动旧训练。新代码只通过 Git 进入 `MahjongAI_sl_rl_repair_20260907`，评测提交为 `6d03037`，旧 runner 的修改和产物原样保留。其 native 实际 SHA-256 为 `a55a81337bfa28c943394687e3687545fc5b77393ce53ec342e30abb1ad582cf`。

新正式目录为该 checkout 的 `logs/formal_confirmation_20260907_laptop_r1`。shortlist 是 S70、Phase C 100k、Phase C 50k。**Phase C 50k 与旧全局 finalist Phase B adaptive best 的 actor 哈希相同**（`b15dd238de18e95ef367e751e9e20876306bfb1dd9b01f5ff06ee5026e08af67`），二者 checkpoint 文件不同但未形成第四个不同策略；没有遗漏全局 finalist，也不重复分配它的预算。

正式流水线以隐藏 supervisor 运行：每臂 A/A 8 局，四臂各 16k 筛选，再为唯一 finalist 与 reference 各做独立 64k 确认。每个完整 chunk 可续跑，Apex 时暂停；不以未完成片段给出最终结果，不自动替换 canonical。

## 5. 数据边界和剩余实验

[跨阶段 ledger](../../logs/sl_rl_fixes/20260905_desktop_r1/cross_stage_split_ledger.json) 仅检查索引：当前 SL train pool 与 Oracle dev 有 **15878** 个原始游戏交集，与 Oracle test 有 **11921** 个交集。**这是可获取的当前 pool，不是已证实的 S70 历史实际训练集。** S70 的完整训练/选择祖先索引尚缺，因此既不能断言历史直接泄漏，也不能把这些 test 叫作全流水线独立。确认强度使用新模拟 seeds；sealed test 内容保持封存。

尚需实际完成的科学验证：

1. 三个固定 SL actor 的正式筛选及独立确认；到上限仍跨零则保留 canonical，不能强选 winner。
2. 在最终 actor 的独立模拟分布上，以真实 return 确认 critic 的 p0/all_players、nonzero/tails、校准和 true/zero/shuffled；预声明三个补全种子，不放松 guard 以选出模型。
3. 从同一 actor 与完整状态比较近 on-policy PPO reference 和合格 Oracle 分支，至少三个训练种子；按 500→1500→3000→20k→40k 门控。两个准备配置有隔离的输出/replay/端口，尚未启动在线长训。
4. 同一 A best 分出的保持 A 分布与转 B/C 的匹配实验、固定状态的完整策略 drift 评估、架构及 replay 增益仍是研究问题；本次不把未完成研究写成已证实收益。

代码和准备工具使这些检查可执行；它们的模型结论仍由原始结果决定。主树用户已有修改未被归入本次成果，具体差异与后续验证状态记录在 [实施 journal](../../logs/sl_rl_fixes/20260905_desktop_r1/implementation_status.json)。
