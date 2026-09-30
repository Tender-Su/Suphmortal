# 在线 RL 当前状态

> 核验：2026-09-08 · 依据：审计修复、资源相关89项回归及多轮真实三角色容量测试；诊断副本已停止，尚未启动正式PPO训练。

历史 RL 候选尚未证明稳定强于 SL。保留可复验的实现与实验记录，下一轮先明确奖励目标、建立可靠的 Oracle 资格，再用独立正式评测判断增益。

## 已有证据

| 对象 | 核验结果 | 当前用途 |
| --- | --- | --- |
| 历史 C/D/E | 33 个完整评测含 canonical，共 66,000 局；32 个 RL 候选的名义配对 95% CI 下界均未为正 | 探索记录，不能宣布正式增益 |
| E 9k / E 10k policy-stop | actor 与 policy 权重逐张量完全相同 | 两次对局点估计差不能解释为策略学习变化 |
| 独立 Oracle critic | 已切换固定输入和新奖励校准，p0 资格仍待确认 | 见 [Oracle 状态](oracle-critic-mainline.md) |
| replay / actor 契约修复 | PPO/V-trace actor 分离；拒绝旧 hybrid 静默续跑；提前过滤未知/过旧/未来行为版本，增加 KL / clip fraction 门槛 | 解析梯度与加载测试通过，未证明对局收益 |

证据、名义区间的多重比较限制和已实施代码改动见 [独立审计](../research/sl-rl-audit-2026-09-05.md)。旧路线的完整结果保留在 [历史 RL 快照](../archive/status/online-rl-mainline-before-doc-refactor-2026-09-05.md)。

9月8日完成资源诊断：GAE物理inference block可显式配置，默认2048；DataLoader在下一块构建前释放上一块dataset。已测配置为batch192、每轮128局、inference block512、Rayon4、Torch1和独立后台HighQoS；logical chunk50、完整轨迹及PPO/GAE目标保持不变。最后后台测试256个计数步中，AdamW实际执行255次，1次AMP跳过；按真实更新计算0.621次/秒，最低可用RAM4.42GiB，356条行为版本检查全部通过、版本差最大0，KL/clip无拒绝。详情与资源边界见[资源报告](../research/rl-resource-tuning-2026-09-08.md)。

现有在线主循环的step及optimizer_steps会在AMP跳步时继续计数，此次已由scaler和AdamW内部时钟核实。资源harness现按实际AdamW调用停止并统计，旧结果保持原始255计数；本次未改变正式RL的scheduler/版本推进定义。下一轮正式训练前需统一成功更新时钟，不能把日志计数直接当作全部成功更新。容量测试所用critic尚未完成资格确认。

## 奖励与评价口径

旧 run 使用 `env.pts=[6,4,2,0]`，中心化为 `[3,1,-1,-3]`；正式 `avg_pt` 使用 `[90,45,0,-135]`。新批准分支和配置模板采用 `[2,1,0,-3]`、gamma 1，与正式 pt 效用一致。旧新效用不能靠常数平移或正比例缩放互换。

新目标已用于独立 Oracle 校准，不覆盖旧 run。PPO reference / Oracle 配置有独立 checkpoint、replay、日志和端口，首段上限 500 steps；Oracle 配置要求已确认的 critic 文件。它们尚未启动，不能把配置准备写成已有 RL 收益。详见 [实施报告](../research/sl-rl-fixes-2026-09-07.md)。

最小参照使用近 on-policy PPO，默认关闭 V-trace 与 dual clip。KL 0.02、clip fraction 0.5 是保守起始门槛，尚未证明最优。显式 V-trace 分支仅用一次已校正 actor advantage，不再叠加 PPO ratio；恢复签名绑定 actor 目标版本、奖励和 critic 架构。

## 下一轮最小协议

1. 明确主目标，冻结可执行配置、SL 起点、对手、源码/扩展摘要与随机 seeds。保留 no-update SL 对照。
2. 先完成 [Oracle 资格](oracle-critic-mainline.md#下一步与通过条件)，再在同一恢复状态和数据条件下比较 visible value 与 Oracle critic；checkpoint 必须包含 optimizer / scaler / scheduler / data cursor。
3. 先验证 replay 新旧策略对应、reward 语义、`value / GAE`、IS / V-trace 和 train / eval 模式，再扩大训练。步数不自动赋予晋级资格。
4. 正式比较固定对手、四座轮换和独立 seeds，按 seed 组做配对统计；候选筛选结束后做独立确认。`test_play=200/400` 与短 ranked 曲线只作诊断。
5. 报告主指标、全部 guardrail、样本数、不确定性和失败/超时处理。不能只报最高点、单个 seed 或一次赢家。

## 运行契约

- `server` 分发参数并管理 replay；`trainer` 执行更新并发布参数；`client` 自博弈并回传 replay。壳进程存在不代表三者有进展。
- 初始化优先级与 checkpoint 语义见 [仓库规则](../../AGENTS.md#训练与评测)。`1v3` challenger 使用 `[1v3.challenger].state_file`。
- 训练默认关闭 `search`；推理增强独立 A/B。对手池用 `--opponent-pool-preset` 明确选择，不手改 `baseline.train` 冒充相同实验。
- 排因配置与正式训练配置分开记录；旧的 `ms_rl*` profile 名字只是预设，不能代替当前批准的实验协议。

命令见 [运行流程](../agent/workflows.md#在线-rl)，历史文献和方案见 [研究索引](../research/README.md)。
