# Oracle critic 当前状态

> 核验：2026-09-08 · 两臂均正常完成40k及保留候选的配对比较；未产生可晋级候选，现无Oracle训练进程。

保留独立 Oracle critic、`all_players` 和真实 `score_rank` return。输入复现、native fold 标签时钟和硬护栏已修复；p0 与 actor 接入资格仍需独立证据。详见 [实施报告](../research/sl-rl-fixes-2026-09-07.md)。

## 当前运行

本轮完成根：`logs/oracle_critic_formal/sl_rl_repair_sl_init_matched_desktop_20260908_resource_r2`。SL初始化臂于12:27:49正常退出0，完整40000步，四次gate因`exact_zero_loss`未决，best_step=0。它的保留候选相对warm no-update的p0 MSE差+1.191281、95% CI [1.129074,1.253489]，不能替代reference。完整状态与协议审查见[完成报告](../research/oracle-matched-completion-2026-09-08.md)；该完成根不自动恢复。

原 `sl_rl_repair_sl_init_matched_desktop_20260907_r1` 在完整31144步移交；第一次资源续跑 `sl_rl_repair_sl_init_matched_desktop_20260908_resource_r1` 又实际推进到33069。两个父根保留源码、配置与checkpoint作为来源证据，不再启动。当前根继承824组optimizer、scheduler/scaler和数据游标，2472个状态张量有限；文件组安全游标不承诺与不中断运行逐bit同轨。见[恢复核验](../../logs/oracle_critic_formal/sl_rl_repair_sl_init_matched_desktop_20260908_resource_r2/resource_resume_preflight.json)及[资源实测报告](../research/rl-resource-tuning-2026-09-08.md)。

| 项目 | 有效配置或产物 |
| --- | --- |
| 配置 / 恢复入口 | `critic_config.toml`、`runtime_manifest.json`、`apex_supervisor_spec.json` |
| 初始化与结构 | 已验证 Long-ABC s70 SL policy（内部 step 390000）迁移两塔，hand-aligned Oracle 首层，新建 value head；`dual_tower`、GN、residual MLP fusion 1024、value hidden 256 |
| 标签 | `all_players`、`score_rank_mc`；`env.pts=[2,1,0,-3]`、gamma 1，与正式 pt 效用一致 |
| 优化 | ScheduleFree AdamW，lr 0.0002、weight decay 0、warmup 2000；评测导出使用 optimizer.eval |
| 输入 / 选择 | 完整决策时钟先算标签再取 fold；验证补全 seed 20260905；step 0 建立 no-update baseline，每 10k gate，全部六项 guard 同时通过 |
| 上限 / 检查点 | 最多 40k、连续 4 次未决停止；`critic/checkpoints/latest.pth`、`adaptive_best.pth`，以内部字段为准 |
| 资源 | batch640、workers2、file batch2、prefetch2、Rayon4、Torch1；`windows_high_qos=true`，CUDA allocator上限0.72；验证保留原批布局，`eval_prefilter_games=false` |

9月7日首次启动是 **SL 权重迁移的新实验**：当时optimizer、scaler、scheduler、data cursor与验证基线重新建立。此次资源续跑继承该臂已经训练的完整状态。相对旧权重臂，目标、数据、seed、优化器、验证与停止规则匹配；另有上述显式资源参数差异，比较的是相同增量预算上限，不是相同历史总训练量。`strict_init_checkpoint=false` 仅用于最初已验证的SL encoder迁移入口，不是来源指纹豁免；两塔各407个参数键完整迁移，最初未加载旧critic或旧value head。

旧权重对照根 `sl_rl_repair_formal_pt_desktop_20260907_r1` 已于 19:30 正常完成 **40000** 步：四次 gate 未决，最终 `inconclusive`，保留 step 0 no-update。最终 p0/all_players MSE 为 **2.353649 / 2.367076**，但 `abs_ge_4` 相对基线恶化 **0.651025**，95% CI **[0.587865, 0.714184]**，不能据 primary 改善选优。完整状态和张量核验见[对照审计](../../logs/oracle_critic_formal/sl_rl_repair_sl_init_matched_desktop_20260907_r1/warm_control_final_audit.json)。该完成根仅作对照，不自动续训。

更早的 broad-to-recent run 已于 9 月 7 日安全退役，最后完整状态为 Phase C **170446**；证据在旧 run 的 `retired_by_sl_rl_audit_20260907.json`。旧 A/B/C 使用阶段步数及 selected best 初始化，不能据阶段名推断累计更新。旧配置和 checkpoint 全部保留，禁止按旧启动命令误恢复旧目标。

运行源码为当前根内的 `source`，沿用 Git `9a4de1f5b01671da29e7b8b905e46f2f2a79c764` 对应修复 runtime。相对原317文件，当前318文件增加进程资源helper，修改Oracle资源入口及runner的Rayon环境传递；旧runtime保持完整。原始SL迁移证据仍见[初始化预检](../../logs/oracle_critic_formal/sl_rl_repair_sl_init_matched_desktop_20260907_r1/initialization_preflight.json)。数据split、目标、验证输入和原step0基线继承不变，实时进度读取当前根日志和checkpoint。当前加载与后台身份见[启动核验](../../logs/oracle_critic_formal/sl_rl_repair_sl_init_matched_desktop_20260908_resource_r2/resource_startup_verified.json)。

运行资源见 [机器页](machine-benchmarks.md)。不要把基础 `mortal/config.toml` 的 loader 默认覆盖到已启动的 case；状态进度从产物读取，不在文档逐次追加 step。

本轮由 [独立启动器](../agent/workflows.md#独立后台启动) 托管，现已正常完成：Explorer 桌面启动隐藏 PowerShell 7 supervisor，宿主及训练子进程不属于 Windows Job，保留 Apex 暂停/恢复。迁移证据见实施报告；实时 PID 以新 run 的 `apex_supervisor_status.json` 和真实进程为准。该入口不提供注销/重启后的自动恢复。

## 历史反例与证据边界

9月5日的随机补全变化、native折扣时钟压缩及guard不能否决best等反例属于旧版本，完整证据保留在下述审计。修复版本真实测试覆盖601状态、16/64 folds、三个gamma，完整与抽样标签最大差为0，固定补全A/A完全一致；独立p0资格仍未完成。

详见 [独立审计](../research/sl-rl-audit-2026-09-05.md) 及其链接的原始 JSON。此前 p0 输出权重选择和旧 C/D/E 记录属于历史证据，不能替代本轮新输入上的资格验证。

## 下一步与通过条件

1. 两臂40k及约定的保留候选比较已完成，当前保留warm no-update作为reference。继续训练或关键对照需要先预声明科学问题、guard与新增预算；本轮未决不代表完整SL起点训练无效。
2. 32 游戏的旧 150k / 当时 latest 1419948 复验已完成，p0 差 +0.00549、CI 跨零。最终 SL actor 冻结后，在独立模拟分布上重新检查 all_players、p0、尾部、校准和 Oracle 输入依赖。
3. 用预先固定的多个隐藏补全版本检查稳健性。人类牌谱未知牌山是随机补全，不能称为全量真实 Oracle；不能直接开启 `trust_seed=true`。原生模拟日志也需先核对引擎版本和 seed 重建语义。
4. 最终候选仅按预声明 selection / 历史回归 guard 决策，人类 sealed test 保持关闭，actor-replay sid0/sid1 不复筛。独立离线资格成立后，再进入受控 actor 对照，验证 `value / GAE` 和真实牌力。

用户已批准正式 pt 对齐及每臂最多 40k 的关键匹配补跑；扩大网络、无差别重跑历史搜索或直接转入长期 RL 仍缺证据。跨阶段索引有重叠且 S70 历史祖先不全，不能把旧 sealed test 称为全流水线独立确认集。训练期补全是否最优仍待实验，工程验证不等于强度提高。

源码入口：[pretrain_oracle_critic.py](../../mortal/online/pretrain_oracle_critic.py)；操作见 [运行流程](../agent/workflows.md#oracle-critic)，源码切换见 [活跃训练边界](../agent/code-health.md#活跃训练边界)。旧资源试验已移入 [历史记录](../archive/status/oracle-critic-resource-benchmarks-before-doc-refactor-2026-09-05.md)。
