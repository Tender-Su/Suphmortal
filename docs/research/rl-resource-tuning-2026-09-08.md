# 台式机 Oracle critic / 在线 RL 资源调优

> 核验：2026-09-08 · 范围：i5-13600KF、RTX 5070 Ti 16 GB、32 GB RAM；资源诊断，不构成模型强度或 Oracle 资格结论。资源测试已结束，正式Oracle已恢复，在线诊断副本全部停止。

## 实验边界

用户要求先优化正在运行的 Oracle critic，再测试在线 RL 的 server、trainer、client 整条流水线。所有测试使用独立源码和产物，保留 RiichiLab CPU 客户端，Apex 出现时停止资源测试。未设置 CPU affinity，未操作笔记本。

原 SL 初始化匹配臂在完整 checkpoint **31144** 步暂停；模型、824 组 optimizer state、scheduler、scaler 和数据进度均已核验。基准分支从该固定副本起步，测试产生的训练权重不回灌正式臂。正式臂仍受原 **40k** 上限、原目标、固定 monitor 和全部 guardrail 约束；恢复游标为文件组安全边界，不承诺与一次不中断运行逐 bit 同轨。

所有证据在 [资源产物目录](../../logs/oracle_resource_tuning/20260908_desktop_r1/)，包括脚本/配置/source/native/输入 checkpoint 摘要、过程采样、失败原因及每步时间。原 runtime 未热修改；资源续跑使用独立冻结目录。

## Oracle 训练：采用 Rayon 4、prefetch 2、Torch 1

保持 logical batch **640**、workers **2**、file batch **2**、native fold、样本顺序、学习率、optimizer、目标和 gate 不变。原默认训练 worker 各使用 Rayon 2。实测系统暴露 14 个逻辑 CPU，进程 affinity 覆盖全部可用 CPU。

以下均从同一个 31144 checkpoint 执行 1000 次真实更新，丢弃前 50 次计时，包含正常的 step 32000 保存。不是用不同时段的生产日志拼接倍率。

| Rayon / prefetch / Torch threads | 稳态更新/秒 | 状态/秒 | 等待数据秒数 | RAM 采样峰值 | 最低可用 RAM |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2 / 1 / 14 | 2.9637 | 1896.8 | 146.26 | 71.1% | 9.19 GiB |
| **4 / 2 / 1** | **4.3730** | **2798.7** | **17.83** | **72.4%** | **8.76 GiB** |
| 4 / 4 / 1 | 4.3948 | 2812.7 | 16.96 | 73.8% | 8.33 GiB |

同一启动环境下，采用档相对基线快 **47.55%**。prefetch 4 只快约 0.5%，但进程树多占约 0.6 GiB；短测本身已有约 6% 时间波动，因此不为这点未确认收益增加预取。Rayon 6/8 的 200-step 探针也没有超过 Rayon 4。提高父进程 Torch 线程数不能解决 native 数据供给不足。

采用档训练 GPU 利用率中位数由约 33% 提升到 65%，全局显存采样峰值约 10.64 GiB；CUDA 实际 allocated / reserved 峰值分别为约 7.67 / 8.31 GiB。续跑采用已测试的 allocator 上限 0.72，为其他桌面 GPU 使用留出余量。

输入证明包括 12800 个状态的全部字段顺序哈希相同，以及各更新探针前四批的完整输入哈希相同；最终数据进度及各时钟一致。200-step A/A 已出现 CUDA 浮点差异；1000-step 权重比较也记录了最大约 0.0002985 的差异，另一个资源档与基线逐张量相同。因此只声明数据和计算定义一致，不声明浮点训练轨迹逐 bit 一致。见 [长窗口结果](../../logs/oracle_resource_tuning/20260908_desktop_r1/long_update_equivalence.json) 和 [A/A 记录](../../logs/oracle_resource_tuning/20260908_desktop_r1/cuda_rounding_aa.json)。

## 后台托管：显式 HighQoS 才能保持吞吐

第一次部署资源续跑后，正式日志仅约 **2.43 step/s**，不能直接宣称生产获得 4.37。正式状态先完整保存到 **33069**；后续入口探针均从它的独立副本起步，正式科学进度没有回退。

直接调用、加入 production prelude、实际 `python -m`、隐藏控制台四种独立探针均约 4.4 step/s；同一脚本改由既有 Explorer 独立托管后可复现约 2.3。真实进程 priority=Normal、affinity 覆盖全部 14 CPU；训练/loader 输入哈希一致。对同一个后台诊断进程树做默认→HighQoS→默认对照，避开切换边界后分别为 **2.311 / 4.153 / 2.303 step/s**。这是同进程连续窗口，支持调度响应的归因，不把三段当作完全相同的实现样本。完整时间和 API 状态见 [QoS 对照](../../logs/oracle_resource_tuning/20260908_desktop_r1/windows_qos_aba.json)。

Windows 可在未显式指定 QoS 时按启发式决定后台进程的服务等级；本次只关闭当前进程的 execution-speed throttling，保留其他电源控制、Normal priority、全 CPU affinity 和系统电源计划。[Microsoft API 文档](https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-setprocessinformation)

新增 [进程资源入口](../../mortal/core/process_resources.py)，默认关闭。Oracle 使用 `oracle_critic_pretrain.windows_high_qos=true`，父训练进程和两个 DataLoader worker 各自设置；在线 role runner 提供显式 `--windows-high-qos`。没有把高优先级或绑核设为默认。在线角色启动前的 helper 日志也须保持 root logging 配置不变，相关启动回归已覆盖。

新冻结 Oracle runtime 在相同 Explorer 托管方式下完成 **500 次实际更新**，丢弃前 50 次后 **4.114 step/s**，原四批输入哈希相同。GPU 利用率中位数 **63%**，RAM 峰值 **71.8%**、最低可用 **8.96 GiB**；CUDA allocated/reserved 仍 **7.67/8.31 GiB**。并行的 100 秒系统探针 p95 **0.81ms**、最大约 16.76ms；此值仍是调度代理，不是实测 UI 帧延迟。见 [后台验收](../../logs/oracle_resource_tuning/20260908_desktop_r1/final_qos_smoke/result.json)。

## 完整 monitor 验证

三个方案均覆盖原定 **3186 游戏、64676 状态**，五个字段的完整顺序哈希及标签完全一致；未读取 sealed test payload。

| 方法 | 用时 | 相对首行 | 输入文件 |
| --- | ---: | ---: | ---: |
| Torch 14、Rayon 2、原批布局 | 394.08 秒 | 1.00× | 16194 |
| Torch 1、Rayon 4、原批布局 | 360.11 秒 | 1.09× | 16194 |
| Torch 1、Rayon 4、先过滤 monitor 文件 | 57.51 秒 | 6.85× | 3186 |

计时包含相同的输入哈希和输出采集。原批布局仍主要耗在未入选文件的特征编码。提前过滤的主指标变化小于 0.000001，六项 guard 指标最大变化约 0.0000674；同批布局但不同线程配置也出现 cuDNN/TF32 单预测值差异。因此当前匹配续跑保留原批布局，`eval_prefilter_games=false`；提前过滤保留为未来固定验证协议的显式选项。bounded-batch 与 shuffled-input 诊断自动保留原批布局。

验证过程中捕获的 CUDA allocated 峰值约 **9.52 GiB**，高于一些瞬时 `nvidia-smi` 样本。不能用低频轮询的显存最大值冒充所有瞬态峰值。完整输入、预测和六项 guard 对拍见 [验证等价性记录](../../logs/oracle_resource_tuning/20260908_desktop_r1/full_validation_equivalence.json)。

## 在线 RL：内存生命周期比增加并行数更关键

基准固定 S70 actor、同一 Oracle checkpoint、GN/dual tower、`all_players`、`score_rank`、formal pt `[2,1,0,-3]`、PPO/GAE、batch 192、chunk 50、零 replay reuse、KL 0.02、clip fraction 0.5 和行为版本差上限 1。选择近 on-policy 顺序协议，学习率计划不因诊断停止步数而缩短。角色使用 loopback、独立 replay/output，并有 RAM、VRAM、时间和 Apex 护栏。

旧 GAE 路径忽略 dataset 的 worker 数，且物理 inference block 固定为 2048。现已增加显式 `online.gae_inference_batch_size`，默认仍为 2048，不改变完整轨迹或 logical chunk。64 局条件下 512 与 2048 各完成 96 次实际 optimizer 更新，均约 179.5 秒、0.535 次/秒；512 的 CUDA reserved 峰值少约 0.75 GiB，故选择 512。

已保留的内存修复位于 [train_online.py](../../mortal/online/train_online.py)：旧 DataLoader 在下一块构建期间仍持有上一块 dataset，现在在迭代结束立即释放 loader。

另外测试过将 `tensor[perm]` 的整块复制改为按同一个 `randperm` 索引取样。该方案通过了字段、部分末批、样本顺序和 RNG 状态对拍，但持续资源测试未证明足够的内存余量，已从主树撤回。实验代码和失败记录保留在产物中的 `online_source_v3`，不属于当前采用版本。

第一项修复后，128 局从 RAM 护栏停止变为完成 96 次实际更新，154.59 秒、0.621 次/秒；零 KL 拒绝，已读取行为版本差为 0。修复前后同 seed 的 **128/128** 份原生对局、动作和策略概率一致，仅排除计时字段 `meta.eval_time_ns`。见 [对局等价性记录](../../logs/oracle_resource_tuning/20260908_desktop_r1/online_loader_lifetime_replay_equivalence.json)。

初始容量试验使用保守的 **5 GiB** 可用 RAM 门槛。256 局触发该护栏；64 局长测在第 5 轮、203 次实际更新后降到 **4.90 GiB** 并停止。此时没有 OOM 报错，停止的是预留门槛，不能据此声称这些配置必然 OOM。14/14 线程基线在第二轮仅完成 53 个训练 batch，也不与成功配置计算虚假的完整吞吐倍率。

随后单独验证 **4 GiB** 预留档，没有继续降低门槛。采用原 shuffle 加 loader 释放修复、Rayon 4、Torch 1、inference block 512、**128 局/轮（32 个 seed 组）**，完成 **256 次实际 optimizer 更新**及多轮参数交换：**378.14 秒、0.677 次/秒**。356 条行为检查全部通过、版本差最大 0，256 个 batch 无 KL/clip 拒绝，最大 KL 0.000607。RAM 峰值 86.9%、最低可用 **4.16 GiB**，全局显存采样峰值约 **9.08 GiB**，调度延迟 p95 约 **0.62ms**。

这是已测的高吞吐配置，RAM 余量紧，须保留资源监控与停止条件。64 局用于同时操作其他程序时留更多余量，但它的 5 GiB 长测并未完整通过。256 局没有在新的 4 GiB 档完成长测，不把它排除为绝对不可行，也不推荐一个未完成验证的更大档。不同窗口的 96/256 次更新数据不用于宣传精确的端到端提升百分比。

最终还将同一资源档放入真实Explorer独立托管，三个角色均核验为 `in_job=false`、Normal priority、全14 CPU affinity和显式HighQoS。三轮参数交换共 **410.90秒**，训练计数为256，但AdamW实际执行 **255次**，有效吞吐 **0.621次/秒**；RAM峰值86.1%、最低可用 **4.42GiB**，全局显存采样峰值 **9.67GiB**，调度p95 **1.63ms**。356条行为检查全部通过、版本差最大0，256个batch无KL/clip拒绝，最大KL0.002235。不同托管/时间窗口不计算精确性能倍率。

这里出现 **1次AMP跳过**：初始scaler65536降至32768，checkpoint的AdamW内部步数为255。现有在线主循环的`steps`和`optimizer_steps`仍按尝试次数推进；此前按日志声称256次实际更新已更正。最终资源harness改为按真实AdamW调用数停止并以非零退出码报告未完成，历史v5产物仍如实保留255次；没有为凑满数字重标记历史结果或缩小初始scaler。正式RL开始前还须统一AMP成功更新、scheduler及版本时钟，此次资源调优未改变这些科学定义。

最终参数见[资源配置记录](../../logs/oracle_resource_tuning/20260908_desktop_r1/selected_resources.json)，原256次实更长测见[早期128局长测](../../logs/oracle_resource_tuning/20260908_desktop_r1/online_v2_r4_t1_g128_i512_reserve4/qualification_summary.json)，最终后台结果见[三角色验收](../../logs/oracle_resource_tuning/20260908_desktop_r1/online_v5_detached_g128_i512_reserve4/qualification_summary.json)。逻辑batch为192、GAE chunk上限仍为50；trainer/client allocator上限分别为0.68/0.17。科学初始化、LR、奖励、资格门槛不能随资源参数一起照搬为已批准的正式RL训练。

## 系统响应与证据限制

资源采样由独立进程/低频 psutil 与 `nvidia-smi` 完成，不使用 WMI Trace。在线 RAM 初筛门槛 5 GiB，后续单独验证 4 GiB 档；每个 case 的 identity/result 记录对应阈值，不能混写为同一安全条件。CUDA 使用 allocator 上限，周期采样仍可能漏掉瞬态。

Oracle 240 秒调度探针的 20ms sleep 额外延迟 p95 为约 2.07ms；修复后 128 局在线探针为约 0.62ms。这是调度代理指标，不是鼠标、键盘或游戏帧延迟。最初在线探针使用 Windows Event.wait，约 12ms 的 p95 含计时粒度影响，不能与后续 sleep 数字作性能倍率比较。

这些结果只支持当前硬件、共存负载、模型和协议下已测配置的选择，不证明全局最优、长期无泄漏或牌力提高。更换 GPU、同时运行其他训练或改变逻辑 batch/模型后，应重新做容量验收。

## 验证与恢复

采用版本的相关回归为 **89 项**：`test_oracle_resource_tuning`、`test_pretrain_oracle_critic`、`test_online_audit_regressions`、`test_sl_rl_fix_contracts`、`test_online_role_runner`。已撤回索引 shuffle 的实验版本曾通过 86 项，不借用该计数说明当前源码。loader 释放修复还有真实 server/trainer/client 实测，不用单元测试数量替代流水线证据。

原 31144 checkpoint 和 validation manifest 的副本逐字节一致；第一次资源续跑仅变更资源参数及隔离路径。最终 runtime 为 318 文件：相对原 317 文件增加进程资源 helper，修改 Oracle 资源入口及传递 Rayon 环境的 runner；原目录保持完整。新的 33069 科学状态含 824 组 optimizer、scheduler/scaler 和数据游标，后续继续保留原 40k 总上限。

第一次资源runtime的恢复副本完成200次真实更新，前四个输入batch与基线完全一致、各时钟继续推进；再次完整验证3186游戏/64676状态，并保持约0.183GiB optimizer状态常驻GPU，CUDA allocated/reserved峰值约 **11.40/11.45GiB**、RAM峰值63%。这覆盖了0.72 allocator上限下的验证峰值，不能用之前0.60上限的9.52GiB数字替代。随后HighQoS版本只增加进程调度设置，模型和样本计算不变，另完成500次后台更新和真实启动核验。

2026-09-08 11:03第一次资源续跑从原31144步开始，实际推进后完整保存33069。11:53最终`resource_r2`从 **33069** 恢复；11:56核验已到33600，正式日志稳态 **4.03 step/s**。训练进程及两个worker的HighQoS均生效，恢复源未采用任何probe权重，原两个父根已标记退役。见[最终恢复审计](../../logs/oracle_critic_formal/sl_rl_repair_sl_init_matched_desktop_20260908_resource_r2/resource_resume_preflight.json)和[实际启动/吞吐](../../logs/oracle_critic_formal/sl_rl_repair_sl_init_matched_desktop_20260908_resource_r2/resource_startup_verified.json)，当前身份只在[Oracle状态](../status/oracle-critic-mainline.md)维护。原小时监控已切到新根，保留Apex暂停及原40k科学预算。

最终可复用基准脚本保存在[脚本快照](../../logs/oracle_resource_tuning/20260908_desktop_r1/benchmark_code_final_v2/manifest.json)。各历史case保留当时的脚本摘要及冻结source，包含被撤回实验和护栏停止。v4首次后台三角色因早期root日志初始化而未进入训练，v5修复后完成上述验收；修复保持角色原logging设置，已有独立回归。
