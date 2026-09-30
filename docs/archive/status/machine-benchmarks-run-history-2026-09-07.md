> 历史归档 · 2026-09-07。保留原事故、修复与完成流水，不作为当前运行依据；当前配置见 [机器状态](../../status/machine-benchmarks.md)。

# 机器与资源

> 核验：2026-09-05 · 区分有效配置、代码内建默认和历史 benchmark。参数表不是新 run 的自动授权或吞吐保证。

## 台式机

本机为 i5-13600KF + RTX 5070 Ti。解释器和环境设置见 [运行流程](../../agent/workflows.md#环境与构建)。

| 来源 / 用途 | batch | train workers / file batch / prefetch | validation |
| --- | --- | --- | --- |
| 本地基础配置的 SL | 1024 | 4 / 10 / 3 | workers 0，file batch 8，prefetch 5 |
| 本地基础配置的 Oracle 预训 | 512 | 4 / 10 / 3 | file batch 8，prefetch 5；其余按生成配置解析 |
| 当前 Oracle case | 640 | 2 / 6 / 2 | workers 0，file batch 8，prefetch 5 |

当前 Oracle run 的身份和配置位置只在 [Oracle 状态](../../status/oracle-critic-mainline.md#当前运行) 维护。workers 为 0 时，不要将配置里保存的 prefetch 数字解读成实际多进程预取量。

旧 GRP / SL / Oracle 的吞吐结论均有特定输入规模和运行环境；重新用于新数据或模型前必须复测。历史表格见 [机器快照](machine-benchmarks-before-doc-refactor-2026-09-05.md) 与 [Oracle 资源记录](oracle-critic-resource-benchmarks-before-doc-refactor-2026-09-05.md)。

## 1v3 默认来源

[one_vs_three.py](../../../mortal/eval/one_vs_three.py) 内建：

| GPU | seed_count | shard_count |
| --- | --- | --- |
| RTX 5070 Ti | 1024 | 4 |
| RTX 4060 Laptop GPU | 640 | 3 |

解析优先级是环境变量 → machine profile → GPU 配置 → 内建 GPU 默认 → 通用 `[1v3]`。具体生效值必须看运行输出。

`seed_count` 是 seed 组数量，四座轮换产生四倍对局；shard 数量负责并行。正式样本量由预声明协议决定，不能把机器默认当成统一的评测预算。覆盖示例见 [1v3 流程](../../agent/workflows.md#1v3-与配对评测)。

## 笔记本与资源边界

已记录硬件为 i9-13900HX + RTX 4060 Laptop 8 GB + 32 GB RAM。2026-03-31 的 SL loader `4 / 10 / 4` 与 validation `7 / 5` 只是历史操作点，不能照搬到当前 Long-ABC。

2026-09-05 14:30 已通过原主机密钥重新确认笔记本 SSH 身份；先前对旧地址的 Meta 隧道 TCP 探针不足以证明远端在线。Long-ABC supervisor 在同日 03:58 超过重试上限退出；最后已验证的恢复点为 step 151890，14:36 检查时 latest 文件未更新。本地 23:43 / 03:34 快照中的低可用 RAM 不再代表当前状态。

14:39 查明主要内存压力来自前一天 20:34 遗留的监控 PowerShell PID 11548：结束前私有提交约 80.5 GiB、驻留约 13.0 GiB。命令对带 PSDrive / PSProvider 附加属性的 Get-Content 日志对象执行 ConvertTo-Json -Depth 10；有界浅层复现确认了额外对象展开。14:42 按创建时间和命令哈希核验后仅结束该监控进程，14:48 可用 RAM 从约 223 MiB 恢复至 21.20 GiB，系统提交从约 98.43 GiB 降至 16.37 GiB。非分页池从约 7.24 GiB 降至 5.70 GiB，尚未证实另有驱动泄漏。证据：[原始占用与命令](../../../logs/sl_monitor/20260905_143922_memory_incident.json)、[处理后复测](../../../logs/sl_monitor/20260905_144839_memory_resolution.json)。

本次系统内存归因不能直接用来解释全部 CUDA OOM。恢复前仍须核对真实 checkpoint step、RAM / VRAM 峰值与并存进程；先排除监控自身的异常占用，再根据训练短测决定是否降低 workers / file batch / prefetch。不能仅因旧 benchmark 快就原样反复重启。恢复和机器操作见 [远程流程](../../agent/remote-ops.md)。

15:16 按用户要求重新排查，未发现新的异常残留：可用 RAM 23.82 GiB、系统提交 11.82 GiB、非分页池 2.18 GiB，10 秒整机 CPU 约 1.4%，GPU 空闲；保留正常常驻程序。15:18 在核验并独立备份 step 151890 checkpoint 后，以冻结源码和原训练配置单次续跑，supervisor 首次异常即停止。15:26 日志确认 Phase C 到 step 152890，loss 0.4778、LR 1e-5，Phase B 已跳过。15:27-15:28 负载样本可用 RAM 4.86-5.18 GiB、系统提交约 50.3-50.8 GiB、显存 7624/8188 MiB；启动 SSH 已关闭而训练继续运行。显存余量较窄，首段通过不能排除后续峰值或长期泄漏。证据：[占用复查](../../../logs/sl_monitor/20260905_151604_cleanup_audit.json)、[续跑核验](../../../logs/sl_monitor/20260905_152818_resume_verified.json)。

15:32 复测可用 RAM 5.89 GiB、系统提交 51.45 GiB、非分页池 2.23 GiB；trainer 私有提交稳定在约 13909.6 MiB，没有新错误或超大监控 PowerShell。GPU 报告的实际 free 为 333 MiB，不能直接用 total-used 代替 free。该次读取的 latest 尚未到下一次常规保存，独立备份仍对应 step 151890。15:33 日志继续到 step 153890、loss 0.4800、LR 1e-5，最近 1000 steps 约 2.31 step/s；15:35 再次确认 trainer 和 supervisor 存活。详见 [负载复测](../../../logs/sl_monitor/20260905_153244_resume_health.json)。

16:22 完成恢复后的首个 monitor 验证并保存 step 160000：monitor policy 0.450112、action accuracy 0.824618、rank accuracy 0.474528；这不是新的 full-recent 或配对门控，控制器仍在 150k、futile 1/2、phase-best 50k、LR 1e-5。16:50 训练已到 163890，约 2.31 step/s，无新致命错误。16:53 / 16:57 的 trainer 私有提交均为 19432.58 MiB，高于首轮验证前的约 13910 MiB；可用 RAM 3.30-3.77 GiB、实际空闲显存 258 MiB，未发现超大监控进程。该增量值得继续观察，但这些样本不足以认定训练泄漏。160k checkpoint 在 CPU 上完整读取，模型、optimizer、scaler、scheduler 和 plan_id 均核验；它是 epoch 中途保存，冻结版本 `94d942f` 不含 batch cursor / RNG，恢复会从该 epoch 重建 loader，不能声称逐 batch 精确接续。151890 epoch 边界独立副本继续保留；未修改或重启活跃训练。详见 [160k 验证与恢复边界](../../../logs/sl_monitor/20260905_165305_monitor_160k.json)。

19:08 发现新的验证 OOM：18:52 保存 step 180000 后，monitor validation 的 forward 报 CUDA out of memory，清理时再次报错，180k 验证没有完成。内部 `run_training` 将它当作 transient failure，6 秒后以相同配置自动重启 trainer；新 PID 4628 在 18:53 加载 180k，19:08 已到 182000。外层 runner / supervisor 身份未变。更正上述“首次异常即停”的范围：`MaxUnexpectedRestarts=1` 只限制外层 runner 退出，冻结版本内部仍是无上限重试，不能保证 trainer 首次异常就停。180k checkpoint 已 CPU 读取和摘要核验，但仍不含 batch cursor / RNG；恢复后较低的 train loss 不能直接当作泛化改善。当前可用 RAM 约 7.78 GiB、GPU free 794 MiB，不足以反推失败时峰值或判定泄漏。已保留证据，未修改活跃源码或额外重启；受控暂停后修复验证显存路径与内部重试边界需用户决定，源码仅在台式机修改并通过 Git 同步。见 [验证 OOM 与自动恢复](../../../logs/sl_monitor/20260905_190827_validation_oom.json)。

### 9 月 5 日受控修复与恢复

用户授权后，20:19:51 在 step 191178 / optimizer step 191109 受控保存并退出 75；supervisor、runner、trainer 和 loader 均已退出，恢复副本与生产 latest 的 SHA256 均为 `afba203fdc431c6f80858806a86f14034b0b2467375cbb3abe3ce21320555873`。台式机 main 提交 `d7d54c7` 仅包含本次修复及测试，既有未提交重构没有被带入；笔记本既有 handoff / loader 差异通过 Git 保留，原始状态和运行差异另有审计，不能把运行源码描述为纯净的旧 `94d942f`。

修复保留训练 batch、loader 分片、采样和指标语义：验证 logical batch 仍为 1024，但 CPU obs 以 256 样本分批执行 Brain 前向，再整批计算 heads / loss / action / scenario / rank 和 cluster 统计；关闭验证 CUDA 预取，不改训练预取。验证清理保留原异常和暂停码；validation loader 与 trainer 的 transient 重试各最多一次，GPU OOM 不再由内部 trainer 自动重试。supervisor 固定持有子进程 handle 并等待退出码，运行时继续 `MaxUnexpectedRestarts=1`。

主工作树、隔离部署副本、笔记本各通过 94 项相关测试。真实 checkpoint / 数据的 1024 样本 GPU 对拍在两批训练输入和 1236 个 optimizer 张量驻留下，完整与分片 Brain 验证峰值已分配显存分别为 577.06 / 516.88 MiB，峰值 reserved 为 618 / 542 MiB；动作一致率 100%，policy loss 差 0.00003234，前后训练反向的 loss 相同。这是短测结果，不是对事故峰值或长期泄漏的排除。

隔离 trainer 从 191178 完成两次更新和两次各 8 logical batches 的真实 monitor 验证，保存到测试目录；随后预算尾部 fallback full validation 在 120 秒测试上限通过暂停协议退出 75，未完成的全量评测不能算通过。测试产物不进入生产候选。退出后无 Python / loader 残留，可用 RAM 24.62 GiB、GPU used 262 MiB / free 7695 MiB。20:51:50 启动单个修复后 supervisor，20:52:19 确认从未改变的正式 191178 checkpoint 恢复；原生产配置、plan_id、S140 与 canonical 未改。旧 checkpoint 仍缺 batch cursor / RNG，恢复从该 epoch 重建 loader，不能宣称逐 batch 精确续跑。详见 [修复与测试审计](../../../logs/sl_monitor/20260905_205044_validation_repair.json)。

20:59:41 正式训练已推进到 step 192178，loss 0.4495、policy loss 0.4053、LR 1e-5；21:00 可用 RAM 6.40 GiB、系统提交 51.59 GiB、GPU used 7171 MiB / free 786 MiB。启动 SSH 已关闭，仍只有一组 runner / trainer 和四个 loader worker；未见本次恢复的新错误。下一完整配对门控仍为 200k。见 [正式推进核验](../../../logs/sl_monitor/20260905_210006_repaired_resume_progress.json)。

22:04 完成修复后的 200k 正式门控：monitor 1024 batches、full-recent 323 batches、old-regression 167 batches，均无 OOM。full-recent policy loss 从 150k 的 0.446243 升至 0.471786，action accuracy 0.820921、action score -0.218541、scenario score -0.287609、rank accuracy 0.479626；old-regression policy loss 从 0.479804 升至 0.510237。控制器按配对证据耗尽 level 0，于 22:04:02 将 LR 从 1e-5 降为 5e-6，best_step 仍 50000，futile 计数归零，阶段未完成；portfolio.json 仍只有 50k / 100k 两个候选，未晋升 200k。200k latest 已 CPU 读取并核对 optimizer 两组 LR / scheduler tail_lr 均为 5e-6、plan_id 不变，checkpoint_id 为 `aef37713bccf47fdbe2d7b49b5a5d1ff`，SHA256 为 `7b676cc4d464e53a984345aa00782f4b9392a9c11bb21e2cb540e802795c6b16`。

22:05 已恢复训练至 step 200178、loss 0.4465、LR 5e-6，原 trainer / supervisor 未更换。验证采样 GPU used 989 MiB / free 6968 MiB；回到训练后 used 7255 MiB / free 702 MiB，可用 RAM 2.44 GiB、commit 57.57 GiB，内存余量仍偏紧，应继续观察而非据此认定泄漏。近期纯训练吞吐约 2.306 step/s；到下一 250k 门控仅训练部分约 6 小时，另加中间 monitor、完整门控和暂停时间，最终收敛 ETA 未知。详细证据见 [200k 门控审计](../../../logs/sl_monitor/20260905_220803_gate200k.json)。

### 9 月 7 日核验正常完成

11:26-11:30 直连确认 Phase C 已于 9 月 6 日 11:10 在 300k 正常结束：runner exit 0，manifest / checkpoint completed=true。250k 为 futile 1/2，300k 因 LR 5e-6 未改善配对 phase-best 而停止；这是 d7d54c7 之前已有的规则，与关闭无限重试无关。300k checkpoint 已安全 CPU 读取，full-recent / old policy 为 0.456654 / 0.493998，修复后无新 OOM 日志。summary 选择 phase_b_adaptive_best，下一门槛 formal_1v3，sealed test 未打开。当前无 trainer，GPU 空闲、可用 RAM 25.19 GiB；未重启或修改 checkpoint。原 200k 监控基线失效，详见 [完成、资源与退出审计](../../../logs/sl_monitor/20260907_112958_phase_c_completed.json)。

## 新 benchmark 的记录要求

记录机器、源码/扩展 hash、模型、有效配置、输入规模、运行前的并存任务、峰值 RAM/VRAM、吞吐、验证和恢复结果。性能提升必须通过数据与训练语义等价性检查。

不要清理运行中的日志或 checkpoint 来做 benchmark；不要在活跃训练环境中重新安装原生扩展。新结果替换本页的适用结论，详细采样放入带日期的报告或 run 产物。
