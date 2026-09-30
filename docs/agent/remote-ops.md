# 双机与远程流程

> 核验：2026-09-07 · 资源原则按用户要求更新；SSH / Git 操作约定沿用 2026-09-05 核验。

本页记录 SSH、Git 同步与 runner 操作；授权和源码修改边界统一见 [AGENTS.md](../../AGENTS.md#双机与提交)。

## 目标与连接

按任务需要查看 SSH 别名与同步目标，IP 以当前配置为准：

```powershell
ssh -G mahjong-laptop | Select-String '^(hostname|user|port) '
git remote get-url laptop-sync
```

需要核对远端运行状态时，可使用以下只读探针。PowerShell 通过单引号 here-string 传给远端，防止 `$` 在本机展开：

```powershell
@'
$ErrorActionPreference = 'Stop'
$env:COMPUTERNAME
Get-Location
Get-Process python, pythonw -ErrorAction SilentlyContinue | Select-Object Id, Path, StartTime
'@ | ssh mahjong-laptop powershell -NoProfile -Command -
```

启动或恢复时，按目标 run 核对解释器、资源、supervisor 和完整 checkpoint。进程名只供定位，不能据此批量终止 Python。

连接失败时，明确记录“远端未核验”和最后可用快照时间；不要把本地同步日志说成实时进度。

监控本身也要有资源边界。Windows PowerShell 5 的 `Get-Content` 字符串可能携带 `PSDrive` / `PSProvider` 等属性，不能直接嵌入对象后深层 `ConvertTo-Json`。使用 `System.IO.File` 读取有界纯文本，JSON 先解析，只序列化显式挑选的标量、纯字符串和简单数组；深度不超过 4，单次输出不超过 64 KiB。一次性采样设置不超过 60 秒的远端超时，记录 PID、创建时间和命令指纹，并在 `finally` 清理自身采样子进程；本地 SSH 退出不等于远端进程退出。先核实遗留采样是否占用资源，再把故障归因于训练。已发生的案例见 [资源记录](../status/machine-benchmarks.md#笔记本与资源边界)。

## 源码同步与 runner

2026-09-30 起按用户要求，本轮源码修复统一在云端完成并提交，再通过 Git 同步到台式机与笔记本的独立 checkout。项目脚本、测试和配置模板也走同一路径；两台 runner 均不编辑、打补丁或复制未提交源码搭建 overlay。机器专属运行配置、日志、构建和进程管理可在对应运行端进行。私有 Git bundle 可用于有界提交传输；不得向公开远端发布此前未公开代码或实验材料。

[sync_laptop_repo.ps1](../../scripts/sync_laptop_repo.ps1) 的两个动作不同：

| 调用方式 | 实际动作 |
| --- | --- |
| `-SkipWorktreeUpdate` | 仍会 `git push laptop-sync`，只跳过远端工作树更新 |
| 不传该参数 | push 后在指定远端仓库 fetch、checkout、pull --ff-only |

脚本默认目标是远端 `Desktop\MahjongAI`，可能不同于训练 checkout；同步时依据目标 run 的进程和 manifest 选择 `-LaptopRepo`。

本地提交后，可仅推送 Git 提交而保持远端工作树不变：

```powershell
.\scripts\sync_laptop_repo.ps1 -Branch main -LaptopHost mahjong-laptop -SkipWorktreeUpdate
```

活跃 runner 的源码与恢复链路保持冻结；需要新代码时，通过 Git 准备独立 checkout，记录 commit 与源码指纹，在安全 checkpoint 后切换。不要把旧 manifest 改写成“原源码未变”。

## 启动与观察

- 优先复用 [分布式控制模块](../../mortal/supervised/distributed_dispatch.py) 和阶段 runner；复杂参数按现有 JSON / Base64 传输，避免多层 shell 手拼转义。
- 非交互后台进程使用 `Start-Process -WindowStyle Hidden`，指定工作目录，并分开记录 stdout / stderr。完整记录 PID、命令、有效配置和源码指纹。
- Session 0、交互前台和远程会话的吞吐可能不同；benchmark 要记录会话类型。
- 完成判定使用 worker 结果文件、退出码、metrics 与 checkpoint，不只看 SSH 命令返回或父壳进程存活。
- 在线 RL 的跨机 worker 只有在明确设计了可信连接和隔离方式后才启用；默认仍是独立 run。

## 性能与容量验收

适用于当前及后续所有笔记本计算任务，依据 [AGENTS.md](../../AGENTS.md#双机与提交) 的资源利用原则执行。

1. 启动长任务或改变并存负载前，用有界、代表性的输入核验 RAM / VRAM 峰值，覆盖训练、全量验证、模型加载、阶段切换和恢复中适用的高占用路径；把并存进程、预取、缓存和暂存副本计入总量。根据实测波动保留安全余量，逐档验证容量，不以反复触发 OOM 寻找上限。
2. 以相同输入和运行语义下的持续有效吞吐选择配置，记录成功更新/秒、样本/秒或完整对局/小时以及资源峰值。通过有序数据准备、预取、批量推理或独立任务并行减少等待；低 CPU/GPU 利用率须分解耗时、记录瓶颈并继续优化，资源百分比和单次提速不能单独作为完成依据。
3. 调整 workers、batch、缓存或并发时，验证样本/标签、逻辑 batch、训练消费游标和恢复状态、评测对局及指标的等价性。保持既定数据划分、辅助目标和模型强度门槛；优化方法不得缩减科学比较来换吞吐。当前训练游标要求串行消费时，分别设计验证并行或有序预取，不能直接全局提高 workers。
4. 经吞吐、等价性和代表性峰值验证后，在完整 checkpoint/chunk 边界部署最快的安全配置；遵守独立 Git runtime 和唯一 writer 边界。运行中观察阶段峰值与内存趋势，并存任务或输入规模变化后重新核验容量；出现压力先降低有界预取/并发或按安全边界暂停，保留完整恢复点。

每次验收保留配置、输入规模、并存任务、会话类型、峰值、安全余量依据和吞吐证据，位置遵守[机器页](../status/machine-benchmarks.md#新-benchmark-的记录要求)。仍有明显资源空闲时，说明可复核的瓶颈及下一步，不默认沿用保守配置到任务结束。

## 中断与恢复

1. 先保存错误、supervisor 状态、metrics、资源快照和最后完整 checkpoint。不要为诊断覆盖旧日志。
2. 用 checkpoint 内部 `steps`、optimizer、AMP scaler、scheduler、data cursor 判定可恢复位置；文件名和修改时间只能辅助。
3. 核对运行时源码与原生扩展摘要，检查恢复入口和输出目录。缺少必要状态时，明确记录这是重新分支而非 exact resume。
4. 确认旧 worker 已退出，再由对应 supervisor / runner 恢复；不重叠启动两个写同一目录的 trainer。
5. 观察恢复后确实越过保存点，记录回退步数和损失的进度，再恢复常规运行。

OOM 后先核对 RAM / VRAM 峰值与并存任务，再调整 loader 和做短验证；历史快配置不能作为原样重启的理由。当前资源证据见 [机器页](../status/machine-benchmarks.md#笔记本与资源边界)。

只移动或清理明确授权的 run 路径；先解析绝对路径并确认位于目标根目录内。源码同步不顺带删除 checkpoints、replay 或浏览器 profile。
