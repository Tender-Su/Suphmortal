# 机器与资源

> 核验：2026-09-08 · 依据：台式机Oracle/在线RL资源实测，以及笔记本有序准备、持续吞吐与恢复收据。参数不构成吞吐保证。

## 台式机

硬件为 i5-13600KF + RTX 5070 Ti 16 GB。解释器见 [运行流程](../agent/workflows.md#环境与构建)。

| 来源 / 用途 | batch | train workers / file batch / prefetch | validation |
| --- | --- | --- | --- |
| 基础配置的 SL | 1024 | 4 / 10 / 3 | workers 0，file batch 8 |
| 基础配置的 Oracle | 512 | 4 / 10 / 3 | file batch 8，其余按生成配置解析 |
| Oracle 匹配臂资源续跑 | 640 | 2 / 2 / 2；Rayon4、Torch1、HighQoS | workers0、file batch2、native fold32、完整monitor；保留原批布局 |
| 在线 RL 已测资源档 | 192 | GAE路径workers0；每轮128局、Rayon4、Torch1、HighQoS | GAE物理inference block512，logical chunk50；顺序近on-policy协议 |

Oracle相同状态的1000-step资源对照为2.964→4.373 updates/s（+47.55%）；独立后台另需显式HighQoS，500-step验收为4.114 updates/s，最低可用RAM8.96GiB。训练allocated/reserved峰值7.67/8.31GiB，完整验证含optimizer常驻时11.40/11.45GiB；allocator上限0.72。

在线三角色在4GiB RAM预留档完成多轮交换；后台256计数步中有255次实际AdamW更新、1次AMP跳过，有效吞吐0.621 updates/s，最低可用RAM4.42GiB。trainer/client allocator上限0.68/0.17，RAM余量紧，保留资源监控与停止条件。相关细节、失败档位和测量限制见[资源实测报告](../research/rl-resource-tuning-2026-09-08.md)。

运行身份、源码/native摘要与暂停边界只在[Oracle状态](oracle-critic-mainline.md#当前运行)维护；RL资格见[在线状态](online-rl-mainline.md)。RiichiLab CPU客户端保持运行。吞吐实测不能替代模型强度判断。

workers 为 0 时，保存的 prefetch 参数不代表启用了多进程预取。运行中的配置和恢复源码不得按基础 TOML 热覆盖；使用独立运行目录和原生扩展，不在活跃环境执行 maturin develop。

## 1v3 默认来源

[one_vs_three.py](../../mortal/eval/one_vs_three.py) 的旧通用内建值：

| GPU | seed_count | shard_count |
| --- | --- | --- |
| RTX 5070 Ti | 1024 | 4 |
| RTX 4060 Laptop GPU | 640 | 3 |

解析优先级为环境变量→machine profile→GPU 配置→内建 GPU 默认→通用配置。seed_count 表示四座 seed 组，不能当作总局数。

9 月 7 日独立正式协议另行明确每臂 16k 筛选/64k 确认。当前新运行固定 chunk 为64 seed组，不套用上表通用默认；各次旧运行不改写。实际 GPU/native/源码和 chunk 数进入评测 provenance，同协议各臂必须一致；吞吐以完整 chunk 记录估算。当前运行与 A/A 状态见 [SL 状态](supervised-mainline.md)。

## 笔记本与资源边界

硬件为 i9-13900HX + RTX 4060 Laptop 8 GB + 32 GB RAM。9 月 7 日已直连确认旧 Phase C 于 9 月 6 日 11:10 在 300k 正常停止，runner exit 0；详见 [SL 状态](supervised-mainline.md) 与 [完成审计](../../logs/sl_monitor/20260907_112958_phase_c_completed.json)。

9 月 5 日已修复的验证 OOM 路径保留 logical batch 1024，以 256 样本分批执行 Brain 前向，整批计算 loss/head/cluster 指标；内部 GPU OOM 不自动无限重试。旧 SL checkpoint 缺 batch cursor/RNG，不能称逐 batch exact resume。事故、证据、修复、94 项当时回归与恢复流水完整保留于 [9 月 7 日历史快照](../archive/status/machine-benchmarks-run-history-2026-09-07.md)，不借用当时计数说明当前代码状态。

当前 SL 和正式评测分别位于独立 Git checkout `MahjongAI_sl_ordered_20260908`、`MahjongAI_1v3_bulk_20260907`，使用隐藏 supervisor 与按需计划任务，保留 Apex 完整 checkpoint/chunk 暂停，不恢复无限异常重试。运行身份只在 [SL 状态](supervised-mainline.md) 与其执行记录维护；旧 runner 源码/配置/权重保持原样。

本轮同输入实测：SL 验证文件批4/Rayon4 比串行快82.85%，既有四 draw 双视图输入准备为2.537倍，所有字段/指标与真实恢复状态精确一致，训练 allocated 峰值仍1.81769GiB。1v3 chunk32/64 同256局为617.627/369.265秒，吞吐提高67.26%，全部事件相同，allocated峰值138.38/161.24MiB。仅为已测输入和阶段的证据，非全程或最大容量保证；详情见 [性能诊断](../../logs/sl_curriculum_audit_20260907/performance_diagnosis_20260907.md)。

9 月 8 日续测采用 train/val 有序准备进程 **4/4**、每块 **4** 个 draw、文件批 **4**、Rayon **4**，消费游标仍在训练进程，常规 DataLoader workers 仍为0。匹配 U6–U36 真实训练吞吐由0.17797升至0.29499 updates/s（+65.8%）；延长至 U68 为0.29658，最低系统可用 RAM 11.71GiB。六进程延长复测0.29213，未见稳定增益且多占约3.4GiB RAM，故选四进程。保持全部输入/辅助标签、micro256×4、确定性设置和成功更新语义，完整恢复状态/指标精确一致；新 SL 任务为经核验的 Normal 优先级。原始对照、限制和恢复链见 [实测报告](../../logs/sl_curriculum_audit_20260907/ordered_preparation_report_20260908.md)。

更大 chunk128 独立诊断触发 `nvcuda64.dll` 原生 fail-fast `0xc0000409`，仅该进程退出，正式负载保持健康。已保留 Windows 事件、WER 和 dump，停止扩大及复现；尚未确定根因，不等同于已证实 CUDA OOM、RAM 泄漏或纯驱动缺陷。当前只采用已验证64档，不在活跃训练时改驱动或原生扩展。

## 新 benchmark 的记录要求

记录机器、源码/native hash、模型、配置、输入规模、并存任务、峰值 RAM/VRAM、吞吐、验证和恢复结果。性能变化需要数据与训练语义等价性证据，单次低显存不能排除历史事故峰值或长期泄漏。

不清理运行日志/checkpoint 来做 benchmark。新结果替换当前适用结论，完整时间线放带日期的报告或运行产物。更早数据见 [旧机器快照](../archive/status/machine-benchmarks-before-doc-refactor-2026-09-05.md) 和 [旧 Oracle 资源记录](../archive/status/oracle-critic-resource-benchmarks-before-doc-refactor-2026-09-05.md)。
