# SL 同起点课程探针实施

> 2026-09-07 · 用户已批准 [机制方案](sl-curriculum-mechanism-proposal-2026-09-07.md)。同起点实验于 15:36 在独立冻结 runtime 启动，目前先重算完整验证基线，尚无课程优劣结论。模型强度仍由独立正式 1v3 决定。

## 对照回答什么

从没有经历 A→B 辅助目标漂移的 A 2.88M checkpoint 出发，比较下一段计算继续 A、转 B、转 C 的目标域收益。A 本身已有 60% 近期数据，因此这不是老数据与新数据的二选一实验。

源文件 SHA-256 为 `489364d8c0dc4f5573e3f9a1fa4e49c97f8724c4e93c3c44503aa2405e77cf90`；已从笔记本复制并核对。内部成功 optimizer steps 为 2,878,912，完整辅助头、Adam 状态和 AMP scaler 均存在。

各组保留同一 actor、rank/opponent/danger 辅助头、Adam moments、scaler、辅助时钟；将当前 LR **5e-6** 固定，不 rewarm。实验局部计数归零，历史 patience、选优结果和数据游标不继承。这是显式分支，不能称为旧 A 训练的逐样本续训。新分支自身保存实际消费游标与 RNG。

## 输入与计算契约

| 项目 | 首轮设置 |
| --- | --- |
| A 配方 | 60% 2023–2024，25% 2021，15% 2009–2020 |
| B 配方 | 90% 2023–2024，10% 2009–2021 replay |
| C 配方 | 98% 2024，2% 2009–2021 replay |
| 2022 | 保留作旧域验证，不进入训练 |
| 域覆盖 | 每个域内不放回轮换；穷尽后重新排列，无固定 12 万 / 8 万池 |
| 完整域大小 | early 1,613,207；mid 187,313；recent 353,201；latest 184,424；replay 1,800,520 |
| 采样单位 | 游戏概率；实际决策比例和重复量另行记录，不把配置比例当成状态曝光比例 |
| 增强与混合 | 每次游戏抽取包含两种增强视图，每 4 个游戏的样本共同打乱 |
| 每次成功更新 | 1,024 个决策；4 个 256 microbatch 累积梯度，GN，search 关闭 |
| 训练种子与轮转 | 20260907、20260917；全体先完成短跨度，再进入长跨度；种子间反转、跨度间轮换顺序 |
| 观测跨度 | 1,024 和 4,096 次成功 optimizer updates；是测量窗口，不是 ABC 阶段总预算 |
| 延迟迁移 | 每个种子比较 A→C 与 C→C，两段各 4,096 次更新；第二段使用共同的新 C 采样种子 |
| controller-dev | 固定 2026-01 的 512 局及 2022 的 256 局；逐文件字节哈希绑定 |
| selection-dev | 单独保留 2025 的 512 局及另一组 2022 的 256 局，本 runner 不打开牌谱正文 |

这些是原有验证年代的新 controller/selection 划分。A 的历史训练过程已使用这些年代进行验证，不能把它们重新声称为从未看过的 sealed test。未来若引入 2025/2026 训练数据，必须重建隔离协议。

所有探针和延迟迁移成本都计入实验总成本：完整计划最多 40,960 次成功更新、41,943,040 个成功更新决策，另记录 AMP 跳步实际消费、验证和各次进程运行时间。决策计数依据持久化 cursor；故障前尚未保存的重做计算无法精确还原，不能把这部分称为零成本。重复量 `decisions_beyond_one_augmented_draw` 表示超出一轮双视图样本数的决策，不是每个独立状态逐一去重后的精确重复率。该观察计划结束后输出信号或未决，不会把全部候选短程无改善自动解释为全局收敛。

## 判断规则

主指标是近期 policy NLL，相同验证游戏逐局配对；准确率和旧域 policy NLL 使用原有 2e-4 非劣容差。改善阈值为 2e-4。预声明最多 84 个标量对比，使用 Bonferroni 校正的近似 game-cluster normal 区间；这依赖独立游戏和足够样本的渐近近似，不是无条件的精确统计保证。

只有同一配方在两个跨度、两个训练种子中都明确优于共同 parent 和另两种配方，且通过护栏，才输出一致的课程信号。若 C 获胜但 A→C 的等计算后续明显更好，拒绝短视的自动接入。两颗种子的重复性要求并不能估计完整训练种子分布。

输出 `inconclusive` 时先区分测量精度、观测跨度和 LR 的问题，继续对竞争者对称扩展；没有静默回退到“永远先训练 A”。连续混合权重、放宽旧域护栏、主网重置及模型发布均不在这个 runner 的自动行为内。

## 实现与恢复

- [实验入口](../../scripts/run_sl_curriculum_probe.py)：准备、冻结输入和源码、串行执行、累计成本、成对报告。
- [采样与观察](../../mortal/supervised/curriculum_probe.py)：域内轮换、逐游戏实际消费统计、观测点、数据与 RNG 游标。
- [训练器](../../mortal/supervised/train_supervised.py)：新增显式 probe hook，正常训练入口不启用；停止计数使用成功 optimizer updates。
- [回归测试](../../mortal/tests/test_curriculum_probe.py)：完整域覆盖、跨轮次恢复、batch 消费游标、辅助和 optimizer 继承、未决判断。

probe 只使用单进程 DataLoader，关闭 CUDA prefetch，避免保存的 cursor 超前于实际消费。恢复时按相同 seed 重建每个域的排列，最多重读当前 4 个游戏；校验当前牌谱字节摘要，跳过已消费样本。checkpoint 在完整 optimizer update 后原子保存；非有限 loss 失败时保留最后一个完整状态。验证不会推进训练 RNG。

第一轮真实启动暴露了 `SearchDistillConfig.from_config` 对 keyword-only 布尔默认值使用位置参数的问题。`mortal/eval/search_runtime.py` 还属于在跑 RiichiLab 程序的源码边界，因此用入口中的精确替换函数修复新冻结 runtime，manifest 记录主树原摘要与新摘要。待主树该运行边界解除后可将调用改为 `default=True` 并删除适配；没有改动活跃对战程序或旧实验 manifest。

## 验证与运行回执

- [137 项相关测试](../../logs/sl_curriculum_audit_20260907/probe_tests_final.log) 通过，包含辅助目标继承、恢复、旧课程控制与新轮转协议。
- [真实小规模轮转](../../logs/sl_curriculum_audit_20260907/probe_smoke_r4/comparison.json) 完成 A/B/C 各 1、2 次成功更新，完整走过跨进程恢复、两类验证和配对报告；少量游戏明确判为不具备统计资格，没有据此选配方。
- [共同起点核验](../../logs/sl_curriculum_audit_20260907/probe_parent_verification.json)：全部 actor/辅助头和 scaler 与原 A 相同，414 个 optimizer 参数状态逐项核对通过，辅助时钟为 2,878,912，三臂基线指标完全相同。
- [真实恢复对拍](../../logs/sl_curriculum_audit_20260907/probe_resume_verification.json)：连续 U2 与从 U1 恢复到 U2 的 learned state、全部 RNG、采样游标、实际消费与验证指标逐项完全相同。训练峰值 allocated 显存约 **1.819 GiB**。
- [microbatch 对照](../../logs/sl_curriculum_audit_20260907/probe_microbatch_verification.json)：64 与 256 使用相同消费数据和逻辑 batch，AMP 累积顺序产生小数值差异，backbone 两步后最大参数差约 7.73e-7；不声称两种 microbatch 的长程轨迹逐位等价。本轮所有正式观察固定使用 256。

运行根为 `logs/sl_curriculum_probe/20260907_desktop_r1`，协议身份 `f0a4112ba72538fde0573755453c2fd1afdfd3de94933755262ffd4b78aabbd7`。见 [冻结协议](../../logs/sl_curriculum_probe/20260907_desktop_r1/manifest.json)、[轮转进度](../../logs/sl_curriculum_probe/20260907_desktop_r1/progress.json)、[supervisor 状态](../../logs/sl_curriculum_probe/20260907_desktop_r1/apex_supervisor_status.json)。启动回执确认 supervisor 在 Windows Job 外持续运行，原 Oracle 和 RiichiLab 进程仍存活。新进程 allocator 上限为 GPU 的 20%，torch/rayon 各 2 线程，保留独立 Apex 暂停与恢复。

最终机器可读结果写到运行根的 `comparison.json`。在全部预声明观测完成前，单臂更新和短程片段不构成自动转段或发布依据。

## 旁证：旧 finalist 对 S70

独立两臂各 2,000 局已完成，SL 平均 pt +0.7425、顺位 2.4905，pt 差值 95% 配对 bootstrap CI [-2.1825, +3.6675]。目前不能确认其强于 S70，原 canonical 身份保留。详见 [完整结果](../../logs/sl_curriculum_audit_20260907/s70_1v3_2000/comparison.json) 及 [直接对战审计](sl-curriculum-budget-audit-2026-09-07.md#sl-对三家-s70-的独立-1v3)。
