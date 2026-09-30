# 接手摘要

> 核验：2026-09-30 · 依据：恢复的 b20ffc3 工作副本、可见历史和 CPU 权重核验回报；未新增 GPU 训练。旧实验细节以对应状态页为准，启动前复核真实进程。

云端已从 bundle 恢复 `b20ffc3`；丢失的后续文档提交未逐字恢复，现重新整理 [研究意图](../research/research-intent-2026-09-30.md)。actor visible、critic 完整 Oracle 已是主线；GRP 指真实结果预测任务。成熟度与当前策略对齐不能用任意小 warmup 或步数上限替代。

| 主题 | 当前边界 | 详情 |
| --- | --- | --- |
| SL | C50k 独立确认通过、canonical 发布身份不变；旧十臂晚期课程仍未决，新问题是 A 过训/可塑性与早期 B/C | [SL 状态](../status/supervised-mainline.md) |
| Oracle critic | 旧两臂 40k 按旧协议均未决；先重审存量长训权重，再补当前策略校准，不重跑旧资格赛 | [Oracle 状态](../status/oracle-critic-mainline.md) |
| 在线 RL | 未启动本轮正式 PPO；从成熟且对齐的 critic 做小闭环，保留奖励、GAE、更新时钟核验 | [RL 状态](../status/online-rl-mainline.md) |
| 运行 | ModelKits 正整理指标及 import preflight，尚无本轮新 GPU 计算；5070 Ti 负责 critic/RL，4060 负责 SL | [研究窗口](../research/research-window-2026-09-30.md) |

下一步顺序：

1. 复核现成 checkpoint 的血缘、奖励目标、结构、指标和加载能力，优先复用 [已核验资产](../research/weight-reuse-2026-09-30.md)；旧新奖励下的 loss 不能直接比较。
2. 实现并验证 primary MSE 保存与 MAE / zero / tail 诊断否决及停止逻辑解耦。保留旧协议结果，不追溯放宽规则。Oracle 依赖诊断、高置信 advantage 门槛不作为新增资格要求。
3. 先完成必要的小规模当前策略 critic 校准，再进入受控 RL，评估 actor 收益。SL 从早期 A 资产设计有明确决策价值的探针；已存在 sealed test 不反复用于调参。

源码仅在云端开发与提交，两台 runner 同步到独立固定 commit 工作树，不热覆盖活跃 worker。算力授权到北京时间 2026-10-08 19:00（UTC 11:00），所有本轮进程必须有独立本机截止管理及保存停止余量；不为填满时窗增设实验。

启动与恢复见 [运行流程](workflows.md)、[活跃训练边界](code-health.md#活跃训练边界) 和 [远程流程](remote-ops.md)。
