# Agent 入口

按任务选择相关页面；需要整体背景时读 [接手摘要](handoff.md)。长期规则在 [AGENTS.md](../../AGENTS.md)，全量导航在 [文档地图](../README.md)。

| 当前任务 | 主要入口 |
| --- | --- |
| GRP 或标签来源 | [GRP 状态](../status/grp-mainline.md) |
| SL 训练、selector、发布 | [SL 状态](../status/supervised-mainline.md) |
| Oracle critic 训练、验证、接入 | [Oracle 状态](../status/oracle-critic-mainline.md) |
| PPO、replay、value / GAE | [RL 状态](../status/online-rl-mainline.md) |
| 启动或评测 | [运行流程](workflows.md) + [机器与资源](../status/machine-benchmarks.md) |
| 笔记本、同步或恢复 | [远程流程](remote-ops.md) |
| 修改或重构代码 | [代码地图](code-map.md) + [验证要求](code-health.md) |
| 修改文档结构 | [文档维护](doc-maintenance.md) |
| 追溯旧判断 | [研究与证据](../research/README.md)，再按链接查历史 |

不要默认通读所有研究和归档。日期较新的运行摘要也不能覆盖未满足的评测门槛；先核对有效配置、checkpoint 内部字段和原始结果。
