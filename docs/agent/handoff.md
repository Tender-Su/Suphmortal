# 接手摘要

> 核验：2026-09-12 · 本次仅更新SL课程与独立正式确认完成状态；其他条目以对应状态页为准，启动或恢复前复核真实进程。

SL/RL 审计修复已落到主树并完成双机回归；新 Oracle 校准与独立正式评测已部署。当前实现、证据和未完成资格见 [实施报告](../research/sl-rl-fixes-2026-09-07.md)，原始诊断保留于 [独立审计](../research/sl-rl-audit-2026-09-05.md)。

| 主题 | 当前边界 | 详情 |
| --- | --- | --- |
| SL | C50k独立正式确认通过，canonical保留发布身份；两seed十臂课程对照仍inconclusive | [SL 状态](../status/supervised-mainline.md) |
| Oracle critic | 两臂 40k 及候选比较已完成，均未决；保留 warm reference，独立资格未定 | [Oracle 状态](../status/oracle-critic-mainline.md) |
| 在线 RL | 完成独立三角色容量测试；正式PPO待critic资格及AMP成功更新时钟核对，历史候选未证明稳定强于SL | [RL 状态](../status/online-rl-mainline.md) |
| 笔记本 | 本轮SL课程与正式确认均正常结束并完整验收；不重启旧运行 | [SL 状态](../status/supervised-mainline.md) |
| 平台客户端 | 台式机 RiichiLab 客户端在运行；ranked 结果不替代正式 `1v3` | [接入说明](../../integrations/riichilab/README.md) |

下一步顺序：

1. SL独立正式确认已通过，后续发布另行安排；Oracle本轮未决，保留no-update，后续实验先明确问题与新增预算。
2. 最终 actor 冻结后在独立模拟分布验证 critic 的主指标、全部 guard、校准与 Oracle 输入依赖；offline finalist 前保持 sealed test 封存。
3. 合格 critic 才进入近 on-policy PPO 与 Oracle 的多训练种子对照；不直接沿用旧 hybrid 或混用奖励/replay。

本轮SL课程与正式确认已经完成；critic资格与RL强度仍以对应状态页为准。其他活跃运行的源码及恢复链路保持冻结，后续主树修正不能热覆盖活跃worker。

启动、恢复或评测参考 [运行流程](workflows.md)；涉及运行中源码参考 [活跃训练边界](code-health.md#活跃训练边界)，涉及另一台机器参考 [远程流程](remote-ops.md)。当前数值和实验路径只在对应状态页维护，不在本页追加流水记录。
