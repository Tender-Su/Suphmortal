# 文档地图

按问题直接进入对应主题；需要整体背景时读 [接手摘要](agent/handoff.md)。每项事实只在一个主页面维护；研究与历史通过链接追溯。

## 当前状态

| 主题 | 回答的问题 |
| --- | --- |
| [GRP](status/grp-mainline.md) | 前置模型、checkpoint 用途与标签边界 |
| [监督学习](status/supervised-mainline.md) | canonical、S70 与后续 SL 的资格和选择口径 |
| [Oracle critic](status/oracle-critic-mainline.md) | 独立预训练、验证缺口与接入条件 |
| [在线 RL](status/online-rl-mainline.md) | 已有收益证据、奖励契约和下一轮实验门槛 |
| [机器与资源](status/machine-benchmarks.md) | 有效配置、历史 benchmark 与资源边界 |

## 操作与开发

| 文档 | 职责 |
| --- | --- |
| [Agent 入口](agent/README.md) | 按任务选择阅读范围 |
| [运行流程](agent/workflows.md) | 环境、配置、训练和评测命令 |
| [双机与远程](agent/remote-ops.md) | 同步、runner、恢复与资源检查 |
| [代码地图](agent/code-map.md) | 模块职责和接口契约 |
| [重构与验证](agent/code-health.md) | 源码冻结、恢复语义和验证要求 |
| [文档维护](agent/doc-maintenance.md) | 归属、更新规则和自动检查 |

接入说明：[雀魂](../integrations/majsoul/README.md) · [RiichiLab](../integrations/riichilab/README.md) · [MahjongCopilot 宿主](../MahjongCopilot/readme.md)。

## 证据与历史

| 入口 | 使用方式 |
| --- | --- |
| [研究与证据](research/README.md) | 当前有效报告及未决问题；先看结论适用范围 |
| [完整方案与专家审阅](research/project-expert-review-2026-09-05.md) | 从头讲解 SL / GRP / Oracle / RL，附公式、配置、证据和审阅清单 |
| [历史归档](archive/README.md) | 旧计划、结果和原文；不作为启动依据 |
| [复盘](reflections/README.md) | 判断过程与协作经验，保留当时语境 |
| [SL 自动 snapshot](status/supervised-fidelity-results.md) | runner 生成的单次运行摘要；不覆盖当前状态页 |

事实依据与文档优先级见 [AGENTS.md](../AGENTS.md#阅读入口)，维护方式见 [文档维护](agent/doc-maintenance.md)。
