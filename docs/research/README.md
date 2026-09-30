# 研究与证据

研究报告解释判断依据；[当前状态](../README.md#当前状态) 维护今天的结论和下一步。报告开头必须标明适用范围、证据日期、已实施内容和未决问题。

## 当前有效报告

| 报告 | 影响范围 |
| --- | --- |
| [独立 critic-only 校准阶段](critic-only-calibration.md) | 固定 actor 状态、精确成功更新预算与重新生成 GAE 的阶段切换 |
| [2026-09-30 至 10-08 研究窗口](research-window-2026-09-30.md) | 云端统一开发、两机分工、指标复议、分阶段实验与截止管理 |
| [源码同步与研究边界 · 2026-09-30](source-sync-2026-09-30.md) | 精简证据、当前代码语义、旧评测门槛待复议和源码传输范围 |
| [SL 独立正式确认完整结果 · 2026-09-12](sl-formal-confirmation-result-2026-09-12.md) | C50k对三家canonical的独立64k确认通过、两臂全原始验收与冻结配对复核；未自动发布 |
| [SL 冻结课程完整结果 · 2026-09-10](sl-curriculum-result-2026-09-10.md) | 两seed十臂完整验收、AC/CC及总预算；inconclusive，无课程推荐，候选牌力确认另见9月12日报告 |
| [Oracle 初始化匹配补跑完成审查 · 2026-09-08](oracle-matched-completion-2026-09-08.md) | 两臂 40k、全部 guard 与保留候选配对比较；没有可晋级的新候选 |
| [SL 同起点课程探针实施 · 2026-09-07](sl-curriculum-probe-implementation-2026-09-07.md) | 用户批准后的 A/B/C 分支、完整域轮换、实际消费游标、延迟迁移与未决规则 |
| [SL 课程机制方案 · 2026-09-07](sl-curriculum-mechanism-proposal-2026-09-07.md) | 共通能力与年代差异、C 的早期正信号、学习进展论文、同起点转段探针及完整辅助目标修复 |
| [SL 阶段预算与近期数据审计 · 2026-09-07](sl-curriculum-budget-audit-2026-09-07.md) | A/B/C 真实消耗、固定池覆盖、辅助目标变化、继承修复及 SL 对 S70 独立 1v3 |
| [台式机 Oracle / 在线 RL 资源调优 · 2026-09-08](rl-resource-tuning-2026-09-08.md) | 固定恢复点吞吐、完整 monitor 对拍、在线流水线容量与 GAE 内存生命周期 |
| [SL / RL 审计修复实施 · 2026-09-07](sl-rl-fixes-2026-09-07.md) | 固定输入、标签时钟、硬护栏、PPO/V-trace 分离、独立正式协议、双机切换及验证边界 |
| [项目完整方案与专家审阅请求 · 2026-09-05](project-expert-review-2026-09-05.md) | 从任务到 SL / GRP / Oracle / RL 的完整说明、60 项审阅问题，以及新复现的 native fold 标签时钟问题 |
| [SL / RL 独立审计 · 2026-09-05](sl-rl-audit-2026-09-05.md) | SL 发布边界、C/D/E 配对结果、Oracle 输入复现性、guardrail 与奖励契约 |

该报告保留完整证据链和审计时的源码/产物指纹。本地 `logs/` 中的附件通常不随 Git 分发；缺少附件时应标为无法本机重验，不能凭报告文字补造结果。

## 未决问题与状态归属

| 问题 | 当前决策位置 | 旧材料的用途 |
| --- | --- | --- |
| S70 / Long-ABC 是否替换 canonical | [SL 状态](../status/supervised-mainline.md) | [selector 统计](../archive/research/supervised/selector-stat-audit.md) 与 [S70 外部基准调研](../archive/research/supervised/s70-open-weight-benchmark-2026-08-09.md) 只解释当时依据 |
| Oracle 验证、p0 资格、输出权重 | [Oracle 状态](../status/oracle-critic-mainline.md) | [旧 Oracle 文献备忘](../archive/research/online-rl/oracle-critic-literature-observations-2026-04-16.md) 不再下达 actor ramp 的当前优先级 |
| RL 奖励和下一轮对照 | [RL 状态](../status/online-rl-mainline.md) | [旧 PPO 计划](../archive/research/online-rl/rl-ppo-improvement-plan.md)、[baseline / reward 备忘](../archive/research/online-rl/online-baseline-refresh-and-reward-target-report-2026-04-13.md) 已退役为运行依据 |
| 更大 GRP 是否值得训练 | [GRP 状态](../status/grp-mainline.md) | [GRP 旧试验](../archive/research/stage0/grp-experience.md) 保留数据窗口、采样与容量的限制 |

其余旧文献综述、ablation 方案和辅助头研究统一在 [历史索引](../archive/README.md)。归档表示退出当前运行依据，不表示论文或观察被一概否定。

## 新报告写法

先写一个可证伪的问题，再写与当前基线的关系、原始证据位置、实际观察、局限与可执行决策。论文结论、开源实现事实、本地复现和工程推断分开标记。会改变主线时，同次更新对应状态页及本索引。

旧报告不叠加新的运行流水；独立新证据用带日期的文件承接。维护和检查规则见 [文档维护](../agent/doc-maintenance.md)。
