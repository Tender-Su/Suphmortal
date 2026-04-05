# 文档导航

这套文档现在分成两层：

- 活跃 handoff 文档：给无上下文 agent 和当前操作者，回答“现在默认是什么、停在哪、下一步怎么做”。
- 历史 / 研究文档：给回溯、复盘和方法论使用，不作为当前默认入口。

## 默认读取顺序

1. `../AGENTS.md`
2. `agent/current-plan.md`
3. `agent/mainline.md`
4. `status/supervised-verified-status.md`
5. `status/p1-selection-canonical.md`
6. 任务涉及双机或 Git 同步时，再读 `agent/laptop-remote-ops.md` / `agent/code-sync.md`

## 活跃文档分工

- `agent/current-plan.md`
  - 当前主线、真实停点、启动前清理状态、下一步。
- `agent/mainline.md`
  - 当前冻结默认。只放稳定规则，不放 run 过程叙事。
- `agent/experiment-workflow.md`
  - 当前阶段应该怎么跑、在哪些点强制停下来。
- `agent/code-sync.md`
  - 台式机与笔记本的 Git 同步。
- `agent/laptop-remote-ops.md`
  - 笔记本 shell、数据根、远端运行前提和踩坑结论。
- `status/supervised-verified-status.md`
  - 当前监督学习阶段的人工核对结论。
- `status/p1-selection-canonical.md`
  - `P1` 唯一有效的选模与解释口径。
- `status/supervised-fidelity-results.md`
  - 历史 run snapshot。只回答“那次 run 写出了什么”，不回答“现在默认是什么”。

## 目录分工

- `agent/`
  - 当前默认、当前流程、当前运维入口。
- `status/`
  - 当前状态、当前口径、当前 benchmark 结论。
- `research/`
  - 方法、证据、专项实验与工程经验。
- `reflections/`
  - 人机协作、复盘和长期记录。
- `archive/`
  - 已退役入口、旧快照、旧长文。默认不参与当前判断。

## 使用纪律

- 先读活跃 handoff 文档，不要先从 `research/` 或 `archive/` 开始。
- 如果某份 run snapshot、旧研究文档或旧状态摘要与活跃 handoff 文档冲突，以活跃 handoff 文档为准。
- `P1` winner 的解释一律以 `status/p1-selection-canonical.md` 为准。
