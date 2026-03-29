# 文档导航

这套文档现在按受众和用途拆开了。默认原则只有两条：

- 想知道“现在默认怎么做”，先看当前入口，不看历史长文。
- 想知道“为什么这样做”，再进入研究记录、复盘或归档。

## 先看哪里

- 当前默认主线与续工入口：`agent/current-plan.md`
- 当前稳定默认、阶段地图、冻结结论：`agent/mainline.md`
- 当前实验流程、人工停点、运行纪律：`agent/experiment-workflow.md`
- 当前真实进度与在跑什么：`status/stage05-verified-status.md`
- `P1` 唯一有效选模口径：`status/p1-selection-canonical.md`
- 自动生成的 run 快照：`status/stage05-fidelity-results.md`

## 按问题找文档

- 想快速上手仓库：`../README.md`
- 想看 agent 工程规范和硬约束：`../AGENTS.md`
- 想看 `Stage 0 / GRP` 研究结论：`research/stage0/grp-experience.md`
- 想看 `Stage 0.5` 当前仍有效的工程方法：`research/stage05/engineering-playbook.md`
- 想看 `P1` selector 的统计审计：`research/stage05/selector-stat-audit.md`
- 想看某次专项实验：`research/stage05/`
- 想看人机协同、个人成长和方法复盘：`reflections/README.md`
- 想看已经退役的阶段快照或旧入口：`archive/README.md`

## 目录分工

- `agent/`
  - 给 agent 的短入口文档，只放当前默认、稳定规则和续工顺序。
- `status/`
  - 给人和 agent 共用的“当前状态”与“当前口径”。
  - 其中 `stage05-fidelity-results.md` 是自动生成的 run 快照，不是默认手册。
- `research/`
  - 给人看的方法、实验设计、专项研究和仍有参考价值的技术结论。
- `reflections/`
  - 给人看的复盘、协同经验、论文素材和长期成长记录。
- `archive/`
  - 历史快照、旧入口、旧长文。可回查，但不能当当前默认。

## 使用规则

- `current-plan.md`、`mainline.md`、`stage05-verified-status.md`、`p1-selection-canonical.md` 之间如果冲突，优先检查是否有人误用了历史文档。
- `stage05-fidelity-results.md` 只回答“这次 run 写出了什么”，不回答“现在默认是什么”。
- `archive/` 下的任何内容都默认不是当前主线，除非当前入口文档明确重新启用。
