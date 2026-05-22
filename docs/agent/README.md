# Agent 入口

先读一屏接手摘要，再按任务打开对应文档。不要从研究长文或归档里拼当前默认。

## 默认阅读顺序

1. `docs/agent/handoff.md`
2. `docs/status/online-rl-mainline.md`
3. `docs/status/supervised-mainline.md`
4. `docs/status/machine-benchmarks.md`

## 按任务加读

| 任务 | 加读 |
| --- | --- |
| 跑训练、评测或生成配置 | `docs/agent/workflows.md` |
| 调度笔记本、同步代码、远程排障 | `docs/agent/remote-ops.md` |
| 找 Python 模块、入口脚本或放置新文件 | `docs/agent/code-map.md` |
| 改文档结构或新增长文 | `docs/agent/doc-maintenance.md` |
| 追溯某个研究判断 | `docs/research/README.md` |

## 冲突规则

- `status/` 覆盖 `agent/` 的摘要。
- `agent/` 覆盖 `research/` 的旧计划。
- `archive/` 和 `reflections/` 默认不指导当前运行。
