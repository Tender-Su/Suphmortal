# 文档导航

文档按四层组织：入口层 → 状态层 → 证据层 → 背景层。

## 文档分层

### 入口层：`agent/`

| 文件 | 职责 |
|------|------|
| `current-plan.md` | 当前停点与下一步 |
| `mainline.md` | 冻结默认、命名族、机器默认 |
| `experiment-workflow.md` | 当前主线怎么跑、人工确认点 |
| `laptop-remote-ops.md` | 远程 shell、数据根、双机运行坑点 |
| `code-sync.md` | 台式机到笔记本的 Git 同步 |

### 状态层：`status/`

| 文件 | 职责 |
|------|------|
| `supervised-verified-status.md` | 人工核对后的监督学习阶段真实结论 |
| `p1-selection-canonical.md` | `P1` 唯一有效评估口径 |
| `supervised-formal-triplet-playoff-canonical.md` | formal triplet → `formal_1v3` → 官方 winner |
| `supervised-fidelity-results.md` | 自动生成的 run snapshot（run-scoped，非当前默认） |
| `laptop-sl-loader-benchmark-2026-03-31.md` | 笔记本 loader 证据 |
| `1v3-multishard-benchmark-2026-04-02.md` | 双机 `1v3` 吞吐证据 |

### 证据层：`research/`

| 文件 | 职责 |
|------|------|
| `supervised-evolution.md` | 监督学习阶段演进全记录 |
| `stage0/grp-experience.md` | GRP 候选对比与探索方向 |
| `supervised/engineering-playbook.md` | 监督学习工程经验 |
| `supervised/selector-stat-audit.md` | selector 统计支持证据 |
| `supervised/a2y-aux-shape-freeze-2026-03-25.md` | 三类辅助头内部 shape 冻结 |
| `supervised/p1-aux-adjustment-2026-03-22.md` | P1 auxiliary 搜索调整记录 |
| `rl-ppo-improvement-plan.md` | 强化学习 PPO 改进草案 |

### 背景层：`reflections/` 与 `archive/`

- `reflections/`：复盘、人机协同方法、个人研究判断
- `archive/`：已退役入口、旧快照、旧长文

## 使用规则

入口层和状态层是当前真相；证据层、背景层（含自动摘要、研究长文、复盘、归档）不能覆盖前两层。
