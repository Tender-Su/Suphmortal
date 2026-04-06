# 文档导航

这套文档按“入口层 -> 状态层 -> 证据层 -> 背景层”组织。先读入口层和状态层，再决定是否需要深入证据或背景材料。

## 默认读取顺序

1. `../AGENTS.md`
2. `agent/current-plan.md`
3. `agent/mainline.md`
4. `status/supervised-verified-status.md`
5. `status/p1-selection-canonical.md`
6. 任务涉及 formal triplet / `formal_1v3` 时，再读 `status/supervised-formal-triplet-playoff-canonical.md`
7. 任务涉及双机时，再读 `agent/laptop-remote-ops.md` / `agent/code-sync.md`
8. 需要监督学习阶段演进背景时，再读 `research/supervised-evolution.md`

## 文档分层

### 1. 入口层：`agent/`

- `current-plan.md`
  - 当前停点、当前下一步、当前工作边界
- `mainline.md`
  - 冻结默认、命名族、机器默认、脚本入口
- `experiment-workflow.md`
  - 当前主线怎么跑、在哪里人工确认
- `laptop-remote-ops.md`
  - 远程 shell、数据根、双机运行资产与已验证坑点
- `code-sync.md`
  - 台式机到笔记本的 Git 同步边界

### 2. 状态层：`status/`

- `supervised-verified-status.md`
  - 当前监督学习阶段已经核对过的真实结论
- `p1-selection-canonical.md`
  - `P1` 唯一有效的评估与解释口径
- `supervised-formal-triplet-playoff-canonical.md`
  - pre-formal 第一梯队如何通过 formal triplet 与 `formal_1v3` 产生官方 winner
- `supervised-fidelity-results.md`
  - 自动生成的 run snapshot；它只描述某次 run 产物
- `laptop-sl-loader-benchmark-2026-03-31.md`
  - 笔记本监督学习 loader 的当前证据
- `1v3-multishard-benchmark-2026-04-02.md`
  - 双机 `1v3` 并发参数的当前证据

### 3. 证据层：`research/`

- `supervised-evolution.md`
  - 监督学习阶段从实验探索到当前结构的统一演进记录
- `stage0/`
  - `GRP` 相关经验与候选对比
- `supervised/`
  - 监督学习阶段的工程经验、selector 统计与专项实验
- `rl-ppo-improvement-plan.md`
  - 强化学习阶段的研究性方案草稿

### 4. 背景层：`reflections/` 与 `archive/`

- `reflections/`
  - 复盘、人机协同方法、个人研究判断
- `archive/`
  - 已退役入口、旧快照、旧长文

## 当前命名族

- 监督学习阶段脚本：`sl_*`
- 监督学习阶段日志：`logs/sl_*`
- 监督学习阶段 checkpoint：`sl_canonical*.pth`

## 使用规则

- `current-plan.md` 只回答“现在停在哪、接下来做什么”
- `mainline.md` 只回答“当前设计是什么”
- `supervised-verified-status.md` 只回答“当前真实结论是什么”
- `p1-selection-canonical.md` 只回答“`P1` 如何比较和解释”
- `supervised-formal-triplet-playoff-canonical.md` 只回答“正式 winner 如何由 triplet / `formal_1v3` 产生”
- `supervised-evolution.md` 统一负责“监督学习阶段是如何一步步演进到当前结构的”
- 自动摘要、研究长文、复盘文档和归档文档都不能覆盖入口层与状态层
