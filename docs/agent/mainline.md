# 当前主线与冻结默认

这份文档只保留当前设计和冻结默认。当前停点看 `docs/agent/current-plan.md`，当前真实结论看 `docs/status/supervised-verified-status.md`。

## 命名族

- 项目主线命名：`监督学习阶段` / `强化学习阶段`
- 监督学习阶段脚本族：`run_sl_*`
- 监督学习阶段日志族：`logs/sl_*`
- 监督学习阶段 checkpoint 族：`sl_canonical*.pth`

## 项目主线

- 项目由两个阶段构成：`监督学习阶段` → `强化学习阶段`
- `GRP` 是监督学习阶段的前置辅助模型
- 监督学习阶段内部结构：`P0 -> P1 -> formal_train -> formal_1v3`
- `P1` 主线结构：`calibration -> protocol_decide -> winner_refine`
- `P1 ablation`：手动诊断轮

## 仓库与机器默认

- canonical branch：`main`
- 源码真源：台式机 `main` 工作树
- 双机同步入口：`docs/agent/code-sync.md`
- 主节点：`i5-13600KF + RTX 5070 Ti`
- 副节点：`i9-13900HX + RTX 4060 Laptop GPU 8GB + 32GB DDR5`

### 监督学习 loader 默认

- 台式机：train `4/10/3`，val `8/5`
- 笔记本：train `4/10/4`，val `7/5`

证据来源：笔记本见 `docs/status/laptop-sl-loader-benchmark-2026-03-31.md`；台式机为本地基准确认值。

### `1v3` 默认

- 台式机 `RTX 5070 Ti`：`seed_count=1024, shard_count=4`
- 笔记本 `RTX 4060 Laptop GPU`：`seed_count=640, shard_count=3`

证据来源：`docs/status/1v3-multishard-benchmark-2026-04-02.md`。

## `GRP` 冻结默认

- 默认主线：`384x3 fp32`
- 默认下游 checkpoint：`best_loss`
- `best_acc`：受控对照 checkpoint
- `latest`：续训 checkpoint

## 监督学习阶段冻结默认

### `P0`

- `top3` 顺序固定为：
  - `C_A2y_cosine_broad_to_recent_strong_12m_6m`
  - `C_A2x_cosine_broad_to_recent_strong_24m_12m`
  - `C_A1x_cosine_broad_to_recent_mild_24m_12m`

### `P1` 各阶段冻结参数

搜索默认、center 规则和局部细搜规则的完整定义见 `docs/status/p1-selection-canonical.md`。

### `formal_train` 与 `formal_1v3`

- 发布结构：`formal_train -> checkpoint pack(best_loss/best_acc/best_rank) -> formal_1v3 -> canonical alias 落位`
- `formal_train` 产物：`best_loss / best_acc / best_rank`
- formal 长度：`phase_a/phase_b/phase_c = 45000/30000/15000`
- 证据入口：`docs/status/supervised-formal-triplet-playoff-canonical.md`

### supervised checkpoint 规则

- canonical checkpoint：`./checkpoints/sl_canonical.pth`
- `best_loss_state_file` 与 `best_state_file` 指向当前 `formal_1v3` 胜者
- `best_acc / best_rank` 作为 secondary candidate 保留

## 强化学习阶段默认

- RL 默认起点：当前 canonical supervised winner
- 启动顺序：
  - 先读 `[control].state_file = ./checkpoints/mortal.pth`
  - 其次读 `[online].init_state_file = ./checkpoints/sl_canonical.pth`
- 当前工作重点：强化学习阶段方案定义
