# 监督学习阶段演进记录

这份文档统一记录监督学习阶段如何一步步形成当前结构。活跃入口文档只描述当前设计；演进过程、方案切换和收敛路径都集中放在这里。

## 当前结构

监督学习阶段当前采用以下主线：

1. `P0`
2. `P1 calibration`
3. `P1 protocol_decide`
4. `P1 winner_refine`
5. `formal_train`
6. `formal_1v3`

当前官方结论：

- 协议 winner：`C_A2x_cosine_broad_to_recent_strong_24m_12m`
- winner 点位：`0.12 + A2x`
- supervised winner：`anchor*1.0`
- 第一替补：`opp_lean*0.85`
- canonical checkpoint：`./checkpoints/sl_canonical.pth`

## 演进主线

### 1. `P0` 入口池收敛

- 早期监督学习探索先围绕候选协议池、训练脚手架和验证口径建立可比较的入口池
- 经过清洗与复核后，`P0` 的稳定入口收敛为：
  - `A2y`
  - `A2x`
  - `A1x`
- 这一步解决的是“哪些协议值得进入 `P1`”

### 2. `P1` 从多轮筛选结构收敛为三段主线

- 监督学习阶段一度尝试过更细的分层筛选结构
- 随着对指标职责和 downstream 决策边界的梳理，`P1` 收敛为：
  - `calibration`
  - `protocol_decide`
  - `winner_refine`
- 这一步解决的是“定标、选协议、调配比”三件事的职责分离

### 3. `calibration` 与 downstream 冠军角色分离

- `calibration` 最终承担统一 budget 量纲与估计 combo factor 的职责
- 当前定标结构固定为 `A2y-only + combo_only`
- 这一步明确了“定标代表协议”和“downstream winner 协议”是两种角色

### 4. `protocol_decide` 收敛到 `A2x @ 0.12`

- `protocol_decide` 在统一三头脚手架下比较协议承载能力
- 多 seed 结果与补跑验证完成后，协议 winner 固定为：
  - `A2x`
- 当前写入默认文档的 winner 点位固定为：
  - `0.12 + A2x`

### 5. `winner_refine` 收敛到 pre-formal 第一梯队

- 在协议 winner 固定后，`winner_refine` 只在 `A2x` 内部继续
- center 集合收敛为：
  - `anchor`
  - `opp_lean`
  - `rank_lean`
  - `danger_lean`
- pre-formal 第一梯队最终收敛为：
  - `opp_lean*0.85`
  - `anchor*1.0`
  - `opp_lean(rank--/danger++)`

### 6. formal triplet 与 `formal_1v3` 负责官方 winner

- 监督学习阶段把第一梯队送入 formal triplet
- child formal run 完成后，再通过 cross-run `formal_1v3` 产生官方 supervised winner
- 这一层最终固定的顺序为：
  1. `anchor*1.0`
  2. `opp_lean*0.85`
  3. `opp_lean(rank--/danger++)`

### 7. 监督学习阶段完成，项目转入 RL 方案定义

- 监督学习阶段的官方产物固定为当前 canonical checkpoint
- 后续项目工作转入强化学习阶段方案定义与起跑准备

## 关键设计收敛

### 角色分离

- `calibration`：负责定标
- `protocol_decide`：负责选协议
- `winner_refine`：负责 pre-formal 第一梯队
- `formal_1v3`：负责官方 winner

### 命名统一

- 文档语义：
  - `监督学习阶段`
  - `强化学习阶段`
- 监督学习阶段脚本与产物命名：
  - `run_sl_*`
  - `logs/sl_*`
  - `sl_canonical*.pth`

### 文档分层

- 入口与当前默认：`docs/agent/`
- 当前状态与结论：`docs/status/`
- 演进与方法：`docs/research/`
- 复盘与历史材料：`docs/reflections/` / `docs/archive/`

## 相关文档

- 当前停点：`docs/agent/current-plan.md`
- 当前设计：`docs/agent/mainline.md`
- 当前真实结论：`docs/status/supervised-verified-status.md`
- `P1` 评估口径：`docs/status/p1-selection-canonical.md`
- formal triplet / `formal_1v3` 证据：`docs/status/supervised-formal-triplet-playoff-canonical.md`
