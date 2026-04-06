# 监督学习阶段工程手册

这份文档只记录当前仍有效、会直接影响实验质量的工程经验。监督学习阶段如何形成当前结构，统一看 `docs/research/supervised-evolution.md`。

## 当前默认前提

- 当前主线入口：
  - `docs/agent/current-plan.md`
  - `docs/status/supervised-verified-status.md`
  - `docs/status/p1-selection-canonical.md`
- 当前 formal triplet / `formal_1v3` 证据：
  - `docs/status/supervised-formal-triplet-playoff-canonical.md`
- `P1` 结构：
  - `calibration -> protocol_decide -> winner_refine`
- `P1 ablation`：
  - 手动诊断轮
- `P0 top3`：
  - `A2y / A2x / A1x`
- downstream 协议：
  - `A2x`

## 当前工程结论

### 1. 正式训练和筛选实验分目录运行

- 正式训练入口：`scripts/run_supervised.bat`
- 单轮协议 A/B：`mortal/run_sl_ab.py`
- 串联保真流程：`mortal/run_sl_fidelity.py`
- 正式训练目录与筛选实验目录分别承载不同职责

### 2. 重要重启使用全新输出目录

- 新 run 对应新输出目录
- 同名 run 先显式停掉，再重新启动

### 3. 默认规则写进代码、测试和主线文档

- `protocol_decide` seed2 扩展规则：
  - `flip_or_gap @ 0.001`
- `winner_refine` center 选择规则：
  - `top_ranked_keep = 4`
- 这些规则同时落在代码默认、测试和主线文档中

### 4. run snapshot 只承担 run 级别说明

- `docs/status/supervised-fidelity-results.md` 记录 run 产物
- 当前默认解释由 `docs/agent/` 与 `docs/status/` 提供

### 5. `P1` 的比较、第一梯队和 downstream 入口使用同一套口径

- `P1` 的排序、第一梯队与 downstream 入口解释统一走 `docs/status/p1-selection-canonical.md`
- 自动摘要只提供辅助信息

### 6. invalid arm 先补齐，再比较结论

- 多 seed round 中的 invalid arm 需要补齐
- 补齐后再解释排序与 winner

### 7. calibration、协议 winner 与官方 winner 分层解释

- `calibration`：定标角色
- `protocol_decide`：协议 winner
- `winner_refine`：pre-formal 第一梯队
- `formal_1v3`：官方 winner

### 8. Windows 快路径优先保护

- 当前训练默认：`4 / 10 / 3`
- 当前验证默认：`8 / 5`
- 验证异常优先处理 worker 生命周期与共享映射释放

### 9. 文档与 run artifact 统一使用结构别名

- `winner_refine` 和 downstream 文档统一使用：
  - `anchor`
  - `opp_lean`
  - `rank_lean`
  - `danger_lean`
  - `*0.85 / *1.0 / *1.15`
  - `rank-- / danger++`

## 当前推荐阅读顺序

1. `docs/status/supervised-verified-status.md`
2. `docs/status/p1-selection-canonical.md`
3. `docs/research/supervised-evolution.md`
4. 本文档
5. `docs/research/supervised/selector-stat-audit.md`
