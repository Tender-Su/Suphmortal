# Stage 0.5 工程手册

这份文档只保留当前仍有效、会直接影响后续实验质量的工程经验。旧的长版记录已归档到：

- `docs/archive/research/stage05/engineering-playbook-legacy-2026-03-29.md`

## 这份文档解决什么问题

- 正式训练和 A/B 该怎么分工
- 当前有哪些工程纪律已经被证明确实重要
- 哪些历史坑现在仍然需要主动避免

## 当前默认前提

- 当前默认主线以 `docs/agent/current-plan.md` 与 `docs/status/p1-selection-canonical.md` 为准
- `P1` 当前结构固定为：`calibration -> protocol_decide -> winner_refine -> ablation`
- 当前 `P0 top3` 以 clean rerun 后的 `A2y / A2x / A1x` 为准

## 当前仍有效的工程结论

### 1. 正式训练和筛选实验必须分开

- 正式训练入口用 `scripts/run_supervised.bat`
- 单轮协议 A/B 用 `mortal/run_stage05_ab.py`
- 串联保真流程用 `mortal/run_stage05_fidelity.py`
- 不要把“为了筛选而开的 run”与“要留作正式下游候选的 run”混成同一目录

### 2. 重要重启必须用全新输出目录

- 不要复用旧 run 目录继续解释新实验
- 旧缓存、旧排序键和旧摘要都可能滞后
- 同名 run 需要显式停掉后重开，不做隐式接续

### 3. 先保证公平，再谈 winner

- selector、摘要字段、重验逻辑一旦改了，历史 `arm_result.json` 需要支持重算
- `P1` 的 winner 解释必须走 `docs/status/p1-selection-canonical.md`
- 自动摘要只能做辅助，不代替人工核对

### 4. Windows 快路径要优先保护

- 当前训练默认：`4 / 10 / 3`
- 当前验证默认：`8 / 5`
- 验证出问题时，优先修 worker 生命周期和共享映射释放
- 不要因为一时资源错误就把验证永久降级成单进程慢路径

### 5. 不要重新引入已证实会拖慢的绕路

- 不要回退到错误的 `numpy collate` 绕路
- 不要把重型 action/scenario 统计重新塞回训练热路径
- 训练热路径只保留轻量优化与基本监控；完整 selection 指标属于验证

### 6. 选模不能只看总 loss

- `P0 / P2 / formal` 与 `P1` 使用的是不同层次的比较口径
- 当前 `P1` 只按 `policy_quality` 规则解释，不把各类 `acc` 再塞回排序键
- `full_recent_loss` 在 `P1` 里只是诊断字段，不是主判胜字段

## 当前推荐阅读顺序

1. `docs/status/stage05-verified-status.md`
2. `docs/status/p1-selection-canonical.md`
3. 本文档
4. `docs/research/stage05/selector-stat-audit.md`
5. 需要专项背景时，再看同目录下的日期文档

## 仍有参考价值的专项研究

- `p1-aux-adjustment-2026-03-22.md`
- `a2y-aux-shape-freeze-2026-03-25.md`
- `selector-stat-audit.md`
