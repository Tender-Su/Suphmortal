# Stage 0.5 工程手册

这份文档只保留当前仍有效、会直接影响后续实验质量的工程经验。旧的长版记录已归档到：

- `docs/archive/research/stage05/engineering-playbook-legacy-2026-03-29.md`

## 当前默认前提

- 当前默认主线以 `docs/agent/current-plan.md`、`docs/status/stage05-verified-status.md`、`docs/status/p1-selection-canonical.md` 为准
- `P1` 当前结构固定为：`calibration -> protocol_decide -> winner_refine`
- `P1 ablation` 当前只保留为 `backlog / manual only`
- 当前 `P0 top3` 事实顺序固定为 `A2y / A2x / A1x`
- 当前 downstream 默认协议固定为 `A2x`

## 当前仍有效的工程结论

### 1. 正式训练和筛选实验必须分开

- 正式训练入口用 `scripts/run_supervised.bat`
- 单轮协议 A/B 用 `mortal/run_stage05_ab.py`
- 串联保真流程用 `mortal/run_stage05_fidelity.py`
- 不要把“为了筛选而开的 run”与“要留作正式下游候选的 run”混成同一目录

### 2. 重要重启必须用全新输出目录

- 不要复用旧 run 目录继续解释新实验
- 旧缓存、旧摘要字段和旧默认都可能滞后
- 同名 run 需要显式停掉后重开，不做隐式接续

### 3. 当前默认必须写成显式规则，不能靠口头约定

- `protocol_decide` 当前默认 seed2 扩展规则是 `flip_or_gap @ 0.001`
- `winner_refine` 当前默认是从 rerun `protocol_decide` effective-coordinate 排名里取 `top4`
- 这些规则必须同时写进代码默认、测试和主线文档
- 如果未来要改，必须显式更新 `stage05_current_defaults.py` 与对应文档，而不是让脚本默默漂移

### 4. run snapshot 不等于当前默认

- `docs/status/stage05-fidelity-results.md` 只是某次 run 的快照
- 当前默认解释必须优先看 `current-plan / verified-status / p1-selection-canonical`
- 历史 run 里出现旧 `ambig` 或旧 center 口径，不代表它们仍然是当前默认

### 5. 先保证公平，再谈 winner

- selector、摘要字段、重验逻辑一旦改了，历史 `arm_result.json` 必须支持重算或被明确标成历史快照
- `P1` 的 winner 解释必须统一走 `docs/status/p1-selection-canonical.md`
- 自动摘要只能做辅助，不代替人工核对

### 6. invalid arm 不补跑，会系统性扭曲证据

- 多 seed round 如果还有 invalid arm，不要直接把“最后一名”当真实弱点
- 当前这轮 `protocol_decide` 里，唯一失败的 `A2x danger_lean 0.12` 在补跑前被压到总榜第 `27`
- 补跑成功后，它回到了总榜第 `6`
- 经验规则是：只要补跑成本还合理，就先把 invalid arm 补齐，再正式收口

### 7. calibration 代表协议与 downstream winner 可以不同

- 当前瘦版 `calibration` 仍固定用 `A2y`
- 这是定标角色，不是 downstream winner
- 当前真正经过 `protocol_decide` 验证、要写死到下游默认里的协议是 `A2x`
- 写默认时必须把这两个角色拆开，避免新 agent 把 `A2y-only calibration` 误读成“当前主线仍是 A2y”

### 8. Windows 快路径要优先保护

- 当前训练默认：`4 / 10 / 3`
- 当前验证默认：`8 / 5`
- 验证出问题时，优先修 worker 生命周期和共享映射释放
- 不要因为一时资源错误就把验证永久降级成单进程慢路径

## 当前推荐阅读顺序

1. `docs/status/stage05-verified-status.md`
2. `docs/status/p1-selection-canonical.md`
3. 本文档
4. `docs/research/stage05/selector-stat-audit.md`
5. 需要专项背景时，再看同目录下的日期文档
