# CLAUDE.md

给其他代理的最小入口说明；agent 入口保留在根目录与 `docs/agent/`，不要再把长文档混回根目录。

## 读取顺序

1. `AGENTS.md`
2. `docs/agent/current-plan.md`
3. `docs/status/stage05-verified-status.md`
4. `docs/status/p1-selection-canonical.md`
5. `docs/agent/mainline.md`
6. `docs/status/stage05-fidelity-results.md`
7. `docs/research/stage0/grp-experience.md`
8. `docs/research/stage05/engineering-playbook.md`

## 最小操作约定

- 使用 `conda activate mortal`
- 构建入口：`.\scripts\build_libriichi.bat`
- `Stage 0.5` 正式训练入口：`.\scripts\run_supervised.bat` 或 `python mortal/run_stage05_formal.py`
- 只使用 `scripts/` 下的批处理入口；根目录旧 wrapper 已移除
- 目标优先级：最终模型强度优先，其次才是训练效率和便利性

其余架构、常量、路径、硬件假设与训练约束均以 `AGENTS.md` 为准。
