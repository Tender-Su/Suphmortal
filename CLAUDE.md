# CLAUDE.md

给其他代理的最小入口说明。

## 读取顺序

1. `AGENTS.md`
2. `docs/agent/current-plan.md`
3. `docs/agent/mainline.md`
4. `docs/status/supervised-verified-status.md`
5. `docs/status/p1-selection-canonical.md`
6. 任务涉及 triplet / `formal_1v3` 时，再读 `docs/status/supervised-formal-triplet-playoff-canonical.md`
7. 任务涉及双机或远程运行时，再读 `docs/agent/laptop-remote-ops.md` 与 `docs/agent/code-sync.md`
8. 需要监督学习阶段演进背景时，再读 `docs/research/supervised-evolution.md`

## 最小操作约定

- 使用 `conda activate mortal`
- 构建入口：`.\scripts\build_libriichi.bat`
- 当前监督学习阶段主入口：`.\scripts\run_supervised.bat`
- 当前强化学习阶段主入口：`.\scripts\run_online.bat`
- `.\scripts\run_sl_p1_only.bat` 用于手动 `P1` 实验
- 监督学习阶段采用 `run_sl_*`、`logs/sl_*`、`sl_canonical*.pth` 这组标准命名
- 目标优先级：最终模型强度优先，其次才是训练效率和便利性

其余架构、常量、路径、硬件假设与训练约束均以 `AGENTS.md` 为准。
