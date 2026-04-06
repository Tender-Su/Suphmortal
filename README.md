# MahjongAI

面向本机 `i5-13600KF + RTX 5070 Ti` 的麻将 AI 训练仓库。项目目标是最终牌力优先。

## 当前状态

- 项目主线由两个阶段构成：
  - `监督学习阶段`
  - `强化学习阶段`
- 监督学习阶段的官方结果已经固定：
  - canonical winner：`anchor*1.0`
  - 第一替补：`opp_lean*0.85`
  - canonical checkpoint：`./checkpoints/sl_canonical.pth`
- 项目当前位于强化学习方案定义与起跑准备阶段
- 强化学习 baseline 默认从当前 canonical supervised checkpoint 起跑

## 项目结构

- `libriichi/`：Rust 麻将引擎、规则、特征提取、PyO3 扩展
- `mortal/`：PyTorch 模型、训练脚本、A/B 工具、在线自博弈
- `scripts/`：项目入口脚本与辅助工具
- `checkpoints/`：模型权重与训练状态
- `logs/`：实验日志与正式 run 产物
- `docs/`：入口文档、状态结论、研究记录、复盘与归档

## 文档入口

默认读取顺序：

1. `AGENTS.md`
2. `docs/agent/current-plan.md`
3. `docs/agent/mainline.md`
4. `docs/status/supervised-verified-status.md`
5. `docs/status/p1-selection-canonical.md`

按需补充：

- formal triplet / `formal_1v3` 证据：`docs/status/supervised-formal-triplet-playoff-canonical.md`
- 双机运行与远程 shell：`docs/agent/laptop-remote-ops.md`
- 台式机到笔记本的 Git 同步：`docs/agent/code-sync.md`
- 监督学习阶段演进过程：`docs/research/supervised-evolution.md`
- 研究与证据索引：`docs/README.md`

## 当前命名族

- 监督学习阶段脚本：`run_sl_*`
- 监督学习阶段日志目录：`logs/sl_*`
- 监督学习阶段 checkpoint：`sl_canonical*.pth`

## 快速开始

### 1. 环境

```powershell
conda env create -f environment.yml
conda activate mortal
```

### 2. 编译引擎

```powershell
.\scripts\build_libriichi.bat
python -c "import libriichi; print('OK')"
```

### 3. 本地配置

- 以 `mortal/config.example.toml` 为模板生成本地 `mortal/config.toml`
- 填写本机数据路径
- 不要提交真实路径

### 4. 训练入口

```powershell
.\scripts\run_grp.bat
.\scripts\run_supervised.bat
.\scripts\run_online.bat
```

手动入口：

- `.\scripts\run_sl_p1_only.bat`
  - 用于 `P1` selector / refine 实验

## 结果与运维

- 监督学习主流程日志：`logs/sl_fidelity/`
- 监督学习 A/B 与 formal child run：`logs/sl_ab/`
- 自动生成的监督学习 run snapshot：`docs/status/supervised-fidelity-results.md`
- 当前人工核对后的真实结论：`docs/status/supervised-verified-status.md`
- 当前唯一有效的 `P1` 评估口径：`docs/status/p1-selection-canonical.md`
- 当前 formal triplet / `formal_1v3` 证据：`docs/status/supervised-formal-triplet-playoff-canonical.md`

## 工程约定

- 项目入口统一在 `scripts/`
- 当前默认、当前停点和当前操作纪律由 `docs/agent/` 与 `docs/status/` 提供
- 研究长文提供方法、证据与演进背景
- 特征通道数与架构硬约束见 `AGENTS.md`
