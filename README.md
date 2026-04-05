# MahjongAI

面向本机（`i5-13600KF + RTX 5070 Ti`）的麻将 AI 训练仓库。当前目标是做出尽可能强的模型，而不是最低训练成本的版本。

## 文档布局

- `README.md`：项目简介、快速启动、训练入口
- `docs/README.md`：给人看的总文档导航
- `AGENTS.md`：给 agent 的权威工程规范、架构常量与硬约束
- `docs/agent/current-plan.md`：给 agent 的短续工入口
- `docs/agent/mainline.md`：当前稳定默认、阶段地图与冻结结论
- `docs/agent/experiment-workflow.md`：当前实验流程、人工停点与运行纪律
- `CLAUDE.md`：其他代理的最小读取顺序

根目录只保留项目入口和 agent 入口；研究长文、状态快照、复盘和历史归档统一放进 `docs/`。

## 项目结构

- `libriichi/`：Rust 麻将引擎、规则、特征提取、PyO3 扩展
- `mortal/`：PyTorch 模型、训练脚本、A/B 工具
- `scripts/`：唯一保留的入口与辅助脚本目录
- `checkpoints/`：权重与中间产物
- `docs/`：研究文档、状态快照、复盘与 agent 当前计划
- `logs/`：训练、A/B、正式实验日志

## 当前训练主线

1. `GRP`：监督学习阶段的前置辅助模型
2. `监督学习阶段`：内部结构固定为 `P0 -> P1 -> formal_train -> formal_1v3`，正式入口为 `.\scripts\run_supervised.bat`
3. `强化学习阶段`：PPO 在线自博弈，入口为 `.\scripts\run_online.bat`

当前监督学习阶段已经收口：

- 当前正式 winner：`anchor*1.0`
- 当前第一替补：`opp_lean*0.85`
- 监督学习阶段不再继续做 Oracle 路线
- 强化学习阶段方案仍在研究，暂不在 README 里写死
- 当前 RL 入口默认会先续跑 `./checkpoints/mortal.pth`，不存在时再从 canonical supervised checkpoint `./checkpoints/stage0_5_supervised.pth` 起跑

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

## 监控与结果

- 监督学习正式训练与协议 A/B 日志位于 `logs/stage05_ab/`
- 保真串联流程日志位于 `logs/stage05_fidelity/`
- 每个具体 run 的 TensorBoard 位于对应目录下的 `tb/`
- `GRP` 的 TensorBoard 位于 `mortal/tb_log_grp`
- 人工核对后的当前进度以 `docs/status/supervised-verified-status.md` 为准
- `P1` 的唯一有效选模口径见 `docs/status/p1-selection-canonical.md`
- selector 哪些部分已可统计化见 `docs/research/stage05/selector-stat-audit.md`
- 自动生成的监督学习阶段 run snapshot 保留在 `docs/status/supervised-fidelity-results.md`
- `scripts/` 目录只保留当前支持的训练入口与在用辅助脚本，不再推荐历史 GRP helper

## 说明

- 只使用 `scripts/` 下的批处理入口；根目录旧 wrapper 已移除
- 不要修改特征通道数，相关约束见 `AGENTS.md`
- 当前训练策略会持续演进，默认口径与路线图以 `docs/agent/current-plan.md` 为准
- 监督学习阶段当前仍有效的工程经验见 `docs/research/stage05/engineering-playbook.md`
