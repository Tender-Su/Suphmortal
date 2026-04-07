# MahjongAI

面向本机 `i5-13600KF + RTX 5070 Ti` 的麻将 AI 训练仓库。项目目标是最终牌力优先。

## 当前状态

- 监督学习阶段已完成，强化学习阶段方案定义进行中
- canonical winner：`anchor*1.0`，canonical checkpoint：`./checkpoints/sl_canonical.pth`
- 详见 `docs/status/supervised-verified-status.md`

## 项目结构

- `libriichi/`：Rust 麻将引擎、规则、特征提取、PyO3 扩展
- `mortal/`：PyTorch 模型、训练脚本、A/B 工具、在线自博弈
- `scripts/`：项目入口脚本与辅助工具
- `checkpoints/`：模型权重与训练状态
- `logs/`：实验日志与正式 run 产物
- `docs/`：入口文档、状态结论、研究记录、复盘与归档

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
.\scripts\run_grp.bat           # GRP 前置模型
.\scripts\run_supervised.bat    # 监督学习阶段
.\scripts\run_online.bat        # 强化学习阶段
.\scripts\run_sl_p1_only.bat    # 手动 P1 实验
```

## 文档入口

Agent 读取顺序见 `CLAUDE.md`。人类快速导航见 `docs/README.md`。
