# 当前实验流程

这份文档只回答“当前主线怎么跑”。当前停点看 `docs/agent/current-plan.md`，当前设计看 `docs/agent/mainline.md`。

## 总体纪律

- 每个大阶段完成后人工确认
- 新 run 使用新输出目录
- run snapshot 记录 run 产物
- 默认不设置 `MORTAL_CPU_AFFINITY`

## 选择入口

- 正式 `GRP` 训练：

```powershell
.\scripts\run_grp.bat
```

- 正式监督学习阶段：

```powershell
.\scripts\run_supervised.bat
```

- 手动 `P1` 实验：

```powershell
.\scripts\run_sl_p1_only.bat
```

- 在线 RL：

```powershell
.\scripts\run_online.bat
```

## 监督学习阶段流程

1. `P0`
   - 目标：产出 `P1` 入口 `top3`
2. `P1 calibration`
   - 目标：统一三头 budget 量纲并估计 combo factor
3. `P1 protocol_decide`
   - 目标：确定下游协议 winner
4. `P1 winner_refine`
   - 目标：在 winner 协议内部产出 pre-formal 第一梯队
5. `formal_train`
   - 目标：产出 checkpoint pack：`best_loss / best_acc / best_rank`
6. `formal_1v3`
   - 目标：在 checkpoint pack 上确定 canonical winner
7. `P1 ablation`
   - 目标：手动诊断三头边际贡献

## 人工确认点

- `P0` 结束后确认
- `P1 protocol_decide` 结束后确认
- `P1 winner_refine` 结束后确认
- `formal_train` 结束后确认
- `formal_1v3` 结束后确认

## 当前分布式入口

### `winner_refine`

```powershell
python mortal/run_sl_winner_refine_distributed.py dispatch `
  --run-name sl_fidelity_p1_top3_cali_slim_20260329_001413
```

### formal triplet

```powershell
python mortal/run_sl_formal_distributed.py dispatch `
  --run-name sl_formal_triplet_20260405 `
  --source-run-name sl_fidelity_p1_top3_cali_slim_20260329_001413 `
  --candidate-arm 'opp_lean*0.85' `
  --candidate-arm 'anchor*1.0' `
  --candidate-arm 'opp_lean(rank--/danger++)'
```

### `formal_1v3`

```powershell
python mortal/run_sl_formal_1v3_distributed.py dispatch `
  --run-name <run_name>
```

## 结果解释边界

- `protocol_decide` 回答协议 winner
- `winner_refine` 回答 pre-formal 第一梯队
- `formal_train` 回答 checkpoint pack
- `formal_1v3` 回答 canonical winner
- `P1` 评估口径见 `docs/status/p1-selection-canonical.md`
- formal triplet / `formal_1v3` 证据见 `docs/status/supervised-formal-triplet-playoff-canonical.md`

## 当前入口命名

- 监督学习阶段入口使用 `run_sl_*` 系列脚本
- 当前 supervised checkpoint 使用 `sl_canonical*.pth` 命名族
