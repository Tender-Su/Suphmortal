# 当前实验流程

这份文档只回答“现在应该怎么跑”，不重复保存所有冻结常量。  
冻结默认看 `docs/agent/mainline.md`。  
实时停点看 `docs/agent/current-plan.md`。  
当前 triplet playoff 口径看 `docs/status/supervised-formal-triplet-playoff-canonical.md`。

## 总体纪律

- 每个阶段结束后都停下来，等人工确认
- 默认不使用一键串到底
- 新 run 用新输出目录，不复用旧 run 解释新实验
- run snapshot 只记录那次 run 的产物，不代替当前默认
- 默认不设置 `MORTAL_CPU_AFFINITY`

## 监督学习阶段当前流程

1. `P0`
   - 目标：选出值得进入 `P1` 的事实 `top3`
   - 当前已冻结，不需要为继续主线而重跑
2. `P1 calibration`
   - 目标：统一三头 budget 量纲并估计 combo factor
   - 当前默认：`A2y-only + combo_only`
   - 这一步只能定标，不能宣布 winner
3. `P1 protocol_decide`
   - 目标：在统一三头脚手架下尽早选出协议 winner
   - 当前主线这一步已经完成并收口
4. `P1 winner_refine`
   - 目标：只在 winner 协议内部细调三头全开配比
   - 当前只允许在 `A2x` 内部继续
   - 当前只保留 pre-formal 第一梯队与内部排序证据
5. `formal_train`
   - 目标：把当前正式候选送进长预算监督学习 formal
   - 这一步只产出 checkpoint pack：`best_loss / best_acc / best_rank`
   - 当前有效长度固定为：`45000 / 30000 / 15000`
6. `formal_1v3`
   - 目标：在 checkpoint pack 上用 `1v3` 决定最终 canonical winner
   - 判定口径：`avg_pt` 为主，`avg_rank` 为辅
7. `P1 ablation`
   - 当前不属于默认主线
   - 只作为 `backlog / manual only` 保留

## 当前人工停点

- 当前默认已停在 `manual formal triplet playoff + cross-run formal_1v3` 收口点
- 当前 winner 已固定为 `anchor*1.0`
- 当前第一替补已固定为 `opp_lean*0.85`
- 当前监督学习阶段已经完成
- 默认不再继续监督学习 Oracle 或旧 `stage1` 路线
- 强化学习阶段方案尚未写死，因此当前默认继续停在这里

## winner_refine 当前推荐跑法

### 单机

- 如果只在台式机跑，直接在当前主线 run 上启动 `winner_refine`
- 运行前不要再恢复任何旧 `winner_refine` 残留；当前主线 run 已完成启动前清理

### 双机

- 不要让两台机器各自独立写同一个 `run_dir`
- 当前推荐由桌面机统一调度：

```powershell
python mortal/run_stage05_winner_refine_distributed.py dispatch `
  --run-name s05_fidelity_p1_top3_cali_slim_20260329_001413
```

## formal triplet 当前推荐跑法

- 当前这一步是 manual downstream playoff，不会回写 `winner_refine` 内部排序
- 当前 `3` 个候选与解释边界固定看：
  - `docs/status/supervised-formal-triplet-playoff-canonical.md`
- 当前推荐由桌面机统一调度：

```powershell
python mortal/run_stage05_formal_distributed.py dispatch `
  --run-name s05_formal_triplet_20260405 `
  --source-run-name s05_fidelity_p1_top3_cali_slim_20260329_001413 `
  --candidate-arm 'opp_lean*0.85' `
  --candidate-arm 'anchor*1.0' `
  --candidate-arm 'opp_lean(rank--/danger++)'
```

## formal_1v3 当前推荐跑法

- `formal_train` 完成后，不要直接手工做 canonical 落位
- 当前推荐仍由桌面机统一调度：

```powershell
python mortal/run_stage05_formal_1v3_distributed.py dispatch `
  --run-name <run_name>
```

## 当前默认入口

- 正式监督学习入口：

```powershell
.\scripts\run_supervised.bat
```

- 在线 RL 入口：

```powershell
.\scripts\run_online.bat
```

说明：

- `run_supervised.bat` 仍然会调用现有的 `run_stage05_formal.py`，这是历史脚本名，不代表项目仍按 `stage0.5` 命名
- `run_online.bat` 会先尝试续跑 `./checkpoints/mortal.pth`；如果这个 RL checkpoint 不存在，就回退到 canonical supervised checkpoint `./checkpoints/stage0_5_supervised.pth`
- 旧 `run_stage1_refine.bat` / `run_stage1_ab.bat` 已退役，不再使用

## 结果解释边界

- 是否入围、谁留在第一梯队，一律按 `docs/status/p1-selection-canonical.md`
- `protocol_decide` 只回答“哪个协议继续往下走”
- `winner_refine` 只回答“winner 协议下三头该怎么配，以及谁还留在 pre-formal 第一梯队”
- `formal triplet playoff` 负责把第一梯队 arm 送进长预算 formal，并把当前正式 winner 收口到 `anchor*1.0`
- `opp_lean*0.85` 当前固定为第一替补与争 `1` 型预备队
- `formal_train` 只回答“正式训练后三个 checkpoint 候选分别是谁”，不直接做 canonical 落位
- `formal_1v3` 才回答“哪一个 formal checkpoint 应该成为 canonical winner，并触发 canonical alias落位”

## 不再使用的旧流程

- 旧 `stage1 oracle-dropout` 路线已退役
- 旧 `solo -> pairwise -> joint refine` 仍有诊断价值，但已经不是当前主线
