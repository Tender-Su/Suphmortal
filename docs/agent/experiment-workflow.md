# 当前实验流程

这份文档只回答“现在应该怎么跑”，不重复保存所有冻结常量。  
冻结默认看 `docs/agent/mainline.md`。  
实时停点看 `docs/agent/current-plan.md`。

## 总体纪律

- 每个阶段结束后都停下来，等人工确认
- 默认不使用一键串到底
- 新 run 用新输出目录，不复用旧 run 解释新实验
- run snapshot 只记录那次 run 的产物，不代替当前默认
- 默认不设置 `MORTAL_CPU_AFFINITY`

## Stage 0.5 当前流程

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
   - 当前 center 与局部搜索规则看 `docs/agent/mainline.md`
5. `P1 ablation`
   - 目标：验证 `all_three / drop_* / ce_only` 的边际贡献
   - 必须等 `winner_refine` 结束并人工确认后再启动

## 当前人工停点

- 当前停在 `protocol_decide` 收口点
- 不自动进入 `winner_refine`
- `winner_refine` 跑完后再次停下
- `ablation` 跑完后再决定是否写入更下游默认

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

- 这条入口的执行语义是：
  - `seed1` 全量跑完全部候选
  - `seed2` 只补 `seed1` 后仍在竞争带里的候选
  - selector 默认：`min_keep = 4`、`selection_gap = 0.001`、`max_keep = 12`
- 它只改变调度方式，不改变候选空间和选模口径

### 运行中控制

- 查看调度状态：

```powershell
python mortal/run_stage05_winner_refine_distributed.py status `
  --run-name s05_fidelity_p1_top3_cali_slim_20260329_001413
```

- 暂停某个 worker：

```powershell
python mortal/run_stage05_winner_refine_distributed.py pause-worker `
  --run-name s05_fidelity_p1_top3_cali_slim_20260329_001413 `
  --worker-label laptop
```

- 暂停并中断某个 worker 的当前任务：

```powershell
python mortal/run_stage05_winner_refine_distributed.py pause-worker `
  --run-name s05_fidelity_p1_top3_cali_slim_20260329_001413 `
  --worker-label laptop `
  --stop-active
```

- 恢复某个 worker：

```powershell
python mortal/run_stage05_winner_refine_distributed.py resume-worker `
  --run-name s05_fidelity_p1_top3_cali_slim_20260329_001413 `
  --worker-label laptop
```

## 结果解释边界

- 是否入围、谁是 winner、谁只是诊断信息，一律按 `docs/status/p1-selection-canonical.md`
- `protocol_decide` 只回答“哪个协议继续往下走”
- `winner_refine` 只回答“winner 协议下三头该怎么配”
- `ablation` 只回答“三个头是否都还有边际贡献”

## 不再使用的旧流程

- 旧 `solo -> pairwise -> joint refine` 仍有诊断价值
- 但它已经不是当前主线，也不能覆盖当前 `P1` winner 的解释
