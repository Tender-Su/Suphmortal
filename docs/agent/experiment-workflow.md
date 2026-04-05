# 当前实验流程

这份文档只回答“现在应该怎么跑”，不重复保存所有冻结常量。  
冻结默认看 `docs/agent/mainline.md`。  
实时停点看 `docs/agent/current-plan.md`。  
当前 `formal triplet playoff` 口径看 `docs/status/formal-triplet-playoff-canonical.md`。

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
   - 当前只保留 pre-formal 第一梯队与内部排序证据
   - 当前正式 `P1 winner` 已由后续 `formal triplet playoff / formal_1v3` 固定为 `anchor*1.0`
   - 当前 center 与局部搜索规则看 `docs/agent/mainline.md`
5. `formal_train`
   - 目标：用当前 downstream 正式候选的完整 recipe 跑正式版 Stage `0.5` supervised
   - 这一步只产出 checkpoint pack：`best_loss / best_acc / best_rank`
   - 当前 `latest` 不再进入后续发布链路
   - 当前 `formal` 有效长度已上调为：`45000 / 30000 / 15000`
6. `formal_1v3`
   - 目标：在正式 checkpoint pack 上用 `1v3` 决定最终 canonical winner
   - 判定口径：`avg_pt` 为主，`avg_rank` 为辅
   - 流程：先 `1 iter` 粗筛；若 `top2` 仍 close，再换新 `seed_key` 连跑 `3~5` 轮
7. `P1 ablation`
   - 当前不属于默认主线
   - 目标仍然是验证 `all_three / drop_* / ce_only` 的边际贡献
   - 只作为 `backlog / manual only` 保留

## 当前人工停点

- 当前默认已停在 `manual formal triplet playoff + cross-run formal_1v3` 收口点
- 当前 winner 已固定为 `anchor*1.0`
- 当前第一替补已固定为 `opp_lean*0.85`
- 默认不再自动重跑 `manual formal triplet playoff`
- `formal_train` 跑完后也再次停下
- 默认不再把 `ablation` 当成写入下游默认的前置条件

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
- 它只改变调度方式，不改变候选空间和评估口径

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

## formal triplet 当前推荐跑法

- 当前这一步是 manual downstream playoff，不会回写 `winner_refine` 内部排序
- 当前 `3` 个候选与解释边界固定看：
  - `docs/status/formal-triplet-playoff-canonical.md`
- 当前推荐由桌面机统一调度：

```powershell
python mortal/run_stage05_formal_distributed.py dispatch `
  --run-name s05_formal_triplet_20260405 `
  --source-run-name s05_fidelity_p1_top3_cali_slim_20260329_001413 `
  --candidate-arm 'opp_lean*0.85' `
  --candidate-arm 'anchor*1.0' `
  --candidate-arm 'opp_lean(rank--/danger++)'
```

- `run_stage05_formal_distributed.py` 会在 source run 的 `winner_refine_round` 内部把这些 alias 解析回实际 arm id

- 查看状态：

```powershell
python mortal/run_stage05_formal_distributed.py status `
  --run-name s05_formal_triplet_20260405
```

- 暂停某个 worker：

```powershell
python mortal/run_stage05_formal_distributed.py pause-worker `
  --run-name s05_formal_triplet_20260405 `
  --worker-label laptop
```

- 恢复某个 worker：

```powershell
python mortal/run_stage05_formal_distributed.py resume-worker `
  --run-name s05_formal_triplet_20260405 `
  --worker-label laptop
```

## formal_1v3 当前推荐跑法

- `formal_train` 完成后，不要直接手工做 canonical 落位
- 如果当前在跑 `formal triplet playoff`，就对每个 child formal run 分别复用下面这条入口
- 当前推荐仍由桌面机统一调度：

```powershell
python mortal/run_stage05_formal_1v3_distributed.py dispatch `
  --run-name <run_name>
```

- 查看状态：

```powershell
python mortal/run_stage05_formal_1v3_distributed.py status `
  --run-name <run_name>
```

- 暂停或恢复某个 worker 的入口与 `winner_refine` 相同，只是脚本名改为 `run_stage05_formal_1v3_distributed.py`

## Stage 1 当前推荐跑法

- `Stage 1` 不再做 `recipe / S1-A~S1-D` screening
- 当前固定 recipe：
  - `CE + rank aux + opponent_state aux + danger aux`
  - 直接继承 canonical `Stage 0.5` winner `anchor*1.0`
- 当前要筛的只有：
  - `oracle-dropout` 的 decay ratio
  - `linear` vs `cosine`
- 当前 profile runner 已改成 `Suphx` 风格两段式：
  - `oracle-dropout transition`
  - `gamma=0 normal continuation`

### 查看可用 profiles

```powershell
python mortal/run_stage1_ab.py --list-arms
```

### 默认 screening

```powershell
python mortal/run_stage1_ab.py `
  --ab-name stage1_profile_screen
```

- 默认 shortlist：
  - `linear_075`
  - `cosine_075`
  - `linear_050`
  - `cosine_050`

### 带 control 的 screening

```powershell
python mortal/run_stage1_ab.py `
  --ab-name stage1_profile_screen_ctrl `
  --include-controls
```

- `--include-controls` 会额外加入：
  - `linear_100`
  - `cosine_100`

### 单 profile 正式长跑

```powershell
python mortal/run_stage1_ab.py `
  --ab-name stage1_formal_linear_075 `
  --profile-arm linear_075 `
  --step-scale 1.0
```

- 如果只想跑低层单配置 baseline，也可以直接用：
  - `scripts/run_stage1_refine.bat`
- 但那条入口不会自动拆成 `transition + continuation` 两段，因此不是当前主线首选

## 结果解释边界

- 是否入围、谁留在第一梯队，一律按 `docs/status/p1-selection-canonical.md`
- `protocol_decide` 只回答“哪个协议继续往下走”
- `winner_refine` 只回答“winner 协议下三头该怎么配，以及谁还留在 pre-formal 第一梯队”
- `winner_refine` 当前历史内部 `top1` 是 `opp_lean*0.85`，但它不单独决定官方 `P1 winner`
- `formal triplet playoff` 负责把第一梯队 arm 送进长预算 formal，并把当前正式 winner 收口到 `anchor*1.0`
- `opp_lean*0.85` 当前固定为第一替补与争 `1` 型预备队
- `formal_train` 只回答“正式训练后三个 checkpoint 候选分别是谁”，不直接做 canonical 落位
- `formal_1v3` 才回答“哪一个 formal checkpoint 应该成为 canonical winner，并触发 canonical alias 落位”；当前已收口为 `anchor*1.0`
- `ablation` 只回答“三个头是否都还有边际贡献”，而且当前只作 backlog 诊断

## 不再使用的旧流程

- 旧 `solo -> pairwise -> joint refine` 仍有诊断价值
- 但它已经不是当前主线，也不能覆盖当前 `P1` 解释
