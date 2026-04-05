# 笔记本远程操作默认

这份文档只回答四件事：

1. 笔记本现在怎么连
2. 数据根和仓库在哪
3. 当前笔记本默认 loader / 远端运行口径是什么
4. Windows + PowerShell + SSH 的稳定做法是什么

Git 同步细节不在这里，见 `docs/agent/code-sync.md`。

## 当前机器事实

- 笔记本主机名：`abandon`
- 笔记本仓库路径：`C:\Users\numbe\Desktop\MahjongAI`
- 笔记本 Python：`C:\Users\numbe\miniconda3\envs\mortal\python.exe`
- 台式机 SSH 别名：`mahjong-laptop`
- 台式机 SSH key：`$HOME\.ssh\mahjong_laptop_ed25519`
- 笔记本 OpenSSH 默认 shell：`PowerShell`

结论：

- 现在默认直接发 PowerShell 命令
- 不要再默认包一层 `powershell -Command`
- 也不要把 SSH `Session 0` 里的后台进程当正式 benchmark 口径

## 数据根

- 当前笔记本数据根：`C:\Users\numbe\mahjong_data_root`
- 当前保留的数据目录：
  - `dataset_rebuilt`
  - `dataset_json_rebuilt`
- 旧坏目录已经退役，不要再用：
  - `dataset`
  - `dataset_json`
  - `dataset_bad_20260330`
  - `dataset_rebuilt_stale_*`
  - `dataset_json_rebuilt_stale_*`

## 当前笔记本默认

### 监督学习阶段 loader

- train：`num_workers = 4`
- train：`file_batch_size = 10`
- train：`prefetch_factor = 4`
- val：`val_file_batch_size = 7`
- val：`val_prefetch_factor = 5`

这组默认来自 `2026-03-31` 的交互前台窗口复核，优先级高于旧的 `6 / 7 / 3` 与 `7 / 6` 口径。

### 1v3

- `seed_count = 640`
- `shard_count = 3`

## 当前双机 winner_refine 口径

- 不要让两台机器各自独立写同一个 `run_dir`
- 当前推荐由桌面机统一调度：

```powershell
python mortal/run_stage05_winner_refine_distributed.py dispatch `
  --run-name s05_fidelity_p1_top3_cali_slim_20260329_001413
```

- 当前远端默认：
  - 远端仓库：`C:\Users\numbe\Desktop\MahjongAI`
  - 远端 Python：`C:\Users\numbe\miniconda3\envs\mortal\python.exe`
  - 远端启动方式：笔记本交互会话里的可见窗口
  - remote screening 默认：`4 / 10 / 4`
  - remote val screening 默认：`7 / 5`

### 远端必备资产

- `mortal/config.toml`
- `logs/stage05_fidelity/<run_name>/state.json`
- `mortal/checkpoints/file_index_supervised_json.pth`

缺任何一个，远端 `winner_refine` 都可能直接失败。

## 当前双机 formal_1v3 口径

- 同样不要让两台机器各自独立写同一个 `run_dir`
- 当前推荐仍由桌面机统一调度：

```powershell
python mortal/run_stage05_formal_1v3_distributed.py dispatch `
  --run-name <run_name>
```

- 当前 `formal_1v3` 与 `winner_refine` 的远端依赖基本一致：
  - `mortal/config.toml`
  - `logs/stage05_fidelity/<run_name>/state.json`
- 但它额外依赖当前 run 已经存在 `formal` checkpoint pack；没有 `formal.status = completed` 时不要启动

## 当前双机 formal_train triplet 口径

- 当前已固定官方 `P1 winner = anchor*1.0`
- 当前已固定第一替补 = `opp_lean*0.85`
- 当前 formal triplet 仍由桌面机统一调度：

```powershell
python mortal/run_stage05_formal_distributed.py dispatch `
  --run-name s05_formal_triplet_20260405 `
  --source-run-name s05_fidelity_p1_top3_cali_slim_20260329_001413 `
  --candidate-arm 'opp_lean*0.85' `
  --candidate-arm 'anchor*1.0' `
  --candidate-arm 'opp_lean(rank--/danger++)'
```

- `run_stage05_formal_distributed.py` 会根据 source run 的 `winner_refine_round` 把这些 alias 解析回实际 arm id
- 当前远端默认：
  - 远端仓库：`C:\Users\numbe\Desktop\MahjongAI`
  - 远端 Python：`C:\Users\numbe\miniconda3\envs\mortal\python.exe`
  - 远端启动方式：笔记本交互会话里的可见窗口
  - remote formal train 默认：`4 / 10 / 4`
  - remote formal val 默认：`7 / 5`

### 远端必备资产

- 当前代码与脚本必须已经同步到笔记本 Git 工作树
- `mortal/config.toml`
- `mortal/checkpoints/file_index_supervised_json.pth`

与当前 `winner_refine` / `formal_1v3` 的一个关键区别是：

- `run_stage05_formal_distributed.py` 会主动把 dispatch 状态同步到远端
- 远端 formal 完成后，也会把 child run 与 `stage05_ab/<child_run_name>_formal` 产物拉回台式机
- 所以后续继续跑 child run 的 `formal_1v3` 时，默认直接在台式机本地 child run 上继续即可

## 最稳的远程操作方式

### 短命令

```powershell
ssh -i "$HOME\.ssh\mahjong_laptop_ed25519" numbe@<laptop-ip> "Get-Location"
ssh -i "$HOME\.ssh\mahjong_laptop_ed25519" numbe@<laptop-ip> "Get-ChildItem 'C:\Users\numbe\mahjong_data_root' -Directory"
```

### 复杂脚本

- 优先把脚本 `scp` 到远端再执行
- 不要默认依赖 `stdin -> powershell -Command -`

### 传文件

```powershell
scp -i "$HOME\.ssh\mahjong_laptop_ed25519" local.ps1 `
  "numbe@<laptop-ip>:C:/Users/numbe/Desktop/MahjongAI/logs/remote.ps1"
```

## 已确认的坑

### PowerShell 变量会被本地提前展开

- `$p`
- `$_`
- `$input`

规避方式：

- 短命令直接一行发
- 复杂脚本用 `scp`
- 必要时显式转义远端变量

### 命令太长会撞 Windows 上限

- 不要把大段脚本 base64 后直接塞进一条远端命令
- 长脚本优先 `scp`

### 不要误杀所有 `python.exe`

- 清理任务时按命令行过滤具体脚本
- 不要做全局 `python.exe` 粗暴清理

### 远端上下文恢复依赖文件索引

- `run_stage05_ab.load_all_files()` 会先读 `file_index_supervised_json.pth`
- 所以后续 agent 启动双机 `winner_refine` 或 `formal triplet` 前，不能只查代码和 `state.json`

## 后续 agent 的起手检查

```powershell
ssh -i "$HOME\.ssh\mahjong_laptop_ed25519" numbe@<laptop-ip> "Get-Location"
ssh -i "$HOME\.ssh\mahjong_laptop_ed25519" numbe@<laptop-ip> "Get-ChildItem 'C:\Users\numbe\mahjong_data_root' -Directory"
ssh -i "$HOME\.ssh\mahjong_laptop_ed25519" numbe@<laptop-ip> "Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' } | Select-Object ProcessId,CommandLine | Format-Table -AutoSize"
```
