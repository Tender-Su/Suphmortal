# 笔记本远程操作默认

这份文档只回答远程运行本身，不解释当前 winner 或当前实验结论。代码同步细节看 `docs/agent/code-sync.md`。

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
- 不要把 SSH `Session 0` 里的后台进程当正式 benchmark 口径

## 数据根

- 当前笔记本数据根：`C:\Users\numbe\mahjong_data_root`
- 当前保留的数据目录：
  - `dataset_rebuilt`
  - `dataset_json_rebuilt`
- 已退役目录：
  - `dataset`
  - `dataset_json`
  - `dataset_bad_20260330`
  - `dataset_rebuilt_stale_*`
  - `dataset_json_rebuilt_stale_*`

## 当前笔记本默认

### 监督学习 loader

- train：`num_workers = 4`
- train：`file_batch_size = 10`
- train：`prefetch_factor = 4`
- val：`val_file_batch_size = 7`
- val：`val_prefetch_factor = 5`

这组默认来自 `docs/status/laptop-sl-loader-benchmark-2026-03-31.md`。

### `1v3`

- `seed_count = 640`
- `shard_count = 3`

这组默认来自 `docs/status/1v3-multishard-benchmark-2026-04-02.md`。

## 双机任务的运行资产

### `winner_refine` / `formal_1v3`

必备资产：

- 当前代码已经同步到笔记本 Git 工作树
- `mortal/config.toml`
- `logs/sl_fidelity/<run_name>/state.json`
- `mortal/checkpoints/file_index_supervised_json.pth`

### formal triplet

必备资产：

- 当前代码已经同步到笔记本 Git 工作树
- `mortal/config.toml`
- `mortal/checkpoints/file_index_supervised_json.pth`
- source run 已经具备可解析的 `winner_refine_round`

说明：

- `run_sl_formal_distributed.py` 会把 dispatch 状态同步到远端
- child formal 完成后也会把 child run 与 `sl_ab/<child_run_name>_formal` 产物拉回台式机

## 当前推荐远程调用模式

### 短命令

```powershell
ssh -i "$HOME\.ssh\mahjong_laptop_ed25519" numbe@<laptop-ip> "Get-Location"
ssh -i "$HOME\.ssh\mahjong_laptop_ed25519" numbe@<laptop-ip> "Get-ChildItem 'C:\Users\numbe\mahjong_data_root' -Directory"
```

### 长脚本

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

- `run_sl_ab.load_all_files()` 会先读 `file_index_supervised_json.pth`
- 所以后续 agent 启动双机任务前，不能只查代码和 `state.json`

## 起手检查

```powershell
ssh -i "$HOME\.ssh\mahjong_laptop_ed25519" numbe@<laptop-ip> "Get-Location"
ssh -i "$HOME\.ssh\mahjong_laptop_ed25519" numbe@<laptop-ip> "Get-ChildItem 'C:\Users\numbe\mahjong_data_root' -Directory"
ssh -i "$HOME\.ssh\mahjong_laptop_ed25519" numbe@<laptop-ip> "Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' } | Select-Object ProcessId,CommandLine | Format-Table -AutoSize"
```
