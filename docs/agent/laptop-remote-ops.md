# 笔记本远程操作备忘

这份文档给后续 Agent 用，目标是让新的会话能立刻明白：

- 笔记本现在能不能直接远程操作
- 数据集现在落在哪
- 笔记本当前默认该怎么跑 Stage 0.5 loader / 训练 benchmark
- Windows + PowerShell + SSH 这套链路里哪些坑已经踩过，哪些做法最稳

## 当前机器状态

- 笔记本主机名：`abandon`
- 笔记本仓库路径：`C:\Users\numbe\Desktop\MahjongAI`
- 笔记本 Python：`C:\Users\numbe\miniconda3\envs\mortal\python.exe`
- 桌面机到笔记本的 SSH：
  - `ssh -i "$HOME\.ssh\mahjong_laptop_ed25519" numbe@<laptop-ip>`
- 笔记本 OpenSSH 默认 shell 已改为 `PowerShell`
- 笔记本执行策略已放开：
  - `CurrentUser = Bypass`
  - `LocalMachine = Bypass`
- 笔记本 `LongPathsEnabled = 1`

结论：现在笔记本已经不是之前那个默认掉进 `cmd.exe` 的脆弱状态了。后续远程执行时，优先直接发 PowerShell 命令，不要再默认套一层 `powershell -Command`。

## 数据集现状

当前笔记本数据根：`C:\Users\numbe\mahjong_data_root`

保留目录只有两套：

- `C:\Users\numbe\mahjong_data_root\dataset_rebuilt`
- `C:\Users\numbe\mahjong_data_root\dataset_json_rebuilt`

旧坏目录已经清掉，不要再去用旧的这些名字：

- `dataset`
- `dataset_json`
- `dataset_bad_20260330`
- `dataset_rebuilt_stale_*`
- `dataset_json_rebuilt_stale_*`

这次重建的根因与修复：

- 原问题：早期 `extract_data.py` 把 zip 里本来已经是 gzip 流的 `.mjson` 又 gzip 了一次，导致双层 gzip，后续解压出来很多“看起来是 json，实际还是 gzip 头”的坏文件
- 修复后逻辑：如果 zip 条目首字节已经是 gzip 魔数 `1f8b`，就原样写到 `.json.gz`
- 已验证：笔记本 `dataset_rebuilt` 的样本文件哈希和桌面机 `D:\mahjong_data\dataset\...` 一致

## 当前默认运行口径

### 笔记本 Stage 0.5 训练 loader 默认

当前决定保留为：

- train：`num_workers = 6`
- train：`file_batch_size = 7`
- train：`prefetch_factor = 3`

验证默认仍保留：

- val：`val_file_batch_size = 7`
- val：`val_prefetch_factor = 6`

说明：

- 之前的 loader 搜索是在笔记本本地 `bench_data` 代表子集上做的，不是全量数据根
- 但 `6/7/3` 已经在同一口径下复现成功
- `4/10/4` 很接近，可以作为近似备选
- `6/8/3` 跑过，但不是那轮最终最优

相关产物：

- `logs/laptop_stage05_loader_bench/summary.json`
- `logs/laptop_stage05_loader_bench/confirm_summary.json`
- `logs/laptop_pure_train_compare_manual_20260330/repro_best_nw6_fb7_pf3_20260330/summary.json`

## 最稳的远程操作方式

### 1. 直接发 PowerShell 命令

现在默认 shell 已经是 PowerShell，可以直接这样跑：

```powershell
ssh -i "$HOME\.ssh\mahjong_laptop_ed25519" numbe@<laptop-ip> "Get-Location"
ssh -i "$HOME\.ssh\mahjong_laptop_ed25519" numbe@<laptop-ip> "Get-ChildItem 'C:\Users\numbe\mahjong_data_root' -Directory"
```

### 2. 复杂脚本优先走 stdin

如果命令里有很多管道、脚本块、引号，最稳的是：

```powershell
@'
Get-ChildItem 'C:\Users\numbe\mahjong_data_root' -Directory |
  Select-Object Name, LastWriteTime
'@ | ssh -i "$HOME\.ssh\mahjong_laptop_ed25519" numbe@<laptop-ip> powershell -NoProfile -Command -
```

### 3. 传文件优先用 `scp`

Windows PowerShell + SSH 引号很容易炸。需要把脚本或配置扔到笔记本时，优先用：

```powershell
scp -i "$HOME\.ssh\mahjong_laptop_ed25519" local.ps1 "numbe@<laptop-ip>:C:/Users/numbe/Desktop/MahjongAI/logs/remote.ps1"
```

比起把长脚本拼进一条 SSH 命令里，这个更稳。

## 已踩过的坑

### 1. 本地 PowerShell 会提前展开远端变量

例如这些变量如果不转义：

- `$p`
- `$_`
- `$input`

就会在桌面机本地先被吃掉，远端收到的脚本会变残。

规避方式：

- 用 here-string 走 stdin
- 或显式写成 `` `$p ``、`` `$_ `` 这类转义

### 2. 命令太长会撞 Windows 上限

把大段脚本 base64 编进一条远端命令时，容易触发“命令行太长”。

规避方式：

- 优先 `scp` 上传脚本
- 或 stdin 管道喂给远端 PowerShell

### 3. `Write-Host` 不适合拿来当日志判断

后台跑脚本时，`Write-Host` 不一定按预期落进重定向日志。后续如果靠日志判断进度，优先用：

- `Write-Output`
- 直接看目录时间戳 / 文件数
- 直接查进程树

### 4. 不要误杀所有 `python.exe`

早期做过粗暴清理，容易把不相关进程一起干掉。后续清理远端任务时，按命令行过滤：

- `extract_data.py`
- `decompress_dataset_json.py`
- 指定 runner 脚本名

## 建议的操作纪律

- 桌面机和笔记本同时跑实验时，不要共用同一个 checkpoint 路径
- 两台机器的 run name / 输出目录必须带机器标签
- 需要同步脚本时，优先同步代码和配置，不要同步大 checkpoint
- 需要复现实验时，先确认口径：
  - 是 `bench_data` 子集
  - 还是 `dataset_json_rebuilt` 全量根
- 看到 tqdm 即时 `batch/s` 不要直接下结论，优先看一段稳定区间的中位数，或者看 `240 steps / elapsed_sec`

## 后续 Agent 的起手检查

新的会话接手笔记本任务时，建议先跑这几步：

```powershell
ssh -i "$HOME\.ssh\mahjong_laptop_ed25519" numbe@<laptop-ip> "Get-Location"
ssh -i "$HOME\.ssh\mahjong_laptop_ed25519" numbe@<laptop-ip> "Get-ChildItem 'C:\Users\numbe\mahjong_data_root' -Directory"
ssh -i "$HOME\.ssh\mahjong_laptop_ed25519" numbe@<laptop-ip> "Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' } | Select-Object ProcessId,CommandLine | Format-Table -AutoSize"
```

如果这三步都正常，再继续具体实验。
