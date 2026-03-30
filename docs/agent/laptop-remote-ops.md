# 笔记本远程操作备忘

这份文档给后续 Agent 用，目标是让新的会话能立刻明白：

- 笔记本现在能不能直接远程操作
- 数据集现在落在哪
- 笔记本当前默认该怎么跑 Stage 0.5 loader / 训练 benchmark
- Windows + PowerShell + SSH 这套链路里哪些坑已经踩过，哪些做法最稳

这份文档不负责 Git 同步方案；代码同步细节看 `docs/agent/code-sync.md`。

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

- train：`num_workers = 4`
- train：`file_batch_size = 10`
- train：`prefetch_factor = 4`

验证默认仍保留：

- val：`val_file_batch_size = 7`
- val：`val_prefetch_factor = 5`

说明：

- `2026-03-31` 已经在笔记本交互前台窗口里重新确认过训练默认
- 这次最小确认组里：
  - `4/10/4` = `1.9564 steps/s`
  - `6/7/3` = `1.8891 steps/s`
  - `6/8/3` = `1.8872 steps/s`
- 同一轮最小验证确认里：
  - `7/5` = `0.8093 steps/s`
  - `7/6` = `0.7562 steps/s`
- 所以当前默认应更新为 `4 / 10 / 4` + `7 / 5`

相关产物：

- `logs/laptop_stage05_loader_bench/summary.json`
- `logs/laptop_stage05_loader_bench/confirm_summary.json`
- `logs/laptop_pure_train_compare_manual_20260330/repro_best_nw6_fb7_pf3_20260330/summary.json`
- `logs/stage05_loader_ab/laptop_stage05_loader_bench_interactive_20260331/`
- `docs/status/laptop-stage05-loader-benchmark-2026-03-31.md`

注意：

- `2026-03-30` 那批 `6/7/3` 结论现在只能视为历史参考
- 后续确认到：通过 SSH `Session 0` 拉起的训练进程，在笔记本上可能被 Windows / 厂商调度偏向小核心
- 当前新的 `4/10/4 + 7/5` 默认，是在笔记本交互会话的可见窗口里重跑确认后的结果

### 双机 winner_refine 入口

如果 `P1 winner_refine` 要让桌面机和笔记本同时参与，当前不要让两台机器共同写一个共享 `run_dir`。当前仓库已经提供桌面机调度入口：

```powershell
python mortal/run_stage05_winner_refine_distributed.py dispatch `
  --run-name s05_fidelity_p1_top3_cali_slim_20260329_001413
```

默认假设：

- 桌面机是调度器和本地 worker
- 笔记本是通过 SSH 拉起的 remote worker
- 远端仓库路径默认：`C:\Users\numbe\Desktop\MahjongAI`
- 远端 Python 默认：`C:\Users\numbe\miniconda3\envs\mortal\python.exe`
- SSH key 默认：`$HOME\.ssh\mahjong_laptop_ed25519`

当前这条链路的语义是：

- `seed1` 先把 `winner_refine` 的全部候选跑完
- `seed2` 不再固定全量双 seed，而是只补 `seed1` 后仍处在竞争带里的候选
- 最终 round 只在补过 `seed2` 的 decision 候选里比较 winner，避免单 seed 尾部候选混回最终榜单
- 当前远端正式口径应是：
  - 在笔记本交互用户会话里弹出可见窗口
  - 不再接受 SSH `Session 0` 里直接后台拉起训练进程作为正式 benchmark / 训练口径
- 当前调度器支持对任一 worker 单独暂停 / 恢复：
  - `python mortal/run_stage05_winner_refine_distributed.py pause-worker --run-name <run_name> --worker-label laptop`
  - `python mortal/run_stage05_winner_refine_distributed.py pause-worker --run-name <run_name> --worker-label laptop --stop-active`
  - `python mortal/run_stage05_winner_refine_distributed.py resume-worker --run-name <run_name> --worker-label laptop`

### 真实启动前检查结果

这轮真实跨机启动前检查已经确认：

- 笔记本可以通过 SSH 正常执行远端命令
- 笔记本当前数据根可用：
  - `C:\Users\numbe\mahjong_data_root\dataset_json_rebuilt`
- 笔记本当前运行配置应使用：
  - train：`6 / 7 / 3`
  - val：`7 / 6`
- 当前 `winner_refine` 远端上下文探针已经通过：
  - `protocol = C_A2x_cosine_broad_to_recent_strong_24m_12m`
  - `candidate_count = 27`
  - `center_count = 3`
  - `seed_offsets = [0, 1009]`
  - `step_scale = 1.5`

这说明：远端代码、配置、`state.json` 和文件索引已经能支撑这轮 `winner_refine` 正常展开候选。

补充：

- 上面的预检查只说明代码 / 配置 / 路径链路可达
- 不说明笔记本在远端训练时一定会被调到合适的大核心 / 前台口径
- 真正要确认笔记本训练默认，仍然要看交互前台窗口里重跑出的 benchmark

### 当前双机调度器的远端必备资产

当前 `winner_refine` 远端 worker 不只需要代码，还需要下面三类本地资产：

- `C:\Users\numbe\Desktop\MahjongAI\mortal\config.toml`
- `C:\Users\numbe\Desktop\MahjongAI\logs\stage05_fidelity\<run_name>\state.json`
- `C:\Users\numbe\Desktop\MahjongAI\mortal\checkpoints\file_index_supervised_json.pth`

缺任何一个，真实启动都可能失败：

- 缺 `config.toml`：读不到笔记本本地数据根和 loader 配置
- 缺 `state.json`：`load_refine_context()` 无法恢复当前 `winner_refine` 上下文
- 缺 `file_index_supervised_json.pth`：远端 `load_all_files()` 会直接报错

### 当前笔记本专用运行配置

这轮为双机 `winner_refine` 已验证可用的笔记本本地配置是：

- `mortal/config.toml`
  - `[dataset].globs = C:/Users/numbe/mahjong_data_root/dataset_json_rebuilt/**/*.json`
  - `[dataset] = 6 / 7 / 3`
  - `[supervised] = train 6 / 7 / 3, val 7 / 6`

注意：

- 这是笔记本本地运行配置，不应该进 Git
- 它属于运行资产，同步方式应该是 `scp` 或远端脚本生成
- 如果后面重新做了 loader 搜索，这里的默认要跟着改

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

### 1.5. `stdin -> powershell -Command -` 首行可能吃到 BOM

这轮真实预检查里，出现过首个 cmdlet 名被 BOM 污染的情况，例如：

- `Set-Location` 变成异常命令名
- `Get-Content` 变成异常命令名
- `Test-Path` 变成异常命令名

实践上更稳的顺序是：

- 短命令：直接一行 SSH
- 稍复杂命令：本地先拼成单行再发
- 真正复杂的探针或修复脚本：先 `scp` 到远端，再执行

不要默认假设 here-string 通过 stdin 喂给远端 PowerShell 一定稳定。

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

### 5. `winner_refine` 的远端上下文恢复依赖文件索引

这轮真实探针里，远端首次失败不是代码问题，而是：

- 缺 `mortal/checkpoints/file_index_supervised_json.pth`

错误位置在：

- `run_stage05_ab.load_all_files()`
- 它会先读 `BASE_INDEX_PATH`

所以后续 agent 在启动双机 `winner_refine` 前，不要只查代码和 `state.json`，一定要查这个索引文件是否在远端。

## 建议的操作纪律

- 桌面机和笔记本同时跑实验时，不要共用同一个 checkpoint 路径
- 两台机器的 run name / 输出目录必须带机器标签
- 需要同步脚本时，优先同步代码和配置；需要同步运行资产时，优先同步 `state.json` 和文件索引，不要默认同步大 checkpoint
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
