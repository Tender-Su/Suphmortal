# 双机同步与远程运行

这份文档把原来的代码同步、远程 shell 和笔记本在线 RL 纪律合到一起，只保留当前有效做法。

## 当前拓扑

- 台式机：
  - 仓库：`C:\Users\numbe\Desktop\MahjongAI`
  - 源码真源：`main`
  - 解释器：`C:\ProgramData\anaconda3\envs\mortal\python.exe`
- 笔记本：
  - 主机名：`abandon`
  - SSH 别名：`mahjong-laptop`
  - 仓库：`C:\Users\numbe\Desktop\MahjongAI`
  - 解释器：`C:\Users\numbe\miniconda3\envs\mortal\python.exe`
  - 数据根：`C:\Users\numbe\mahjong_data_root`
- 台式机 bare mirror：
  - `C:\Users\numbe\repos\MahjongAI-desktop.git`
- 台式机 remote：
  - `laptop-sync = mahjong-laptop:C:/Users/numbe/repos/MahjongAI-desktop.git`

## 默认代码同步

在台式机仓库根目录执行：

```powershell
.\scripts\sync_laptop_repo.ps1
```

它会：

1. 把台式机 `main` 推到笔记本 bare mirror
2. 让笔记本工作树对 `main` 做 `pull --ff-only`

如果同时要同步 GitHub：

```powershell
git push origin HEAD:main
```

如果只想更新笔记本 bare mirror，不拉笔记本工作树：

```powershell
.\scripts\sync_laptop_repo.ps1 -SkipWorktreeUpdate
```

## Git 与运行资产的边界

应该进 Git：

- Python / Rust 代码
- 测试
- 当前文档
- 通用调度脚本

不应该进 Git：

- `mortal/config.toml`
- `logs/**`
- `checkpoints/**`
- `mortal/checkpoints/file_index_supervised_json.pth`
- 某轮 run 的 `state.json`

## 监督学习分布式任务的运行资产

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

补充说明：

- 这些都属于运行资产，不应进 Git
- `run_sl_formal_distributed.py` 会把 dispatch 状态同步到远端
- child formal 完成后会把 child run 与 `logs/sl_ab/<child_run_name>_formal` 产物拉回台式机

## 在线 RL 的两种笔记本模式

### 独立 arm

- 笔记本自己跑 `server + trainer + client`
- 用于并行实验臂、Oracle 对照、独立 smoke
- 不与台式机共享 replay / checkpoint / TensorBoard

入口：

```powershell
.\scripts\start_laptop_online_independent_arm.ps1 `
  -Arm current_config `
  -ExperimentProfile ms_rl1_minimal_500 `
  -OpponentPoolPreset validation
```

### 远端 worker

- 台式机跑 `server + trainer`
- 笔记本只跑 `client`
- 用于给同一条 run 增加 self-play 吞吐

入口：

```powershell
.\scripts\start_laptop_online_worker.ps1 `
  -Arm current_config `
  -ExperimentProfile default `
  -OpponentPoolPreset validation `
  -DesktopHost 192.168.1.3 `
  -DesktopPort 5000
```

## 当前冻结纪律

- 双机 `client` 默认一律全 GPU：
  - `control=cuda:0`
  - `baseline_train=cuda:0`
- 不再把 CPU 当文档内建排障分支
- 笔记本运行前，除了 Git 同步，还会额外覆盖：
  - `mortal/**/*.py`
  - `scripts/start_interactive_remote_python.ps1`
- 这层覆盖是当前脏工作树研发口径的一部分，不要删

## 已确认的坑

- 不要把 PowerShell 壳进程还在，当成 `train_online.py` 还在
- 长参数链路不要再走 JSON 串；当前稳定做法是 `Base64 + NUL`
- 远端任务不要依赖 `Start-Process -NoNewWindow`
- 不要全局误杀所有 `python.exe`
- 笔记本历史上出现过 `python.exe -> nvcuda64.dll -> 0xc0000409 / BEX64`
  - 当前处理策略是保留 CUDA 配置、抓 WER、直接重拉 run
  - 不是把设备切回 CPU

## 远程健康检查

优先检查这些，而不是只看命令有没有返回：

- `server_task / trainer_task / client_task` 下的 `done.json`
- 对应 `stdout.log / stderr.log`
- Windows 事件日志里的 `Application Error / Windows Error Reporting`

起手检查命令：

```powershell
ssh -i "$HOME\.ssh\mahjong_laptop_ed25519" numbe@<laptop-ip> "Get-Location"
ssh -i "$HOME\.ssh\mahjong_laptop_ed25519" numbe@<laptop-ip> "Get-ChildItem 'C:\Users\numbe\mahjong_data_root' -Directory"
ssh -i "$HOME\.ssh\mahjong_laptop_ed25519" numbe@<laptop-ip> "Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' } | Select-Object ProcessId,CommandLine | Format-Table -AutoSize"
```

## 继续阅读

- 当前实验入口：`docs/agent/workflows.md`
- 机器级默认：`docs/status/machine-benchmarks.md`
- 当前在线 RL 结论：`docs/status/online-rl-mainline.md`
