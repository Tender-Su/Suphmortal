# 代码同步默认

这份文档只回答一个问题：台式机和笔记本现在应该如何同步代码。

这份文档只负责 Git 同步；远程 shell / 数据集 / PowerShell 操作细节看 `docs/agent/laptop-remote-ops.md`。

## 当前默认

- 台式机是源码真源：
  - 工作树：`C:\Users\numbe\Desktop\MahjongAI`
  - 当前分支：`main`
  - 跟踪分支：`origin/main`
- 笔记本不是直接从台式机工作树拉代码，而是走一个局域网 bare mirror：
  - bare mirror：`C:\Users\numbe\repos\MahjongAI-desktop.git`
  - 笔记本工作树：`C:\Users\numbe\Desktop\MahjongAI`
  - 当前分支：`main`
- 台式机已配置 SSH 别名：
  - `mahjong-laptop`
- 台式机仓库已配置 remote：
  - `laptop-sync = mahjong-laptop:C:/Users/numbe/repos/MahjongAI-desktop.git`
  - `origin = https://github.com/Tender-Su/Suphmortal.git`

## 当前仓库状态

- 当前唯一正式开发主线是：`main`
- 旧的过渡分支 `github-main` 已经清理，不再使用
- 旧实验分支 `exp/stage05-utilization-pass1` 不属于当前主线历史；它只是本地旧档案，不参与当前同步方案
- 笔记本 bare mirror 的默认 `HEAD` 已经指向 `main`

## 为什么这样配

- 台式机当前没有 `sshd`，所以不让笔记本直接反向从台式机 `pull`
- 台式机可以稳定 SSH 到笔记本，所以默认流程是：
  - 台式机 `push` 到笔记本 bare mirror
  - 然后让笔记本工作树从它本机 bare mirror `pull --ff-only`
- 这样仍然保持“台式机是真源”的工程纪律，同时不需要额外折腾台式机系统服务

## 默认用法

在台式机仓库根目录执行：

```powershell
.\scripts\sync_laptop_repo.ps1
```

它会做两件事：

1. `git push laptop-sync main:refs/heads/main`
2. 通过 SSH 进入笔记本，把 `C:\Users\numbe\Desktop\MahjongAI` fast-forward 到最新 `main`

如果这次改动也要同步到 GitHub，再额外执行：

```powershell
git push origin HEAD:main
```

如果只想更新笔记本 bare mirror，不想动笔记本工作树：

```powershell
.\scripts\sync_laptop_repo.ps1 -SkipWorktreeUpdate
```

## 当前推荐 Git 边界

- 要长期保留、要让新 agent 自动知道的东西，放进 Git：
  - Python / Rust 代码
  - 测试
  - agent 文档
  - 调度器入口与通用调度模块
- 机器相关、run 相关、会随环境漂移的东西，不放进 Git：
  - `mortal/config.toml`
  - `logs/**`
  - `checkpoints/**`
  - `mortal/checkpoints/file_index_supervised_json.pth`
  - 某一轮 run 的 `state.json`

## 当前对双机调度器的建议

- 当前建议把下面这些文件正式纳入 Git，并通过 `main -> laptop-sync` 同步：
  - `mortal/distributed_dispatch.py`
  - `mortal/run_stage05_winner_refine_distributed.py`
  - `mortal/run_stage05_formal_distributed.py`
  - `mortal/run_stage05_formal_1v3_distributed.py`
  - 对应测试与文档
- 不要继续长期依赖“代码只在桌面机本地、再手工 `scp` 到笔记本”这种方式
- 手工 `scp` 只适合：
  - 当前还没来得及提交、但需要立刻做一次临时远端验证
  - run 相关运行资产同步，例如 `state.json` / `file_index_supervised_json.pth`

## 当前推荐顺序

- 如果改动是代码逻辑：
  - 先在桌面机工作树完成修改和测试
  - 再提交到 `main`
  - 再运行 `.\scripts\sync_laptop_repo.ps1`
- 如果改动是笔记本专用运行资产：
  - 不要提交 Git
  - 用 `scp` / 远端脚本单独同步

## 双机调度器的运行资产纪律

- 当前 `winner_refine` 双机调度器代码应该走 Git
- 但下面这些东西仍然应该视为“运行资产”，不能要求 Git 负责：
  - 笔记本本地 `mortal/config.toml`
  - 目标 run 的 `logs/stage05_fidelity/<run_name>/state.json`
  - 笔记本本地 `mortal/checkpoints/file_index_supervised_json.pth`
- 原因很简单：
  - 它们不是通用代码
  - 它们依赖机器路径、数据根和当前 run 状态
  - 把它们放进 Git 只会让同步边界变脏

## 笔记本端当前状态

- 当前笔记本工作树已经是 Git 仓库
- 当前 `origin` 指向：
  - `C:\Users\numbe\repos\MahjongAI-desktop.git`
- 当前工作分支：
  - `main`
- 笔记本工作树更新方式：
  - `git fetch origin`
  - `git checkout main`
  - `git pull --ff-only origin main`
- 旧的非 Git 工作树已整目录备份到：
  - `C:\Users\numbe\Desktop\MahjongAI_pre_git_backup_20260330_231850`

## 操作纪律

- 默认只在台式机提交代码，笔记本只同步和跑实验
- 当前默认顺序是：
  - 先在台式机 `main` 提交
  - 再运行 `.\scripts\sync_laptop_repo.ps1`
  - 需要公开同步时，再 `git push origin HEAD:main`
- 如果确实在笔记本上做了临时改动，不要直接长期分叉；尽快整理回台式机主线
- 双机并行跑实验时，代码同步和实验输出是两回事：
  - 代码走 Git / bare mirror
  - 日志、checkpoint、数据目录不要走 Git
- 笔记本工作树更新默认使用 `pull --ff-only`，避免把它变成另一个独立开发主线
