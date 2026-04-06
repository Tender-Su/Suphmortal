# 代码同步默认

这份文档只回答一个问题：台式机和笔记本现在应该如何同步代码。远程 shell、数据根和运行资产细节不在这里，见 `docs/agent/laptop-remote-ops.md`。

## 当前拓扑

- 台式机是源码真源：
  - 工作树：`C:\Users\numbe\Desktop\MahjongAI`
  - 当前分支：`main`
  - 跟踪分支：`origin/main`
- 笔记本代码不同步自台式机工作树，而是走局域网 bare mirror：
  - bare mirror：`C:\Users\numbe\repos\MahjongAI-desktop.git`
  - 笔记本工作树：`C:\Users\numbe\Desktop\MahjongAI`
  - 当前分支：`main`
- 台式机 SSH 别名：
  - `mahjong-laptop`
- 台式机仓库 remote：
  - `laptop-sync = mahjong-laptop:C:/Users/numbe/repos/MahjongAI-desktop.git`
  - `origin = https://github.com/Tender-Su/Suphmortal.git`

## 默认同步命令

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

## Git 与运行资产的边界

应该进 Git：

- Python / Rust 代码
- 测试
- 当前生效的文档
- 双机调度器入口与通用调度模块

不应该进 Git：

- `mortal/config.toml`
- `logs/**`
- `checkpoints/**`
- `mortal/checkpoints/file_index_supervised_json.pth`
- 某轮 run 的 `state.json`

## 当前推荐顺序

- 如果改动是代码逻辑：
  - 先在台式机完成修改和测试
  - 再提交到 `main`
  - 再运行 `.\scripts\sync_laptop_repo.ps1`
- 如果改动是笔记本专用运行资产：
  - 不要提交 Git
  - 用 `scp` / 远端脚本单独同步

## 当前纪律

- 默认只在台式机提交代码，笔记本只同步和跑实验
- 笔记本工作树默认使用 `pull --ff-only`
- 双机并行跑实验时：
  - 代码走 Git / bare mirror
  - 日志、checkpoint、数据目录不要走 Git
- 不要继续长期依赖“桌面机写代码，再手工 `scp` 成堆脚本到笔记本”的方式
