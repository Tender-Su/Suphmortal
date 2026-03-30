# 代码同步默认

这份文档只回答一个问题：台式机和笔记本现在应该如何同步代码。

## 当前默认

- 台式机是源码真源：
  - 工作树：`C:\Users\numbe\Desktop\MahjongAI`
  - 当前分支：`main`
- 笔记本不是直接从台式机工作树拉代码，而是走一个局域网 bare mirror：
  - bare mirror：`C:\Users\numbe\repos\MahjongAI-desktop.git`
  - 笔记本工作树：`C:\Users\numbe\Desktop\MahjongAI`
- 台式机已配置 SSH 别名：
  - `mahjong-laptop`
- 台式机仓库已配置 remote：
  - `laptop-sync = mahjong-laptop:C:/Users/numbe/repos/MahjongAI-desktop.git`

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

如果只想更新笔记本 bare mirror，不想动笔记本工作树：

```powershell
.\scripts\sync_laptop_repo.ps1 -SkipWorktreeUpdate
```

## 笔记本端当前状态

- 当前笔记本工作树已经是 Git 仓库
- 当前 `origin` 指向：
  - `C:\Users\numbe\repos\MahjongAI-desktop.git`
- 当前工作分支：
  - `main`
- 旧的非 Git 工作树已整目录备份到：
  - `C:\Users\numbe\Desktop\MahjongAI_pre_git_backup_20260330_231850`

## 操作纪律

- 默认只在台式机提交代码，笔记本只同步和跑实验
- 如果确实在笔记本上做了临时改动，不要直接长期分叉；尽快整理回台式机主线
- 双机并行跑实验时，代码同步和实验输出是两回事：
  - 代码走 Git / bare mirror
  - 日志、checkpoint、数据目录不要走 Git
- 笔记本工作树更新默认使用 `pull --ff-only`，避免把它变成另一个独立开发主线
