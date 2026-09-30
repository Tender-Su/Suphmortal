# RiichiLab 客户端

> 核验：2026-09-05 · 依据 [run_mortal_bot.py](run_mortal_bot.py)；平台协议变更后需重新 validate。

模型与权重留在本机，客户端经 WebSocket 收发 MJAI 事件/动作。它复用本仓库 `1v3` 的 checkpoint 加载器，支持当前 categorical-policy checkpoint；接入成功和 ranked 成绩不能替代指定对手的正式 `1v3`。

## 单局验证

从仓库根运行。先在 [RiichiLab bots](https://riichi.dev/bots) 创建 bot 并取得 token。下面的 PowerShell 7 示例从隐藏输入读取 token，避免把值写进文件或 shell 历史：

```powershell
$mortalPython = 'C:\ProgramData\anaconda3\envs\mortal\python.exe'
$checkpoint = '.\mortal\checkpoints\sl_canonical.pth'
$env:RIICHILAB_BOT_TOKEN = Read-Host -MaskInput 'RiichiLab bot token'
& $mortalPython -m integrations.riichilab.run_mortal_bot `
  --checkpoint $checkpoint --mode validate --device cpu `
  --output-dir .\logs\external_model_eval\riichilab_validate_review
```

选择实际 checkpoint 后执行；S70 的本地产物位置见 [SL 状态](../../docs/status/supervised-mainline.md)。默认 CPU 推理；改用 GPU 前核对训练任务与显存余量。服务端拒绝动作时客户端断开并记录失败，不能把该次尝试计为正常完成。

## 连续 ranked 与恢复

validate 通过后才使用 ranked；每次独立实验选择新输出目录：

```powershell
& $mortalPython -m integrations.riichilab.run_mortal_bot `
  --checkpoint $checkpoint --mode ranked --device cpu `
  --games 1000 --output-dir .\logs\external_model_eval\riichilab_ranked_review
```

checkpoint 在进程内复用，各局重新连接队列；临时失败退避重试，不消耗目标完成局数。输出目录内：

| 文件 | 含义 |
| --- | --- |
| `progress.json` | 原子更新的累计进度 |
| `games.jsonl` | 每次完成或失败尝试的简要记录 |
| `protocol/` | 每次连接的完整协议记录 |

中断后，确认旧进程已退出，保持模型/配置/输出目录一致并加 `--resume`。`--games` 表示累计目标，不是新增局数。结束使用后清除当前 shell 的 token：`Remove-Item Env:RIICHILAB_BOT_TOKEN`。

## 评级停止与网络参数

参数定义以 `--help` 和代码为准：

- `--stop-rating-at` 与实际 `--rating-bot-id` 一起设置；启用后忽略 `--games`。
- 每局后等待 rating API 的 `total_games` 前进，再判断阈值与是否入队。API 失败会暂停入队。
- 输出目录的 `stop_policy.json` 可设置停止策略及 `activation_total_games`；改变策略前检查当前文件，区分历史峰值与从指定完成局数以后达标。
- `--proxy-url`、`--no-proxy` 和物理连接参数用于对应网络环境，不照抄其他机器地址。

评级策略会改变采样终止时间。研究比较保留失败、重连和完整样本，预声明停止规则；不要把达到阈值的局部峰值当成正式模型胜率。指定对手、四座轮换与配对统计见 [评测流程](../../docs/agent/workflows.md#1v3-与配对评测)。
