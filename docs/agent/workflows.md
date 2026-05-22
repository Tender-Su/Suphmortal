# 运行流程

这份文档只回答“现在该怎么跑”，不重复解释 winner 和研究结论。

## 环境与构建

- 优先使用解释器：`C:\ProgramData\anaconda3\envs\mortal\python.exe`
- `conda` 在新 PowerShell 里不一定在 `PATH`，只要不需要交互激活，就优先直接用上面的绝对路径
- 这台机器的 plain `python` 当前可能解析到 `C:\Python314\python.exe`；PyO3 `0.23.4` 不支持 Python `3.14`，所以 PyO3 / Rust 测试不要依赖裸 `python` 自动发现
- Rust / PyO3 相关测试前，记得把 `PYO3_PYTHON` 与 `mortal` 环境的 `PATH` 补齐

常用入口：

```powershell
.\scripts\build_libriichi.bat
C:\ProgramData\anaconda3\envs\mortal\python.exe -c "import libriichi; print('OK')"

$env:PYO3_PYTHON="C:\ProgramData\anaconda3\envs\mortal\python.exe"
$env:PATH="C:\ProgramData\anaconda3\envs\mortal;C:\ProgramData\anaconda3\envs\mortal\Library\bin;C:\ProgramData\anaconda3\envs\mortal\Scripts;$env:PATH"
cargo test -p libriichi state::test
```

## 主入口脚本

```powershell
.\scripts\run_grp.bat
.\scripts\run_supervised.bat
.\scripts\run_sl_p1_only.bat
.\scripts\run_online.bat
.\scripts\run_online_fidelity.bat
.\scripts\run_oracle_critic_pretrain.bat
.\scripts\run_oracle_dependency_eval.bat
```

说明：

- `run_online.bat [arm] [suffix]`
  - 直接跑本地在线 RL
  - `arm` 可传 `current_config / visible_only / actor_true / actor_shuffled / critic_only`
- `run_online_fidelity.bat`
  - 包装 `mortal/research/run_online_fidelity.py`
- `run_oracle_critic_pretrain.bat`
  - 包装 `mortal/online/pretrain_oracle_critic.py`
- `run_oracle_dependency_eval.bat`
  - 包装 `mortal/eval/oracle_dependency_eval.py`

## 监督学习相关

监督学习阶段已完成，但这几类入口仍保留：

- 正式训练：`.\scripts\run_supervised.bat`
- 手动 `P1`：`.\scripts\run_sl_p1_only.bat`
- 自动 snapshot：`docs/status/supervised-fidelity-results.md`
- 完整阶段链路：`P0 -> P1 calibration -> protocol_decide -> winner_refine -> formal_train -> formal_1v3`
- `P1 ablation` 仍只作为手动诊断轮

监督学习当前口径统一看 `docs/status/supervised-mainline.md`，不要再从旧的 `P1` / formal 分散文档里拼接结论。

## 监督学习下游 / 分布式入口

只有在要重跑、复核或调度旧的监督学习下游阶段时，才需要这些入口：

### `winner_refine`

```powershell
C:\ProgramData\anaconda3\envs\mortal\python.exe -m mortal.supervised.run_sl_winner_refine_distributed dispatch `
  --run-name sl_fidelity_p1_top3_cali_slim_20260329_001413
```

### formal triplet

```powershell
C:\ProgramData\anaconda3\envs\mortal\python.exe -m mortal.supervised.run_sl_formal_distributed dispatch `
  --run-name sl_formal_triplet_20260405 `
  --source-run-name sl_fidelity_p1_top3_cali_slim_20260329_001413 `
  --candidate-arm 'opp_lean*0.85' `
  --candidate-arm 'anchor*1.0' `
  --candidate-arm 'opp_lean(rank--/danger++)'
```

### `formal_1v3`

```powershell
C:\ProgramData\anaconda3\envs\mortal\python.exe -m mortal.supervised.run_sl_formal_1v3_distributed dispatch `
  --run-name <run_name>
```

结果边界：

- `protocol_decide` 回答协议 winner
- `winner_refine` 回答 pre-formal 第一梯队
- `formal_train` 产出 checkpoint pack：`best_loss / best_acc / best_rank`
- `formal_1v3` 决定官方 supervised winner

## 在线 RL：本地独立 arm

标准流程是先生成 machine-scoped config，再启动 `server + trainer + client`。

```powershell
C:\ProgramData\anaconda3\envs\mortal\python.exe -m mortal.online.online_machine_modes `
  --mode independent_arm `
  --base-config .\mortal\config.toml `
  --experiment-profile ms_rl1_minimal_500 `
  --opponent-pool-preset validation `
  --output .\logs\online_modes\independent_arm\demo\config.toml `
  --runtime-root .\logs\online_modes\independent_arm\demo

.\scripts\start_online_independent_arm.bat .\logs\online_modes\independent_arm\demo\config.toml current_config
```

使用纪律：

- 排因 / 微探针 / Oracle 对照默认用 `validation`
- 更长窗口和冲上限训练再用 `formal`
- 当前不推荐一上来就做 `RL-1 / RL-2` pair；先让 `RL-1` 单臂加回跑通

在线 self-play 进程职责：

- `server`：监听 `127.0.0.1:5000`，管理参数分发、replay buffer、drain / submit 目录
- `trainer`：从 server drain replay，做 PPO 更新，然后提交新参数
- `client`：拉取最新参数，运行 self-play，并把 replay 交回 server

## 在线 RL：本地 worker

当台式机自己跑 `server + trainer`，只想再起一个本地 worker 时：

```powershell
C:\ProgramData\anaconda3\envs\mortal\python.exe -m mortal.online.online_machine_modes `
  --mode worker `
  --base-config .\mortal\config.toml `
  --output .\logs\online_modes\worker\demo\config.toml `
  --runtime-root .\logs\online_modes\worker\demo `
  --remote-host 127.0.0.1 `
  --remote-port 5000

.\scripts\start_online_worker.bat .\logs\online_modes\worker\demo\config.toml current_config
```

## 在线 RL：成对 smoke

`scripts/start_rl_oracle_sanity_pair.ps1` 仍然可用，但它是“已经有稳定共享底座后做匹配对照”的工具，不是当前第一步默认入口。

```powershell
.\scripts\start_rl_oracle_sanity_pair.ps1 `
  -PairName rl1_rl2_smoke `
  -DesktopProfile ms_rl2_smoke_10k `
  -LaptopProfile ms_rl1_smoke_10k
```

## RL 版 fidelity

`mortal/research/run_online_fidelity.py` 当前主要服务于 `value.weight + oracle_critic` 这条线。

```powershell
.\scripts\run_online_fidelity.bat calibration `
  --source-run ..\logs\online_modes\independent_arm\some_run `
  --output-json ..\logs\online_fidelity\calibration.json

.\scripts\run_online_fidelity.bat emit_protocol_decide `
  --base-config .\config.toml `
  --base-experiment-profile ms_rl1_add_value_gae_is_500 `
  --calibration-json ..\logs\online_fidelity\calibration.json `
  --runtime-root ..\logs\online_fidelity\protocol_decide `
  --output-json ..\logs\online_fidelity\protocol_decide_manifest.json `
  --opponent-pool-preset validation `
  --oracle-critic-mode both

.\scripts\run_online_fidelity.bat rank_protocol_decide `
  --results-json ..\logs\online_fidelity\protocol_decide_results.json `
  --output-json ..\logs\online_fidelity\protocol_decide_ranking.json

.\scripts\run_online_fidelity.bat emit_winner_refine `
  --base-config .\config.toml `
  --base-experiment-profile ms_rl1_add_value_gae_is_500 `
  --protocol-json ..\logs\online_fidelity\protocol_decide_ranking.json `
  --runtime-root ..\logs\online_fidelity\winner_refine `
  --output-json ..\logs\online_fidelity\winner_refine_manifest.json `
  --opponent-pool-preset validation
```

当前判读纪律：

- `protocol_decide / winner_refine` 最终排名优先看正式 `1v3`
- `formal_avg_pt / formal_avg_rank` 比训练内 `avg_pt / avg_rank` 更可信

## Oracle 依赖度评测

```powershell
.\scripts\run_oracle_dependency_eval.bat
.\scripts\run_oracle_dependency_eval.bat .\checkpoints\best_actor_true.pth --games 3000 --output-json .\logs\oracle_dependency\actor_true_eval.json
```

## TensorBoard

TensorBoard 路径以当前 config 为准；常见本地目录如下：

```powershell
tensorboard --logdir .\mortal\tb_log_supervised_main
tensorboard --logdir .\mortal\tb_log
tensorboard --logdir .\mortal\tb_log_grp
```

machine-scoped online run 通常把 TensorBoard 写到对应 runtime root 下的 `tb_log`。

## 常见产物语义

- `checkpoints/grp.pth`：`GRP best_loss`，默认下游使用。
- `checkpoints/grp_best_acc.pth`：`GRP best_acc`，只做受控对照候选。
- `checkpoints/grp_latest.pth`：`GRP latest`，只用于续训。
- `checkpoints/sl_canonical*.pth`：监督学习 canonical pack，语义见 `docs/status/supervised-mainline.md`。
- `checkpoints/online_ppo/` 或 machine-scoped run 的 `checkpoints/`：在线 RL 周期性保存和 best export。

checkpoint 通常是包含 `mortal`、`policy_net`、`config` 等字段的 dict；具体加载口径以对应训练 / 评测入口为准。

## `1v3` 评测坑位

- `mortal/eval/one_vs_three.py` 真正读取 challenger 的路径是 `[1v3.challenger].state_file`，不是 `[control].state_file`
- 当前这台 `RTX 5070 Ti` 会按代码默认扩成 `seed_count=1024 / shard_count=4`
- 如果要强制回到受控 `1v3 = 2000`：

```powershell
$env:MORTAL_1V3_SEED_COUNT = '500'
$env:MORTAL_1V3_SHARD_COUNT = '1'
```

## 当前运行原则

- 训练阶段默认关闭 `search`
- `search` 只作为推理期系统增强单独评估，不和 actor 训练收益混算
- 不要手改 `baseline.train` 去切 opponent pool，优先用 `--opponent-pool-preset`
- 不要只看壳进程是否还在；在线 RL 真正要确认 `server / trainer / client` 对应 Python 进程都在工作
