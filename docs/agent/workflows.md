# 运行流程

> 核验：2026-09-05 · 命令依据当前仓库入口。运行参数以对应状态页和有效配置为准，输出使用独立 run 目录。

所有示例从仓库根目录执行。长输出按 [RTK 约定](../../RTK.md) 包装；以下使用 PowerShell。

## 环境与构建

本机非交互解释器：

```powershell
$mortalPython = 'C:\ProgramData\anaconda3\envs\mortal\python.exe'
& $mortalPython -c "import sys, torch; print(sys.executable); print(torch.__version__, torch.version.cuda); print(torch.cuda.is_available())"
```

新机器以 [environment.yml](../../environment.yml) 为环境模板；它不是锁定依赖文件，不能仅凭安装完成就认定 PyTorch / CUDA 与目标 GPU 兼容。已有环境先检查，不重复安装。

```powershell
conda env create -f environment.yml
conda activate mortal
```

构建需要已激活目标环境，且该环境没有正在运行的训练/客户端依赖原生扩展。Rust 测试也使用同一解释器和 DLL 路径：

```powershell
$env:PYO3_PYTHON = $mortalPython
$mortalEnvDir = Split-Path $mortalPython
$env:PATH = "$mortalEnvDir;$mortalEnvDir\Library\bin;$mortalEnvDir\Scripts;$env:PATH"
& $mortalPython -m maturin develop --release --manifest-path libriichi/Cargo.toml
& $mortalPython -c "import libriichi; print(libriichi.__file__)"
& $mortalPython -m mortal.tests.test_greedy
```

活跃训练期间的替代构建和验证边界见 [重构与验证](code-health.md#活跃训练边界)。部分 `.bat` 使用裸 `python`，使用前确认 PATH；直接模块调用更容易固定解释器。

## 配置与路径

初次配置才复制模板，已有本地配置不要覆盖：

```powershell
if (-not (Test-Path .\mortal\config.toml)) {
    Copy-Item .\mortal\config.example.toml .\mortal\config.toml
}
$env:MORTAL_CFG = (Resolve-Path .\mortal\config.toml).Path
```

填写本机数据、输出和 checkpoint 路径；真实配置不提交。路径语义以 [config.py](../../mortal/config.py) 为准：

- 未设置 `MORTAL_CFG` 时读取 `mortal/config.toml`；相对的 `MORTAL_CFG` 按进程工作目录解析。
- 配置内 `PATH_KEYS` / `GLOB_KEYS` 列出的相对路径，以配置文件所在目录解析；不是所有字符串字段都会自动转换。
- 因此默认配置的 `./checkpoints/sl_canonical.pth` 对应仓库根下的 `mortal/checkpoints/sl_canonical.pth`。
- CLI 路径由各入口处理；从仓库根运行时使用根相对路径或绝对路径。新生成的 config 要检查路径归一化结果。

新 run 启动时核对输出独立、初始化优先级、数据范围、seed、对手与资源预算；恢复现有 run 核对 checkpoint 内部状态及源码指纹。

## GRP 与 SL

| 任务 | 模块 / 参数 | 说明 |
| --- | --- | --- |
| GRP | `mortal.supervised.train_grp` | [GRP 状态](../status/grp-mainline.md) |
| formal SL | `mortal.supervised.run_sl_formal` | 可能刷新 canonical，先核对发布与输出配置 |
| 手动 P1 | `mortal.supervised.run_sl_p1_only` | 诊断入口 |
| adaptive SL | `mortal.supervised.run_sl_ab --ab adaptive` | 必须明确 `--ab`；省略时默认 `all`，会扩展实验范围 |

调用形式为 `& $mortalPython -m <模块> <参数>`；不要对没有 argparse 的训练入口试跑 `--help`。以下两个入口可安全查看参数：

```powershell
& $mortalPython -m mortal.supervised.run_sl_formal --help
& $mortalPython -m mortal.supervised.run_sl_ab --help
```

新 adaptive SL 从 checkpoint bootstrap 时，默认继承 `aux`、`supervised.aux` 和 `supervised.rank_aux` 及辅助 ramp 时钟。仅在明确做目标消融的新 run 中使用 `--adaptive-bootstrap-auxiliary-policy current`，同时明确重置辅助日程；配方和日程选择进入 bootstrap 记录。直接 `state_file` 恢复还会核对损失系数和目标配置。通用训练器的 weights-only 初始化默认 `supervised.init_auxiliary_schedule = 'reset'`，课程交接设置为 `inherit`，使用该项时必须保持辅助配方一致。旧运行继续使用冻结源码，不能改写历史 manifest。证据见 [课程机制方案](../research/sl-curriculum-mechanism-proposal-2026-09-07.md)。

旧完整链路 `P0 → P1 calibration → protocol_decide → winner_refine → formal_train → formal_1v3` 保留在代码中。分布式入口分别为 [winner_refine](../../mortal/supervised/run_sl_winner_refine_distributed.py)、[formal_train](../../mortal/supervised/run_sl_formal_distributed.py)、[formal_1v3](../../mortal/supervised/run_sl_formal_1v3_distributed.py)；先读其 `--help`，再按实际 manifest 传 run name。发布口径见 [SL 状态](../status/supervised-mainline.md)。

## Oracle critic

先将 `MORTAL_CFG` 指向本次独立配置，再选择训练或评测。参数入口：

```powershell
& $mortalPython -m mortal.online.pretrain_oracle_critic --help
```

训练调用 `mortal.online.pretrain_oracle_critic`；已有运行按对应 supervisor / manifest 恢复，不额外启动同一输出目录。`--fresh` 不是普通续训参数。

### 独立后台启动

交互式 Windows 桌面使用 [独立启动器](../../scripts/start_oracle_critic_detached.ps1)，不要从编辑器直接 `Start-Process` supervisor 并假设隐藏窗口就已隔离：

```powershell
pwsh -NoProfile -File .\scripts\start_oracle_critic_detached.ps1 `
  -SpecPath .\logs\your_run\apex_supervisor_spec.json
```

启动器通过现有 Explorer 桌面创建隐藏的 PowerShell 7 宿主；宿主必须验证自己不属于任何 Windows Job，才调用原 Apex supervisor。没有计划任务或服务注册，不继承调用端的临时环境变量；运行环境应由有效 spec / runner 明确提供。已有 supervisor 的独占锁会拒绝重复启动，不能用此命令接管仍在运行的旧 supervisor。

- `detached_launch_<id>.json` 记录宿主 PID、session、Job 隔离结果及启动错误；训练状态仍看 `apex_supervisor_status.json`，宿主错误看同名 `.log`。启动确认超时先查这些文件，不能盲目重试。
- `-Probe` 只启动 12 秒空载探针，完成后 receipt 为 `probe_completed`。不启动训练、不占 GPU。
- Apex 暂停、exact 保存、恢复和 runner 有限重试仍由原 supervisor 负责；启动器不修改模型、学习率、数据游标或源码指纹。
- 这是登录会话内的独立运行，不承诺注销、重启后自动恢复，也不负责重启意外死亡的 supervisor。无 Explorer 桌面或 Job 隔离失败时明确拒绝，不回退到绑定编辑器的启动方式。

空载回归测试覆盖终止调用端 Job 后宿主仍存活，以及独占锁拒绝重复启动。仅在已授权的 Windows 桌面会话执行：

```powershell
$env:MORTAL_TEST_DETACHED_LAUNCH = '1'
& $mortalPython -m unittest scripts.test_start_oracle_critic_detached -v
Remove-Item Env:MORTAL_TEST_DETACHED_LAUNCH
```

### 单次评测

只评估已选择的 checkpoint 时，使用 dev：

```powershell
& $mortalPython -m mortal.online.pretrain_oracle_critic `
  --eval-only --eval-split dev `
  --eval-checkpoint .\logs\oracle_review\checkpoints\candidate.pth
```

上面的路径需替换为实际产物；有效配置也必须匹配。当前生产 loader 的输入复现性问题尚未修复，命令成功不能消除这个限制。sealed test 的开放条件见 [Oracle 状态](../status/oracle-critic-mainline.md#下一步与通过条件)。

actor 的 Oracle 依赖度可用另一入口诊断；它不等同于独立 critic 的离线资格检查：

```powershell
& $mortalPython -m mortal.eval.oracle_dependency_eval `
  --state-file .\logs\oracle_review\checkpoints\actor.pth `
  --games 400 --device cpu --modes true zero shuffled `
  --output-json .\logs\oracle_review\actor_dependency.json
```

## 在线 RL

[online_machine_modes.py](../../mortal/online/online_machine_modes.py) 生成机器配置，`independent_arm` 建独立 run，`worker` 连接已有 server。先确认 [RL 协议](../status/online-rl-mainline.md)，再生成配置。下面只示范排因用的短 profile：

```powershell
& $mortalPython -m mortal.online.online_machine_modes `
  --mode independent_arm --base-config .\mortal\config.toml `
  --experiment-profile ms_rl1_minimal_500 --opponent-pool-preset validation `
  --output .\logs\online_modes\desktop_review\config.toml `
  --runtime-root .\logs\online_modes\desktop_review
```

检查生成结果后，在三个终端分别设置解释器变量，并分别执行一条角色命令：

```powershell
& $mortalPython -m mortal.online.online_role_runner server --config .\logs\online_modes\desktop_review\config.toml --arm current_config
& $mortalPython -m mortal.online.online_role_runner trainer --config .\logs\online_modes\desktop_review\config.toml --arm current_config
& $mortalPython -m mortal.online.online_role_runner client --config .\logs\online_modes\desktop_review\config.toml --arm current_config
```

需要额外 worker 时，生成 `--mode worker` 配置并明确 `--remote-host` / `--remote-port`；仅启动 client。TCP / pickle 仅接可信本机输入，不直接暴露到公网。远程场景见 [远程流程](remote-ops.md)。

[run_online_fidelity.py](../../mortal/research/run_online_fidelity.py) 保留 `calibration / emit_protocol_decide / rank_protocol_decide / emit_winner_refine` 等编排入口；使用 `--help` 查参数，不要复用归档里的历史 run name 和旧配方。

## 1v3 与配对评测

在独立评测配置中设置 `[1v3.challenger].state_file` 与 champion。设置 `[control].state_file` 不能替代 challenger。机器默认会影响局数，受控示例明确指定 500 个 seed 组，即四座轮换共 2000 局：

```powershell
$env:MORTAL_CFG = (Resolve-Path .\logs\eval_review\config.toml).Path
$env:MORTAL_1V3_SEED_COUNT = '500'
$env:MORTAL_1V3_SHARD_COUNT = '1'
& $mortalPython -m mortal.eval.one_vs_three
Remove-Item Env:MORTAL_1V3_SEED_COUNT, Env:MORTAL_1V3_SHARD_COUNT
```

这是固定局数的调用示例，正式样本预算仍按协议决定。比较时冻结 seed/key、对手、四座轮换、搜索开关、源码与 checkpoint hash，保留完整原始对局。

对已有匹配对局计算配对区间：

```powershell
& $mortalPython -m mortal.eval.paired_1v3 `
  --candidate-log-dir .\logs\eval_review\candidate `
  --reference-log-dir .\logs\eval_review\reference `
  --output .\logs\eval_review\paired.json
```

默认 challenger 名为 `mortal`，实际日志名称不同时显式传 `--candidate-name` / `--reference-name`。脚本按 seed 组配对；不匹配的日志必须排查，不能拿独立样本冒充配对结果。

## 观察、接入与检查

- TensorBoard 使用实际 run 中配置的目录：`& $mortalPython -m tensorboard.main --logdir <实际目录>`。
- 看运行进度时交叉检查 metrics、supervisor、最新完整 checkpoint 和恢复后的实际更新；不只看进程存在。
- 平台接入由各自 README 维护：[雀魂](../../integrations/majsoul/README.md)、[RiichiLab](../../integrations/riichilab/README.md)、[MahjongCopilot](../../MahjongCopilot/readme.md)。
- 代码测试命令在 [重构与验证](code-health.md#按改动验证)；文档检查在 [文档维护](doc-maintenance.md#自动检查)。
