# 真实 critic-only 两更新工程检查

> 范围：2026-09-30 新增独立 runner；云端只做 CPU 轻量验证。真实 checkpoint / Torch / native 集成结果须由固定 commit 的 Windows runner 另行提供。本检查不产生研究候选，不证明 critic 成熟度或牌力。

## 要回答的问题

已有 8 局 frozen C50k 对 canonical 的完整日志，是否能通过真实 `train_online.train()` 完成两次成功的 critic-only optimizer 更新，同时保持 actor 的参数、buffers、eval 行为和 logits 完全不变？直接复用现有输入，不增加对局或重做三个 critic 对照。

入口：[critic_only_update_check.py](../../mortal/research/critic_only_update_check.py)。它不修改 production train/server，不调用会创建子进程的 `train_online.main()`。

## 输入与运行

- 完整训练 TOML；`aux` / `supervised` 两个完整 section 从精确 actor checkpoint 继承，不清空辅助目标配方
- 原始 probe 的 C50k actor、canonical opponent 和已注册 warm40k critic；SHA256 必须分别匹配原 provenance 中 `actor` / `opponent` / `warm40k`，内部 steps 必须是 50000 / 40000
- 原始 8 局 probe 输出根；不是重用结果的二次链。复用既有 `load_verified_rollout` 检查源码/native/采样身份、完整日志、注册 SHA256、两个完整四座 seed groups，仍只选择 `['trainee']`
- 从原 provenance 核对的 seed-start / seed-key / sampling-seed、device 与线程数；固定 imputation seed 为 20260905
- 从未存在的新输出目录，且位于原 rollout 目录之外；明确指定正数工程 LR

命令示意，所有路径、seed 和 LR 必须替换成已核验值：

```powershell
& C:\ProgramData\anaconda3\envs\mortal\python.exe -m mortal.research.critic_only_update_check `
  --config <full-training.toml> --actor <C50k.pth> --critic <warm40k.pth> `
  --opponent <canonical.pth> --reuse-rollout-dir <original-eight-game-root> `
  --output-dir <never-existing-engineering-output> `
  --seed-start <original-start> --seed-key <original-key> --sampling-seed <original-seed> `
  --engineering-lr <explicit-positive-engineering-lr> `
  --device cuda --batch-size 32 --torch-threads 1 --rayon-threads 4
```

真实执行必须使用[独立 deadline supervisor](../agent/deadline-supervisor.md)，在 Git 同步的固定、干净 commit 上作为一个 role 运行，不启动 server/client。上面的参数数组放进 lease 的 `argv`；算力截止和保存余量仍适用。不要设置 `MORTAL_ORACLE_ARM` / `MORTAL_ORACLE_ARTIFACT_SUFFIX`。原训练的协作停止信号保留；提前停止不会伪报两更新通过。

## 实际训练契约

- 新 state 路径 + `online.init_state_file` 权重初始化 actor，独立 `value.oracle_critic_state_file` 初始化 critic；optimizer / scheduler / scaler / exact update clock 是新阶段
- 实际 online GAE，gamma=1、lambda=1、all_players、score_rank；沿用 MSE 和配置中的 zero-sum 项。结构、fusion、head 和 exact_zero_sum 由已核验 critic 元数据确定
- actor visible、guiding/search off；保留真实 canonical `TestPlayer` 构造，关闭初始和周期 test_play
- `critic_only=true` 保持 actor/policy eval + no_grad，生产 `policy_step_active=false` 门控全部 policy/aux objectives。完整 aux/supervised 配方与生产有效对齐逻辑不被绕过
- 封闭日志没有 pv 前缀，且 actor 完全相同：显式关闭 importance sampling / V-trace，不伪造行为版本
- `opt_step_every=1`、精确成功更新预算=2、`stop_at_max_steps=false`、不 compile；显式固定非零工程 LR，绝不视为研究推荐

只设四处边界：`common.drain` 返回一次注册原目录，第二次调用失败；`common.submit_param` 观察真实模型及 runtime，返回本地传输版本而不联网；轨迹方法 wrapper 仅加入固定 seed，原样调用实际方法；step wrapper 原样调用真实 `observed_scaler_step` 一次，并只读观察真实 `train_batch` 局部 loss / gradients / clock。没有替换模型、loss、GAE、optimizer、scaler、clock 或 checkpoint saver。

## 验收与产物

初次 publication 前核对 live actor/policy 与输入完整 state 相等，并核对 live critic 已载入指定权重。每次实际更新前要求 actor/policy gradients 全 None、policy gate 关闭、全部 aux loss 为零、实际 value/total loss 有限。成功更新的 critic gradients 和更新后 state 必须有限。

production 原子保存 latest 并退出 86 后，严格 reload actor、policy、critic、value；要求 actor 参数/buffers 与输入相等、同一真实 visible obs/mask 在同 device/eval 精度下 logits 完全相等，Oracle brain 与 value head 各至少一个 tensor 改变。真实 checkpoint 的 exact clock 必须 successes=2、attempts=successes+skips，与观察记录一致且无 inherited/legacy offset。原 checkpoint、输入 config、provenance、outcomes 和八份日志的 hash 前后必须不变。

新目录含 request、effective config、provenance、observed_updates、input_immutability、result 与 `engineering_only_latest.pth`；失败写 failure。只有全部检查完成才写 passed result，provenance complete 仍须与 result/immutability/日志一并审阅。底层退出 86 被 runner 验证后正常退出 0；其它退出码不吞掉。

这仅证明两个真实更新的工程连接。fixed-imputed Oracle 不是完整真实牌山重建；两次更新不证明 critic readiness、不证明 PPO 集成或模型强度，不自动追加 256 局或作为后续 RL 默认起点。

## 轻量测试

```sh
python -m unittest mortal.tests.test_critic_only_update_check mortal.tests.test_critic_calibration mortal.tests.test_update_clock mortal.tests.test_frozen_actor_critic_probe
python -m py_compile mortal/research/critic_only_update_check.py mortal/tests/test_critic_only_update_check.py
```

轻量测试覆盖配置不可变与完整配方、路径碰撞、角色绑定、单次 drain、文件增删改、透明 seed/step wrapper、异常与 skip 不伪造 clock，以及仅允许四处 patch 的实际 train() 调用。没有 Torch 的环境通过这些测试，不能称为真实训练通过。
