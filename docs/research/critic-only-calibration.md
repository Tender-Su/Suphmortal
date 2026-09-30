# 独立 critic-only 校准阶段

`[value].critic_only = true` 是显式 opt-in。它让整个运行保持真正的 actor-freeze：actor / policy 使用 `eval()` 和 `no_grad()`，只训练 value head，以及启用时的独立 Oracle critic。visible value head 读取 detached actor features；普通 PPO 的 policy-inactive 交替步仍允许 value loss 更新共享 trunk，语义不变。

这不是已观测到的 v4 GN actor 参数漂移修复：原有 true warmup 已使用 `no_grad()`。独立阶段避免在同一 GAE chunk 中解除冻结，以及在同一梯度累积窗口跨越 warmup / PPO 边界。新模式同时固定 BN buffers、dropout 的 eval 行为；有限 warmup 的旧行为不改。

## 配置单位

- `[value].critic_only`：布尔值，默认 false；true 要求 `value.enabled=true`。整段运行不自动切换 PPO
- `[online].max_successful_optimizer_steps`：非负整数，默认 0 表示不限制。只统计本阶段实际成功的 optimizer 更新，AMP overflow skip 不计入；历史推定 / inherited progress 不计入
- `[control].opt_step_every`：每次 optimizer 尝试所需的训练 microbatch 数；新上限仅在完整 optimizer 边界检查。终止前先原子保存 latest，再使用已有终止退出码 86，不在累积中途截断
- `[optim.scheduler].max_steps`、`warm_up_steps`、有限 `[value].critic_warmup_steps`：仍是旧的训练 microbatch 时钟，不因新上限改单位。成功更新预算不改变 LR 时钟，overflow 仍推进 microbatch LR
- `[online].stop_at_max_steps`：旧 scheduler/data 上限开关，默认 true。若本次只想受成功更新预算约束，显式设 false；两种上限同时开启时，任一都可能先结束，旧上限行为不改

上限必须由实验计划给出；没有任意默认校准步数。运行结束不代表 critic 已可用。

## 续训与阶段切换

1. 校准用独立输出路径、run name、显式成功更新预算和 `critic_only=true`。保留已批准的 reward / value / GAE 契约与机器截止管理
2. 同阶段 full checkpoint resume 保留 exact attempts / successes / skips，因此预算是整个阶段累计值，重启不会续赠额度。没有 `optimizer_update_clock` 的旧 checkpoint 不允许启用有界 full resume，应作为新阶段 weights-only 初始化。full resume 不提供精确 replay cursor / old-policy snapshot 恢复
3. weights-only 路径重置 exact attempts / successes / skips。旧 `control.state_file` 的 weights-only fallback 可能保留历史 microbatch 与兼容 progress，因此新阶段应使用不存在的新 `control.state_file`，并以 `[online].init_state_file` 指向上阶段 latest，使优化器、scheduler、scaler、microbatch 和 exact update 计数从新阶段起点开始
4. 检查保存 checkpoint 的 actor / policy state（参数及 buffers）与校准起点逐项相等，并在固定输入、eval 模式比较 logits；独立评估 critic。达到步数不能替代 readiness 证据
5. 通过后才启动独立 PPO 进程：新输出 state 路径、`critic_only=false`、`critic_warmup_steps=0`，通过 init_state_file 载入校准权重，按 PPO 计划设置或取消成功更新上限。新进程从数据重新计算 GAE；不复用校准阶段 chunk。full resume 中改变 critic_only 会被拒绝。避免配置另一个 `value.oracle_critic_state_file` 覆盖刚校准的 critic

## 验证

云端轻量检查：

```sh
python -m unittest mortal.tests.test_critic_calibration mortal.tests.test_update_clock
python -m py_compile mortal/online/critic_calibration.py mortal/online/train_online.py
```

在 Git 同步的固定 commit runner 上执行真实 Torch 测试：

```sh
python -m unittest mortal.tests.test_critic_calibration_torch mortal.tests.test_update_clock_torch
```

覆盖实际冻结 / mode helper、GN 与 BN buffers、固定输入 logits、visible value head 与 Oracle 分支可训练性、eval 后恢复、普通交替步共享 trunk 梯度，以及实际 Brain / CategoricalPolicy / ValueHead。没有 Torch 或 libriichi 时相应测试 skip；skip 不是 runner 通过证据。

注意：现有辅助损失对齐逻辑仍会继承 checkpoint 中的 aux / supervised 配置。独立阶段重置 optimizer 等运行状态，不意味着自动重置全部损失配方；启动前必须记录并核对实际生效配方。
