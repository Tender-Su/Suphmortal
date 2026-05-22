# Oracle 四臂对照与依赖度评测协议

> 原始长文写于 `2026-04-10`。  
> 当前默认以 `docs/status/online-rl-mainline.md` 为准；本文只保留 Oracle 对照的核心实验设计。

## 要回答的问题

actor Oracle 的收益到底来自哪里：

- 是否依赖 hidden info 的真实性？
- 是否只是“加一条强通道，再在 continuation 里删掉”的正则化效果？
- 如果 Oracle 只放在 critic，是否已经足够甚至更稳？

## 四个实验臂

| 实验臂 | actor | critic | 作用 |
| --- | --- | --- | --- |
| `visible_only` | 只看公开信息 | 无 Oracle | 朴素基线 |
| `actor_true` | 真实 Oracle hidden info，随后退火到 zero Oracle | Oracle | 检验当前 actor Oracle recipe |
| `actor_shuffled` | 错配 hidden info，退火配置对齐 `actor_true` | Oracle | 检验收益是否依赖真实 hidden info |
| `critic_only` | 只看公开信息 | Oracle | 检验 Oracle 是否更适合只做 critic 信号 |

## 判读规则

- `actor_true >> actor_shuffled` 且 `actor_true > critic_only`：actor Oracle 确实有独立信息价值。
- `actor_shuffled ~= actor_true`：优先怀疑收益来自噪声通道和 continuation，而不是真实 curriculum。
- `critic_only >= actor_true`：Oracle 更适合留在 critic / teacher / planner，不该优先直接喂 deployable actor。
- 任一结论都必须看正式 `1v3`，训练内 `test_play` 只做诊断。

## Dependency Eval

对同一份 actor-shaped checkpoint，固定对手和种子，比较三种输入口径：

- `true`
- `zero`
- `shuffled`

关注：

- `avg_rank`
- `avg_pt`
- `agari_rate`
- `houjuu_rate`
- `delta_vs_zero_avg_pt`
- `delta_vs_zero_avg_rank`

判读：

- `true >> zero`：仍依赖真实 Oracle，部署风险高。
- `shuffled << zero`：模型会被错配 hidden info 带偏。
- `true ~= zero` 且 `shuffled ~= zero`：模型基本退回 visible-only，可部署性更好。

## 运行入口

```powershell
.\scripts\run_online.bat visible_only
.\scripts\run_online.bat actor_true
.\scripts\run_online.bat actor_shuffled
.\scripts\run_online.bat critic_only

.\scripts\run_oracle_dependency_eval.bat
.\scripts\run_oracle_dependency_eval.bat .\checkpoints\best_actor_true.pth --games 3000 --output-json .\logs\oracle_dependency\actor_true_eval.json
```

## 当前执行纪律

- 第一轮用 `validation` opponent pool。
- 训练阶段保持 `search=false`。
- `rank / opponent / danger` 可以作为背景能力保留，但 Oracle 对照的主变量必须清楚。
- 如果结果指向“噪声 + 去噪”，再追加 `visible_only + feature_dropout` 或 structured visible corruption 对照。

## 与当前主线的关系

- 当前更紧急的问题是让共享优化栈站稳，不是马上扩大完整四臂长跑。
- Oracle critic 方向短期更适合先做 `critic influence ramp`，避免还没学会用 Oracle 的 critic 过早支配 actor。
