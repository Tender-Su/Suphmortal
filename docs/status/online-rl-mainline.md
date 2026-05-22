# 在线 RL 主线结论

这份文档只回答三件事：

1. 当前代码已经接了什么
2. 当前默认验证路线是什么
3. 目前已经确认了哪些结果

它不再混入旧版 Phase 记录和大段历史过程。

## 当前代码能力

### 机器模式与对手池

- `mortal/online/online_machine_modes.py`
  - 支持 `independent_arm / worker`
  - 支持 `default / validation / formal` opponent-pool preset
  - 已内置一组短窗、长窗和 smoke profile 家族
- `mortal/online/online_role_runner.py`
  - `server / trainer / client` 三角色统一入口

### 已接线的训练能力

- `mortal/online/train_online.py`
  - actor Oracle guiding
  - Oracle critic
  - Step-Level GAE
  - 版本级 replay importance sampling
  - 单边 `entropy_floor`
  - chosen-action `logit_thres` gradient gate
- `mortal/online/server.py` / `mortal/core/common.py` / `mortal/online/client.py`
  - trainer 发布 `param_version`
  - worker / replay 会携带策略版本元信息

### 已接线的推理与评测能力

- `mortal/eval/engine.py` + `mortal/eval/search_runtime.py`
  - local belief search / planner
- `mortal/eval/oracle_experiments.py`
  - `visible_only / actor_true / actor_shuffled / critic_only`
- `mortal/eval/oracle_dependency_eval.py`
  - `true / zero / shuffled` dependency eval
- `mortal/research/run_online_fidelity.py`
  - RL 版 `calibration / protocol_decide / winner_refine`

### 导出与评测口径

- actor Oracle 开启时：
  - `state_file` 可以保留 live 训练态
  - `best_state_file` 导出 same-structure zero-oracle deploy checkpoint
- `1v3` / `test_play` 默认都按 visible-only zero-oracle 口径评测

## 当前默认验证路线

### 当前底座

- visible-only actor
- `policy.online_action_scope='all'`
- `grp.label_smoothing=0`
- `search.enabled=false`
- `search_distill.enabled=false`
- `value.oracle_critic=false`
- `oracle_dependency_eval.enabled=false`
- `online.importance_sampling.vtrace_mode=auto`

### 当前执行顺序

- 默认先做 `RL-1` 单臂加回，不把 `RL-1 / RL-2` pair 当第一步
- 默认先过三层门：
  - `500 -> 1500 -> 3000`
- 通过后再扩到 `20k / 40k`

### 当前常用 profile 家族

- `minimal`
- `add_rank_opp_danger`
- `add_value_gae_is`
- `add_value_gae_is_oracle_critic`
- `add_value_gae_is_rank_opp_danger`
- `shared_stack`

### 当前 opponent pool 纪律

- `validation`
  - 排因、Oracle 对照、短窗微探针
- `formal`
  - 更长窗口、冲上限训练

### 当前冻结 opponent pool preset

这两套 preset 当前都不是占位示意，而是已经冻结到仓库配置里的真实成员。

- 当前代码行为：
  - `baseline.train` 仍是 session 级抽样
  - 每次 `train_play session` 先从池里抽一个 checkpoint
  - 该 session 里的三家对手共用这个 checkpoint
  - 不是“每个座位独立抽样”

- `validation`
  - `state_file = ./checkpoints/opponent_pool/baseline_anchor_legacy.pth`
  - `champion_state_file = ./checkpoints/opponent_pool/sl_canonical_supervised_champion.pth`
  - `anchor_state_file = ./checkpoints/opponent_pool/baseline_anchor_legacy.pth`
  - `history_state_files = ['./checkpoints/opponent_pool/rl1_aux_only_500_20260415_best.pth']`
  - `champion_prob = 0.20`
  - `anchor_prob = 0.70`
  - `history_prob = 0.10`
  - 作用：固定排因、Oracle 对照、短窗门测
  - 当前成员来源：
    - `baseline_anchor_legacy.pth <- mortal/checkpoints/baseline.pth`
    - `sl_canonical_supervised_champion.pth <- mortal/checkpoints/sl_canonical.pth`
    - `rl1_aux_only_500_20260415_best.pth <- logs/online_modes/independent_arm/rl1_add_rank_opp_danger_500_20260415_020802/checkpoints/best.pth`

- `formal`
  - `state_file = ./checkpoints/opponent_pool/baseline_anchor_legacy.pth`
  - `champion_state_file = ./checkpoints/opponent_pool/rl1_vgi_500_validation_20260416_best.pth`
  - `anchor_state_file = ./checkpoints/opponent_pool/baseline_anchor_legacy.pth`
  - `history_state_files = [`
  - `  './checkpoints/opponent_pool/sl_canonical_supervised_champion.pth',`
  - `  './checkpoints/opponent_pool/rl1_aux_only_500_20260415_best.pth',`
  - `  './checkpoints/opponent_pool/rl1_mortal_policy_smoke_4k_20260411_best.pth',`
  - `]`
  - `champion_prob = 0.50`
  - `anchor_prob = 0.25`
  - `history_prob = 0.25`
  - 作用：更长窗口、冲上限训练
  - 当前成员来源：
    - `baseline_anchor_legacy.pth <- mortal/checkpoints/baseline.pth`
    - `rl1_vgi_500_validation_20260416_best.pth <- logs/online_modes/independent_arm/20260416_011753__rl1_add_value_gae_is_500_validation/checkpoints/best.pth`
    - `sl_canonical_supervised_champion.pth <- mortal/checkpoints/sl_canonical.pth`
    - `rl1_aux_only_500_20260415_best.pth <- logs/online_modes/independent_arm/rl1_add_rank_opp_danger_500_20260415_020802/checkpoints/best.pth`
    - `rl1_mortal_policy_smoke_4k_20260411_best.pth <- logs/online_modes/independent_arm/rl1_mortal_policy_smoke_4k_20260411_231608/checkpoints/best.pth`

- `baseline.test`
  - 不属于这两套训练池
  - 继续作为受控评测基线使用

### 当前四个稳定器口径

- `logit_thres`
  - 当前实现是 chosen-action gradient gate
  - 不再做前向 logits clamp
- `entropy_floor`
  - 单边熵下限
  - 只在熵低于下限时加大熵权重，不再双边追 target
- `importance_rho_clip / importance_c_clip`
  - 当前在线 `value + GAE` 路径已接入真 `V-trace`
  - `online.importance_sampling.vtrace_mode=auto` 时，只对足够陈旧的 replay 启用
  - fresh replay 继续走 plain GAE
- `warmup`
  - 微探针固定为 `500 -> 50`
  - `1500 -> 100`
  - `3000 -> 150`

## 当前已经确认的结果

### 最小 PPO 底座

- `rl1_mortal_policy_smoke_4k_20260411_231608`
  - `step 0`: `avg_rank=2.5325`, `avg_pt=-3.0375`
  - `step 1000`: `avg_rank=2.5225`, `avg_pt=-2.475`
  - 结论：统一模型、all-action、无 label smoothing 的最小 PPO 已经拿到早期正向

- `rl2_mortal_policy_smoke_4k_20260411_234157`
  - `step 0`: `avg_rank=2.5325`, `avg_pt=-3.0375`
  - `step 1000`: `avg_rank=2.57625`, `avg_pt=-6.13125`
  - `step 2000`: `avg_rank=2.6150`, `avg_pt=-9.45`
  - 结论：短窗里 `RL-2` 明显没优于 `RL-1`，但这仍不足以单独判死完整 Oracle curriculum

### 三辅助头 vs `value / GAE / replay IS`

- `2026-04-15` 的正式 `1v3=2000` 对照：
  - `sl_canonical step0`: `avg_rank=2.507`, `avg_pt=-0.945`
  - `add_rank_opp_danger_500`: `avg_rank=2.5075`, `avg_pt=+0.3825`
  - `add_value_gae_is_500`: `avg_rank=2.53`, `avg_pt=-2.295`
  - `add_value_gae_is_rank_opp_danger_500`: `avg_rank=2.511`, `avg_pt=-1.485`
- 当前解读：
  - 三辅助头本身已经出现了短窗微正信号
  - 负向更像来自 `value / GAE / replay importance sampling` 这组三项在当前 critic/target 条件下仍未站稳

### 冻结 validation 池上的后续补跑

- `ms_rl1_add_value_gae_is_1500`
  - 训练内 `test_play=400`：`avg_rank=2.615`, `avg_pt=-10.35`
  - 正式 `1v3=2000`：`avg_rank=2.5515`, `avg_pt=-3.6675`
- `ms_rl1_add_value_gae_is_rank_opp_danger_500`
  - 训练内 `test_play=200`：`avg_rank=2.585`, `avg_pt=-6.525`
  - 正式 `1v3=2000`：`avg_rank=2.54`, `avg_pt=-4.095`
- 当前解读：
  - `value / GAE / replay IS` 的正向信号目前无法稳定放大
  - 一旦拉长到 `1500` 或叠上三辅助头，就回到明显负收益

### Oracle critic protocol decide 重测

- `2026-04-16` 的 `oracle_critic=true` 五臂 `protocol_decide` 已按正式 `1v3=2000` 口径重测
- 形式 winner 仍是 `w004220`，但结论不显著：
  - top-2 gap 只有 `0.09 pt`
  - `ambiguous=true`
  - `winner_flipped_by_stderr=true`
- 当前解读：
  - 不能继续按这个 winner 推 `winner_refine`
  - 这不等价于“Oracle critic 没价值”
  - 更合理的怀疑是：从 visible-only SL bridge 出来的 `oracle_brain` 一开始还不会稳定利用 Oracle 信息，直接接入 `value / GAE` 太猛

## 当前最重要的判读规则

- 是否“过门”优先看正式 `1v3`
- 训练内 `test_play=200/400` 只保留为诊断信号，不再直接当 protocol ranking 依据
- `RL-1 / RL-2` 成对对照要后置到共享优化栈已经站稳之后
- 当前更值得优先排查的是：
  - 为什么 `value / GAE / replay IS` 的早期正向不能持续
  - 如何让 `Oracle critic` 在还没成熟时不要过早强力支配 actor
  - 相关文献观测见 `docs/research/online-rl/oracle-critic-literature-observations-2026-04-16.md`

## 当前下一步

1. 不再把 `ms_rl1_add_value_gae_is_20k` 当当前下一步判断依据；它已经越过短窗验证阶段且仍明显负增长。
2. 先继续扩 `add_rank_opp_danger`，当前最自然的下一门是 `1500`。
3. `add_value_gae_is_rank_opp_danger` 先不要直接扩窗；应先回头检查为什么 `value / GAE / replay IS` 会把已有的微正信号拉坏。
4. 对 `Oracle critic`，短期更推荐先实现轻量 `critic influence ramp`：`value_loss` 可以从 step 0 学，但 critic 对 actor / GAE 的影响从小到大；hard `oracle_critic_warmup` 暂时只作为后续备选，不作为论文公认默认。
5. 只有某一层过了 `3000` 门，才把它扩到 `20k`；只有 `20k` 仍为正，才继续看 `40k`。
6. 只有当共享优化栈已经稳定前进时，才在同一层做匹配的 `RL-1 / RL-2` 对照，不再在裸 smoke 上直接判死 `RL-2`。

## 当前操作提醒

- `one_vs_three.py` 真正读取 challenger 的路径是 `[1v3.challenger].state_file`
- 这台 `RTX 5070 Ti` 默认会放大到 `seed_count=1024 / shard_count=4`
- 若要强制回到受控 `1v3 = 2000`：

```powershell
$env:MORTAL_1V3_SEED_COUNT = '500'
$env:MORTAL_1V3_SHARD_COUNT = '1'
```

- 做 opponent pool 切换时，不要手改 `baseline.train`，优先用 `--opponent-pool-preset`
- 训练阶段默认关闭 `search`；后续若要评估 `search`，按推理期系统增强单独做 A/B
