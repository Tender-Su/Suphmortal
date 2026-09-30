# 在线 RL 主线结论

> 历史归档 · 归档整理：2026-09-05。正文保留当时的事实、判断和命令，不作为当前运行依据。当前入口见 [文档地图](../../README.md)；旧 SL / RL 强度及 Oracle 验证结论须结合 [独立审计](../../research/sl-rl-audit-2026-09-05.md) 阅读。

这份文档只回答三件事：

1. 当前代码已经接了什么
2. 当前默认验证路线是什么
3. 目前已经确认了哪些结果

它不再混入旧版 Phase 记录和大段历史过程。

## 2026-09-05 审计后的当前结论

- 完整证据和执行建议见 [SL / RL 独立审计](../../research/sl-rl-audit-2026-09-05.md)。本节和文末“当前下一步”覆盖下面历史探索中的强弱推断。
- 37 个 C/D/E 评测反复使用同一批 500 seed × 四座位。对保留完整原始日志的 32 个 RL 候选重新做配对聚类 bootstrap，没有一个相对 canonical 的名义 95% CI 下界大于零；不能据旧点估计认定最佳 clip/value 或稳定增益，也不能据此证明 RL 无效。
- E 9k 保留为历史探索候选，不再表述为已经证明更强的推荐起点。它与 E 10k policy-stop 的 actor/policy 张量哈希完全相同，原评测分差不能说明两个 actor 的学习强度不同；E 9k 原始逐局日志缺失。
- 当前 Oracle 使用真实 `score_rank_mc`、`all_players`，已有预测能力且通过真实牌谱 CPU 接入 smoke。但默认 loader 随机补全未知牌山，实际 dev 的同一输入重复解码会变化；必须先固定验证特征、做 no-update A/A，再重建 best 比较基准。
- 自适应控制器的 guardrails 目前只作延长阶段的补偿，不否决 `update_best`。当前微小 primary 门限与验证功效也不匹配；在修复验证前，不凭这些微小差距扩大训练或发布。
- `env.pts=[6,4,2,0]` 与正式 `avg_pt=[90,45,0,-135]` 不是仿射等价目标。新奖励方向应同时对齐 Oracle、value、GAE 和评测；本次已添加奖励契约和 resume 签名检查，未修改活跃 run 的目标。
- 旧 V-trace probe 的 value 递推和最终 PPO actor surrogate 应分开审计，不能把该组合的一次负结果推广为否定 V-trace。
- 本次直接修复了未跟踪 replay 整批漏过滤、未知版本误保留及单样本 advantage NaN；335 项相关 CPU 测试通过。活跃预训、loader 和自适应源码保持冻结。

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

## 历史运行结果与当时判读

以下保留原始点估计和实验经过；其中“明显改善”“winner”“路线偏负”等属于当时判读，强度与因果结论以 2026-09-05 审计限定为准。2000 局运行完成不等于已通过独立统计确认。

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

### 双塔 Oracle critic C/D/E 长窗

- 已新增独立 C/D/E 配置与 runner：
  - `mortal/online/oracle_cde_configs.py`
  - `scripts/build_oracle_cde_config.py`
  - `scripts/run_oracle_cde_online.py`
- `dual_D_s20000`
  - batch `384` 跑到 `15500` 后触发 `system_mem_percent>=85.0x3`
  - 改 batch `320` 从 `15500` 续到 `20000`
  - 最终 `value_loss=0.343165`，`ratio_var=0.004302`
  - 同 seed 小 `1v3=2000`：`logs/oracle_cde/dual_D_s20000_seed2026052350_1v3`
    - `seed_key=2026052350`, `seed_count=500`, `shard_count=1`
    - `challenger rankings=[480, 489, 491, 540]`, `avg_rank=2.5455`, `avg_pt=-3.8475`
- `dual_E_w3000_s20000_b320`
  - 从双塔 outcome/score-rank Oracle critic best checkpoint 初始化
  - `critic_warmup_steps=3000`，warmup 期间 actor freeze
  - 干净跑到 `20000`
  - 最终 `value_loss=0.336286`，`ratio_var=0.006959`
- `dual_E_w3000_s50000`
  - 从 `dual_E_w3000_s20000_b320` 续跑
  - 修复了 resume 时 scheduler horizon 被旧 checkpoint 固定在 `20000` 的问题；续训会把 scheduler 参数对齐当前 config
  - batch `320` 跑到 `48000` 后触发 `gpu_mem_mb>=15000.0x3`，这是资源 guard 生效，不是训练错误
  - batch `288` 从 `48000` 续到 `50000` 干净退出
  - 最终 `value_loss=0.324371`，`ratio_var=0.003890`，`coverage=0.985375`
  - 50k 末段策略更新非常稳，但 value loss 主要是平台震荡；不能只按 value loss 判定强度
  - 小 `1v3=2000` 门槛已跑：`logs/oracle_cde/dual_E_w3000_s50000_1v3`
    - challenger: `logs/oracle_cde/dual_E_w3000_s50000_b288_resume48000/checkpoints/mortal.pth`
    - champion: `mortal/checkpoints/baseline.pth`
    - `seed_key=2026052350`, `seed_count=500`, `shard_count=1`
    - `challenger rankings=[480, 502, 494, 524]`, `avg_rank=2.531`, `avg_pt=-2.475`
- 同 seed 初始化对照：
  - `sl_canonical_seed2026052350_1v3`
    - challenger: `mortal/checkpoints/sl_canonical.pth`
    - `seed_key=2026052350`, `seed_count=500`, `shard_count=1`
    - `challenger rankings=[495, 483, 519, 503]`, `avg_rank=2.515`, `avg_pt=-0.81`
  - 相对初始化底座，`dual_E_w3000_s50000` 在同 seed 小窗里低了 `1.665pt`
  - 相对初始化底座，`dual_D_s20000` 在同 seed 小窗里低了 `3.0375pt`
- `dual_D_s3000_fix_old_policy_entropy`
  - 修复点：
    - `dist.entropy().view(-1, 1)` 与 `[B]` 的 `clip_loss` 相加会静默广播成 `[B, B]`；已改成同 shape 检查的 `compute_policy_objective_loss`
    - `old_update_every=400` 之前被包在 `save_every=500` 块内，实际每 `2000` step 才更新 old policy；已改成独立按 `400/800/1200/...` 更新
  - batch `288` 跑到 `2500` 后触发 `system_mem_percent>=85.0x3`，这是资源 guard 生效，不是训练错误
  - batch `224` 从 `2500` checkpoint 续到 `3000` 干净退出；但后续 20k 中窗到 `5000` 时观察到 role 级系统内存瞬时 `87.97%`
  - 3000 末端训练指标：`value_loss=0.363858`，`ratio_var=0.003816`，`coverage=1.0`
  - 同 seed 小 `1v3=2000`：`logs/oracle_cde/dual_D_s3000_fix_old_policy_entropy_seed2026052350_1v3`
    - challenger: `logs/oracle_cde/dual_D_s3000_fix_old_policy_entropy_b224_resume2500/checkpoints/mortal.pth`
    - `seed_key=2026052350`, `seed_count=500`, `shard_count=1`
    - `challenger rankings=[492, 507, 496, 505]`, `avg_rank=2.507`, `avg_pt=-0.54`
  - 相对初始化底座，修正版 `dual_D_s3000` 在同 seed 小窗里高了 `0.27pt`
- `dual_D_s5000_fix_scheduler_guard_b224_resume3000`
  - 直接把 3000-step checkpoint 续到更长 `max_steps` 时发现 scheduler resume bug：旧 checkpoint 在 3000 step 已到 `1e-5`，但换成 5000/20000 horizon 后会按新 cosine schedule 把 LR 抬回高位
  - 已加 `lr_increase_guard`：若 reconcile 后的 LR 高于 checkpoint 中已加载 optimizer LR，则保留 checkpoint LR，只记录被忽略的 scheduler 参数变更
  - 本 run 的 guard 日志：loaded LR `[1e-05, 1e-05]`，proposed LR `[4.395315e-05, 4.395315e-05]`
  - 训练干净到 `5000`，`value_loss=0.359654`，`ratio_var=0.004399`，`coverage=1.0`
  - 资源：`steps/sec=3.80`，GPU memory max `10.9 GB`，system RAM max `82.0%`
  - 同 seed 小 `1v3=2000`：`logs/oracle_cde/dual_D_s5000_fix_scheduler_guard_seed2026052350_1v3`
    - `challenger rankings=[457, 521, 520, 502]`，`avg_rank=2.5335`，`avg_pt=-1.5975`
  - 对比未 guard 的坏 5k：`dual_D_s5000_fix_old_policy_entropy_seed2026052350_1v3` 为 `avg_pt=-7.8075`
  - 对比未 guard 的坏 20k：`dual_D_s20000_fix_old_policy_entropy_b192_resume5000_seed2026052350_1v3` 为 `avg_pt=-10.53`
  - 当前判读：scheduler bug 解释了 3k -> 5k 的大回退，但 guard 后 5k 仍低于 3k 和 `sl_canonical`；不能直接放大到 20k，先做最小变量 probe，优先降低 `value.weight`
- `dual_D_s5000_value_w002_cudnnbench_b192_resume3000`
  - 从同一个 3000-step checkpoint 续跑，只把 `[value].weight` 从 `0.05` 降到 `0.02`
  - 资源排查中发现两点：
    - 每 `400` step 更新 old policy 时不能 `deepcopy(mortal)` / `deepcopy(policy_net)`；应复用旧对象并 `load_state_dict`，否则 GPU 上会出现额外模型副本峰值
    - `repro.enabled=true` 默认会令 `cudnn_benchmark=false`，在 dual-tower online 上会选到高显存卷积路径；受控 seed 实验如果要稳定跑长窗，应显式 `allow_cudnn_benchmark=true`
  - 有效 run：batch `192`，`allow_cudnn_benchmark=true`，干净跑到 `5000`
  - 资源：`steps/sec=4.25`，GPU memory max `9.4 GB`，system RAM max `83.85%`
  - 同 seed 小 `1v3=2000`：`logs/oracle_cde/dual_D_s5000_value_w002_cudnnbench_seed2026052350_1v3`
    - challenger: `logs/oracle_cde/dual_D_s5000_value_w002_cudnnbench_b192_resume3000/checkpoints/mortal.pth`
    - `seed_key=2026052350`, `seed_count=500`, `shard_count=1`
    - `challenger rankings=[511, 489, 504, 496]`，`avg_rank=2.4925`，`avg_pt=+0.5175`
  - 当前判读：`value.weight=0.02` 明显改善 3k -> 5k 回落；这是当前 `dual_D` 的 5k winner
- `dual_D_s20000_value_w002_cudnnbench_b192_resume5000`
  - 从 5k winner 续到 20k；训练本身干净退出
  - 但 `online.importance_sampling.max_policy_versions=8` 不够：到后段 replay 行为策略缓存覆盖掉光，`replay_is/coverage=0`、`missing_fraction=1`，`version_gap_max` 约 `11.6`
  - 同 seed 小 `1v3=2000`：`logs/oracle_cde/dual_D_s20000_value_w002_cudnnbench_seed2026052350_1v3`
    - `challenger rankings=[492,476,516,516]`，`avg_rank=2.528`，`avg_pt=-1.98`
  - 判读：20k 退化不只是 value loss 问题，replay IS cache 窗口是明确 bug/机制问题
- `dual_D_s20000_value_w002_cache16_cudnnbench_b192_resume5000`
  - 修复/设置：`online.importance_sampling.max_policy_versions=16`
  - 新增 runner 安全性：
    - C/D/E runner 的 `--arm D/E/C` 会归一为旧 Oracle experiment 的 `current_config`，避免把 C/D/E 标签误传给 `MORTAL_ORACLE_ARM`
    - 配置生成器支持 `--resume-state-file`，会把 5k online checkpoint staged 到新 run 的 `[control].state_file`，避免误从 `sl_canonical` 启动
  - 训练从 5k winner 续到 20k，干净退出：`logs/oracle_cde/dual_D_s20000_value_w002_cache16_cudnnbench_b192_resume5000`
  - 训练末段机制指标：
    - `replay_is/coverage_min=1`，`replay_is/missing_fraction_max=0`
    - `replay_is/version_gap_max` 末段约 `7.3-9.4`，低于 cache16 窗口
    - `value_loss` 末段约 `0.340-0.352`
    - `entropy/entropy` 从 5k 后约 `0.466` 缓慢降到 `0.411`
    - `important_ratio/max` 末段约 `1.37-1.44`
  - 资源：`steps/sec=2.63`，GPU memory max `10.64 GB`，system RAM max `90.59%`；未触发 `92%` hard guard
  - 同 seed 小 `1v3=2000`：`logs/oracle_cde/dual_D_s20000_value_w002_cache16_cudnnbench_seed2026052350_1v3`
    - challenger: `logs/oracle_cde/dual_D_s20000_value_w002_cache16_cudnnbench_b192_resume5000/checkpoints/mortal.pth`
    - `seed_key=2026052350`, `seed_count=500`, `shard_count=1`
    - `challenger rankings=[486,522,487,505]`，`avg_rank=2.5055`，`avg_pt=-0.4725`
  - 当前判读：cache16 修住了 20k 后段 replay IS coverage 掉光的问题，并把 20k 从 cache8 的 `-1.98` 拉回到 `-0.4725`，也略高于 `sl_canonical -0.81`；但仍低于 5k winner 的 `+0.5175`，所以不能直接拉 40k/50k
- true V-trace probe
  - 初始 run：`dual_D_s20000_value_w002_vtrace_r1c1_cache16_cudnnbench_b192_resume5000`
    - 从 5k winner 续跑，`vtrace_target_rho_clip=1.0`、`vtrace_target_c_clip=1.0`、`vtrace_mode=auto`、`vtrace_min_version_gap=2`
    - trainer 正确启动 true V-trace：`true V-trace recursion armed for stale replay trajectories only`
    - 8000 step 后中断，原因不是资源越界，而是 `client` 一次 `connect()` 超时导致 runner 收掉三角色
    - 中断前资源安全：GPU memory max `10.67 GB`，system RAM max `89.04%`
    - 中断前训练信号已经可疑：`value_loss` 从 5500 step 的约 `0.365` 跳到 6000 step 的约 `0.906`，8000 step 仍约 `0.840`
  - 代码修复：
    - `mortal/online/client.py` 增加有上限的 server connect retry，避免一次瞬时 TCP 超时打断长跑
    - 连接成功后恢复 blocking socket，避免把 connect timeout 意外带到后续 `send_msg/recv_msg`
    - 修正 `replay_is/version_gap_max` 监控口径：以后记录窗口内最大 gap，而不是 batch-max 的均值；该修复只影响观测，不改变训练行为
  - 续跑完成：`dual_D_s20000_value_w002_vtrace_r1c1_cache16_cudnnbench_b192_resume8000_retryconn`
    - 从上条 8000-step checkpoint 续到 20000，干净退出：trainer `0`，errors 全 false
    - 资源：`steps/sec=3.10`，GPU memory max `12.31 GB`，system RAM max `89.94%`；未触发 `92%` hard guard
    - 末段机制指标：`coverage_min=1`，`missing_fraction_max=0`，`important_ratio/max` 约 `1.36-1.43`
    - 末段 `value_loss` 约 `0.78-0.88`，明显高于 plain cache16 20k 的约 `0.34-0.35`
  - 同 seed 小 `1v3=2000`：`logs/oracle_cde/dual_D_s20000_value_w002_vtrace_r1c1_cache16_seed2026052350_1v3`
    - challenger: `logs/oracle_cde/dual_D_s20000_value_w002_vtrace_r1c1_cache16_cudnnbench_b192_resume8000_retryconn/checkpoints/mortal.pth`
    - `seed_key=2026052350`, `seed_count=500`, `shard_count=1`
    - `challenger rankings=[491,467,523,519]`，`avg_rank=2.535`，`avg_pt=-2.43`
  - 当前判读：true V-trace 递推不是下一条该加码的路线；它资源上可跑、coverage 也健康，但训练 loss 与 `1v3` 都弱于 plain cache16 20k，更弱于 5k winner
- E-style actor-freeze calibration 12k probe
  - 代码/配置修正：
    - `value.critic_warmup_steps` 在训练器里是绝对 global step 阈值；配置生成器在 `--resume-state-file` 场景下会读取 checkpoint `steps`，把用户给的相对 warmup 转成绝对阈值
    - 本 run 从 5k winner resume，`critic_warmup_requested_steps=3000`、`critic_warmup_resume_steps=5000`，实际 `value.critic_warmup_steps=8000`
  - 训练完成：`logs/oracle_cde/dual_E_w3000_s12000_value_w002_cache16_cudnnbench_b192_resume5000`
    - trainer 正常退出，errors 全 false，`latest_step=12000`
    - 资源：`steps/sec=3.40`，GPU memory max `10.63 GB`，system RAM max `90.42%`
    - freeze 段到 step `8000`：`value_loss` 从 `0.364559` 到 `0.353675`，`coverage_min=1`，`missing_fraction_max=0`
    - 解冻后到 step `12000`：`value_loss=0.361135`，`entropy/entropy` 从约 `0.4699` 降到 `0.450834`，`important_ratio/max` 末段约 `1.34-1.38`
    - 全程 `replay_is/coverage_min=1`、`replay_is/missing_fraction_max=0`，cache16 机制健康
  - 同 seed 小 `1v3=2000`：`logs/oracle_cde/dual_E_w3000_s12000_value_w002_cache16_seed2026052350_1v3`
    - challenger: `logs/oracle_cde/dual_E_w3000_s12000_value_w002_cache16_cudnnbench_b192_resume5000/checkpoints/mortal.pth`
    - `seed_start=10000`，`seed_key=2026052350`，`seed_count=500`，`shard_count=1`
    - 首次主进程 stdout 没拿到最终 ranking 行，但 arena logs 完整生成 `2000` 局；随后用 `one_vs_three --worker` 同参数复算，结果一致
    - `challenger rankings=[492,506,500,502]`，`avg_rank=2.506`，`avg_pt=-0.36`
  - 当前判读：E 12k 略高于 plain cache16 20k 的 `-0.4725` 和 `sl_canonical -0.81`，但仍低于 5k winner `+0.5175`；它不是坏路，但还不能直接放大到 20k/50k。下一步应解释解冻后 entropy / policy drift 为什么仍然吃掉 5k winner 的优势。
- E-style freeze/unfreeze isolation probes
  - 8k freeze-only：`logs/oracle_cde/dual_E_w3000_s8000_value_w002_cache16_cudnnbench_b192_resume5000`
    - 从 5k winner resume，`critic_warmup_requested_steps=3000`、`critic_warmup_resume_steps=5000`、实际 `value.critic_warmup_steps=8000`，`max_steps=8000`
    - 训练完成：`latest_step=8000`，trainer `0`，errors 全 false
    - 资源：`steps/sec=5.06`，GPU memory max `9.65 GB`，system RAM max `90.76%`
    - freeze 段指标：`value_loss` 在 `5500/6000/6500/7000/7500/8000` 为 `0.364846 / 0.364346 / 0.364375 / 0.353398 / 0.359742 / 0.354032`，`coverage_min=1`，`missing_fraction_max=0`
    - 同 seed 小 `1v3=2000`：`logs/oracle_cde/dual_E_w3000_s8000_value_w002_cache16_seed2026052350_1v3`
    - `challenger rankings=[511,489,504,496]`，`avg_rank=2.4925`，`avg_pt=+0.5175`
    - 当前判读：8k freeze-only 与 5k winner 完全同分，基本排除 freeze/export/checkpoint 把 actor 打坏。
  - 9k unfreeze-only：`logs/oracle_cde/dual_E_w0_s9000_value_w002_cache16_cudnnbench_b192_resume8000`
    - 从 8k freeze checkpoint resume，`critic_warmup_requested_steps=0`、`critic_warmup_resume_steps=8000`、实际 `value.critic_warmup_steps=8000`，`max_steps=9000`
    - 训练完成：`latest_step=9000`，trainer `0`，errors 全 false
    - 资源：`steps/sec=11.54`，GPU memory max `11.16 GB`，system RAM max `82.63%`
    - 解冻后指标：`8500 -> 9000` 的 `value_loss` 为 `0.351535 -> 0.352951`，`entropy/entropy` 为 `0.465820 -> 0.460515`，`important_ratio/max` 为 `1.255338 -> 1.319682`，`coverage_min=1`，`missing_fraction_max=0`
    - 同 seed 小 `1v3=2000`：`logs/oracle_cde/dual_E_w0_s9000_value_w002_cache16_seed2026052350_1v3`
    - `challenger rankings=[501,483,498,518]`，`avg_rank=2.5165`，`avg_pt=-1.5525`
    - 当前判读：只解冻 1000 step 就从 `+0.5175` 掉到 `-1.5525`，主嫌疑转为 PPO actor drift；继续盲目 10k/20k 不划算，应先做 `entropy_floor` / `clip_ratio` / actor LR 的最小变量 probe。
  - 9k entropy-floor probe：`logs/oracle_cde/dual_E_w0_s9000_ef047_er1e3_value_w002_cache16_cudnnbench_b192_resume8000`
    - 从同一个 8k freeze checkpoint resume，`entropy_floor=0.47`、`entropy_adjust_rate=0.001`、`entropy_floor_start_step=8000`，其他条件匹配普通 9k
    - 训练完成：`latest_step=9000`，trainer `0`，errors 全 false
    - 资源：`steps/sec=11.77`，GPU memory max `9.95 GB`，system RAM max `82.58%`
    - 解冻后指标：`8500 -> 9000` 的 `value_loss` 为 `0.351546 -> 0.353569`，`entropy/entropy` 为 `0.465804 -> 0.460356`，`entropy/dynamic_weight` 只从 `0.001008` 到 `0.001017`，`important_ratio/max` 为 `1.259590 -> 1.316473`，`coverage_min=1`，`missing_fraction_max=0`
    - 同 seed 小 `1v3=2000`：`logs/oracle_cde/dual_E_w0_s9000_ef047_er1e3_value_w002_cache16_seed2026052350_1v3`
    - `challenger rankings=[494,478,517,511]`，`avg_rank=2.5225`，`avg_pt=-1.5075`
    - 当前判读：这版轻量 `entropy_floor` 几乎没有改变普通 9k 的 entropy/ratio 轨迹，`1v3` 也几乎同等下跌；下一步应优先试更直接的 PPO 更新幅度控制，例如 `clip_ratio=0.1`。
  - 9k `clip_ratio=0.1` probe：`logs/oracle_cde/dual_E_w0_s9000_clip010_value_w002_cache16_cudnnbench_b192_resume8000_retry`
    - 从同一个 8k freeze checkpoint resume，只把 `policy.clip_ratio` 从 `0.2` 降到 `0.1`
    - 第一次同名启动被外层 shell 中断在第一轮更新块中，没有训练错误、没有 `summary.json`；正式采用干净 `retry` 目录和新端口复跑结果
    - 训练完成：`latest_step=9000`，trainer `0`，errors 全 false
    - 资源：`steps/sec=11.68`，GPU memory max `10.67 GB`，system RAM max `83.51%`
    - 解冻后指标：`8500 -> 9000` 的 `value_loss` 为 `0.352429 -> 0.355314`，`entropy/entropy` 为 `0.467991 -> 0.465063`，`important_ratio/max` 为 `1.186009 -> 1.219353`，`important_ratio/variance` 为 `0.001403 -> 0.001827`，`coverage_min=1`，`missing_fraction_max=0`
    - 同 seed 小 `1v3=2000`：`logs/oracle_cde/dual_E_w0_s9000_clip010_value_w002_cache16_seed2026052350_1v3`
    - `challenger rankings=[496,494,535,475]`，`avg_rank=2.4945`，`avg_pt=+1.3725`
    - 当前判读：`clip_ratio=0.1` 同时压住 ratio、保住 entropy，并把 9k 正式小窗推到当前最高；但它只是 E 线信号，不能替代 C/D/E 的初始化与训练顺序对照。
  - C warmup 与 D 3k statsfix 对照：
    - 目的：暂停继续拉长 9k winner，先确认 `GRP/outcome pretrain` 进入 online PPO 的口径是否健康。
    - 共同口径：dual tower，`batch_size=192`，`cache16`，`value.weight=0.02`，`clip_ratio=0.1`，`allow_cudnn_benchmark=true`，`opponent_pool_preset=validation`，`seed_key=2026052350`，同 seed 小 `1v3=2000`。
    - C 口径：`SL bridge + critic warmup + PPO`。C 不加载 Oracle critic pretrain，从 `sl_canonical.pth` 启动，必须设置正的 `critic_warmup_steps`，warmup 期间 actor freeze。
    - C 基准：`dual_C_w3000_s5000_value_w002_clip010_cache16_cudnnbench_b192`
      - `critic_warmup_steps=3000`，0-3k actor freeze、3k-5k PPO；`oracle_critic_state_file=""`
      - 训练干净到 `latest_step=5000`，trainer `0`、errors 全 false；资源峰值 GPU `10.84 GB`，system RAM `86.71%`
      - 同 seed 小 `1v3=2000`：`logs/oracle_cde/dual_C_w3000_s5000_value_w002_clip010_cache16_seed2026052350_1v3`
      - `challenger rankings=[469,523,498,510]`，`avg_rank=2.5245`，`avg_pt=-1.5525`
      - 当前判读：这条 C 5k 低于 `sl_canonical -0.81`，也低于当前 D/E 正信号；在当前参数下，只有 online critic warmup、没有 human outcome pretrain 的路线不够强。
    - C freeze-only 3k / 4k 隔离：
      - `dual_C_w3000_s3000_freezeonly_value_w002_clip010_cache16_cudnnbench_b192` 只跑到 warmup 结束，正式 `1v3=2000` 为 `rankings=[495,483,519,503]`，`avg_pt=-0.81`，与 `sl_canonical` 完全一致
      - `dual_C_w3000_s4000_value_w002_clip010_cache16_cudnnbench_b192` 在 3k freeze 后只多解冻 1k，正式 `1v3=2000` 为 `rankings=[491,496,507,506]`，`avg_pt=-0.9`
      - 当前判读：C 的 warmup 阶段本身没有伤 actor，也没有带来可见增益；掉分主要发生在 3k 之后的 PPO 解冻阶段，5k 的更深回落说明问题不在 warmup 初始化，而在后续 policy 更新怎么接上。
    - D 训练：`logs/oracle_cde/dual_D_s3000_value_w002_clip010_cache16_cudnnbench_b192_statsfix`
      - 加载 Oracle critic pretrain：`logs/oracle_critic_resource_probe/dual_b640_w2_f6_p2_full_s30000/checkpoints/best.pth`
      - 干净到 `3000`，errors 全 false；资源峰值 GPU `10.32 GB`，system RAM `88.07%`
      - `3000` 点：`value_loss=0.359313`，`entropy=0.515378`，`important_ratio/ratio=0.999995`，`variance=0.026265`，`batch_max_mean=1.982543`，`window_max=7.873874`，`clipped_window_max=1.1`，`coverage_min=1`，`missing_fraction_max=0`，`version_gap_max=9`
      - 同 seed 小 `1v3=2000`：`logs/oracle_cde/dual_D_s3000_value_w002_clip010_cache16_statsfix_seed2026052350_1v3`
      - `challenger rankings=[490,508,493,509]`，`avg_rank=2.5105`，`avg_pt=-0.8775`
    - 当前判读：D 的实现口径符合定义，可复用；D statsfix 0->5k 路径偏弱，但仍是 D 线的有效弱结果。
- 2026-05-25 D 5k、clip/value 搜索和 8k/12k gate：
  - 运行资源口径：本轮本机可能同时开浏览器/app，在线 runner 以 `--max-system-mem-percent 95 --hard-system-mem-percent 98 --resource-breach-samples 5` 运行；所有列入结论的训练均 `trainer=0`、errors 全 false，资源波动未当作配置失败。
  - D statsfix 5k：
    - 共同口径：dual tower，`batch_size=192`，`cache16`，`value.weight=0.02`，`clip_ratio=0.1`，`allow_cudnn_benchmark=true`，`opponent_pool_preset=validation`，从 3k statsfix checkpoint resume，`seed_key=2026052350`，同 seed 小 `1v3=2000`。
    - D：`dual_D_s5000_value_w002_clip010_cache16_cudnnbench_b192_statsfix_resume3000`，5k 指标 `value_loss=0.363442`、`entropy=0.518932`、`important_ratio/window_max=1.64575`、`coverage_min=1`、`missing_fraction_max=0`、`version_gap_max=8`；`1v3 rankings=[479,528,483,510]`，`avg_rank=2.512`，`avg_pt=-0.99`。
    - 判读：D statsfix 5k 未过 `sl_canonical -0.81`，所以单靠这条 0->5k 路径不够强；但它仍是符合 D 定义的数据，可作为 D 线弱结果保留。
  - 差异审计和早期起点结论：
    - 从新 D statsfix 3k 起点改 3k->5k `clip_ratio=0.2`：`dual_D_s5000_value_w002_clip020_cache16_cudnnbench_b192_statsfix_resume3000`，`1v3 rankings=[458,529,491,522]`，`avg_rank=2.5385`，`avg_pt=-2.7225`；后段放宽 clip 不是复现旧 5k winner 的原因。
    - 从旧强 3k 起点 `dual_D_s3000_fix_old_policy_entropy_b224_resume2500` 续 3k->5k，改用当前 `cache16/batch192/cudnnbench` 和 `clip_ratio=0.1`：`dual_D_s5000_value_w002_clip010_cache16_cudnnbench_b192_oldstart_resume3000`，`1v3 rankings=[488,513,514,485]`，`avg_rank=2.498`，`avg_pt=+0.765`；这是本轮当前最强 5k checkpoint。
    - 受控复刻旧早期配方，从 0->3k 用 `value.weight=0.05 / clip_ratio=0.2 / cache16 / batch192 / cudnnbench / repro seed`：`dual_D_s3000_value_w005_clip020_cache16_cudnnbench_b192_statsfix`，`1v3 rankings=[487,507,490,516]`，`avg_rank=2.5175`，`avg_pt=-1.5075`，未复现旧 3k 的 `-0.54`。旧强 3k 起点的优势不能简单归因到 `value.weight=0.05 / clip_ratio=0.2`，更可能与早期非 repro 轨迹、batch `288->224` 分段、cache8/旧运行轨迹共同有关；先把它当强势起点使用，不把它误写成可稳定复刻的配方。
  - 在旧强 3k 起点上的 `clip_ratio` 搜索，均为 `value.weight=0.02 / cache16 / batch192 / cudnnbench`，同 seed 小 `1v3=2000`：
    - `clip_ratio=0.05`：`dual_D_s5000_value_w002_clip005_cache16_cudnnbench_b192_oldstart_resume3000`，`rankings=[488,494,500,518]`，`avg_pt=-1.89`。
    - `clip_ratio=0.075`：`dual_D_s5000_value_w002_clip0075_cache16_cudnnbench_b192_oldstart_resume3000`，`rankings=[466,533,496,505]`，`avg_pt=-1.125`。
    - `clip_ratio=0.1`：`dual_D_s5000_value_w002_clip010_cache16_cudnnbench_b192_oldstart_resume3000`，`rankings=[488,513,514,485]`，`avg_pt=+0.765`。
    - `clip_ratio=0.15`：`dual_D_s5000_value_w002_clip015_cache16_cudnnbench_b192_oldstart_resume3000`，`rankings=[468,536,476,520]`，`avg_pt=-1.98`。
    - 旧参考 `clip_ratio=0.2`：`dual_D_s5000_value_w002_cudnnbench_b192_resume3000`，`rankings=[511,489,504,496]`，`avg_pt=+0.5175`。
    - 判读：`clip_ratio=0.1` 是当前旧强 3k 起点上的 5k winner；更窄/更宽都明显变差。
  - 在旧强 3k 起点、`clip_ratio=0.1` 上的 `value.weight` 搜索，同 seed 小 `1v3=2000`：
    - `value.weight=0.01`：`dual_D_s5000_value_w001_clip010_cache16_cudnnbench_b192_oldstart_resume3000`，`rankings=[468,534,498,500]`，`avg_pt=-0.675`。
    - `value.weight=0.015`：`dual_D_s5000_value_w0015_clip010_cache16_cudnnbench_b192_oldstart_resume3000`，`rankings=[480,497,511,512]`，`avg_pt=-1.7775`。
    - `value.weight=0.02`：`dual_D_s5000_value_w002_clip010_cache16_cudnnbench_b192_oldstart_resume3000`，`rankings=[488,513,514,485]`，`avg_pt=+0.765`。
    - `value.weight=0.03`：`dual_D_s5000_value_w003_clip010_cache16_cudnnbench_b192_oldstart_resume3000`，`rankings=[477,492,517,514]`，`avg_pt=-2.16`。
    - 判读：`value.weight=0.02` 仍是当前局部最优；`0.01` 虽略高于 `sl_canonical -0.81`，但明显弱于 `0.02`。
  - 继续训练 gate：
    - `dual_D_s8000_value_w002_clip010_cache16_cudnnbench_b192_oldstart_resume5000`：8k 指标 `value_loss=0.358170`、`entropy=0.482090`、`important_ratio/window_max=1.83504`、`coverage_min=1`、`missing_fraction_max=0`、`version_gap_max=9`；`1v3 rankings=[497,474,500,529]`，`avg_rank=2.5305`，`avg_pt=-2.6775`。
    - `dual_D_s12000_value_w002_clip010_cache16_cudnnbench_b192_oldstart_resume5000`：12k 指标 `value_loss=0.361095`、`entropy=0.476376`、`important_ratio/window_max=1.74539`、`coverage_min=1`、`missing_fraction_max=0`、`version_gap_max=9`；`1v3 rankings=[491,513,483,513]`，`avg_rank=2.509`，`avg_pt=-0.99`。
    - 当时判读：8k/12k 退化时 replay IS coverage、value loss 和资源都健康，因此不是 cache/资源 bug；更像 5k 后继续 PPO actor 更新损伤策略。不要继续 20k/50k，D 线推荐 checkpoint 是 `dual_D_s5000_value_w002_clip010_cache16_cudnnbench_b192_oldstart_resume3000/checkpoints/mortal.pth`。
- 2026-05-25 E 9k 后 actor-stability probes：
  - E 9k `clip_ratio=0.1` 仍是当前同 seed 小窗最强 checkpoint：
    - `logs/oracle_cde/dual_E_w0_s9000_clip010_value_w002_cache16_cudnnbench_b192_resume8000_retry/checkpoints/mortal.pth`
    - 同 seed 小 `1v3=2000` 为 `rankings=[496,494,535,475]`，`avg_rank=2.4945`，`avg_pt=+1.3725`。
  - 从 E 9k 继续到 12k 的补跑已闭环：
    - 原 `dual_E_w0_s12000_clip010_value_w002_cache16_cudnnbench_b192_resume9000` 停在 `latest_step=11500`，无 `summary.json`；确认 checkpoint `steps=11500` 后，用 `dual_E_w0_s12000_clip010_value_w002_cache16_cudnnbench_b192_resume11500` 补到 12k。
    - 补跑 trainer `0`、errors false，资源峰值 GPU `10.92 GB`、系统 RAM `80.29%`。
    - 同 seed 小 `1v3=2000`：`logs/oracle_cde/dual_E_w0_s12000_clip010_value_w002_cache16_seed2026052350_1v3`，`rankings=[487,492,506,515]`，`avg_rank=2.5245`，`avg_pt=-1.7775`。
    - 判读：9k 的强正信号不能靠继续 plain PPO 延续到 12k；不要把 E 9k 直接拉长窗。
  - 新增可配置的 policy LR scale 机制，用于 actor-stability probe：
    - `policy.actor_lr_scale` / `policy.policy_head_lr_scale` 默认都是 `1.0`，默认 optimizer layout 不变；显式非 1 时把 actor trunk 和 policy head 拆成独立 optimizer groups。
    - 修复了首次实现中的两个问题：helper 作用域不能依赖 `train()` 内部的 `nn` import；LR scale 不能改变模型语义 signature，否则会把 9k resume 误变成 `steps=0` 并重新进入 freeze。当前逻辑是：权重语义匹配但 optimizer layout 不匹配时，保留 checkpoint `steps/optimizer_steps`，只重置 optimizer/scheduler/scaler/best_perf。
    - targeted tests：`python -m unittest mortal.tests.test_train_online mortal.tests.test_oracle_cde_configs`，112 tests passed。
  - 从 E 9k 到 10k 的 LR-scale probes：
    - `dual_E_w0_s10000_clip010_actorlr05_value_w002_cache16_cudnnbench_b192_resume9000_retry2`：`actor_lr_scale=0.5`、`policy_head_lr_scale=1.0`，从 9k 权重、`steps=9000` 正确续到 10k；trainer `0`、errors false，资源峰值 GPU `11.49 GB`、系统 RAM `84.50%`。同 seed 小 `1v3=2000` 为 `rankings=[447,493,543,517]`，`avg_rank=2.565`，`avg_pt=-3.69`。
    - `dual_E_w0_s10000_clip010_policyheadlr05_value_w002_cache16_cudnnbench_b192_resume9000`：`actor_lr_scale=1.0`、`policy_head_lr_scale=0.5`，从 9k 正确续到 10k；trainer `0`、errors false，资源峰值 GPU `11.36 GB`、系统 RAM `84.37%`。同 seed 小 `1v3=2000` 为 `rankings=[436,525,532,507]`，`avg_rank=2.555`，`avg_pt=-2.79`。
    - 判读：只降 actor trunk 或只降 policy head LR 都不能保住 9k 强信号；这两个结果还都包含 optimizer/scheduler reset，因此不能证明“低 LR 本身必坏”，但足够说明 LR scale 不是当前最有希望的第一方向。下一步应优先做 policy update throttle/stop：让 Oracle value 继续校准，同时减少或停止 PPO policy loss 对 actor/policy 的改动。
  - 从 E 9k 到 10k 的 policy update throttle/stop probes：
    - `dual_E_w0_s10000_clip010_policyupd2_value_w002_cache16_cudnnbench_b192_resume9000`：`policy.update_interval=2`、phase `0`，optimizer/scheduler exact resume，PPO policy/aux 约半数 step 更新，Oracle value 每步更新；trainer `0`、errors false，资源峰值 GPU `11.03 GB`、系统 RAM `82.99%`。同 seed 小 `1v3=2000` 为 `rankings=[482,488,517,513]`，`avg_rank=2.5305`，`avg_pt=-1.9575`。
    - `dual_E_w0_s10000_clip010_policystop_value_w002_cache16_cudnnbench_b192_resume9000`：`policy.update_interval=100000`，9k->10k 等价于暂停 PPO policy/aux，只训 Oracle value；trainer `0`、errors false，资源峰值 GPU `10.94 GB`、系统 RAM `84.69%`。同 seed 小 `1v3=2000` 为 `rankings=[503,491,520,486]`，`avg_rank=2.4945`，`avg_pt=+0.8775`。
    - 判读：policy stop 明显优于继续 PPO、LR scale 和 interval=2，但仍低于 E 9k 的 `+1.3725`；说明继续校准 value 本身相对安全，但并没有直接提高 actor。当前最强 checkpoint 仍应停在 E 9k，而不是 10k/12k。
- 当前资源结论：
  - dual-tower 在线短窗 batch `320` 可用，但不能当默认
  - dual-tower 在线长窗/默认应使用 batch `192`
  - 受控 seed 的 dual-tower online C/D/E 需要显式 `allow_cudnn_benchmark=true`；否则 `repro` 会关闭 cuDNN benchmark，显存峰值可升到 `15GB+`
  - batch `224`/`288`/`320` 只作为手动短窗或续跑加速档；不建议通过提高资源 guard 阈值掩盖内存/显存压力
  - `scripts/run_oracle_cde_online.py` 的资源 guard 已改为使用 role 采样中的最大系统内存，避免第一个 role 的较低读数掩盖 trainer/client 瞬时高水位

## 当前最重要的判读规则

- 是否“过门”看预声明协议下、独立确认牌山上的正式 `1v3`，报告配对区间；2000 局仅用于粗筛。
- 训练内 `test_play=200/400` 只保留为诊断信号，不再直接当 protocol ranking 依据
- `RL-1 / RL-2` 成对对照要后置到共享优化栈已经站稳之后
- 成熟且目标对齐的 Oracle critic 应完整使用；保留 `all_players` 与真实 return 标签，先证明在当前 actor 分布上的资格。
- 相同部署 actor 哈希要去重；不同输出需检查推理条件和 provenance，不能误判为 value-only 更新提高或损伤 actor。
- 验证输入、对手、规则、奖励和 source fingerprint 都属于比较契约。随机补全、重复选择和预算结束不能被包装成确定 winner。

## 当前下一步

1. 安全 checkpoint 后修复 Oracle 验证输入可复现性；以同一新快照同时重算 150k best、latest 和 no-update anchor。不要用新输入统计直接减旧随机输入统计。
2. 确认正式 pt 与平均顺位的目标选择，推荐将新实验统一到正式 pt 的归一化奖励 `[2,1,0,-3]`；旧训练目标保持到明确的新实验边界。
3. 固定 actor/对手/引擎/推理条件完成 1v3 A/A；随后在 canonical、S70 和少量 SL finalist 中确定新底座。
4. 对 Oracle 做当前 actor 分布上的 p0 主指标、all_players、尾部、校准与输入依赖确认，明确每条非劣护栏；不自动简化输出或换成 GRP 教师标签。
5. 通过后建立近 on-policy PPO 参照，小窗查契约错误；用独立牌山和多个训练种子决定是否扩大、加入 replay 或改写 V-trace actor 目标。
6. 旧 E 9k / D 5k 和旧 clip/value 配方保留为探索对照，不自动作为新 canonical 或成熟增益路线。建议 16k 局筛选、少量 finalist 独立 64k 局确认，实际预算随最小有意义效应和方差确定；区间跨零允许未决。

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
