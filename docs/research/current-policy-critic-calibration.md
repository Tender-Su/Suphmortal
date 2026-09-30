# 当前策略 Oracle critic 校准入口

本扩展只把固定当前策略牌谱接入现有 `mortal.online.pretrain_oracle_critic`。Actor 不进入训练器；训练独立 critic 的 visible/Oracle 两塔和四头 value。未配置 `current_policy_manifest` 时，原文件分割、全玩家数据、单局聚类与历史 training contract 保持不变。

## 1. 登记已完成的采样块

继续使用现成 `frozen_actor_critic_probe`，接受其三 critic 评分开销，不另造生成器。先预声明各块是 train 还是 dev；仅登记完成且四座旋转完整的块。不要混入 pilot/工程 smoke，也不以预测结果挑选分组。

```powershell
$Python = 'C:\ProgramData\anaconda3\envs\mortal\python.exe'
& $Python -m mortal.data.current_policy_manifest `
  --train-probe-dir '<completed-train-block-1>' `
  --train-probe-dir '<completed-train-block-2>' `
  --train-probe-dir '<completed-train-block-3>' `
  --train-probe-dir '<completed-train-block-4>' `
  --dev-probe-dir '<completed-dev-block>' `
  --output '<new-run>/current_policy_manifest.json'
```

CLI 不推断 split，不覆盖已有 manifest。它不读取 `predictions` / `metrics` 来划分数据。登记内容包括：

- 每局原始绝对路径、SHA256、seed、完整 seed_key、trainee 座位、显式 group ID/int63 cluster ID、split、来源块
- 完成块的 provenance/outcomes 文件身份、原有各 checkpoint/config 权重 SHA256、采样参数
- actor/opponent、native、相关运行源码及采样行为的一致性；各块 sampling seed 可以不同
- 逐文件核验实际 gzip 内容 hash、首尾事件、完整小局数、原生 header 的 seed/key/trainee 座位，并与原 outcomes 对照
- 四座必须完整，重复路径、重复内容、重复 seed group、跨 train/dev 泄漏全部拒绝；dev 至少两独立组

manifest 是文件列表唯一来源。不同块的相同 basename 不会合并；从不靠文件名猜 seed/seat。训练启动重新验证一次全部日志 hash，此后复用进程内已验证 ledger，不在每次 batch/epoch 重算。原始日志、receipt 和 manifest 须保持只读，不边训练边追加。

## 2. 正式训练配置

使用新的 run/output 路径，把已选真实预训练 critic 作为 `init_state_file`，沿用其结构参数（resnet、fusion mode/hidden、head、exact-zero-sum）。新阶段建立新 optimizer/scaler/scheduler、数据游标与 dev step0 基线；不能拿旧预训练 optimizer 或其 dev baseline 冒充本阶段 resume。

以下为合入完整有效配置的字段示例，不是独立可运行 TOML；初始化文件、结构和资源参数须来自已核验来源：

```toml
[env]
pts = [2, 1, 0, -3]

[oracle_critic_pretrain]
current_policy_manifest = "./current_policy_manifest.json"
player_names = ["trainee"]
critic_arch = "dual_tower"
train_scope = "all"
target_mode = "all_players"
return_mode = "score_rank_mc"
discount_gamma = 1.0
exact_zero_sum = true
strict_init_checkpoint = true
init_state_file = "<verified-selected-critic.pth>"
state_fold_count = 1
val_state_fold_count = 1
num_epochs = 1
enable_augmentation = false
reserve_ratio = 0.0
val_batches = 0
test_batches = 0
train_oracle_imputation_seed = 20260905
val_oracle_imputation_seed = 20260905
eval_input_modes = ["true"]
dependency_val_every_steps = 0

[oracle_critic_pretrain.adaptive_curriculum]
enabled = true
phase_name = "current_policy_calibration"
selection_protocol = "primary_with_diagnostics"
```

还须在完整配置中明确批准的 `adaptive_curriculum.gate_every_steps` 和 optimizer/scheduler/LR；不继承未审核的默认长训练上限。`gate_every_steps` 与正式观察间隔一致。这里不规定数据预算、更新数、LR、成熟门槛或新 MAE veto。

```powershell
$env:MORTAL_CFG = '<new-run>/critic_config.toml'
& $Python -m mortal.online.pretrain_oracle_critic `
  --max-steps $ApprovedUpdateBudget `
  --val-every-steps $ApprovedObservationInterval `
  --save-every $ApprovedObservationInterval
```

`current_policy_manifest` 也可经 CLI `--current-policy-manifest` 指定。配置内相对路径按该 TOML 所在目录解析。新模式要求显式正值 max_steps/val_every_steps/save_every，以及 `primary_with_diagnostics`；不允许 max_*_files、验证 batch 截断、额外 file indexes/globs 或 dev state folding，避免完整组被切开。它默认只运行现成 `true`（实际为 imputed Oracle）评估，不增加 zero/shuffled 资格赛。

## 3. 输入、标签与统计含义

- `player_names=['trainee']` 只筛输入决策；每局必须恰好一条非空 trainee trajectory，漏读/空轨迹硬失败。`all_players` 仍是四头相对座位序，p0=trainee
- gamma1、pts `[2,1,0,-3]` 下，MC 等于最终名次效用减当前状态所属小局开始名次效用；使用完整决策时钟，再取样，不是只回归裸终局 rank
- Oracle 保留记录的隐藏手牌/摸牌，加未知牌山补全；`trust_seed=false`，不声称真实牌山已重建
- 沿用训练 `base_seed + 1000003 * stream_pass` 重采样；dev 固定 seed20260905。补全版本、种子与训练 schedule 都在既有 contract 中保存
- primary 沿用全状态等权 p0 MSE，四座组作为独立 CI cluster。它不同于 probe 的组等权估计量；`seed_group_balanced_loss` 是四头组等权诊断
- 新评估输出明确包含 `ci_cluster_unit`、`ci_estimand`、`num_seed_groups`。为兼容现有控制器，旧 `num_games` / `game_balanced_*` 键保留，但 `num_games_key_unit='seed_groups'`；`min_paired_games` 在本模式实际数独立四座组

保留现成 step0/no-update、latest、adaptive_best、best、best_primary、best_observed_primary 角色。MAE/zero/tail 为诊断，不成为新否决条件。不从 CI 未决推出成熟或不成熟。

## 4. 保存与恢复

manifest 内容/分组/player filter、actor/opponent/native/采样源、当前训练 runtime 源码/native 身份都加入训练和数据流 contract；验证 contract 也绑定相同新阶段身份。完整 resume 必须一致，不能替换 actor、加牌谱、重分 split 或更改补全协议后沿用原 best。

同阶段用同一个 latest/state_file 恢复，不加 `--fresh`，继承 optimizer/scaler/scheduler、成功更新步和文件组安全游标。保留现有安全重放语义，不承诺逐样本/RNG/逐 bit 同轨。成功更新预算不是物理消费上限，AMP skip 与恢复重放须另报告。

## 5. 验证入口

```powershell
& $Python -m unittest mortal.tests.test_current_policy_manifest -v
& $Python -m unittest mortal.tests.test_current_policy_training -v
# 使用正式完成数据做CPU读取验收，不生成新的smoke，不更新模型：
$env:MORTAL_CURRENT_POLICY_TEST_MANIFEST = '<new-run>/current_policy_manifest.json'
& $Python -m unittest mortal.tests.test_current_policy_training -v
```

第一套只用标准库，覆盖 ledger/hash/原始header/完整组/split/来源/resume contract。第二套实际导入 Torch/native；缺依赖明确 skip，其中真实 gzip 解码测试还需显式给出 manifest。不能把标准库 pass 或依赖 skip 说成真实训练集成已通过。真实输入测试逐局检查 dev 首组四座均仅输出 trainee、4头/finite/zero-sum 及共同 cluster ID；既有 `test_value_coordinate_contracts` 核对标签算术。

正式启动仍须受本轮独立 deadline supervisor 管理，保护旧源码/权重，先使用新固定 commit 工作树验证。该文档不授权运行 GPU 或发布源码。
