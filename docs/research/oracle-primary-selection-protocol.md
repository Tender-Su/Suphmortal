# Oracle primary-with-diagnostics 选择协议

这是显式 opt-in 的代码协议，不改变历史实验结论，也不启用新的训练任务。
在新分支配置的 `[oracle_critic_pretrain.adaptive_curriculum]` 中设置
`selection_protocol = "primary_with_diagnostics"`。省略时仍为 `legacy_guardrails`；
旧配置、旧 contract 的序列化表示及 SL 默认行为保留。Oracle 新协议要求
primary 为 lower `primary_loss`（p0 MSE），不能用 MAE 偷换 primary。

## 指标与控制器

- primary 的 paired cluster CI、meaningful_delta、min_paired_games，以及既有
  futile / unresolved 次数、LR 和停止阈值不变；本次没有重新设定有意义改善量。
- 原 `guardrails` 列表在新协议中仅为 diagnostics。继续记录各项 paired mean、CI、
  game/sample count；MAE、all_players 与事后 outcome slices 不否决 primary 改善，
  也不通过 compensation 延缓 primary 已判定的 futility。
- 最小 paired games 仅约束 primary。诊断 slice 可以少于该数量；合法空 slice 必须
  明确提供空 records 且没有数值 metric，保存为 null、count=0，不伪造 0 loss/CI。
- 诊断不是忽略数据损坏：非有限值、重复 game id、非法 count、缺失 records、
  配对样本数或 game set 改变仍报错。每次候选验证均检查与 accepted baseline 的
  配对完整性，非 gate 不得跳过这些约束。

## Checkpoint 角色

新协议区分 observation 与 promotion：

- `best_observed_primary`：在所有完整、固定输入的 dev validation 中最低观察
  primary loss 的候选。无需 gate promotion；并列不覆盖。这不是已证实更好的模型。
- `adaptive_best`：由 paired primary 判据接受的当前 phase best，供 curriculum 使用。
- 历史 `best_primary` / `best` 名称保留。新协议下只在 adaptive 接受该 step 且对应
  loss 创低时更新；非 gate 无权绕过 acceptance。旧协议原有保存语义不改写。
- `latest`：恢复训练。所有新协议 checkpoint 带 `best_observed_primary_loss`，
  续训对 companion 文件核对 training contract（含 validation fingerprint）、file splits、
  finite loss，并核对 accepted 文件的 step 身份；保留最低观察值。
- 无更新 baseline 是有效候选；新协议初始化保存上述全部角色，避免没有任何
  update 被接受时缺失可选权重。旧协议 baseline 文件集合保持原样。

候选文件默认名为 `{run_name}_best_observed_primary.pth`，可通过
`best_observed_primary_state_file` 显式配置。新协议要求五种角色路径互异，防止候选
覆盖已接受模型。它不自动进入 sealed test 流程；
既有 sealed-test finalist 约束完全不变。

## 恢复边界

新协议名称进入 training contract 和 adaptive state。旧→新或新→旧不能 exact
resume，也不能默默继承旧 phase baseline。需要新的 run/artifact 路径和显式研究分支，
重新测量无更新 baseline；使用旧权重初始化时记录来源，不重标记旧实验为成功。
不要覆盖旧 checkpoint 或修改旧 manifest 来绕过契约。

CPU 回归入口：
`python -m unittest mortal.tests.test_adaptive_curriculum mortal.tests.test_oracle_primary_selection`。
新测试不导入 Torch 或 libriichi；训练入口的真实加载/落盘仍需固定版本 runner 验证。
