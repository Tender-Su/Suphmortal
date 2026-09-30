# 监督学习当前状态

> 核验：2026-09-12 · 依据：冻结课程完成；C50k独立64k正式确认与两臂全量原始验收通过，未自动发布。

保留现有 canonical 的发布身份。S70 和后续 Long-ABC 是候选；四月协议完成不代表此后的 SL 工作已经结束，也不代表 canonical 对所有候选的优势已被证明。

2026-09-30 双机存量与旧评测已盘点；新训练未启动，详见 [权重复用](../research/weight-reuse-2026-09-30.md)。

## 模型与证据

| 对象 | 当前身份 | 已有证据与边界 |
| --- | --- | --- |
| canonical / `anchor*1.0` | 已发布基准 | 历史 playoff 对 `opp_lean*0.85` 各 39,936 局，只差 0.087891 pt；combined stderr 0.598451，已触发 close-call |
| S70 | 后续正式候选、当前 Oracle 的初始化来源 | checkpoint 内部 step 390000；保存的 full-recent NLL 为 0.447225，canonical 为 0.476685；历史输入指纹未完全匹配，不能据此直接发布 |
| Long-ABC | C50k独立正式确认通过，未发布 | Phase C 于9月6日11:10在300k正常停止，低LR未改善配对phase-best；summary选择phase_b_adaptive_best，其actor与C50k相同。本轮C50k通过独立formal_1v3，sealed test未打开、发布身份未切换。见 [训练完成审计](../../logs/sl_monitor/20260907_112958_phase_c_completed.json) 与 [正式确认结果](../research/sl-formal-confirmation-result-2026-09-12.md) |

默认 canonical 路径为 `mortal/checkpoints/sl_canonical.pth`。本次读取的 S70 是 `logs/sl_fidelity/sl_anchor_longabc_s70_20260609_r1_1v3_compare/best_action_score.pth`。这些是本地实验产物，不是源码仓库必备文件。

原始数值与适用范围见 [审计报告](../research/sl-rl-audit-2026-09-05.md)；更早阶段记录见 [旧 SL 状态快照](../archive/status/supervised-mainline-before-doc-refactor-2026-09-05.md)。

## 选择与发布

1. 固定数据窗口、validation 输入与 action mask，记录 checkpoint 内部 step、源码/配置/数据摘要。
2. 用 `comparison_recent_loss` / `recent_policy_loss` 比较策略拟合；`full_loss` 含 auxiliary 项，只用于相应诊断，不能跨辅助权重直接裁决策略强弱。
3. `protocol_decide`、`winner_refine` 和离线 checkpoint 选择用于筛选；正式 `1v3` 决定发布资格。`test_play=200/400` 是诊断。
4. 最终候选在独立 seeds、同一对手、四座轮换上比较，报告配对差值与不确定性；重复挑选候选后还需独立确认。近似平局保持未决。
5. 未完成正式确认前，候选不得覆盖 canonical。新发布记录必须能追到 checkpoint、配置、完整对局与决策文件。

历史 selector 参数的统计校准与启发式边界保留在 [selector 审计](../archive/research/supervised/selector-stat-audit.md)。不把其训练 proxy 自动提升为牌力证据。

修复后的 adaptive 选择要求全部非劣护栏通过，样本不足、缺失或连续未决不得选优/晋级。[正式确认入口](../../mortal/eval/confirmation_protocol.py) 限制最多 3 个不同 actor；固定 canonical 对手、四座位、源码/native/权重指纹，先 A/A，再每臂 16k 筛选、一个 finalist 进入独立 64k 确认。完整 chunk 可续跑；到预算 CI 仍跨零则保留 canonical。

本轮冻结 shortlist 为 S70、Phase C 100k、Phase C 50k；后者与 Phase B adaptive best 的 actor 相同，已去重。运行根为笔记本 `MahjongAI_1v3_bulk_20260907` 的 `logs/formal_confirmation_20260907_laptop_batch64_r1`，chunk64、256+256局 A/A 事件完全一致。固定种子、16k/64k预算和判定门槛保持冻结，旧chunk32产物未混入新manifest。完成状态见 [执行记录](../../logs/sl_curriculum_audit_20260907/sl_1v3_execution_state.json)，修复背景见 [实施报告](../research/sl-rl-fixes-2026-09-07.md)。

用户要求的 finalist 对三家 S70 独立比较已完成，两臂各 2,000 局，A/A 通过；SL 平均 pt +0.7425、顺位 2.4905，pt 差值 95% 配对区间 [-2.1825, +3.6675]，尚不能确认强于 S70，不替代上面的 canonical 发布协议。见 [完整结果](../../logs/sl_curriculum_audit_20260907/s70_1v3_2000/comparison.json)。

四臂16k筛选、原始事件与身份均已核验，按冻结screen最高均值选择唯一finalist **C50k**；详见 [筛选核验](../../logs/sl_curriculum_audit_20260907/heartbeat_20260909_0710/report.md)。独立确认两臂各64k全部完成，C50k相对reference **+2.9313 pt，95%配对区间 [2.3801,3.4727]**，顺位2.458359；按冻结下界>0规则为 **qualified**。两臂16000组四座seed、全部原始事件/native名次与冻结身份通过，原decision逐字段复现。该区间不含训练seed或其他对手不确定性，canonical发布身份保持；见 [最终结果](../research/sl-formal-confirmation-result-2026-09-12.md)。

[课程预算审计](../research/sl-curriculum-budget-audit-2026-09-07.md) 确认旧 A/B/C 实际新增约 380/20/25 万步，且 B/C 固定池缩小、A→B 辅助目标改变、完整验证集不同。主树已补完整辅助配方继承（含 `supervised.aux`）、恢复系数校验和辅助 ramp 时钟保留。用户批准的 [机制方案](../research/sl-curriculum-mechanism-proposal-2026-09-07.md) 已迁至笔记本同一 parent 和匹配输入：两 seed 的 A/B/C 先 U1024 再 U4096，AC/CC 延迟迁移各 U4096，总上限 40960 次成功更新。保留 Adam/scaler/辅助时钟，固定 LR 5e-6，完整域轮换采样；不按 A/B/C 先后无限延长 patience。

本轮 SL 已于9月10日正常结束，产物根为 `MahjongAI_sl_ordered_20260908/logs/sl_curriculum_probe/20260908_laptop_ordered_r1`。采用train/val有序准备进程4/4、每块四draw、验证文件批4、Rayon4；保留精确消费游标与迁移来源链，匹配吞吐提高65.8%。参数选择及恢复证据见 [实测报告](../../logs/sl_curriculum_audit_20260907/ordered_preparation_report_20260908.md)，最终验收见 [完成报告](../../logs/sl_curriculum_audit_20260907/heartbeat_20260910_0916/report.md)。

两seed的A/B/C U1024均已完整结束。近期policy loss在两个seed中均为C低于B、B低于A，B/C相对A的主指标校正区间均越过改善阈值；但六个分支相对共同parent均未同时证明主指标增益和全部非劣条件，仍无课程推荐。统计按seed分别报告，未合并或提前选择；见 [短跨度汇总](../../logs/sl_curriculum_audit_20260907/heartbeat_20260908_1650/report.md)。

两seed A/B/C U4096与AC–CC均未通过全部条件，十臂完整结果为 **inconclusive，无课程推荐**。第一seed AC–CC仅旧域非劣通过，第二seed三项均未过；复用旧统计，仅新增最后三个指标。见 [最终研究结果](../research/sl-curriculum-result-2026-09-10.md) 与 [全量验收](../../logs/sl_curriculum_audit_20260907/heartbeat_20260910_0916/report.md)。该结果不证明已劣化、现代数据无价值或模型达到全局上限。

本轮累计40960成功updates、41943040成功决策，含15次AMP跳步的持久消费41958400；U1024不双计，重启丢失/重放的物理成本存在计数缺口。生产约0.296–0.299updates/s，完整验证约19–21分钟；最低可用RAM14.687GiB、显存峰值2595/8188MiB，见 [完整生产验收](../../logs/sl_curriculum_audit_20260907/C_U1024_acceptance_20260908/report.md)。SL与本轮独立正式确认均已结束；容量余量不代表各阶段吞吐最优。

## 产物语义

| 字段或阶段 | 意义 |
| --- | --- |
| `[supervised].best_state_file` / `best_loss_state_file` | 当前 canonical 的下游导出约定；实际配置须一起核对 |
| formal child 的 `best_loss / best_acc / best_rank` | 单次训练内部候选，名称本身不是正式发布判定 |
| `latest` | 恢复模型与 optimizer / scaler / scheduler；data cursor / RNG 是否齐全须核对具体产物，不能仅凭文件名断言逐 batch 精确续跑 |
| [自动 snapshot](supervised-fidelity-results.md) | runner 生成的单次运行摘要，不覆盖本页 |

本轮计算与验收已完成，后续发布或新实验另行安排。命令见 [运行流程](../agent/workflows.md)，勿重启已完成的旧运行。
