# SL 冻结课程完整结果

证据日期：2026-09-10。适用范围：用户批准的同起点课程探针，两训练seed、固定LR 5e-6、全部辅助目标、预声明跨度与控制域。十臂已完整结束，统计与退出已核验；独立正式1v3确认仍在进行。

本轮要检验的是：在同一parent和匹配输入下，A/B/C课程或A预训练后的延迟迁移，是否在近期策略拟合上取得明确改善，同时保持action与旧域非劣。最终结果为 **inconclusive，没有符合冻结规则的课程推荐**。保留现有生产选择，本组课程运行结束；后续模型发布资格继续由独立正式1v3决定。

## 冻结设计与验收

方案和实施边界见 [机制方案](sl-curriculum-mechanism-proposal-2026-09-07.md) 与 [探针实施](sl-curriculum-probe-implementation-2026-09-07.md)。两个seed分别为20260907、20260917；A/B/C各从共同parent训练，在U1024与U4096观察。每个seed的AC/CC分别从其A/C U4096切入C配方，各继续4096成功更新，local seed采用原冻结seed+100000。

固定逻辑batch1024（micro256×4）、LR5e-6、完整域轮换、全部辅助目标和继承的Adam/scaler/辅助时钟。控制域为512近期局、336048决策，以及256旧域局、168827决策；完整观察中的局ID、计数和基线对应一致。运行身份与源码、checkpoint/learned state、消费游标及AMP成本证据见 [本轮验收](../../logs/sl_curriculum_audit_20260907/heartbeat_20260910_0916/verification.json)。

统计采用预声明84项比较、familywise alpha0.05，按训练seed分别计算paired game-cluster近似正态区间，校正z=3.433774986768099。近期policy loss改善阈值0.0002，action accuracy和旧域policy loss非劣margin均为0.0002。各候选必须通过完整规则，不能凭点估计或某一单项晋级。

## 结果

两seed A/B/C在U1024和U4096的十二个parent比较，均未同时通过主指标改善和两项非劣条件。U1024近期loss均呈C低于B、B低于A，但这种相对排序没有使任何候选取得完整课程资格。既有结果见 [短跨度汇总](../../logs/sl_curriculum_audit_20260907/heartbeat_20260908_1650/two_seed_U1024_summary.json) 和 [长跨度汇总](../../logs/sl_curriculum_audit_20260907/heartbeat_20260909_1215/two_seed_U4096_summary.json)。

AC−CC的预声明三指标如下。loss越低越好，accuracy越高越好；区间均使用上述多重比较校正。

| 训练seed | 指标 | AC−CC均值 | 校正区间 | 对应条件 |
| --- | --- | ---: | --- | --- |
| 20260907 | 近期policy loss | -0.000018795 | [-0.000360587, +0.000322997] | 未证明改善 |
| 20260907 | action accuracy | +0.000127958 | [-0.000480852, +0.000736768] | 未证明非劣 |
| 20260907 | 旧域policy loss | -0.000446490 | [-0.000925173, +0.000032192] | 非劣通过 |
| 20260917 | 近期policy loss | +0.000003285 | [-0.000338290, +0.000344859] | 未证明改善 |
| 20260917 | action accuracy | -0.000020830 | [-0.000616667, +0.000575006] | 未证明非劣 |
| 20260917 | 旧域policy loss | -0.000169105 | [-0.000659517, +0.000321307] | 未证明非劣 |

两个seed的AC−CC均未通过全部条件，没有证明A预训练带来完整延迟迁移收益。第一seed统计直接复用，见 [原比较](../../logs/sl_curriculum_audit_20260907/heartbeat_20260909_2122/first_seed_AC_CC_assessment.json)；本轮只新增 [第二seed的三个指标](../../logs/sl_curriculum_audit_20260907/heartbeat_20260910_0916/second_seed_AC_CC_assessment.json)。

[最终汇总](../../logs/sl_curriculum_audit_20260907/heartbeat_20260910_0916/frozen_curriculum_summary.json) 核对24个已存ABC contrast block共576核心字段、第一seed AC−CC及新比较，与冻结运行器结果全部相等。冻结选择逻辑得到recommendation=null、delayed_transfer_favors_A=false、status=inconclusive。没有新增AC-parent/CC-parent比较、合并seed、跨跨度选择、打开selection/sealed或自动发布。

## 完成与成本

最后CC于9月10日08:44完成验证，08:45运行器正常结束；完整checkpoint、observation、segment/completed以及全部SL进程退出均已核验。末checkpoint为4096成功updates、2次AMP skip，模型/Adam共1658个受检tensor有限、nonfinite_batches为0；完整验证1135.047秒。[退出证据](../../logs/sl_curriculum_audit_20260907/heartbeat_20260910_0916/live.json) 与 [CPU验收](../../logs/sl_curriculum_audit_20260907/heartbeat_20260910_0916/complete_checkpoint.json) 保留原始身份和指纹。

[最终账本](../../logs/sl_curriculum_audit_20260907/heartbeat_20260910_0916/budget_ledger.json) 按十臂各4096成功更新计数，U1024不另加：

| 计量 | 数值 |
| --- | ---: |
| 成功updates | 40,960 |
| 成功decisions | 41,943,040 |
| AMP跳步次数 | 15 |
| 跳步额外决策 | 15,360 |
| 持久cursor累计消费 | 41,958,400 |
| 剩余科学更新预算 | 0 |

系统重启前未保存训练和未封存对局可能被重放，这部分额外物理成本无法精确计数。原orphan attempt保留，不能把持久cursor成本称为全部物理计算。已完成attempt的reported wall合计170167.688秒，其中旧runtime19591.656秒；它不是总GPU时间。

## 解释与后续边界

结果仅适用于这两个seed、当前controller局簇、固定LR与观察跨度及完整辅助配方。区间反映各训练seed条件下的局采样不确定性；两个训练seed不足以估计训练seed方差。非劣条件未通过不等于证明劣化；未推荐课程也不证明现代数据无价值或模型达到全局上限。

本组冻结课程已完成，不自动延长或改用新的判定规则。canonical身份保持。此前四臂16k筛选选择C50k作为唯一finalist，独立reference与C50k各64k确认继续，当前进度和下一验收门槛以 [执行状态](../../logs/sl_curriculum_audit_20260907/sl_1v3_execution_state.json) 与 [SL状态页](../status/supervised-mainline.md) 为准。独立确认完成前不发布新模型。
