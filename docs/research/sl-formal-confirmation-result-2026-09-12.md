# SL 独立正式 1v3 确认结果

> 核验：2026-09-12 · 适用范围：冻结Long-ABC候选C50k对三家canonical的独立64k确认。结果、原始证据与判定复核完成；没有自动发布或打开sealed test。

## 结论

**C50k 通过冻结正式确认门槛（qualified）。** 相对reference，平均pt差为 **+2.931328**，配对95%区间 **[+2.380061, +3.472734]**。预声明门槛是mean_pt的95%下界严格超过0；完整预算、四座seed、A/A、冻结身份与推理条件均通过核验。原运行器decision与冻结源码复算结果逐字段完全相等，见 [最终判定核验](../../logs/sl_curriculum_audit_20260907/heartbeat_20260912_0011/final_decision_verification.json)。

canonical的发布身份保持。本次结论支持这个固定C50k checkpoint在本协议下优于reference；不代表对所有对手更强，也不包括重复训练seed的不确定性。此前对三家S70的2000局比较仍为未决。后续发布需要独立的人工决定；本任务没有改变模型别名或下游初始化。

另一个同起点、两seed十臂的 [SL课程对照](sl-curriculum-result-2026-09-10.md) 仍为 **inconclusive，无课程推荐**。本次Long-ABC候选的牌力确认不改变该课程机制实验的结论。

## 冻结协议与完整结果

reference和C50k分别64000局，各16000组完整四座seed。confirmation key为 `207618792752062275`，seed范围10000–25999；screen key为 `2182282524960967648`，与确认独立。四臂各16000局screen按预先冻结的最高均值规则选择唯一C50k，未重新选候选，见 [筛选核验](../../logs/sl_curriculum_audit_20260907/heartbeat_20260909_0710/report.md)。

| 指标 | C50k | reference |
| --- | ---: | ---: |
| 对局数 | 64000 | 64000 |
| 平均pt | 2.931328 | 0.000000 |
| 平均顺位 | 2.458359 | 2.500000 |
| 1位 | 16962 | 16000 |
| 2位 | 15989 | 16000 |
| 3位 | 15801 | 16000 |
| 4位 | 15248 | 16000 |

顺位差为 -0.041641，配对95%区间 [-0.048734, -0.034562]，仅作配套结果；资格按预声明mean_pt判定。rank points为 `[90,45,0,-135]`。bootstrap以完整 `(seed, seed_key)` 的四座均值为独立单元，20000次重采样、固定bootstrap seed `20260905`；未把64000局当作64000个独立样本，也未加入训练seed或其他对手。

运行Git为 `5e3e8a15742f9f7b6155c75ade3146c141006d25`，protocol为 `269f94ece89272962f99b89008e8ed7e7399315725931cb3918c7c2e3d6a45dc`。C50k actor为 `b15dd238de18e95ef367e751e9e20876306bfb1dd9b01f5ff06ee5026e08af67`，checkpoint step50000。28份运行源码、两个模型文件和native SHA均与冻结manifest一致；使用chunk64、确定性FP32、Oracle输入zero，search/AMP/compile/TF32关闭。256+256局A/A完整事件相等，旧chunk32产物未混入。

## 原始验收与运行收尾

reference已完成的64000局全原始事件/native名次证明按不变结果和manifest复用。C50k首256局也复用原先已验的gzip/event/native证明，并与当前收据SHA和完整result逐条关联；剩余63744局本轮逐gzip读取，全部事件SHA与complete/result相等，native名次与result记录一致。250个chunk、16000组四座seed没有缺口或重复，见 [C50k全量原始证明](../../logs/sl_curriculum_audit_20260907/heartbeat_20260912_0011/candidate_raw_verification.json)、[首批复用证明](../../logs/sl_curriculum_audit_20260907/heartbeat_20260912_0011/reused_initial_chunk.json) 和 [reference全量证明](../../logs/sl_curriculum_audit_20260907/heartbeat_20260910_1725/reference_raw_verification.json)。

C50k最后chunk封存于 2026-09-11T22:36:12.318604+08:00，result写于 2026-09-11T23:39:44.378565+08:00；聚合实测 3812.060秒（约63分32秒），取代此前借用reference约44分27秒的条件估计。配对决策写于 2026-09-11T23:40:02.810494+08:00，supervisor于 2026-09-11T23:40:04.6471584+08:00正常完成、runner exit0。SL/formal生产进程均已退出，计划任务Ready，旧SL-Bulk/Matched保持Disabled。最后状态见 [执行记录](../../logs/sl_curriculum_audit_20260907/sl_1v3_execution_state.json)。

本轮原始审计最多两个BELOW_NORMAL CPU进程，远端48秒硬限、本地55秒，每完成chunk后约25秒软停止；保存 64 个新批次回执，coordinator用时 1194.688秒，无失败，不导入Torch/CUDA。没有重放reference、重算旧screen/SL统计、扩大候选或科学预算。

当前chunk64运行的有效对局共192512局：A/A512、四臂screen64000、两臂confirmation128000。此前迁移、中断、未封存重放和审计成本不包含在此有效对局数中；原历史与未知物理成本缺口保留，不能将它当作GPU总耗时或所有尝试总成本。SL最终账本仍为40960成功updates、41943040成功decisions、15次AMP skip、持久消费41958400，U1024不双计。

已验SL有序准备优化和chunk64同输入事件/恢复等价证据继续有效；128-seed native fail-fast根因仍未确定，不构成已证实OOM，也没有重试更大档或宣称达到全局吞吐上限。

全部计算、原始验收、最终判定和报告已完成；`sl-1v3` 小时轮询已删除，见 [删除回执](../../logs/sl_curriculum_audit_20260907/heartbeat_20260912_0011/automation_deleted_receipt.json)。
