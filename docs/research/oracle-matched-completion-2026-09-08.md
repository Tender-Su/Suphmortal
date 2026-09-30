# Oracle 初始化匹配补跑完成审查

> 核验：2026-09-08 · 两臂均正常完成40000步；本报告是固定monitor上的完成审查，不构成独立资格或正式牌力确认。

## 结论

SL初始化臂于12:27:49正常退出0，最终`inconclusive`、四次未决、best_step=0。旧权重臂此前也完成40k并保留step0。按冻结协议比较两臂保留候选后，**没有可晋级的新候选，继续保留warm no-update作为reference**。

四个端点已按checkpoint SHA256去重；源码318/317文件、有效配置、完整training contract、数据signature与验证输入fingerprint均匹配各自冻结记录。两个latest各含824组optimizer、2472个有限状态张量，optimizer/scheduler/scaler/data时钟均40000；ScheduleFree checkpoint均为eval导出。真实训练进程树已退出，Apex未运行，RiichiLab保持运行。

原始证据：[完成审计](../../logs/oracle_critic_formal/matched_completion_20260908_r1/completion_audit.json)、[可复验审查脚本](../../logs/oracle_critic_formal/matched_completion_20260908_r1/audit_completion.py)、[冻结比较协议](../../logs/oracle_critic_formal/sl_rl_repair_sl_init_matched_desktop_20260908_resource_r2/matched_protocol.json)。

## SL臂学到了什么，为什么没有更新best

固定monitor为3186游戏、64676状态。主指标为p0 MSE；分组guard同样按p0定义，不能混用all_players的分组统计。

| 指标 | clean step0 | clean 40k |
| --- | ---: | ---: |
| p0 MSE | 3.819599 | 2.362675 |
| all_players MSE | 3.866165 | 2.384457 |
| p0 exact_zero MSE | 0.081030 | 1.112690 |
| p0 abs_ge_4 MSE | 19.363367 | 7.775366 |

四个10k gate都只有`exact_zero_loss`未通过。40k相对自身step0的p0 MSE差为-1.456924，配对95% CI [-1.515626,-1.398223]；exact_zero差为+1.031659，CI [1.004174,1.059145]。因此是整体误差改善伴随零回报分组误差上升，不是没有学习或异常退出。

低幅度的初始预测在真实回报恰为0的分组上有优势，而尾部误差很大。当前硬规则要求每个分组同时非劣，这个起点会形成严格的权衡。规则按预声明执行；不能看到结果后删除guard、改变目标或按已实现的outcome幅度给样本加权。

旧权重臂则在`abs_ge_4_loss`上失败：最终相对其step0差+0.651025，配对95% CI [0.587865,0.714184]。两种初始化在当前预算下都没有通过全部guard，但失败分组不同。

## 保留候选与共同reference

两臂保存的逐游戏损失和样本数可直接做配对cluster CI；逐项核验game ID与计数一致，复用已有monitor统计，无须重新解码或消耗保留数据。

clean adaptive-best就是clean step0。它相对warm no-update的p0 MSE差为**+1.191281**，95% CI **[1.129074,1.253489]**；只有exact_zero guard通过，另外五个guard均未通过。它不能替代warm reference。

两个40k端点仅为预声明诊断：clean/warm的p0 MSE分别2.362675/2.353649，all_players MSE分别2.384457/2.367076。这些点估计不构成两端点的配对显著性结论，也不能把失败guard的latest改选为winner。

## 后续边界

本轮两臂40k及约定的保留候选比较已完成。reserved selection/regression、sealed test和actor-replay sid0/sid1均未重新打开，未发布模型或启动正式PPO；warm reference本身也不等于已经取得Oracle接入资格。

若继续，应先预声明要检验的权衡、共同reference、全部guard、停止条件和新增预算，再决定是否延长SL起点训练或做关键对照。40k未决不能推出完整SL起点训练无效。在线正式训练还需先修正AMP成功更新时钟；资源配置与计数证据见[资源报告](rl-resource-tuning-2026-09-08.md)。

比较的限制包括：warm臂早于共同协议冻结完成；资源参数与安全恢复边界有记录，训练轨迹不承诺逐bit相同；monitor结果不能替代独立模拟分布资格与正式1v3。
