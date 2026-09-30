# 研究意图与恢复后的决策

> 核验：2026-09-30 · 依据：已导出的用户可见历史、当日决策和 runner 的 CPU 权重核验回报。本文重建当前研究意图，不声称逐字恢复丢失文档或提交；旧实验数值与原协议结论不变。

## 已经确定的路线

actor 始终只看 visible 输入，Oracle 完整进入独立 critic。早期 actor Oracle / dropout 的讨论，经捷径依赖、训练与部署错位、噪声正则等假设，转向 critic 路线；这些假设不是已经完成的因果证明，也不是要求重新比较隐藏信息是否有价值。

研究语境的 GRP 是 game result prediction 真实结果预测任务，不是历史小型 GRP 模型或以它为 teacher。critic 学习真实 outcome / score_rank / return-to-go，保留 all_players；预测的是当前 visible policy 继续行动的回报，不是全知最优策略。

历史路线的设计意图需与实际执行分开：

- C：SL bridge → 冻结 actor 的在线 critic warmup → PPO
- D：SL bridge + outcome 预训练 → PPO
- E：D 的预训练 → 冻结 actor 的当前策略 critic 校准 → PPO

旧 C 的 warmup 实际为 0，削弱的是该次 C 执行对设计的检验，不能据此否定认真训练 critic 的主线。用户“3岁教5岁”的担忧是 critic 成熟度不足；任意固定步数或很短 warmup 不是成熟证据。

## 当前判据与下一步

成熟且对齐的 critic 应充分使用 Oracle。用户已质疑把 Oracle 输入依赖当作独立资格门，并反对高置信 advantage 才影响 actor 的新门槛；不新增此类资格赛。置换隐藏输入可作明确问题下的可选 OOD 诊断，不能成为主线入口，也不能当作合法 visible-only 因果对照。

本轮授权方案是将 primary MSE 的模型保存，与 MAE、exact-zero、事后回报 tail 分组诊断的一票否决及其停止逻辑解耦。保留这些诊断及异常排查；实现和测试完成前不能声称新选择器已经生效。旧协议的未决/失败仍按原判据保留，不追溯放宽门槛宣告旧实验成功。

先复核现成 checkpoint 的曲线、目标、架构和来源，在相同目标/样本上补必要预测；再做小规模当前策略校准和 RL 闭环。若对齐证据仍不足，应继续解决不足，而不是到任意步数就解冻 actor。收益、成本、停止/扩展条件在实验前声明；不为了用满授权时间重复大训练。

SL 的问题是 A 是否训得过久、损害后续可塑性，以及较早切 B/C 是否更好，不是寻找所有阶段共享的固定总预算上限。两 seed 十臂 40,960 次更新只检验晚期 A2.88M 起点，原结论仍为 inconclusive；优先复用已存早期 A 权重，避免重新训练整段 A。

## 恢复与执行边界

云端工作副本从 bundle 恢复到 `b20ffc3`；`e901092` / `381a34b` 的文档改动未随 bundle 保留，本文是依据现有证据重新整理，不把失去的 `381a34b` 写成当前 HEAD。主聊天历史导出已恢复并核验；文件身份保留在私有恢复回执，不与源码或其他证据文件混用。

历史输入为已导出的 predecessor-public-history 用户可见记录及本轮线程；权重清单为 `MahjongAI-weight-inventory-20260930.json`，早期资产清单为 `MahjongAI-early-A-assets-20260930.json`。这些私有输入不随 Git 提交；新核验结果及局限见 [权重复用](weight-reuse-2026-09-30.md)。

源码在云端修改、测试和提交，两机只同步固定 commit：ModelKits / RTX 5070 Ti 负责 critic 与 RL，ABANDON / RTX 4060 Laptop 负责 SL。授权截止为北京时间 2026-10-08 19:00（UTC 11:00），须有本机独立 deadline supervisor 及保存停止余量。

本次更新时 ModelKits 任务仍在整理历史指标和 import preflight，未启动新 GPU 计算；此前 smoke 在 native API 预检处退出，实际 0 局。状态不是新的训练结果。执行预算与安全边界见 [研究窗口](research-window-2026-09-30.md)，当前操作入口见 [接手摘要](../agent/handoff.md)。
