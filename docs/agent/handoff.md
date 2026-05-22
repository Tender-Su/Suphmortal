# 当前接手摘要

这份文档只放接手所需的一屏信息。完整结论以 `docs/status/` 为准。

## 当前阶段

- 源码真源：台式机 `main` 工作树。
- `GRP` 保留为前置模型，默认下游使用 `best_loss`。
- 监督学习阶段已完成；官方 winner 是 `anchor*1.0`，第一替补是 `opp_lean*0.85`。
- 当前 canonical supervised checkpoint 是 `./checkpoints/sl_canonical.pth`。
- 活跃研发主线是在线 RL。

## 现在最该做

1. 继续扩 `add_rank_opp_danger`，下一门优先看 `1500`。
2. 暂停直接扩 `add_value_gae_is_rank_opp_danger`，先查清 `value / GAE / replay IS` 为什么会把已有微正信号拉坏。
3. 不再把 `ms_rl1_add_value_gae_is_20k` 当续工入口；这条线已经越过短窗验证且仍明显负增长。
4. 只有某一层过了 `3000` 门，才扩到 `20k`；只有 `20k` 仍为正，才考虑 `40k`。
5. 共享优化栈稳定前进后，再做匹配的 `RL-1 / RL-2` 对照。

## 当前判读纪律

- 是否“过门”优先看正式 `1v3`。
- 训练内 `test_play=200/400` 只做诊断，不再当排名口径。
- 排因、Oracle 对照、短窗门测默认用 `validation` opponent pool。
- 更长窗口和冲上限训练再切 `formal` opponent pool。
- 训练阶段默认关闭 `search`；推理期增强单独做 A/B。

## 不要踩的坑

- 在线 RL 启动顺序是 `[control].state_file -> [online].init_state_file -> [supervised].best_loss_state_file -> [supervised].best_state_file`。
- `one_vs_three.py` 的 challenger 路径是 `[1v3.challenger].state_file`，不是 `[control].state_file`。
- 双机 quick pair 维持全 GPU 默认，不要把 `client` 或 `baseline_train` 改成 `cpu`。
- 笔记本若再出现 `nvcuda64.dll / 0xc0000409 / BEX64`，先保留 CUDA 配置，查 WER 与任务日志，再重拉 run。

## 去哪里看细节

| 问题 | 文件 |
| --- | --- |
| 在线 RL 已接能力、结果、下一步依据 | `docs/status/online-rl-mainline.md` |
| 监督学习 winner 和 `P1` 口径 | `docs/status/supervised-mainline.md` |
| loader / `1v3` 机器默认 | `docs/status/machine-benchmarks.md` |
| 命令怎么跑 | `docs/agent/workflows.md` |
| 双机怎么调度 | `docs/agent/remote-ops.md` |
