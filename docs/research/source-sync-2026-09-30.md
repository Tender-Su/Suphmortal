# 源码同步与研究边界

> 核验：2026-09-30 · 仅检查当前源码、Git 元数据及现有小型结果摘要；未运行训练或新评测。

本次保存源码、配置模板、依赖/锁文件、测试、必要文档和
[精简结果证据](../evidence/research-snapshot-2026-09-30.json)。
原始日志、数据集、权重、环境缓存、机器配置及聊天记录保留本地，不纳入 Git。
算法用的 `libriichi/src/algo/data/*.bin.gz` 是必要的小型查表资源，保留。
独立 MahjongCopilot checkout 的改动另以 [兼容补丁](../../integrations/mahjongcopilot/README.md) 保存。

## 研究结论的使用方式

用户要求重新研究评测设计与门槛。旧文档中的 `qualified`、`inconclusive`、
`not_established` 只描述旧协议如何判定，不代表这些门槛已经获得用户对后续研究的认可。
保留原数值和规则用于复核，不事后修改旧实验的结论，也不据此禁止讨论替代评测方案。

C50k 固定对手比较的均值差为 +2.931328 pt，配对区间 [2.380061, 3.472734]；
两 seed 十臂课程实验完成 40960 次成功更新，按原规则未选出课程赢家。
RiichiLab 的 S70 运行于 9 月 11 日达到记录评分 1965 后正常停止。
这些分别回答不同问题，不能把平台评分、课程 loss 和固定对手比较当作同一强度尺度。

`score_rank_mc` 在 gamma=1 时是最终名次效用减去当前局初名次效用，
并非直接预测最终 `[2,1,0,-3]`；因此绝对 return 可以达到 5。
`exact_zero` 表示该回报差恰好为零，与四人输出和为零是不同概念。
按未来已实现回报分组的硬非劣 guard 是否适合条件期望预测，是后续需要重新研究的问题。
在线 AMP 跳步仍会推进计数和 scheduler 的问题可直接从当前代码核验；本次未修改训练算法。

## 同步范围

本次检查的 main HEAD 中没有超过 1 MiB 的 blob；提交前 main 可达历史的 blob 总量约18.7 MB。
本地 `.git` 约22.8 GiB 不等于 main 的传输量。Codex 自动快照引用中存在损坏项，
因此不能用 `--all`、mirror 或整个 `.git` 复制代替有界源码同步。
本次不清理对象、不改写历史、不 force push。

现有 GitHub origin 为公开仓库。新增私有源码和实验材料公开前需明确确认；
也可通过用户私有工作区传输经扫描的当前源码快照，避免携带历史个人路径和自动快照。
机器专属 `mortal/config.toml` 不上传，运行时从 `mortal/config.example.toml` 配置。
远程脚本保留整理前的默认主机、用户名、解释器和路径，仍支持显式参数覆盖；本次私有同步不改变原有运行方式。

完整聊天应通过产品官方可见消息接口单独私有导出，不从隐藏推理、内部日志或元数据摘要补造。
本次官方 app-server proxy 请求超时，未获得完整聊天，未访问原始会话目录。

## 运行默认值纠正（2026-09-30）

源码整理曾额外改动运行默认值，现按整理前备份逐字节还原以下生产文件，保留用户此前的开发改动。
机器路径与普通主机配置保留在私有源码快照；真实凭证仍不提交。测试中的临时目录隔离与占位值改进保留。

- `mortal/supervised/run_sl_winner_refine_distributed.py`
- `scripts/laptop_cleanup_runner.ps1`
- `scripts/laptop_decompress_runner.ps1`
- `scripts/laptop_wait_extract_then_decompress_runner.ps1`
- `scripts/run_laptop_rebuild_chain.ps1`
- `scripts/start_laptop_cleanup_and_decompress.ps1`
- `scripts/start_laptop_online_independent_arm.ps1`
- `scripts/start_laptop_online_worker.ps1`
- `scripts/start_rl_oracle_sanity_pair.ps1`
- `scripts/sync_laptop_repo.ps1`
