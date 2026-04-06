# 1v3 Multi-Shard Benchmark 2026-04-02

说明：这份文档只回答吞吐默认，不回答模型强弱结论。

这份文档只回答一个问题：当前双机 `1v3` 的默认并发口径应该定成什么。

## 结论

- 台式机 `RTX 5070 Ti` 默认固定为：`seed_count = 1024`, `shard_count = 4`
- 笔记本 `RTX 4060 Laptop GPU` 默认固定为：`seed_count = 640`, `shard_count = 3`

原因很简单：

- 台式机上 `4 shard + 1024` 是已测最快点，`5 shard` 无论保同总量还是回收 `seed` 都没有翻盘
- 笔记本上 `3 shard + 640` 是已测最快点，`4 shard` 和更高 shard 都已经进入回退区

## 台式机结果

- 单进程历史最佳：`768 / 1 shard = 7.5314 games/s`
- `768 / 2 shard = 10.1708 games/s`
- `768 / 3 shard = 10.2875 games/s`
- `768 / 4 shard = 10.0562 games/s`
- `768 / 5 shard = 9.6054 games/s`
- `1024 / 3 shard = 11.0271 games/s`
- `1024 / 4 shard = 11.0943 games/s`
- `1024 / 5 shard = 10.5282 games/s`
- `1280 / 5 shard = 10.9791 games/s`

解释：

- `2 shard` 已经显著优于单进程
- `3 shard` 继续小幅增益
- `4 shard` 只有在 `1024 seed` 时略微领先 `3 shard`
- `5 shard` 开始被额外调度和尾部开销反噬

## 笔记本结果

- 单进程历史最佳：`512 / 1 shard = 5.2828 games/s`
- `512 / 2 shard = 7.4151 games/s`
- `512 / 3 shard = 7.5176 games/s`
- `512 / 4 shard = 7.1869 games/s`
- `640 / 2 shard = 6.9365 games/s`
- `640 / 3 shard = 7.9247 games/s`
- `640 / 4 shard = 7.4916 games/s`

解释：

- `2 shard` 对笔记本同样有效
- `3 shard + 640` 是当前最好点
- `4 shard` 已经开始回退
- 没有继续测试 `5 shard` 的必要

## 运维口径

- 默认先用代码内建 GPU 默认，不必每次在本地 `config.toml` 单独覆写
- 只有在机器名或 GPU 变动时，才通过 `[1v3.machine_overrides]` 或 `[1v3.gpu_overrides]` 定点覆盖
- 如果只是临时 benchmark，优先用环境变量：
  - `MORTAL_1V3_SEED_COUNT`
  - `MORTAL_1V3_SHARD_COUNT`

## 备注

- 本轮 benchmark 主要目标是测吞吐，不是比较模型强弱
- 为了消掉模型差异干扰，测试时使用了各机本地现成 checkpoint 做自对打
- 单分片 `1v3` 的引擎复用回归和多分片 `Ctrl+C` 清理问题已在同日修复，不影响这里记录的 `iters = 1` 多分片吞吐结果
