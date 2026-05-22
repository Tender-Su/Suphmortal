# 机器级冻结默认

这份文档只保留会影响当前运行口径的机器级默认，不再把 loader 和 `1v3` 的结论分成两份文档。

## 当前默认

硬件口径：

- 台式机：Intel Core i5-13600KF + NVIDIA GeForce RTX 5070 Ti
- 笔记本：Intel Core i9-13900HX + NVIDIA GeForce RTX 4060 Laptop GPU（8 GB VRAM）+ 32 GB DDR5

| 场景 | 台式机 | 笔记本 |
| --- | --- | --- |
| 监督学习 train loader | `num_workers=4, file_batch_size=10, prefetch_factor=3` | `num_workers=4, file_batch_size=10, prefetch_factor=4` |
| 监督学习 val loader | `val_file_batch_size=8, val_prefetch_factor=5` | `val_file_batch_size=7, val_prefetch_factor=5` |
| `1v3` 默认 | `seed_count=1024, shard_count=4` | `seed_count=640, shard_count=3` |
| `GRP 384x3 fp32` loader | `num_workers=10, file_batch_size=50, prefetch_factor=4` | 未冻结 |

补充：

- 台式机监督学习 loader 当前冻结操作点是 train `4/10/3`、val `8/5`
- 如果台式机验证阶段再次出现资源问题，优先重试同一配置；当前确认过的修法是显式 iterator/worker teardown，不是退回单进程
- `GRP` 当前默认仍是 `384x3 fp32`；更大模型是否值得上主线看 `docs/research/stage0/grp-experience.md`，不要把旧的“已完全排除”说法当当前结论

## 监督学习 loader 证据

### 笔记本

`2026-03-31` 交互前台 benchmark 结论：

- 训练确认：
  - `nw4_fb10_pf4 -> 1.9564 steps/s`
  - `nw6_fb7_pf3 -> 1.8891 steps/s`
  - `nw6_fb8_pf3 -> 1.8872 steps/s`
- 验证确认：
  - `vfb7_vpf5 -> 0.8093 steps/s`
  - `vfb7_vpf6 -> 0.7562 steps/s`

冻结结果：

- train：`4/10/4`
- val：`7/5`

产物：

- 训练确认目录：
  - `logs/sl_loader_ab/laptop_sl_loader_bench_interactive_20260331/confirm_train_nw6_fb7_pf3.summary.json`
  - `logs/sl_loader_ab/laptop_sl_loader_bench_interactive_20260331/confirm_train_nw4_fb10_pf4.summary.json`
  - `logs/sl_loader_ab/laptop_sl_loader_bench_interactive_20260331/confirm_train_nw6_fb8_pf3.summary.json`
- 验证确认目录：
  - `logs/sl_loader_ab/laptop_sl_loader_bench_interactive_20260331/confirm_val_nw4_fb10_pf4_vfb7_vpf5_small.summary.json`
  - `logs/sl_loader_ab/laptop_sl_loader_bench_interactive_20260331/confirm_val_nw4_fb10_pf4_vfb7_vpf6_small.summary.json`

## `1v3` 默认来源

代码当前的内建 GPU 默认位于 `mortal/eval/one_vs_three.py`：

- `NVIDIA GeForce RTX 5070 Ti -> seed_count=1024, shard_count=4`
- `NVIDIA GeForce RTX 4060 Laptop GPU -> seed_count=640, shard_count=3`

解析顺序同样由代码固定：

1. 环境变量 `MORTAL_1V3_SEED_COUNT` / `MORTAL_1V3_SHARD_COUNT`
2. `[1v3.machine_overrides.<COMPUTERNAME>]`
3. `[1v3.gpu_overrides."<GPU name>"]`
4. 代码内建 GPU 默认
5. `[1v3]` 配置

## `1v3` 吞吐证据

### 台式机

- `768 / 1 shard = 7.5314 games/s`
- `768 / 2 shard = 10.1708 games/s`
- `768 / 3 shard = 10.2875 games/s`
- `768 / 4 shard = 10.0562 games/s`
- `768 / 5 shard = 9.6054 games/s`
- `1024 / 3 shard = 11.0271 games/s`
- `1024 / 4 shard = 11.0943 games/s`
- `1024 / 5 shard = 10.5282 games/s`
- `1280 / 5 shard = 10.9791 games/s`

结论：

- `1024 / 4 shard` 是当前最好点
- `5 shard` 已经进入回退区

解释：

- `2 shard` 已经显著优于单进程
- `3 shard` 继续小幅增益
- `4 shard` 只有在 `1024 seed` 时略微领先 `3 shard`
- `5 shard` 开始被额外调度和尾部开销反噬

### 笔记本

- `512 / 1 shard = 5.2828 games/s`
- `512 / 2 shard = 7.4151 games/s`
- `512 / 3 shard = 7.5176 games/s`
- `512 / 4 shard = 7.1869 games/s`
- `640 / 2 shard = 6.9365 games/s`
- `640 / 3 shard = 7.9247 games/s`
- `640 / 4 shard = 7.4916 games/s`

结论：

- `640 / 3 shard` 是当前最好点
- `4 shard` 已经回退

解释：

- `2 shard` 对笔记本同样有效
- `3 shard + 640` 是当前最好点
- `4 shard` 已经开始回退

## 备注

- 这组 benchmark 只回答吞吐，不回答模型强弱
- `1v3` 测试时主要目标是测并发口径，而不是做模型比较

## 使用规则

- 默认优先接受代码内建值，不必在本地 `config.toml` 手动重写
- 只有机器名或 GPU 变了，才写 `[1v3.machine_overrides]` / `[1v3.gpu_overrides]`
- 临时 benchmark 优先用环境变量覆盖
