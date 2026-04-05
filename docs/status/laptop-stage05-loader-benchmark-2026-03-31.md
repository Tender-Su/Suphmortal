# 笔记本监督学习 Loader Benchmark（2026-03-31，交互前台口径）

## 前提

- 这轮 benchmark 不再使用 SSH `Session 0` 里直接后台拉起训练进程
- 所有任务都通过笔记本交互会话中的可见窗口启动
- 目标不是重做全量大扫，而是复核最相关的训练 / 验证配置

## 训练确认

- `nw4_fb10_pf4_vfb8_vpf5_bs1024`
  - `240 / 122.6745s = 1.9564 steps/s`
- `nw6_fb7_pf3_vfb8_vpf5_bs1024`
  - `240 / 127.0444s = 1.8891 steps/s`
- `nw6_fb8_pf3_vfb8_vpf5_bs1024`
  - `240 / 127.1751s = 1.8872 steps/s`

结论：

- 当前训练默认应更新为 `num_workers=4`
- 当前训练默认应更新为 `file_batch_size=10`
- 当前训练默认应更新为 `prefetch_factor=4`

## 验证确认

- `nw4_fb10_pf4_vfb7_vpf5_bs1024`
  - `240 / 296.5411s = 0.8093 steps/s`
- `nw4_fb10_pf4_vfb7_vpf6_bs1024`
  - `240 / 317.3805s = 0.7562 steps/s`

结论：

- 当前验证默认应更新为 `val_file_batch_size=7`
- 当前验证默认应更新为 `val_prefetch_factor=5`

## 产物

- 训练确认目录：
  - `logs/stage05_loader_ab/laptop_stage05_loader_bench_interactive_20260331/confirm_train_nw6_fb7_pf3.summary.json`
  - `logs/stage05_loader_ab/laptop_stage05_loader_bench_interactive_20260331/confirm_train_nw4_fb10_pf4.summary.json`
  - `logs/stage05_loader_ab/laptop_stage05_loader_bench_interactive_20260331/confirm_train_nw6_fb8_pf3.summary.json`
- 验证确认目录：
  - `logs/stage05_loader_ab/laptop_stage05_loader_bench_interactive_20260331/confirm_val_nw4_fb10_pf4_vfb7_vpf5_small.summary.json`
  - `logs/stage05_loader_ab/laptop_stage05_loader_bench_interactive_20260331/confirm_val_nw4_fb10_pf4_vfb7_vpf6_small.summary.json`

## 解释边界

- 这轮结果比 `2026-03-30` 那份更可信
- 主要原因不是数据根变化，而是启动口径从 SSH `Session 0` 改成了交互前台窗口
- 后续如果笔记本硬件电源策略再变，仍建议直接沿用这份最小确认流程快速复核
