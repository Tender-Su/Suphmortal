# 笔记本监督学习 Loader Benchmark（2026-03-31，交互前台口径）

## 前提

- 这轮 benchmark 使用笔记本交互会话中的可见窗口作为标准执行环境
- 目标是复核最相关的训练 / 验证配置

## 训练确认

- `nw4_fb10_pf4_vfb8_vpf5_bs1024`
  - `240 / 122.6745s = 1.9564 steps/s`
- `nw6_fb7_pf3_vfb8_vpf5_bs1024`
  - `240 / 127.0444s = 1.8891 steps/s`
- `nw6_fb8_pf3_vfb8_vpf5_bs1024`
  - `240 / 127.1751s = 1.8872 steps/s`

结论：

- 当前训练默认：
  - `num_workers=4`
  - `file_batch_size=10`
  - `prefetch_factor=4`

## 验证确认

- `nw4_fb10_pf4_vfb7_vpf5_bs1024`
  - `240 / 296.5411s = 0.8093 steps/s`
- `nw4_fb10_pf4_vfb7_vpf6_bs1024`
  - `240 / 317.3805s = 0.7562 steps/s`

结论：

- 当前验证默认：
  - `val_file_batch_size=7`
  - `val_prefetch_factor=5`

## 产物

- 训练确认目录：
  - `logs/sl_loader_ab/laptop_sl_loader_bench_interactive_20260331/confirm_train_nw6_fb7_pf3.summary.json`
  - `logs/sl_loader_ab/laptop_sl_loader_bench_interactive_20260331/confirm_train_nw4_fb10_pf4.summary.json`
  - `logs/sl_loader_ab/laptop_sl_loader_bench_interactive_20260331/confirm_train_nw6_fb8_pf3.summary.json`
- 验证确认目录：
  - `logs/sl_loader_ab/laptop_sl_loader_bench_interactive_20260331/confirm_val_nw4_fb10_pf4_vfb7_vpf5_small.summary.json`
  - `logs/sl_loader_ab/laptop_sl_loader_bench_interactive_20260331/confirm_val_nw4_fb10_pf4_vfb7_vpf6_small.summary.json`

## 解释边界

- 这份结果对应笔记本交互前台口径
- 后续如果笔记本硬件电源策略变化，建议沿用这份最小确认流程快速复核
