# 笔记本 Stage 0.5 Loader Benchmark（2026-03-30）

## 机器

- CPU：`Intel Core i9-13900HX`
- GPU：`NVIDIA GeForce RTX 4060 Laptop GPU`
- VRAM：`8 GB`
- RAM：`32 GB DDR5`

## 结论

- 当前笔记本 `Stage 0.5` 训练快路径默认：
  - `num_workers = 6`
  - `file_batch_size = 7`
  - `prefetch_factor = 3`
- 当前笔记本验证快路径默认：
  - `val_file_batch_size = 7`
  - `val_prefetch_factor = 6`
- 很接近的验证备选：
  - `7 / 5`

## 测量摘要

- 首轮训练扫描里，`6 / 7 / 3` 以 `0.9761 steps/s` 领先，略高于 `4 / 10 / 4` 的 `0.9737 steps/s`
- 更长复核里，`6 / 7 / 3` 提升到 `1.1735 steps/s`，相对 `4 / 10 / 4` 的 `1.1385 steps/s` 拉开到约 `3.1%`
- 首轮验证扫描里，`7 / 5` 以 `0.6052 steps/s` 略高于 `7 / 6` 的 `0.6035 steps/s`
- 更长复核里，`7 / 6` 提升到 `0.6381 steps/s`，反超 `7 / 5` 的 `0.6296 steps/s`，因此当前推荐 `7 / 6`
- 所有进入复核的训练 / 验证配置都稳定跑通，没有复现 worker 崩溃或资源错误

## 方法边界

- 这轮 benchmark 没直接使用笔记本上的全量数据根
- 原因：笔记本当前没有完整本地 `D:\mahjong_data\...`，而桌面数据共享尚未打通到“训练脚手架可直接读取”的状态
- 因此本轮采用了从台式机抽取的代表性本地子集：
  - train：`320` files
  - monitor validation：`64` files
- 这意味着结论适合当作笔记本的“当前操作默认”，但在笔记本正式持有全量数据根后，仍建议做一次 full-data 复核

## 产物

- 桌面仓库副本：
  - `logs/laptop_stage05_loader_bench/summary.json`
  - `logs/laptop_stage05_loader_bench/confirm_summary.json`
- 笔记本原始运行目录：
  - `C:\Users\numbe\Desktop\MahjongAI\logs\laptop_stage05_loader_bench\`

## 额外修复

- 为兼容笔记本当前 `torch` 的 `DataLoader` 签名，`mortal/train_supervised.py` 已改为只在运行时支持时才传 `in_order`
