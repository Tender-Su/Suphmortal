# SL early / late A→B→C 工程入口

> 范围：2026-09-30 新分支工程契约，不是训练结果或预算批准。已接入真实 SL trainer；本次云端未运行 GPU、未操作 runner。预算、LR 和 4060 容量须在启动前另行锁定。

## 要回答的问题

在相同后续数据、更新量、LR 和完整辅助目标下，较早离开 A 是否比 A 2.88M 更有利于 B/C 适应？仅比较两个已有 A checkpoint，不重训 A，不增加 parent × LR 网格。

- early：A microsteps 1,366,000 / successful updates 1,365,487；库存 LR 约 4.91502392e-5
- late：A microsteps 2,880,000 / successful updates 2,878,912；库存 LR 5e-6
- 两者均须有 414 个 Adam state、全部 learned heads、参数名分组、scheduler 和 AMP state；只验证结构不是牌力证据
- B 按游戏抽样：90% 2023–24 + 10% 2009–21；C：98% 2024 + 2% 2009–21。复用完整域轮换，无小型固定池；比例不是 decision 比例

## 状态与恢复契约

[新入口](../../scripts/run_sl_early_transition.py) 复用旧 probe 的配置、ordered preparation、消费游标、观察与真实 `train_supervised.train(probe=...)`；[分支契约](../../mortal/supervised/early_transition.py) 与旧 data-only 协议分开。旧 runner 仍要求同父 LR 和 A 2.88M，未放宽旧实验。

新模式 `preserve_adam_declared_phase_lr` 保留 backbone、policy、rank、opponent、danger 的全部权重、Adam moments / per-parameter step、参数名分组、betas / eps / weight decay、AMP 和累计 auxiliary clock。两臂全量辅助 recipe 必须相同；rank base 0.001548 / max 0.00516、opponent 0.00135、danger 0.00804 且 enabled，任何不匹配均失败，不自动修正或关闭辅助头。

每个 phase 显式重置 microsteps、successful-update counter、scheduler clock、历史指标 / patience / controller、sampler 和 RNG。新 phase 的 Python / NumPy / Torch RNG 在模型构造后按该 phase seed 初始化；phase 内恢复沿用已有完整 cursor / RNG。A 原产物没有 RNG/data cursor，因此 A→B 是新实验分支，不声称是 exact resume。累计 auxiliary clock 在 B→C 继续累计，不重走 ramp。

两臂统一采用 late A 的配置模板，所有 learned state 从各自 parent 装载；全 recipe、模型 tensor shapes 和 Adam group options 在准备时匹配。新 trainer 额外检查实际参数名分组完全相同，禁止兼容层默默 remap。输出中的 `template.json`、完整配置与指纹记录此选择。

LR 在 `supervised.lr`、`supervised.scheduler` 及 `optim.scheduler` 同时声明，并使用真实 `LinearWarmUpConstantLR` 序列化。首个 optimizer update 使用 init LR；每次成功 update 后递增 scheduler，warmup 后为 peak。`initial_lr=1` 是 LambdaLR 的基准倍率重置，不是 Adam clock 重置。source LR 和新 phase scheduler 同时写入 provenance。

B 两臂均完成后才进入 C；每个 arm 的 C 只读自己的 B `state_file.pth` latest endpoint，核对 SHA、checkpoint ID、step 和所属 arm。不可改成旧 best rollback。A→B→C parent chain 通过 trainer 已有 `run_provenance` 写入每次 checkpoint，而非仅留在一次性外部回执。

## 准备与执行

只在经过测试、Git 同步的独立固定 commit 工作树准备。`prepare --source-commit` 要求完整 HEAD 且 checkout clean；复制可执行 Python 与显式 native extension 到新实验的 frozen source，并记录逐文件 SHA。默认源码中的 search distillation keyword-only 调用已修正，不再对 frozen copy 做隐式热补丁。

CLI 的必填参数：

- `prepare --directory NEW --early EARLY_A --late LATE_A --index FULL_INVENTORY --validation-index EVAL_INDEX --native LIBRIICHI --source-commit FULL_SHA`
- 每 phase 必填 `--b-updates` / `--c-updates`、`--b-observations` / `--c-observations`、`--b-seed` / `--c-seed`、`--b-lr` / `--c-lr`、`--b-init-lr` / `--c-init-lr`、`--b-warmup` / `--c-warmup`
- 每 phase observation 点递增，最后一点等于该 phase budget；预算和 warmup 单位均为 successful optimizer updates，不是历史日志 microsteps
- 必填 `--microbatch`、`--logical-batch`、`--val-batch-size`、`--gpu-memory-fraction`；logical batch 必须是 microbatch 整数倍。不会默认 50k、4096 或历史 2.5h 吞吐
- 可选有界 `--prepare-workers`、`--val-prepare-workers`、`--prepare-file-batch-size`、`--val-file-batch-size`、`--rayon-threads` 复用 ordered preparation；优化前需测 4060 峰值 RAM/VRAM 与实际有效吞吐

FULL_INVENTORY 是带 `train_files` / 可选 `val_files` 的原始全量索引；按原始 game identity 去重与分域。EVAL_INDEX 明确提供非空 `full_recent_files` 和 `old_regression_files`，两角色互不重叠且不进入训练域。准备时锁定验证 game identity 与内容指纹；所有臂重新跑 U0 baseline，不将旧 B/C loss 当新 baseline，也不打开 sealed test。

训练文件内容采用共享 SQLite ledger，在四游戏 block 即将首次消费时固定 SHA，不在 prepare 时全量读多年语料。普通及 ordered loader 在解析后、yield 前核对已有 block SHA 并提交首次 pin；后续所有 arm / phase / cycle 必须一致。未进入消费 block 的文件尚未固定内容。ledger 与 experiment ID 绑定，缺失时拒绝恢复；保留其与实验目录，不能重建空账本跳过校验。每个 block 只处理当前几行，不重写不断增长的 JSON；阶段全局训练锁保证单 writer。

准备后用 `run --directory NEW` 顺序编排；运行单段时必须调用 `NEW/source/scripts/run_sl_early_transition.py phase --directory NEW --arm early|late --phase B|C --until-update DECLARED_POINT`，来自其他 checkout 的直接 phase 执行会拒绝。每轮先完成两臂同一观察点，再继续更长 horizon；下一轮反转两臂顺序。直接 C 入口同样要求两臂 B 完成。新目录拒绝覆盖，已有 phase config/provenance 不匹配会失败，OS lock 阻止重复 runner 或同时训练两个 phase。观察点模型保留为独立文件，latest 用于安全续训；如果历史结果已存在而 latest 缺失，不会默默从 parent 重启覆盖结果。

真实运行必须放在现有[截止 supervisor](../agent/deadline-supervisor.md) 的独立本机 lease 下，使用本轮授权截止和保存余量。SL 会继承 stop-file，在 optimizer 边界原子保存并退出 75；外层 runner 不自动重启。该入口不替代独立 deadline / process ownership 管理。新实验输出和 lease 输出均放工作树外或已忽略的目录，保留所有历史 checkpoint / manifest。

## 输出与验证边界

每段包含独立观察 JSON、模型、latest、配置及 endpoint 回执；总目录保留 manifest、attempt wall time、progress 和完成回执。沿用 observation 的 per-game cluster records、实际消费 exposure、成功 decision 数及 CUDA peak。训练中 AMP skip 会令实际消费数超过成功 decision 数，需同时查看。此入口没有自动 winner、旧 84-comparison Bonferroni 或旧 guard veto；训练结果须再按当前 ROI/牌力目标分析。

测试：`python -m unittest mortal.tests.test_sl_early_transition -v`。云端标准库测试覆盖预算、全 recipe、Adam mapping、source/new LR、parent chain、文件指纹、collision 和互斥锁。真实 Torch 测试覆盖全部 learned state / Adam / AMP 保留、首更新与 scheduler restart、B→C counters、全域 common order 和 RNG；缺 Torch/toml 时明确 skip，不视为通过。

目标环境还需执行 `mortal.tests.test_curriculum_probe`、`mortal.tests.test_ordered_preparation`、`mortal.tests.test_train_supervised`，再按另行批准的极小真实 native 数据预算验收 trainer。云端契约通过不等于 Windows、native loader、CUDA 或完整训练通过。
