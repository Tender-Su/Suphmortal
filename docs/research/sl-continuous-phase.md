# SL 单主线连续 phase 扩展

> 2026-10-01 工程说明。允许从一个已验证的 B 保存点继续 B；预算由后续曲线和投入产出决定，不把任意小 horizon 当收敛。未部署到当前 3f8 任务，不改原 manifest、观察点、两臂队列、checkpoint 或截止授权。原 C50k 仍是独立强基准。

## 与当前 matched 实验的边界

[入口](../../scripts/continue_sl_phase.py) 在全新目录复制所选 phase checkpoint，建立新 experiment identity、parent SHA / checkpoint ID 和 parent chain。选择 early 或 late 任一已有 B 即可，不检查另一臂是否完成 20k。也可原样扩展已有 C，但不会自动从 B 切 C。

B→C 是另一次明确的 phase 决策，必须另行指定 C 的数据分布、seed、scheduler 和局部时钟行为。此入口没有 LR、warmup、seed、训练 recipe 改写选项；仅下述显式 microbatch 分支可改变 batch 拆分，不通过扩预算偷做转段。旧 3f8 流程和规则保持不变。

## 显式 microbatch 数值协议迁移

`prepare --microbatch 512 --microbatch-change-reason "验收依据"` 声明新数值协议。只允许 `256×4 ↔ 512×2`，logical batch 固定 1024，必须位于完整 optimizer boundary；同 batch、其他 batch、未知保存字段/counter、非整除计数、启用历史 controller/calibration 均拒绝。不传此选项时仍原样续训。

唯一训练配置改动为 `supervised.batch_size`、`control.opt_step_every` 和 `supervised.log_every`。日志间隔按相同 decision 数换算，例如 `128×256 → 64×512`；`max_steps/val_every_steps/save_every` 必须继续为 0，由原有 probe 的 update/wall-time hooks 控制。验证仍是 aggregate1024/Brain256。

顶层 `steps = 已消费 decisions / 新 microbatch`。例如旧 `4004` microsteps 对应 `1000` 成功 + `1` AMP skip、`1,025,024` consumed decisions，迁移后是 `2002` microsteps，成功/skip 仍是 `1000/1`。原/新微步、batch/accum、log 间隔、实际 exposure、source checkpoint ID、理由均写入 manifest、receipt 和持久化 `run_provenance.microbatch_migrations`，重复续训保留转换历史。parent SHA/文件仍保存原 clock。

learned heads、Adam 所有状态和 mapping、AMP、scheduler clock/下一步 LR、auxiliary optimizer clock、成功/skip、sampler RNG/positions、模型 RNG、当前 block/row offset、consumed/files 与 ledger 均逐项原样保留；不重 warmup、不重复样本。只有顶层微步单位转换。`observed`、`continuous_observation` 和保存/验证 cadence 使用成功 update，不转换；epoch、验证次数、best/last 指标与 elapsed time 保留为原数值协议下的历史。未保存的 `running` 日志窗口与普通恢复一样重新累计，不虚构跨中断窗口。

这不是 bitwise 或梯度等价恢复。2026-10-01 ABANDON 三组各 4 个计时成功 update 的实测为 old256×4 `0.28130 updates/s`、fixed256×4 `0.28370`、fixed512×2 `0.51018`；512 相对 old 提升 81.36%，无 OOM，allocated/reserved 为 3.245/3.574 GiB。512 首步 loss 差 `−2.2918e−5`，unscaled gradient relative L2 差 `0.2744%`，max abs `8.266e−5`，finite。证据为私有 Git receipt ref `1ce080ee002673e65684b47d38e345c1aab96441` 的 `.sync/receipts/sl-fixed-benchmark-20261001-task2/benchmark_summary.json`。该短 benchmark 不代表持续吞吐；新分支恢复后的实际吞吐另行记录。与原 late256 比较时须明确数值协议不同，只作工程分配判断，不重跑另一臂来制造 matched 结论。

## 完整性与实际缺口

必须存在真实 saved state：全部 learned heads、Adam moments / 参数 step / name mapping、AMP scaler、scheduler、下一更新 LR、累计 auxiliary clock、完整 microstep 与成功/跳过 update 计数、Python/NumPy/Torch/CUDA RNG、sampler RNG/seed/positions、current 四游戏 block、row offset、已消费样本数和 content ledger。

检验 `microsteps = (成功 + AMP skip updates) × accumulation` 与真实消费 decision 数；不能把成功次数当实际读取次数。当前 block 的 draw 序号、文件和 offset 必须可重建。scheduler `last_epoch` 与成功 update 一致、`_last_lr` 与 optimizer 各组 LR 一致。缺失项或不完整 optimizer boundary 拒绝迁移，不补假 cursor/RNG，不回到 A 重 warmup。

迁移逐项比较除声明的 identity/config/output 元数据及显式 microstep 单位转换外的保存状态，并核对 learned-state digest；用 SQLite 的只读 backup 获取一致 ledger snapshot，保留全部已 pin 文件，包括原实验另一臂或更晚进度保守增加的 pins。仅改副本 metadata identity，原 ledger 不写入。run 对照不可变 snapshot 检查 inherited pins 无删除/改写，且仍覆盖 checkpoint 已消费文件。未消费文件仍未固定字节，不扫描全语料。

优先选择不可变 `update_*.pth`；若选 mutable latest，读、复制、完成准备时 SHA 均必须一致。输出与原目录必须相互独立。缺失必要数据时只能另立明确的 non-exact experimental branch：保留实际可用 learned/Adam/AMP 状态、明确重设缺失 sampler/RNG 并重新衡量数据曝光；需要另行批准，入口不会自动降级。

“保存状态精确迁移”不等于跨 Torch/native/硬件或源码的 bitwise 轨迹已获证明。新 manifest 记录旧/新 Python runtime 指纹差异；若 shared training source 有变化，准备须带 `--runtime-change-reason` 记录验收依据。该文本不会替代真实验收。本集成另含两处 fixed-shape 小热点优化；CPU 数值对拍不代表 CUDA/完整 optimizer update 对拍或提速已完成。

## 保存与验证解耦

[ContinuousPhaseProbe](../../mortal/supervised/continuous_probe.py) 仅通过已有真实 trainer 的 restore/observe/after_update hooks 工作，不另造 trainer。

- 轻量训练统计继续使用保存配置的 `log_every`，不触发验证
- latest 按独立 update 或 wall-time 时钟保存，先到者触发；可声明约 1k updates / 30 分钟。保存包含完整 cursor/RNG，验证前也先保存
- trend 使用固定 128 recent + 64 old 子集，可声明约每 2k updates。只保存指标 JSON，不额外存一份大模型；不拿 trend 代替 full 晋级依据
- full 保留原 512 recent + 256 old 面板，可声明约每 10k updates；终点总会 full，即使不落在整周期。full 产物保留 named checkpoint、SHA 与 per-game records
- phase 中断恢复不重复迁移起点 U0。验证恢复训练 RNG。若结果已落盘而 latest 尚未更新，续训复用该结果，避免重复验证。未配对的 full 模型存档不覆盖，需明确检查
- `run --seal` 在当前 latest 执行/复用 full 后封存，不再做 optimizer update；切 phase 前用这一步。封存后不在原目录重启训练，可显式创建下一次 continuation

固定 trend 是 full 面板的确定子集，随新 index 和 manifest 固定；再次 continuation 保留同一 trend 面板。full 与 trend 同时到期时只跑 full。此入口要求原验证数值协议：config `val_batch_size=1024`，冻结 trainer 的 evaluate 内 `validation_microbatch_size=256`；它不是外部 driver monkeypatch。其他 batch 协议的 checkpoint 不会自动混入本线。

所有 cadence 与 `--until-update` 必填，无隐含 50k 或“已收敛”。update 是该 phase 自起点的累计成功次数，例如从 B1000 扩展到 B200000，不重置成局部 0。2026-10-01 当前运行实测参考为 `1000 / 3926.016 = 0.25471 updates/s`，来自 U1000−U0−validation；不是未来速度保证。每个 trend/full JSON 保留 evaluation_seconds、累计 elapsed_seconds 和曝光统计，新优化是否提速需另测。

## CLI 与验收

准备参数：`prepare --source-run OLD --checkpoint SAVED_POINT --directory NEW --source-commit FULL_SHA`，并显式传 `--until-update`、`--save-every-updates`、`--save-every-seconds`、`--trend-every-updates`、`--full-every-updates`、`--trend-recent-games`、`--trend-old-games`、`--trend-seed`。`--check-only` 只做 state 结构检查，不表示 copy/ledger/实际续训已验收。

准备要求 clean 固定 Git commit，复制冻结 Python/native 源码；只能从 `NEW/source/scripts/continue_sl_phase.py run --directory NEW` 执行。allocator cap 原样保留，run 检查 config 仅包含声明的 metadata relocation 和可选 microbatch 迁移。使用既有[独立截止 supervisor](../agent/deadline-supervisor.md)；外部 stop 保存后退出 75，不自动续租/重启。

测试入口：`python -m unittest mortal.tests.test_sl_continuation mortal.tests.test_sl_early_transition mortal.tests.test_train_fixed_shape`。标准库覆盖缺状态拒绝、计数、cadence、ledger backup、不可覆盖目录、双向 microstep 转换与未知 counter 拒绝；真实 Torch CPU 覆盖 Adam/scaler/scheduler/RNG 序列化后的下一 update、mid-block 下一批顺序、迁移前后 DataLoader 下一 logical1024 样本相同、无 U0/无验证保存、trend/full 与结果中断恢复。native parser 在 cursor 单测中使用 fixture；仍需 Windows/native/CUDA 实际恢复，不以这些单测冒充完整生产训练或要求 512 梯度完全相同。
