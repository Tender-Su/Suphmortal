# 代码健康与重构纪律

> 历史归档 · 归档整理：2026-09-05。正文保留当时的事实、判断和命令，不作为当前运行依据。当前入口见 [文档地图](../../README.md)；旧 SL / RL 强度及 Oracle 验证结论须结合 [独立审计](../../research/sl-rl-audit-2026-09-05.md) 阅读。

这份文档定义长期代码边界和验证要求。目标不是追求文件数量少，而是让每个机制只有一个实现、热路径没有多余工作、训练产物仍可复现。

## 判断标准

按以下顺序取舍：

1. 行为和 checkpoint 合约不变。
2. 热路径少做分配、同步和 Python 循环。
3. 一个机制只有一个所有者，调用方只保留策略。
4. 删除已被替代且没有入口、测试或产物依赖的代码。
5. 只有在职责真正独立时才拆模块，不为缩短文件制造转发层。

## 公共机制所有权

| 机制 | 唯一实现 | 调用方职责 |
| --- | --- | --- |
| 原子 JSON / TOML / checkpoint、文件摘要 | `mortal/core/artifacts.py` | 只决定写什么和写到哪里 |
| 外部程序请求训练安全退出 | `mortal/core/external_pause.py` | 在安全边界检查并保存完整状态 |
| 监督训练标量写入、按 game 聚合 | `mortal/supervised/metric_reporting.py` | 产生指标，不复制 TensorBoard 字段列表 |
| 数据解析和样本折叠 | `libriichi/src/dataset/` | Python 侧只编排文件和 batch |
| 分布式进程启动与 worker 控制 | `mortal/supervised/distributed_dispatch.py` 及稳定控制 API | 各阶段只描述 task 和结果判定 |

阶段脚本不得重新实现上表中的通用机制。确实需要不同的重试、锁或落盘语义时，先把差异写成公共 API 的显式参数并加测试。

## 热路径纪律

- Rust loader 负责日志解析、特征生成和可下推的筛选；不要把逐样本工作搬回 Python。
- batch 内聚合优先使用 tensor 运算；避免逐元素 `.item()`，它在 CUDA 路径会强制同步。
- 训练循环不做目录扫描、配置序列化和重复 hash；这些工作放在启动、验证或 checkpoint 边界。
- 不为微小代码复用增加运行时抽象。热路径中的函数提取必须能被内联理解，且不引入动态分派。
- 优化必须同时给出等价性测试；吞吐优化还应保留基准命令和输入规模。

## 删除代码的证据

满足以下条件才删除：

- `rg` 找不到导入、入口脚本、文档命令和配置引用；
- 不是 checkpoint 反序列化所需的旧类或字段；
- 不是外部机器通过 Git 固定版本运行的入口；
- 对应测试已迁移到唯一实现；
- 删除后最小回归和相关阶段回归通过。

`mortal/research/` 中带 `_tmp_` 的文件不自动视为死代码；先核对运行记录和文档引用。训练产物与 provenance 文件永远不通过源码重构顺手清理。

## 活跃训练边界

训练进程可能在下一次 DataLoader spawn、Apex 恢复或阶段切换时重新导入源码。重构前必须检查进程、supervisor 状态和 source fingerprint：

- 活跃进程已经导入或可能恢复导入的文件保持冻结；
- 在其他模块完成公共 API 和测试，待安全 checkpoint 后再迁移；
- 不修改旧 manifest 来伪装 source fingerprint 一致；
- 外部中断后的恢复必须从 checkpoint 内部 exact step、optimizer、AMP scaler、scheduler 和 data cursor 继续。

## 回归层级

窄改动先跑对应测试；训练公共机制至少跑：

```powershell
C:\ProgramData\anaconda3\envs\mortal\python.exe -m unittest `
  mortal.tests.test_artifacts `
  mortal.tests.test_metric_reporting `
  mortal.tests.test_train_supervised `
  mortal.tests.test_sl_ab
```

分布式编排改动至少跑：

```powershell
C:\ProgramData\anaconda3\envs\mortal\python.exe -m unittest `
  mortal.tests.test_run_sl_winner_refine_distributed `
  mortal.tests.test_run_sl_formal_distributed `
  mortal.tests.test_run_sl_formal_1v3_distributed
```

数据热路径改动还必须运行 `scripts/audit_gameplay_loader.py` 的 binary-equivalence 审计，并按 `docs/status/machine-benchmarks.md` 记录吞吐。Rust 改动运行 targeted test 和 `cargo fmt --check`；发布前再跑完整 Python discovery 与 Rust workspace 测试。

`GameplayLoader` 审计只把确定性输出用于跨二进制逐字节比较。真实牌谱没有可信牌山 seed 时，`invisible_obs` 会随机补齐未知牌，跨进程哈希不同是预期行为；此时必须比较样本数、动作、可见观测、mask、局次和 GRP，并另外验证所有 native sample folds 的样本 multiset 合并后严格等于未折叠全集。活跃训练期间不得用 `maturin develop` 覆盖环境中的原生扩展；在工作区临时目录加载 `cargo build --release` 的产物完成对拍。

## 2026-09-01 重构验证基线

- Python：`739` 项 `unittest` 全部通过，`compileall` 通过。
- Rust：`38` 项单元测试和 `4` 项 doctest 全部通过；`cargo fmt --check` 与 `cargo clippy -p libriichi --all-targets -- -D warnings` 通过。
- loader：三份跨年份牌谱共 `1900` 个样本，确定性字段在已安装基线与临时新版二进制之间逐字节一致；4-fold 并集的 count、XOR 和 256-bit sum 与全集一致。
- 指标聚合：`200000` 条样本、约 `20000` 个 game 的基准由 `0.2381s` 降至 `0.0424s`，约 `5.62x`；算法输出由回归测试锁定。

## 当前剩余结构债务

- `run_sl_fidelity.py` 和三个 distributed runner 仍混合 CLI、状态迁移和阶段策略；后续应先把稳定状态机移入小模块，再保留兼容导出。
- Oracle critic 训练与编排仍有旧的原子写入和暂停 helper；当前活跃训练结束后迁移到 `mortal/core/` 的唯一实现。
- `train_online.py` 与 `pretrain_oracle_critic.py` 仍是大主循环；只有在 checkpoint 合约测试覆盖后才拆，避免表面整洁换来恢复语义漂移。
