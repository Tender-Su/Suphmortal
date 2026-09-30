# 重构与验证

> 核验：2026-09-05 · 长期开发规范。历史测试数量不代表当前工作树已通过测试。

以行为正确、checkpoint 契约稳定和代码可维护为目标，按实际问题选择重构方式。

## 公共机制与热路径

公共机制位置见 [代码地图](code-map.md#关键入口)。原子落盘、文件摘要、暂停、指标聚合和分布式传输优先复用现有实现；需要不同语义时用显式参数与测试表达。

- Rust 负责牌谱解析、特征和可下推的筛选，Python 负责文件和 batch 编排。
- batch 聚合使用 tensor 运算；避免逐元素 `.item()` 造成 CUDA 同步。
- 目录扫描、配置序列化和重复 hash 放在启动、验证或 checkpoint 边界。
- 性能改动同时记录输入规模、等价性和吞吐；不以更快掩盖标签或采样分布变化。

## 活跃训练边界

DataLoader spawn、supervisor 恢复和阶段切换可能重新导入代码。修改前核对进程、supervisor 和 source fingerprint：

1. 保持活跃进程及恢复链路可能读取的源码、配置和原生扩展不变。
2. 在本地台式机的独立源码快照 / runtime overlay 中修复；笔记本所需代码通过 Git 同步到独立 checkout 后验证，在安全 checkpoint 后以新指纹切换。源码修改边界见 [AGENTS.md](../../AGENTS.md#双机与提交)。
3. 不改旧 manifest 来伪装源码一致，不覆盖正在使用的 checkpoint。
4. exact resume 需要 checkpoint 内部 step、optimizer、scaler、scheduler 和 data cursor；缺失时明确标记分支或重启。

活跃环境中不要运行 `maturin develop`。需要对拍 Rust 时，在临时目录构建并加载独立产物，不覆盖已安装 `.pyd`。机器恢复步骤见 [远程流程](remote-ops.md#中断与恢复)。

## 删除和迁移前

用 `rg` 检查导入、CLI、脚本、配置、文档、测试和远端固定版本引用；还要确认旧类/字段不是 checkpoint 反序列化依赖。名称带 `_tmp_` 不代表可删。训练产物与 provenance 不在源码清理范围。

重构主循环时重点验证保存/恢复、数据游标、train / eval 切换和暂停行为；拆分顺序按问题决定。待清理的历史重复 helper 也要遵守活跃源码冻结。

## 按改动验证

按改动和风险选择能验证行为与契约的测试、实测和静态检查；发现失败或未决风险时再扩大范围。下列命令是按需选用的入口。环境与 DLL 路径见 [运行流程](workflows.md#环境与构建)。

Rust 测试通常放同模块 `#[cfg(test)]`；Python 测试放 `mortal/tests/` 或相邻模块。

训练公共机制的常用回归：

```powershell
rtk proxy C:\ProgramData\anaconda3\envs\mortal\python.exe -m unittest `
  mortal.tests.test_artifacts mortal.tests.test_metric_reporting `
  mortal.tests.test_train_supervised mortal.tests.test_sl_ab
```

分布式编排：

```powershell
rtk proxy C:\ProgramData\anaconda3\envs\mortal\python.exe -m unittest `
  mortal.tests.test_run_sl_winner_refine_distributed `
  mortal.tests.test_run_sl_formal_distributed `
  mortal.tests.test_run_sl_formal_1v3_distributed
```

涉及数据热路径时，可用 [audit_gameplay_loader.py](../../scripts/audit_gameplay_loader.py) 核对原生输出等价性与 native sample folds：

- 对确定性字段比较样本数、动作、可见观测、mask、局次与 GRP；所有 folds 的 multiset 合并应等于全集。
- 人类牌谱 `invisible_obs` 随机补全不能直接当成跨进程逐字节一致字段。固定评测输入的要求另见 [Oracle 状态](../status/oracle-critic-mainline.md)。
- 记录 benchmark 输入和资源条件，遵守 [机器页](../status/machine-benchmarks.md#新-benchmark-的记录要求)。

Rust 状态逻辑的测试与格式检查示例：

```powershell
rtk cargo test -p libriichi state::test
cargo fmt --check
```

格式化只保留与当前修改相关的变化。发布验证按影响范围选择 Python discovery、Rust workspace tests、clippy 等检查，覆盖相关行为与契约；模型发布仍遵守对应状态页的正式评测门槛。文档独立修改用 [文档检查](doc-maintenance.md#自动检查) 验证链接与结构。

## 记录验证结果

报告具体工作树/源码摘要、测试命令、结果与未覆盖范围。历史 [2026-09-01 验证记录](../archive/agent/code-health-before-doc-refactor-2026-09-05.md) 和 [2026-09-05 审计](../research/sl-rl-audit-2026-09-05.md) 保留当时的计数，不能借用为之后改动的通过证明。
