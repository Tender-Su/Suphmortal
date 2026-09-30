# mortal Python 包

从仓库根目录以模块方式运行：`python -m mortal.<子包>.<模块>`。实际解释器、配置解析和可执行示例统一见 [运行流程](../docs/agent/workflows.md)。

| 子包 | 职责 |
| --- | --- |
| [core/](core/) | 模型、checkpoint 和训练公共机制 |
| [data/](data/) | 数据迭代器、标签和奖励 |
| [supervised/](supervised/) | GRP、SL 训练及实验编排 |
| [online/](online/) | Oracle critic 预训练、RL server / client / trainer |
| [eval/](eval/) | 推理、`1v3`、Oracle 依赖度、配对统计 |
| [research/](research/) | 探针、审计与辅助实验 |
| [tests/](tests/) | Python 回归测试 |

关键模块与形状契约见 [代码地图](../docs/agent/code-map.md)。配置模板是 [config.example.toml](config.example.toml)，具体实验以所用配置和 manifest 为准。
