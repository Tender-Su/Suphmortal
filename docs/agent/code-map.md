# 代码地图

> 核验：2026-09-05 · 本页维护模块职责；阶段状态、有效参数和命令分别由状态页与运行流程维护。

## 分层

| 目录 | 职责 |
| --- | --- |
| [libriichi/src/](../../libriichi/src/) | Rust 引擎、状态、特征、日志数据、PyO3 |
| [mortal/core/](../../mortal/core/) | 模型、checkpoint、复现与训练公共机制 |
| [mortal/data/](../../mortal/data/) | 数据迭代器、标签和奖励 |
| [mortal/supervised/](../../mortal/supervised/) | GRP、SL 主循环、selector 与阶段编排 |
| [mortal/online/](../../mortal/online/) | Oracle critic 预训练、RL 角色与配置生成 |
| [mortal/eval/](../../mortal/eval/) | 推理、1v3、Oracle 对照、配对统计与搜索 |
| [mortal/research/](../../mortal/research/) | 探针、审计、辅助实验；稳定后再迁入阶段目录 |
| [mortal/tests/](../../mortal/tests/) | Python 回归测试 |
| [scripts/](../../scripts/) | Windows 入口、机器操作与独立审计工具 |

## 关键入口

| 文件 | 所有权 / 契约 |
| --- | --- |
| [config.py](../../mortal/config.py) | 配置读取和指定路径字段归一化；不提供完整 schema 校验 |
| [prelude.py](../../mortal/core/prelude.py) | 日志、warning、stdin、CPU affinity opt-in 等进程预设 |
| [artifacts.py](../../mortal/core/artifacts.py) | 原子 JSON / TOML / checkpoint 与稳定摘要 |
| [config_utils.py](../../mortal/core/config_utils.py) | 配置 section、递归合并、布尔解析 |
| [external_pause.py](../../mortal/core/external_pause.py) | 外部程序触发的安全暂停协议 |
| [model.py](../../mortal/core/model.py) | Brain、OracleDualTowerBrain、policy / value / aux、GRP |
| [dataloader.py](../../mortal/data/dataloader.py) | SL / RL 迭代器与 worker 初始化 |
| [oracle_value.py](../../mortal/data/oracle_value.py) | OracleTerminalValueDataset 与真实结果标签 |
| [metric_reporting.py](../../mortal/supervised/metric_reporting.py) | 指标输出与按 game 聚合，不决定 winner |
| [distributed_dispatch.py](../../mortal/supervised/distributed_dispatch.py) | worker 启动、传输与生命周期 |
| [pretrain_oracle_critic.py](../../mortal/online/pretrain_oracle_critic.py) | 独立 Oracle critic 主循环 |
| [server.py](../../mortal/online/server.py) / [client.py](../../mortal/online/client.py) | 参数 / replay 服务与自博弈 |
| [train_online.py](../../mortal/online/train_online.py) | PPO 更新与参数发布 |
| [engine.py](../../mortal/eval/engine.py) | MortalEngine 的 batch 推理与动作接口 |
| [one_vs_three.py](../../mortal/eval/one_vs_three.py) | challenger / champion 加载与 1v3 |
| [paired_1v3.py](../../mortal/eval/paired_1v3.py) | 原始对局的 seed 组配对统计 |

## 原生与模型接口

PyO3 暴露 `consts / state / dataset / arena / stat / mjai`。Rust loader 解析牌谱和生成特征，Python 迭代器组织文件、batch 与 worker；可下推的逐样本处理优先留在 Rust。

特征尺寸和 norm 约束只在 [AGENTS.md](../../AGENTS.md#环境与代码契约) 维护。模型结构定位：`Brain` 为 1D ResNet；V3/V4 pre-activation ResBlock 含 ChannelAttention；CategoricalPolicy 输出动作分布，旧 DQN 是另一套头。checkpoint 加载不能仅靠输入尺寸相同来判断兼容。

Oracle 验证需区分真实已知隐藏状态与随机补全；相关逻辑在 [invisible.rs](../../libriichi/src/dataset/invisible.rs)，当前问题见 [Oracle 状态](../status/oracle-critic-mainline.md)。

## 平台与宿主

| 入口 | 边界 |
| --- | --- |
| [雀魂](../../integrations/majsoul/README.md) | 浏览器连接与执行适配；当前未包含 protobuf → MJAI 完整解码 |
| [RiichiLab](../../integrations/riichilab/README.md) | 本地权重推理、WebSocket 客户端与日志 |
| [MahjongCopilot](../../MahjongCopilot/readme.md) | 独立宿主，已存在本地 GN / policy 适配；ABI、依赖和端到端接入需在宿主内验证 |

独立 MJAI smoke 只证明该运行时边界，不自动证明宿主集成完成或模型牌力提升。宿主引擎在 [bot/local/engine.py](../../MahjongCopilot/bot/local/engine.py)，兼容测试在 [test_local_engine.py](../../MahjongCopilot/tests/test_local_engine.py)。

## 文件放置与迁移

共享机制放在已有公共模块，阶段文件保留策略；独立职责才拆模块，不加纯转发层。新增研究脚本放 `mortal/research/`，实验产物放独立 run。移动入口时同步调用方、测试、配置和文档；checkpoint 反序列化引用也要检查。具体证据要求见 [重构与验证](code-health.md)。
