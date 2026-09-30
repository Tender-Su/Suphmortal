# GRP 当前状态

> 核验：2026-09-05 · 依据：本地有效配置与训练代码。这里记录下游契约，不把旧 benchmark 当成新模型选择结果。

研究语境中的 `GRP` 指 game result prediction 真实结果预测任务，不限定为历史小型模型，也不默认用它为 Oracle critic 生成伪标签；当前主线见 [研究意图](../research/research-intent-2026-09-30.md)。以下仅记录历史具体 GRP 模型的工程契约。当前配置使用 `hidden_size=384 / num_layers=3 / dtype=float32`；本轮没有重新训练 GRP，也没有重新证明该容量最优。

## checkpoint 用途

下表为仓库根目录下的默认本地路径；实际路径由所用配置解析。

| 产物 | 默认路径 | 用途 |
| --- | --- | --- |
| best_loss | `mortal/checkpoints/grp.pth` | 默认下游输入 |
| best_acc | `mortal/checkpoints/grp_best_acc.pth` | 受控对照候选 |
| latest | `mortal/checkpoints/grp_latest.pth` | 续训，不能因时间较新自动取代 best_loss |

训练实现见 [train_grp.py](../../mortal/supervised/train_grp.py)，模型见 [model.py](../../mortal/core/model.py)。配置来源与路径解析见 [运行流程](../agent/workflows.md#配置与路径)。

## 标签与下游边界

- `GRP_SIZE=7` 是输入契约。改变形状必须同步 Rust 数据生成、Python 模型与 checkpoint 加载。
- GRP 预测指标不等价于最终牌力。更换 GRP 后，需要重新验证受影响的奖励、标签或策略链路。
- Oracle critic 的当前主线学习真实 `score_rank` / return-to-go；不要把它替换为 GRP 生成的标签。详见 [Oracle 状态](oracle-critic-mainline.md)。

## 何时重开实验

只有在输入、数据窗口、训练预算和评测协议可匹配时，再比较更大 GRP 或不同 dtype。至少同时记录 loss、accuracy、校准、吞吐和下游表现；旧实验之间的不同采样规模不能直接证明容量优劣。

历史数据见 [GRP 实验记录](../archive/research/stage0/grp-experience.md)。命令在 [运行流程](../agent/workflows.md)，资源在 [机器页](machine-benchmarks.md)；本页只在模型选择或下游契约变化时更新。
