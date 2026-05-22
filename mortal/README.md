# mortal Python 包结构

`mortal/` 现在按职责分层，根层只保留配置文件和包入口。仓库内入口走 `scripts/` 或 `python -m mortal...`，不再依赖仓库外部命令封装。

| 目录 | 职责 |
| --- | --- |
| `core/` | 模型、checkpoint、配置工具、复现、训练公共工具 |
| `data/` | 数据集迭代器、标签和奖励相关数据处理 |
| `supervised/` | GRP、监督学习训练、P0/P1/formal/fidelity 编排 |
| `online/` | 在线 RL 的 server、client、trainer、机器模式和角色入口 |
| `eval/` | 推理 engine、`1v3`、Oracle 评测、搜索运行时 |
| `research/` | 一次性探针、审计脚本、辅助实验工具 |
| `tests/` | Python 单元测试 |

仓库内常用入口仍从 `scripts/` 运行。直接跑模块时优先使用：

```powershell
C:\ProgramData\anaconda3\envs\mortal\python.exe -m mortal.supervised.run_sl_formal --help
C:\ProgramData\anaconda3\envs\mortal\python.exe -m mortal.online.online_machine_modes --help
```
