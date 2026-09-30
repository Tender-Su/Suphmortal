# 文档维护

以缩短查找、理解和更新成本为目标，按任务直接整理相关文档和规则，合并重复内容并保留可追溯证据。每项事实或规则只设一个维护位置，其他页面用链接引用。

## 归属与长度

| 位置 | 维护内容 | 更新方式 |
| --- | --- | --- |
| 根 README / Agent 入口 | 导航和长期项目定位 | 不写 run、winner、机器参数 |
| [handoff](handoff.md) | 一屏现状、关键限制、下一步顺序 | 替换过时摘要，不追加流水 |
| `docs/status/` | 每条主线的当前结论、配置来源、证据和下一步 | 核验后替换；详细旧结果归档 |
| [workflows](workflows.md) / [remote-ops](remote-ops.md) | 本地与远程操作命令 | 与对应 CLI 同步；测试命令由 [code-health](code-health.md) 维护 |
| 接入目录 README | 该适配器的能力边界、安装与命令 | 与接入代码一起更新 |
| `docs/research/` | 当前有效的证据报告与未决判断 | 文件带日期，开头标明影响范围 |
| `docs/archive/` | 退出当前运行依据的计划、结果和旧文档 | 保留原文，添加历史标记与当前入口 |
| `docs/reflections/` | 人的判断、协作和个人复盘 | 保留当时语境，不改写成当前操作手册 |

篇幅以便于检索和维护为准；检查器的体积限制用于发现膨胀。优先消除重复，不为凑行数制造转发页面或超长行。

## 状态更新

1. 先核验代码、所用配置、checkpoint 内部字段和原始结果，说明未能获取的证据。
2. 页首写 `> 核验：YYYY-MM-DD · 依据与范围`。该日期表示实际核验，不能为了消除过期提示只改日期。
3. 更新对应主线的“结论、证据、下一步”；只有跨阶段影响才同步 handoff。进程 PID、每次 step 和完整控制器流水留在 run 产物。
4. 新研究结论同时更新 [研究索引](../research/README.md) 与状态页；平局、未决和未完成修复要保留。
5. 用历史索引保留需要追溯的旧数据，给退出的页面加历史标记。旧命令仍可保留为证据，但不能留在当前操作页。

事实依据与文档优先级统一见 [AGENTS.md](../../AGENTS.md#阅读入口)。

## 链接、路径与生成文件

- 用相对 Markdown 链接指向文档和源码，让移动可检查；目录、配置键、产物名用行内代码。不要只写一个不可点击的文档路径。
- 当前文档的源码路径和 `-m mortal...` 模块必须存在；占位路径明确标成示例。历史正文保留旧路径，实际 Markdown 导航仍需有效。
- 本地 `logs/`、checkpoints、target 不保证随仓库分发，普通检查将其视为可选附件；复现时可开启附件检查。
- [SL 自动 snapshot](../status/supervised-fidelity-results.md) 由 [run_sl_fidelity.py](../../mortal/supervised/run_sl_fidelity.py) 维护，路径不可随意改名；不手工追加当前决策。
- `CLAUDE.md` 只引用 `AGENTS.md`，不复制第二份规则。新增文档必须从现有索引可达。
- `MahjongCopilot/` 是独立上游宿主，其安装、许可证和 [MJAI 协议原文](../../MahjongCopilot/assets/mjai%20Protocol.md) 保留自身结构；项目文档维护 [集成边界](code-map.md#平台与宿主)，不擅自重写协议规范。

## 自动检查

[check_docs.py](../../scripts/check_docs.py) 只读取文件，使用 Python 标准库，不导入训练代码、不启动 GPU 或网络：

```powershell
C:\ProgramData\anaconda3\envs\mortal\python.exe scripts/check_docs.py
C:\ProgramData\anaconda3\envs\mortal\python.exe scripts/check_docs.py --json
```

它检查项目文档的本地链接、标题锚点、索引可达性、当前源码/模块引用、状态日期、历史标记和页面体积，支持当前使用的行内/引用式 Markdown 链接和 fenced code。默认提示超过 30 天未核验的状态；可用 `--max-age-days` 调整提示窗口。

修改检查器时运行故障样例回归：`rtk proxy C:\ProgramData\anaconda3\envs\mortal\python.exe -m unittest discover -s scripts -p test_check_docs.py`。

需要检查本机证据附件时加 `--check-artifacts`。历史样本缺失时应记录缺失范围；检查器不验证外链可达性、论文真伪或模型效果，绿色结果只代表文档结构与引用通过。

维护范围为根 Markdown、`docs/`、`mortal/README.md` 与 `integrations/` 的 Markdown；不递归扫描训练日志、生成环境或第三方宿主。移动文档后同时更新实际链接和索引，再运行检查及 `git diff --check`。
