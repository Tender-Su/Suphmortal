# 文档地图

这套文档只有一个目标：让接手者先看到当前真相，再按需追证据，不在多个文件里拼结论。

## 先读

| 目的 | 文件 |
| --- | --- |
| 接手当前工作 | `agent/README.md` |
| 看当前下一步 | `agent/handoff.md` |
| 跑命令 | `agent/workflows.md` |
| 双机与远程 | `agent/remote-ops.md` |
| 找代码位置 | `agent/code-map.md` |

## 当前真相

| 主题 | 文件 |
| --- | --- |
| 监督学习最终结论 | `status/supervised-mainline.md` |
| 在线 RL 当前主线 | `status/online-rl-mainline.md` |
| 机器级 loader / `1v3` 默认 | `status/machine-benchmarks.md` |
| 自动生成的监督学习 snapshot | `status/supervised-fidelity-results.md` |

## 追溯材料

| 目录 | 用途 |
| --- | --- |
| `research/` | 设计理由、证据链、仍可能复用的研究长文 |
| `reflections/` | 人机协作、判断过程和个人复盘 |
| `archive/` | 退役文档和旧快照 |

## 维护规则

- 当前结论只写在 `status/`；入口文档只链接和摘要。
- 运行命令只写在 `agent/workflows.md` 或 `agent/remote-ops.md`。
- 研究长文不参与默认接手，除非 `agent/README.md` 明确点名。
- 新文档归属不确定时，先看 `agent/doc-maintenance.md`。
