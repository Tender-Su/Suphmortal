# 当前默认续工入口

这份文档只回答四件事：

1. 当前主线是什么
2. 真实停在哪一步
3. 启动前清理是否已经完成
4. 如果继续，下一步该做什么

稳定默认看 `docs/agent/mainline.md`。  
当前核对状态看 `docs/status/stage05-verified-status.md`。  
`P1` 选模口径看 `docs/status/p1-selection-canonical.md`。  
运行步骤看 `docs/agent/experiment-workflow.md`。  
双机与 Git 细节看 `docs/agent/laptop-remote-ops.md` / `docs/agent/code-sync.md`。

## 当前主线

- 当前代码主分支：`main`
- 当前源码真源：台式机 `main` 工作树
- 当前 `Stage 0.5 / P1` 唯一主线：
  - `calibration -> protocol_decide -> winner_refine`
- `P1 ablation` 当前定位：
  - `backlog / manual only`
  - 不再阻塞 downstream `formal_train -> formal_1v3`
- 当前 `winner_refine` 之后的下游结构：
  - `formal_train -> checkpoint pack(best_loss / best_acc / best_rank) -> formal_1v3 -> canonical alias落位`
- 当前已验证协议 winner：
  - `C_A2x_cosine_broad_to_recent_strong_24m_12m`
- 当前已验证 winner 点位：
  - `0.12 + A2x`

## 当前真实停点

- 当前活跃主线 run：
  - `logs/stage05_fidelity/s05_fidelity_p1_top3_cali_slim_20260329_001413/`
- `2026-04-04` 当前分布式状态：
  - `winner_refine seed1 = 36 / 36 completed`
  - `winner_refine seed2 = 6 completed / 2 running / 4 pending`
  - 当前阶段是 `running_p1_winner_refine`
  - 本轮只在 `A2x` 内部继续
- `2026-04-02` 的 `A2x @ 0.18` 三臂 probe：
  - 只是 `seed1-only negative probe`
  - 不改当前主线
  - 不恢复它的 `seed2`

## 启动前清理

- 已于 `2026-04-03` 完成当前主线 run 的启动前清理
- 当前没有旧 `winner_refine_round / winner_refine_dispatch / winner_refine_centers` 残留
- 已删除退役的旧 `ambig` 状态字段：
  - `state.json`
  - `p1_protocol_decide_round.json`
  - `distributed/protocol_decide_dispatch/dispatch_state.json`
- 当前保留的有效搜索配置仍是：
  - `protocol_decide_progressive_ambiguity_mode = flip_or_gap`
  - `gap_threshold = 0.001`

## 如果继续

1. 只允许在 `A2x` 协议内部启动 `winner_refine`
2. 当前 `winner_refine` 不是任意旧 run 的自动 `top-k`；它已经冻结为：
   - 从当前主线 run 的 `protocol_decide` effective-coordinate 排名取 `top4` centers
3. 如果让笔记本参与，使用桌面机调度入口：
   - `python mortal/run_stage05_winner_refine_distributed.py dispatch --run-name s05_fidelity_p1_top3_cali_slim_20260329_001413`
4. 这个双机入口只改变调度方式，不改变 center、局部搜索点或最终 winner 解释口径
5. `winner_refine` 跑完后，默认直接把 front runner 当作当前 `P1 winner`
6. 后续不要再把 `formal` 理解成“训练完直接做 canonical 落位”；当前代码已经改成：
   - `formal_train` 只产出 `best_loss / best_acc / best_rank` checkpoint pack，`latest` 丢弃
   - 最终 canonical alias 落位由 `formal_1v3` 决胜后完成
7. `ablation` 不再属于默认主线；只有在需要确认边际贡献或评估删头时，才作为 backlog 手动启动

## 不要这样做

- 不要把 `docs/status/stage05-fidelity-results.md` 当成当前默认入口
- 不要把 `0.18` probe 重新当作当前主线，或恢复它的 `seed2`
- 不要把旧 `solo / pairwise / joint refine` 结构重新当当前主线
- 不要默认把 `ablation` 再插回 `P1` 主线
- 不要在没有人工确认的情况下把多个阶段一键串到底
