# 当前默认续工入口

这份文档只回答四件事：

1. 当前主线是什么
2. 真实停在哪一步
3. 启动前清理是否已经完成
4. 如果继续，下一步该做什么

稳定默认看 `docs/agent/mainline.md`。  
当前核对状态看 `docs/status/stage05-verified-status.md`。  
`P1` 评估口径看 `docs/status/p1-selection-canonical.md`。  
当前 `formal triplet playoff` 口径看 `docs/status/formal-triplet-playoff-canonical.md`。  
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
- 当前下游手动方案：
  - `winner_refine first-tier triplet playoff`

## 当前真实停点

- 当前活跃主线 run：
  - `logs/stage05_fidelity/s05_fidelity_p1_top3_cali_slim_20260329_001413/`
- `2026-04-05` 当前真实状态：
  - `winner_refine seed1 = 36 / 36 completed`
  - `winner_refine seed2 = 12 / 12 completed`
  - 当前 `manual formal triplet` 已完成：
    - `logs/stage05_fidelity/s05_formal_triplet_20260405/`
  - 当前 triplet cross-run `formal_1v3` 已完成：
    - `logs/stage05_fidelity/s05_formal_triplet_20260405_winner_playoff_1v3/`
  - 当前正式 `P1 winner`：
    - `anchor*1.0`
  - 当前第一替补：
    - `opp_lean*0.85`
  - 当前历史 pre-formal `top1`：
    - `opp_lean*0.85`
  - 当前 triplet formal 的 cross-run offline 排序：
    1. `opp_lean*0.85`
    2. `opp_lean(rank--/danger++)`
    3. `anchor*1.0`
  - 当前 triplet cross-run `formal_1v3` 的正式顺序：
    1. `anchor*1.0`
    2. `opp_lean*0.85`
    3. `opp_lean(rank--/danger++)`
  - 当前 3 个 child formal 都已收口，且各自 `offline_checkpoint_winner` 都是：
    - `best_loss`
  - 当前下一步不是再开新一轮 triplet，而是：
    - 以 `anchor*1.0` 作为 downstream 默认 winner 继续
    - 保留 `opp_lean*0.85` 作为第一替补与争 `1` 型预备队
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

1. 只允许在 `A2x` 协议内部继续解释当前 `winner_refine`
2. 当前 `winner_refine` 不是任意旧 run 的自动 `top-k`；它已经冻结为：
   - 从当前主线 run 的 `protocol_decide` effective-coordinate 排名取 `top4` centers
   - center 结构名只写 `anchor / opp_lean / rank_lean / danger_lean`
3. 当前 `winner_refine` 只留下 pre-formal 第一梯队：
   - `opp_lean*0.85`
   - `anchor*1.0`
   - `opp_lean(rank--/danger++)`
   它的历史内部 `top1` 仍是：
   - `opp_lean*0.85`
   当前已经写死的 downstream 结论是：
   - 当前 winner = `anchor*1.0`
   - 当前第一替补 = `opp_lean*0.85`
4. 当前 `formal` 默认长度已上调为旧口径 `1.5x`：
   - 当前有效 `phase_a / phase_b / phase_c = 45000 / 30000 / 15000`
   - `2026-04-05` 本次 triplet 实测 wall-clock：
     - 台式机单条约 `4.5 h`
     - 笔记本单条约 `11.2 h`
5. `2026-04-05` triplet formal 已确认的 offline 解释是：
   - `opp_lean*0.85`
     - 当前 `best_full_recent_loss` 最低
     - 当前 `macro_action_acc` 最高
     - 是当前进入 `formal_1v3` 前的 offline front-runner
   - `opp_lean(rank--/danger++)`
     - 当前总 loss 与 front-runner 非常接近
     - 当前 `opp / danger` 头最强
     - 是当前最值得保留的 hedge challenger
   - `anchor*1.0`
     - 当前 `rank_acc` 最强
     - 但按现有 selection policy 的 cross-run offline 排序仍列第 `3`
6. `2026-04-05` 当前 triplet cross-run `formal_1v3` 已确认的正式结论是：
   - 当前判据仍固定为：`avg_pt` 为主，`avg_rank` 为辅
   - 当前 winner：`anchor*1.0`
   - 当前第一替补：`opp_lean*0.85`
   - `opp_lean(rank--/danger++)` 不再作为当前保留候选
7. 如果继续当前主线，默认不是重跑 triplet，而是：
   - 把 `anchor*1.0` 当 downstream 默认 winner
   - 把 `opp_lean*0.85` 当第一替补 / 争 `1` 型预备队
8. 只有在需要做诊断性复跑时，`child_run_name` 才从 triplet formal coordinator 的 `dispatch_state.json` 取；不在当前 canonical 文档里展开 raw `W_r...` 后缀；triplet 的解释边界看：
   - `docs/status/formal-triplet-playoff-canonical.md`
9. 后续不要再把 `formal` 理解成“训练完直接做 canonical 落位”；当前代码已经改成：
   - `formal_train` 只产出 `best_loss / best_acc / best_rank` checkpoint pack，`latest` 丢弃
   - 最终 canonical alias 落位由 `formal_1v3` 决胜后完成
10. `ablation` 不再属于默认主线；只有在需要确认边际贡献或评估删头时，才作为 backlog 手动启动
11. 当前 `Stage 1` 的问题定义已经改写为：
   - 不再问 `recipe / 三头是否开启`
   - 固定使用 `anchor*1.0` 的 full-aux recipe
   - 只问 `oracle-dropout` 的保留比例与 `linear / cosine`
12. 当前 `Stage 1` 默认结构按 `Suphx` 写成：
   - `oracle-dropout transition -> gamma=0 normal continuation`
   - continuation 默认学习率 = transition 的 `0.1x`
13. 当前 `Stage 1` 推荐入口不是旧 `Block C`，而是：
   - `python mortal/run_stage1_ab.py --list-arms`
   - `python mortal/run_stage1_ab.py --ab-name stage1_profile_screen`
   - 默认 shortlist = `linear_075 / cosine_075 / linear_050 / cosine_050`

## 不要这样做

- 不要把 `docs/status/stage05-fidelity-results.md` 当成当前默认入口
- 不要把 `0.18` probe 重新当作当前主线，或恢复它的 `seed2`
- 不要把旧 `solo / pairwise / joint refine` 结构重新当当前主线
- 不要默认把 `ablation` 再插回 `P1` 主线
- 不要把 `formal triplet playoff` 误读成“`P1` 官方 selector 改成 top3 晋级”
- 不要再把 `opp_lean*0.85` 写成当前 winner；它现在是第一替补
- 不要在当前 canonical 文档里继续硬编码原始 `W_r...` 名字
- 不要再把 `Stage 1` 的 `recipe` 当成待回答问题
- 不要在没有人工确认的情况下把多个阶段一键串到底
