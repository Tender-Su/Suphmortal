# 当前默认续工入口

这份文档只回答四件事：

1. 当前主线是什么
2. 真实停在哪一步
3. 启动前清理是否已经完成
4. 如果继续，下一步该做什么

稳定默认看 `docs/agent/mainline.md`。  
当前核对状态看 `docs/status/supervised-verified-status.md`。  
`P1` 评估口径看 `docs/status/p1-selection-canonical.md`。  
当前 triplet playoff 口径看 `docs/status/supervised-formal-triplet-playoff-canonical.md`。  
运行步骤看 `docs/agent/experiment-workflow.md`。  
双机与 Git 细节看 `docs/agent/laptop-remote-ops.md` / `docs/agent/code-sync.md`。

## 当前主线

- 当前代码主分支：`main`
- 当前源码真源：台式机 `main` 工作树
- 项目当前只保留两大阶段：
  - `监督学习阶段`
  - `强化学习阶段`
- 当前监督学习阶段内部唯一主线：
  - `P0 -> P1(calibration -> protocol_decide -> winner_refine) -> formal_train -> formal_1v3`
- `P1 ablation` 当前定位：
  - `backlog / manual only`
  - 不再阻塞监督学习阶段收口
- 当前已验证协议 winner：
  - `C_A2x_cosine_broad_to_recent_strong_24m_12m`
- 当前已验证 winner 点位：
  - `0.12 + A2x`
- 当前正式 `P1 winner`：
  - `anchor*1.0`
- 当前第一替补：
  - `opp_lean*0.85`

## 当前真实停点

- 当前活跃主线 run：
  - `logs/stage05_fidelity/s05_fidelity_p1_top3_cali_slim_20260329_001413/`
- `2026-04-05` 当前真实状态：
  - `winner_refine seed1 = 36 / 36 completed`
  - `winner_refine seed2 = 12 / 12 completed`
  - `manual formal triplet` 已完成：
    - `logs/stage05_fidelity/s05_formal_triplet_20260405/`
  - triplet cross-run `formal_1v3` 已完成：
    - `logs/stage05_fidelity/s05_formal_triplet_20260405_winner_playoff_1v3/`
  - triplet cross-run `formal_1v3` 的正式顺序：
    1. `anchor*1.0`
    2. `opp_lean*0.85`
    3. `opp_lean(rank--/danger++)`
  - 当前 3 个 child formal 都已收口，且各自 `offline_checkpoint_winner` 都是：
    - `best_loss`
- 当前监督学习阶段已经完成，当前默认停在：
  - `anchor*1.0` 作为 canonical supervised winner
  - `opp_lean*0.85` 作为第一替补与争 `1` 型预备队
- 监督学习阶段后续不再继续 Oracle 路线
- 强化学习阶段方案尚未敲定，默认先停在这里

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

1. 当前默认不是再开新一轮监督学习实验。
2. 当前默认也不是恢复监督学习 Oracle 或旧 `stage1` 路线。
3. 当前监督学习阶段已经写死的结论是：
   - 当前 winner = `anchor*1.0`
   - 当前第一替补 = `opp_lean*0.85`
   - 当前 `formal` 默认长度已经固定为旧口径的 `1.5x`
   - 当前 canonical checkpoint 仍沿用历史文件名：
     - `./checkpoints/stage0_5_supervised.pth`
4. 如果只是核对监督学习阶段，默认只看：
   - `docs/status/supervised-verified-status.md`
   - `docs/status/supervised-formal-triplet-playoff-canonical.md`
   - `docs/status/p1-selection-canonical.md`
5. 如果要继续项目，下一步应是：
   - 先明确强化学习阶段的方案
   - 再决定是否从 `anchor*1.0` 直接起跑 RL baseline

## 不要这样做

- 不要再把项目主线写成 `stage0.5 -> stage1 -> stage2`
- 不要再把 `opp_lean*0.85` 写成当前 winner；它现在是第一替补
- 不要再把监督学习 Oracle 当作待回答问题
- 不要再使用 `run_stage1_refine.bat` 或 `run_stage1_ab.bat` 这类退役入口
- 不要把 `docs/status/supervised-fidelity-results.md` 当成当前默认入口
- 不要在没有人工确认的情况下把监督学习阶段和强化学习阶段一键串到底
