# Stage 0.5 保真版 A/B 实时结果

- 运行目录：`C:\Users\numbe\Desktop\MahjongAI\logs\stage05_fidelity\s05_formal_triplet_20260405_winner_playoff_1v3`
- 更新时间：`2026-04-05 21:44:19`
- 当前状态：`completed`
- 自动串联范围：`P0 + P1 + Stage 0.5 formal_train`
- 说明：`formal checkpoint pack` 产出后不再直接做 canonical 落位；当前主线会停在 `formal_1v3` 之前，最终 canonical 落位由 `formal_1v3` 决胜完成。

> run-scoped snapshot: if this file conflicts with the verified/canonical status docs, prefer those docs for current defaults.

## 当前结论

- P0 下游种子 top4：`TBD`
- P0 round3 winner：`TBD`
- P1 协议 winner：`C_A2x_cosine_broad_to_recent_strong_24m_12m`
- P1 最终总胜者：`anchor*1.0`
- P1 第一替补：`opp_lean*0.85`
- P1 winner 来源：`formal_triplet_playoff_formal_1v3`
- P1 ablation 策略：`backlog_manual_only`
- formal_train：`completed`
- formal_1v3：`completed`
- Stage 0.5 canonical 落位：`completed`

## Formal 1v3

- 状态：`completed`
- dispatch：`C:\Users\numbe\Desktop\MahjongAI\logs\stage05_fidelity\s05_formal_triplet_20260405_winner_playoff_1v3\distributed\formal_1v3_dispatch\dispatch_state.json`
- winner：`anchor*1.0`
- close_call：`{'triggered': True, 'comparison_basis': 'avg_pt_primary_avg_rank_secondary', 'leader': 'anchor*1.0', 'runner_up': 'opp_lean*0.85', 'leader_avg_pt': -0.7752403846153846, 'runner_up_avg_pt': -0.8631310096153846, 'avg_pt_gap': 0.087890625, 'leader_avg_rank': 2.5092147435897436, 'runner_up_avg_rank': 2.508213141025641, 'avg_rank_gap': -0.0010016025641026438, 'combined_pt_stderr': 0.5984506429753185, 'close_threshold': 0.5984506429753185, 'stderr_mult': 1.0}`

## 路径

- 状态文件：`C:\Users\numbe\Desktop\MahjongAI\logs\stage05_fidelity\s05_formal_triplet_20260405_winner_playoff_1v3\state.json`
- 文档文件：`C:\Users\numbe\Desktop\MahjongAI\docs\status\stage05-fidelity-results.md`
