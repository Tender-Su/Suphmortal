# 当前默认续工入口

先读 `docs/agent/mainline.md`，再读 `docs/status/supervised-verified-status.md`。这份文档只保留当前停点、当前下一步和当前工作边界。

## 当前停点

- 当前开发主线：`main`
- 当前源码真源：台式机 `main` 工作树
- 当前项目位置：
  - 监督学习阶段已经完成
  - 强化学习阶段方案定义正在进行
- 当前已验证协议 winner：
  - `C_A2x_cosine_broad_to_recent_strong_24m_12m`
- 当前已验证 winner 点位：
  - `0.12 + A2x`
- 当前正式 supervised winner：
  - `anchor*1.0`
- 当前第一替补：
  - `opp_lean*0.85`
- 当前 canonical supervised checkpoint：
  - `./checkpoints/sl_canonical.pth`
- 当前关键证据 run：
  - `logs/sl_fidelity/sl_fidelity_p1_top3_cali_slim_20260329_001413/`
  - `logs/sl_fidelity/sl_formal_triplet_20260405/`
  - `logs/sl_fidelity/sl_formal_triplet_20260405_winner_playoff_1v3/`

## 当前下一步

1. 明确强化学习阶段方案。
2. 以当前 canonical supervised checkpoint 作为 RL baseline 起跑点。
3. 如有新的更强起点证据，再更新 RL 起跑配置。

## 当前工作边界

- 项目主线描述统一写成：
  - `监督学习阶段 -> 强化学习阶段`
- 当前 winner / backup 解释统一写成：
  - winner = `anchor*1.0`
  - 第一替补 = `opp_lean*0.85`
- 当前真相入口统一使用：
  - `docs/status/supervised-verified-status.md`
  - `docs/status/p1-selection-canonical.md`
  - `docs/status/supervised-formal-triplet-playoff-canonical.md`
- 监督学习阶段演进背景统一放在：
  - `docs/research/supervised-evolution.md`
