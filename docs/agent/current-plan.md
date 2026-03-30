# 当前默认续工作入口

这份文档只回答三件事：当前主线是什么、真实停在哪一步、如果继续应该做什么。

稳定默认看 `docs/agent/mainline.md`。  
运行方法看 `docs/agent/experiment-workflow.md`。  
代码同步看 `docs/agent/code-sync.md`。  
笔记本 shell / 数据链路看 `docs/agent/laptop-remote-ops.md`。

## 建议读取顺序

1. `AGENTS.md`
2. 本文档
3. `docs/agent/mainline.md`
4. `docs/status/stage05-verified-status.md`
5. `docs/status/p1-selection-canonical.md`
6. 如果任务涉及笔记本或 Git 同步，再读 `docs/agent/code-sync.md` / `docs/agent/laptop-remote-ops.md`

## 当前主线

- 当前代码主分支固定为：`main`
- 当前源码真源是台式机 `main` 工作树
- 双机代码同步默认入口：`docs/agent/code-sync.md`
- 当前 Stage `0.5 / P1` 唯一主线仍是：
  - `calibration -> protocol_decide -> winner_refine -> ablation`
- 当前已验证的 downstream 协议 winner：
  - `C_A2x_cosine_broad_to_recent_strong_24m_12m`

## 当前真实停点

- 当前主线 run：
  - `logs/stage05_fidelity/s05_fidelity_p1_top3_cali_slim_20260329_001413/`
- 当前状态：
  - `stopped_after_p1_protocol_decide`
- 当前这轮 `protocol_decide` 已收口：
  - 双 seed 已完成
  - 唯一失败 arm 已补跑成功
  - 当前结果是 `27 / 27` 全部有效
  - winner 仍然是 `A2x`
- 当前原则：
  - 停在 `protocol_decide` 收口点
  - 不自动进入 `winner_refine`

## 如果继续

1. 只允许在 `A2x` 协议内部启动 `winner_refine`
2. 直接使用 `mainline.md` 里已经冻结的 winner_refine 默认，不重新自动取 `top-k center`
3. 如果让笔记本参与，不要让两台机器共同写同一个 `run_dir`；改用桌面机调度入口：
   - `python mortal/run_stage05_winner_refine_distributed.py dispatch --run-name s05_fidelity_p1_top3_cali_slim_20260329_001413`
4. 这个双机入口只改变执行方式，不改变当前 `winner_refine` 的 center、局部搜索点或最终选模口径
5. `winner_refine` 跑完后再次停下，再决定是否进入 `ablation`

## 不要这样做

- 不要把 `docs/status/stage05-fidelity-results.md` 当成当前默认入口；它只是 run snapshot
- 不要把旧 `solo / pairwise / joint refine` 结构重新当作当前主线
- 不要在没有人工确认的情况下把多个阶段一键串到底
