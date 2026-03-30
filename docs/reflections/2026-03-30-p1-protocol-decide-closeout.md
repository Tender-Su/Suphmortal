# 2026-03-30 P1 protocol_decide 收口复盘

这份文档记录这轮 `P1 protocol_decide` 为什么可以正式收口，以及这次收口后哪些默认已经被写死。

## 结果

- 当前主线 run：
  - `logs/stage05_fidelity/s05_fidelity_p1_top3_cali_slim_20260329_001413/`
- 当前状态：
  - `stopped_after_p1_protocol_decide`
- 当前已验证协议 winner：
  - `C_A2x_cosine_broad_to_recent_strong_24m_12m`
- 当前下游下一步：
  - 只在 `A2x` 内部做 `winner_refine`

## 这轮真正确认了什么

- `A2x` 是当前最适合继续做三头全开主线载体的协议 winner
- `A2y` 仍然保有更低的裸 `ce_only / recent_policy_loss`，但它不是当前最好的三头主线载体
- `A1x` 仍有帮助，但当前整体不如 `A2x`

## 为什么这轮能收口

- 双 seed 都已经完成
- 唯一失败 arm 已经补跑成功
- 当前结果是 `27 / 27` 全部有效，没有残留 invalid arm
- 补跑后 winner 没有变化，说明当前结论不是建立在偶发故障之上

## 收口后写死了什么

- 当前 downstream 默认协议写死为 `A2x`
- 当前 `protocol_decide` 默认 seed2 扩展规则写死为 `flip_or_gap @ 0.001`
- 当前 `winner_refine` 默认不再是自动 `top-k center`
- 当前 `winner_refine` 冻结三中心：
  - `C_A2x_cosine_broad_to_recent_strong_24m_12m__B_r0046_o0037_d0037`
  - `C_A2x_cosine_broad_to_recent_strong_24m_12m__B_r0034_o0014_d0041`
  - `C_A2x_cosine_broad_to_recent_strong_24m_12m__B_r0052_o0025_d0043`

## 这轮最重要的经验

### 1. 裸协议强，不等于三头主线载体强

- `A2y` 在 `ce_only` 和 `loss` 上仍然很强
- 但当前主问题不是“谁的裸 CE 最低”，而是“谁适合挂上三头继续往下走”
- `protocol_decide` 的价值，就是把这两个问题彻底拆开

### 2. invalid arm 不补跑，会系统性低估协议

- 本轮唯一失败臂是 `A2x danger_lean 0.12`
- 补跑前它在合并榜单里被压到第 `27`
- 补跑成功后，它直接回到第 `6`
- 这说明 invalid arm 即使不翻 winner，也会明显扭曲“协议强弱的证据密度”

### 3. calibration 代表协议与 downstream winner 可以不同

- 当前瘦版 `calibration` 仍保留 `A2y-only + combo_only`
- 这是定标角色，不是 downstream winner
- 当前真正要写死到下游默认里的协议是 `A2x`

## 这轮人机协同里的分工

- 人负责把目标和收口条件定义清楚：先补跑失败臂、先把协议问题和配比问题拆开、先写死当前默认
- AI 负责执行和落地：定位失败 arm、补跑、比较前后差异、同步更新代码、测试和文档
- 这次协同最有效的地方，不是“谁单独更会算分”，而是人持续校正方向，AI 持续把方向转成可执行工程动作
