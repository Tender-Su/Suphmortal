# S70 公开权重强度对照（2026-08-09）

> 历史归档 · 归档整理：2026-09-05。正文保留当时的事实、判断和命令，不作为当前运行依据。当前入口见 [文档地图](../../../README.md)；旧 SL / RL 强度及 Oracle 验证结论须结合 [独立审计](../../../research/sl-rl-audit-2026-09-05.md) 阅读。

## 结论

在本次能够下载、校验并实际跑通的公开四麻权重集合中，S70 位于最强档：

- S70 **远强于 RiichiPPO 的公开 `s80` 权重**，双向对照均为压倒性差距。
- S70 与公开 `mortal-298k` **处于同一大档，点估计略偏向 S70**；当前双向各 2000 半庄的置信区间都跨过零，不能宣称统计显著胜出。
- 结合项目已有 64k `1v3`，S70 已经确定强于 Original SL 和 S8。

因此，最稳妥的强度表述是：**S70 是目前找到的可本地运行公开权重中的上游/第一档模型；与 `mortal-298k` 同档，当前点估计略偏 S70，但尚未被证明达到或超过 Mortal 4.1c、NAGA、Suphx 等非同权重、非同环境的顶级系统。** 不能把当前结果直接换算成天凤段位或雀魂段位。

## 可复查结果

全部 1v3 使用 `[90, 45, 0, -135]` rank pt；置信区间以四个座位共用同一 seed 的 duplicate set 为独立单位。Mortal 官方基准也采用四座轮换的 duplicate 1v3 和相同 rank pt 定义：<https://mortal.ekyu.moe/perf/strength.html>。

| Challenger | Opponents | Sets / games | 1/2/3/4 位 | 平均顺位 | 平均 rank pt | 95% CI |
|---|---|---:|---:|---:|---:|---:|
| S70 | Original SL x3 | 16k / 64k | 见既有 artifact | 2.46630 | +2.30977 | 既有报告未保存 CI |
| S70 | S8 best_loss x3 | 16k / 64k | 见既有 artifact | 2.47986 | +1.64320 | 既有报告未保存 CI |
| `mortal-298k` | S70 x3 | 500 / 2000 | 479/500/512/509 | 2.52550 | -1.55250 | [-4.75698, +1.65198] |
| S70 | `mortal-298k` x3 | 500 / 2000 | 508/478/503/511 | 2.50850 | -0.87750 | [-4.06811, +2.31311] |
| S70 | RiichiPPO x3 | 10 / 40 | 31/6/2/1 | 1.32500 | +73.12500 | [+61.16007, +85.08993] |
| RiichiPPO | S70 x3 | 5 / 20 | 0/4/4/12 | 3.40000 | -72.00000 | [-111.81231, -32.18769] |

两个 `mortal-298k` 方向都略为负并不矛盾。1v3 比较的是“一份 challenger 对三份同种 opponent”的桌面组成，交换双方会改变整桌策略分布；它不是严格的零和镜像。可用的相对信号是 S70 方向高出 0.675 rank pt，但这个差距远小于当前不确定性。

RiichiPPO 的样本小，但效应量极大，且双向 95% CI 都远离零；已经足以判断不在同一强度档。继续跑到数千局只会提高数值精度，不会合理地逆转分档结论。

## 权重和环境

| 模型 | SHA-256 | 实际加载结构 |
|---|---|---|
| S70 `best_action_score.pth` | `873914071538805c548599434e365d626892e45d10924486f087e65b856c378d` | v4, GN, 192 x 40, categorical policy |
| `mortal_298k.pth` | `bfb3a6c072aa0bfd4171a9cdc77cb6c02ae42cde920843f9e5784394f23447d8` | v4, BN, 192 x 40, legacy DQN |
| RiichiPPO `ppo_2025_self_kldistill_s80_v1.pth` | `f2456780433b46b77fbd6756f43e11be6034c07c9038f8c9489a5f1468af155a` | 74 channels, 16 x 192 BN ResNet, 82 actions |

`mortal-298k` 的模型卡明确写明 298k、192 x 40、四麻半庄，并警告不建议用于排位：<https://huggingface.co/VoidShine/mortal-298k>。本报告不采用模型卡的 MAKA 等级作为强度证据，只采用实际对局。

Mortal 对照在本项目原生 `libriichi` 中运行。RiichiPPO 架构与 S70 不共享动作头，因此用 RiichiEnv `4p-red-half`、`default_tenhou()` 规则进行跨架构对照；两个模型各走自己的原生 observation/action 映射。RiichiPPO 权重训练时使用的 RiichiEnv 版本与当前 0.4.8 不完全一致，这是该组结果的主要工程局限，但所有对局均合法完成，且巨大双向差距不太可能只由小版本差异解释。

## Akagi 间接标尺

Mortal-Policy 作者公开的可复核文字结论不是“Policy 对 Akagi 的精确领先值”，而是：其模型排序为 `policy online + BC > value offline-to-online > value offline-only`；其中 value online 相对 offline 约 `+0.6 PT / 500k`，最弱的 value offline-only Mortal-V2 对 `Akagi offline 240308 V4 best/V3 best` 约 `+1.8~2.0 PT / 200k`。作者只说 Policy 的提升“超出预期”，没有公布 Policy 对 Akagi 的数值。因此，S70 对同一 Akagi 权重的结果可以判断它是否越过作者的最弱 value baseline，但不能仅靠减法精确换算 S70 与 Policy 的差距：<https://github.com/Equim-chan/Mortal/discussions/91>。

目标 V4 权重的准确文件名已从第三方 Mortal-Policy 配置中锁定为 `model_v4_20240308_best_min.pth`。完整搜索 Akagi 全部 Git 历史、GitHub Release、GitHub/Sourcegraph/grep.app/Hugging Face 等可索引来源后，没有找到该文件；Akagi 的旧版说明要求从 Discord 的 `#bot-zip` 下载 `bot.zip` 后提取 `mortal.pth`：<https://github.com/shinkuan/Akagi>。

Akagi v2.0.0 公开包内确实有一个 `mortal.pth`，但它只有约 5.1 MB，实际为 v4、BN、`32 x 2` 的 legacy DQN；仓库说明也把它称为 tiny weak model。S70 对该占位权重的 40 局加载/动作/完整对局 smoke 已通过，结果仅 `+1.125 PT`，样本极小且对手不是 240308 权重，故不纳入强度表，也不能用于推算 Mortal-Policy。

## 开源权重覆盖边界

搜索到的仓库不少，但可直接用于公平对局的强权重很少：

- `Mortal-Policy` 公开了训练代码，但 README 明确说明 weights 和部分超参数已从开源仓库移除：<https://github.com/Nitasurin/Mortal-Policy>。
- Kanachan 明确不提供训练数据或训练后模型：<https://github.com/Cryolite/kanachan>。
- NAGA、Suphx 没有发现可下载、可在同一环境运行的官方权重，因此没有把宣传结果或论文结果伪装成直接对战。
- Akochan 按用户指出的定位，只保留为历史弱基线，不拿来评估 S70 上限。

## 线上平台

RiichiLab 是可用的公开 AI 对战平台，采用 WebSocket MJAI 协议和 OpenSkill 排名：<https://riichi.dev/docs>。已实现 `integrations.riichilab.run_mortal_bot`，同一个 loader 可以加载 S70 和后续 categorical-policy RL checkpoint。

S70 在当前机器 CPU 上用 Copilot loader 预热后连续 20 次 forward 的平均时延为 **23.66 ms**；这不是端到端网络时延，但相对平台建议的普通回合 500 ms 内推理目标有充足余量。实际 S70 checkpoint 加载和动作 smoke 也已通过，格式识别为 `policy_net/categorical/GN`，steps 为 390000。

真正进入线上排位仍需模型所有者用 GitHub 登录、创建 bot、保存一次性 token、完成 validation。这个步骤会创建外部账号状态，不能在没有所有者授权和 token 的情况下代办。线上排名形成后，它将是 S70 相对其他在线 bot 的补充证据；在累计足够场次前，不应替代 formal `1v3`。

2026-08-09 已完成首次实际上线：四麻 bot `MahjongAI-S70`（bot 305）用 CPU 加载 S70，validation 一次通过，102/102 个动作请求全部 accepted，`defaulted=0`、`rejected=0`。随后运行一场受控 ranked 半庄，79/79 个动作请求全部 accepted，仍为 `defaulted=0`、`rejected=0`。对局 `06f1bd83-0036-41b0-a613-2f51e5764bd4` 的结算为：`Nodoka v0.1` 51,500（1 位）、S70 39,100（2 位）、`cheese` 9,700（3 位）、`Nodoka v0.3` -300（4 位）；S70 平台评分 `1500 -> 1560`。对局页：<https://riichi.dev/games/06f1bd83-0036-41b0-a613-2f51e5764bd4>。

该结果只证明真实服务端协议兼容、动作合法性和首场运行稳定；`n=1` 没有强度推断价值，`+60` 也主要受新 bot 的初始不确定性影响。RiichiLab 结果应在积累足够场次后单独汇总，不能混入 formal `1v3`。

## 原始证据

- `logs/external_model_eval/s70_vs_mortal298k_2000/result.json`
- `logs/external_model_eval/s70_challenger_vs_mortal298k_2000/result.json`
- `logs/external_model_eval/s70_vs_riichippo_10sets/result.json`
- `logs/external_model_eval/riichippo_vs_s70_5sets/result.json`
- `logs/external_model_eval/riichilab_s70_validate.jsonl`
- `logs/external_model_eval/riichilab_s70_ranked.jsonl`
- `logs/analysis/s70_curves_20260724/s70_report_artifact.json`

当前 Oracle critic 训练在所有比较完成后仍正常推进，比较过程没有停止或替换该训练。
