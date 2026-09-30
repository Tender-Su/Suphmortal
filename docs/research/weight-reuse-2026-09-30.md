# 存量权重与评测复用

> 核验：2026-09-30 · 双机目录、历史 manifest/result 与 18 个 checkpoint 的 CPU 安全元数据读取；没有新增训练或推理。此报告决定哪些计算可以省略，不新增模型发布结论。

## 覆盖与证据

ModelKits 的 2 个项目根、ABANDON 的 13 个项目/备份根，以及配置引用的 2 个现存临时输出根已盘点。共 2,360 个权重格式路径，365 个按名称判为索引/cache，1,995 个模型候选路径，未去重逻辑容量约 330.38 GiB。路径数不等于不同模型数，容量不等于可释放空间。

完整私有产物为 `MahjongAI-weight-inventory-20260930.json`、`MahjongAI-weight-index-20260930.csv` 和 `MahjongAI-weight-summary-20260930.txt`；目录及候选路径以 JSON 为准。没有全量反序列化、重算大权重 hash、解压旧包、删除副本或打开 sealed test。历史 hash、checkpoint ID、仅同大小三类证据不能混用；当前文件可能已覆盖历史 latest。

## 直接复用，避免重复计算

| 资产 | 已有证据 | 下一步用途 |
| --- | --- | --- |
| B adaptive-best / C50k | 同 actor `b15dd238de18e95ef367e751e9e20876306bfb1dd9b01f5ff06ee5026e08af67`；独立每臂 64k 对三家 canonical，+2.931328 pt，配对 95% CI [2.380061, 3.472734] | 作为有证据的研究起点；不重复这场确认，不自动改变发布身份 |
| C100k | actor `0d7301ca55021931b61d0249f32db1528b9c53d943993a0ddd1c0ab59238b6d3`；旧每臂 16k screen，+3.107813 pt，CI [2.010938, 4.199063] | 直接复用原 screen；不是独立确认，也不证明胜过 C50k |
| S70 | 同一旧 screen 已完成；C50k 对三家 S70 的独立各 2k 比较仍未决 | 不因旧小样本未决便重新铺开全筛选 |
| 9月 A/B/C/AC/CC 探针 | 两 seed 十臂共 40,960 逻辑成功更新及逐局结果已有 | 不重复原协议；仅回答 A2.88M 起点的晚期继续训练 |
| 旧 RL 候选 | 已有 32 候选、66k 局回溯；E9k/E10k 同 actor | 不盲扫旧 clip/value 网格；旧目标与新目标不混同 |
| 修复后 Oracle clean/warm 的 0/40k | 权重和 3,186 局 / 64,676 状态 monitor 曲线保留 | 先用现成模型；旧 best cluster 文件属于 step0，不当作40k预测 |

依据与统计边界见 [SL 正式确认](sl-formal-confirmation-result-2026-09-12.md)、[SL 状态](../status/supervised-mainline.md)、[课程结果](sl-curriculum-result-2026-09-10.md)、[旧 RL 审计](sl-rl-audit-2026-09-05.md) 与 [Oracle 匹配结果](oracle-matched-completion-2026-09-08.md)。旧 screen 的普通 CI 不含候选选择校正或训练 seed 不确定性。

## 新找到但不自动重评的资产

- 旧 C latest 为累计 300,000 steps / 299,894 optimizer steps，继承 B50k，实际新增 250,000 / 249,909；不是 C 阶段新训了 300k。现有训练验证未改善继承的 phase-best，范围内未找到该精确终点的 frozen 1v3；离线失败不能自动推导牌力较差，也不因步数更高就优先大评测
- B200k、A3.80M 终点及中间里程碑保留。A末92万步未进入已有 finalist，不等于这部分权重必然无用
- 早期 A1.13M、1.32M、1.36M、1.366M 保留，迁移记录已接上后续 converge。可从这些权重开明确的新分支，省去重复 A 预训练；未确认完整 RNG/数据游标，不声称 exact resume
- A2.88M 到旧动态 B 是 weights-only 初始化，重置主 optimizer/global steps，且验证 split 不同；不能把跨阶段 loss 直接相减或忽略辅助目标变化来解释课程因果收益
- 旧 Oracle A1.55M、B250k、C170446 也保留，只作历史目标下的诊断端点，不替代新目标匹配对照

旧 32 游戏复验比较 150k / 当时 latest 1,419,948，p0 差 +0.00549、CI 跨零；这一已完成历史结果继续保留，不改写成当前长训权重的比较结论。

## 已核验的长训 critic 复用边界

2026-09-30 补充依据为 ModelKits 任务的三份旧权重 CPU 核验回报；云端未加载这些大权重，清单不替代 checkpoint 内部证据。候选 ID 对应原库存定位，尚未进行新推理或 GPU 训练。

| 资产 | 核验值 | 可用结论 |
| --- | --- | --- |
| MK0174，旧 formal best | 内部 steps 2,370,000；记录 loss 2.9381111465 | 有长训候选，不能把旧 formal 全部简化为短 warmup |
| MK0176，旧 formal latest | 内部 steps 2,500,000；实际最后 validation loss 2.9660594597 | 终点与 best 分开，不能用 best_loss 字段冒充最后一次验证 |
| MK0223，constant_low latest | 内部 steps 1,600,000，独立 1.58M + 20k 链路 | 不是从 2.50M 继续训练的后续终点 |

三者均为旧 `env.pts=[6,4,2,0]`、gamma .999，不能与当前 `[2,1,0,-3]`、gamma 1 的 loss 直接比较。旧 formal 为 192×40 towers、`all_players`、`score_rank_mc`；1.60M 为 hand-aligned、`residual_mlp` fusion 1024、head 256，带 exact-zero 配置和 weight decay .03。这些差异先做加载和目标迁移预检，不要求以旧 optimizer 原样续训。

旧 checkpoint 未提供可确认的 norm、成功更新计数、optimizer class 和 ScheduleFree train/eval mode 字段。不能凭文件名、配置惯例或 tensor 数量补出这些身份，也不能宣称 exact resume 或导出模式已验证。下一步先补历史曲线与 import preflight；若需要新评分，明确统一标签、结构和模式后再比较。现有 fixed-imputed 三模型 pilot 入口严格要求新标签与 0/40k 身份，不能把这些长训旧权重直接塞入该接口或关闭契约检查冒充兼容。

## 决策

### 同日既有报告复核（未新增推理）

- `matched_completion_20260908_r1/completion_audit.json`：同一 monitor 3,186 局 / 64,676 状态，clean 的 p0 MSE 3.819599→2.362675，差值 CI [-1.515626,-1.398223]；warm 2.628318→2.353649，CI [-0.295419,-0.253919]。clean 被 exact-zero target 分组 veto，warm 被 abs-ge4 分组 veto；两者 MAE 均改善。10k 至 40k 各 gate 都受对应分组拒绝，原协议最终仍为 inconclusive。
- 对应运行目录保留 step0 adaptive-best 与 40k latest，没有 best-primary 文件。warm 20k 记录值 2.353004 略低于 40k，但该目录未保存其权重；clean 资源迁移副本 31,144 / 33,069 仍存在，未经新评分不能认定更好。
- `s70_sf_finalist_stage10_s190000_20260831_r1/paired_with_legacy_1600k_remainders123.json`：同旧目标下 1.6M / SF200-190k / SF100-190k 的 p0 MSE 为 3.201800 / 3.215777 / 3.218857。SF200−1.6M 差值 +0.013977，CI [0.009006,0.018948]，9,533 局 / 49,482 状态。这场历史比较无需重跑，也不能归因为单一 optimizer 效果。
- 旧 Oracle adaptive B/C 转段均继承 A150k adaptive-best，manifest 的模型状态 hash 相同；不能将阶段名视为各自新胜者。旧 A1.2M 是总体 MSE 改善、主 p0 MSE 未改善，不能归入“主指标改善被诊断否决”的两例。
- 旧 canonical 128局 / 512状态探针以及人类 dev 32局复算，不构成 C50k 当前策略分布上的充分对齐证据。

新实验先写明尚未解决的疑问及它会改变的决定。优先重用原始结果和已存权重；只有缺失的预测、分布诊断或牌力比较确实影响选择时才新增有界推理。离线拟合是廉价诊断，不是对战实力证明；避免用一次筛选最高点或已实现结果分组硬门槛代替最终 actor 收益。当前没有依据重训整段 A 或重新跑两条 Oracle40k。
