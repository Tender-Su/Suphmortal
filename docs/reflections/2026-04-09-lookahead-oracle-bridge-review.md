# 2026-04-09：Look-Ahead / Oracle / Bridge 复盘备忘

这是一份背景层备忘，记录一轮围绕 `Suphx` 论文、当前 RL `Oracle` 主线和 bridge 设计的集中讨论。它不覆盖 `docs/agent/` 和 `docs/status/` 的当前默认口径，但会影响后续实验优先级。

## 0. 同日决策更新

在这份备忘写完之后，项目主线又做了一次明确收束。当前默认决策改成：

- 不再把 actor Oracle checkpoint 额外导出成单独的 normal 结构；
- 改为保留 **同结构 actor-shaped checkpoint**，在评估/部署时统一走 **zero-oracle visible-only**；
- input bridge 继续保留，但默认不再是“新增 slice 全 0”，而是：
  - visible slice 直拷；
  - 新增 oracle slice 用 `small kaiming * tiny_scale`；
- `dual-encoder fusion / sidecar privileged branch` 作为备选路线保留；
- 只有在后续观察到：
  - actor Oracle 退到 0 后性能跌幅过大，
  - 且 continuation 无法拉回，
  - 才考虑切去 dual 路线。

因此，下面正文里凡是把“normal export”或“新增 slice 全 0”写成更强推荐项的部分，都应以这一节的同日更新为准。

## 1. 这轮讨论里形成的核心判断

### 1.1 对 Suphx look-ahead feature 的判断

用户提出的核心怀疑是成立的，而且论文原文本身就支持这种担心：

- `Suphx` 的 look-ahead feature 明确是一个**强进攻先验**。
- 它做的是“给定当前打哪张牌，估计之后在若干次自家摸打演化下，可能和出的概率与打点”。
- 论文原文还明确写了两个强简化：
  - 用 DFS 搜可能和牌形；
  - **忽略对手行为，只考虑自己摸打行为**。

这意味着它不是一个中性的“未来理解模块”，而是一个偏向：

- 进攻；
- 自家手牌效率；
- 和牌可能性与打点组合空间；
- 基于 visible-only 的启发式前瞻。

因此，把它理解为“用手工搜索器把模型往进攻方向推了一把”是合理的。

我对这件事的当前判断是：

- 这类特征很可能确实能显著帮助模型更快学会：
  - 牌效率；
  - 形效率；
  - 攻击型打点意识；
  - 某些 discard 的局部未来收益比较。
- 但它**不太可能自然补齐**立直麻将里更难的部分：
  - 防守；
  - 对他家手牌进度/大小/形态的估计；
  - 攻守判断；
  - 局收支与局况驱动的策略切换。

所以用户的那条主判断我认同：

- 如果模型自己就能较好学会牌效率，那么额外的结构/特权信息更值得投向：
  - 防守建模；
  - 他家状态估计；
  - 攻守转换；
  - 局况驱动的价值判断。

这也解释了为什么当前仓库里已经在做的 `opponent state / danger / critic` 方向，比继续堆更强的“纯进攻前瞻器”更贴近这套判断。

### 1.2 对 Oracle actor 证据强度的判断

这里要把“论文证明了什么”和“我们能从论文外推出什么”分开。

论文公开证据只足够支持下面这句话：

- 在 Suphx 的训练设置里，`RL-2 = RL-1 + oracle guiding`，而 `RL-2` 比 `RL-1` 更强。

但论文**没有**证明下面这些更强的命题：

- `Oracle actor` 作为结构设计本身必然正收益；
- 换到别的 reward / critic / backbone / sample regime 后仍然正收益；
- 如果不用它，PPO 结果一定更差；
- `actor Oracle` 的收益独立于他们整套训练 recipe 的其他部分而存在。

也就是说：

- 论文证明的是“那条完整 recipe 在他们的设置里有效”；
- 没证明“actor Oracle 这个部件单拿出来，在任何近似设置里都稳定有利”。

所以对这条路线持怀疑态度是完全合理的，尤其是在：

- 当前工程不是 Suphx 同一套 reward；
- 当前工程还有 `Oracle critic`、`Step-Level GAE`、`danger/opponent/regret` 等额外结构；
- 训练资源和分布也明显不同。

### 1.3 对 distillation 的判断

论文对蒸馏其实给了两层信息，不能简单总结成“蒸馏不行”。

`rl.tex` 的口径是：

- **simple knowledge distillation does not work well**
- 原因是 normal agent 和 oracle agent 的可观测信息差太大，normal agent 很难模仿一个“强得过分”的 oracle。

但 `remark.tex` 的口径又是：

- 如果同时训练 oracle agent 和 normal agent，
- 并且对两者之间加约束，让 oracle 去 distill normal，
- 他们的 preliminary experiments 里“**also works quite well**”。

因此，更准确的总结应该是：

- **简单/后验式 distillation 不好用**；
- **带联训和约束的 distillation 不是死路，论文甚至明确说 preliminary result 还不错**；
- 只是 Suphx 最终公开主线选择了 perfect-feature dropout，而不是把联训蒸馏做成正式主 recipe。

### 1.4 对 Oracle critic 的判断

Suphx 论文公开内容里：

- `Oracle guiding / actor-side perfect-feature dropout` 是正式做过实验并进主线的东西；
- `Oracle critic` 只在 `remark.tex` 里作为后续可行方向提出。

所以当前仓库更准确的表述不是“照抄 Suphx Oracle”，而是：

- `Suphx-style actor Oracle guiding`
- 加上
- `CTDE-style Oracle critic`

这是在 Suphx 公开主线之上做的扩展，而不是完全复刻。

同日后续决策还补了一条更具体的实现口径：

- `critic` 默认保持 full-oracle 视角，不再额外做屏蔽；
- 如果后续要尝试屏蔽，也只能作为稳定性对照实验，而不再是默认主线。

## 2. 对当前 bridge 方案的重新判断

### 2.1 之前最容易说歪的一点

这轮讨论后，一个需要明确纠正的点是：

- 当前代码里的部署口径，不能简单说成“把 oracle 模型硬裁掉一半”。

当前实现实际上是：

- `mortal/online/train_online.py` 在 actor Oracle 开启时，导出的是 **同结构 zero-oracle checkpoint**；
- `mortal/core/checkpoint_utils.py` 的 `load_brain_state_with_input_bridge()` 会把可对齐参数搬过去；
- 只有第一层卷积因为输入通道数不同，需要做 input bridge；
- 额外的 oracle 输入切片目前是：
  - 先做小尺度随机初始化；
  - 再把原 visible slice 权重原样拷进去。

这意味着当前 bridge 有一个非常重要的性质：

- **它是 function-preserving 的可控扩容/回投影**。

更直白地说：

- `Brain(is_oracle=True)` 和 `Brain(is_oracle=False)` 的结构差异，实质上集中在第一层输入通道数；
- 如果 oracle actor 在前向时看到的 oracle 通道全是 0，
- 那么它的计算结果应当与 bridge 后的 normal brain 对齐。

所以这里真正的风险不是“导出时突然拿刀砍掉一块从未见过的网络”。

真正的风险是：

- 在 `gamma -> 0` 之后，
- 模型是否真的已经学会在“oracle 输入为 0”时也能稳定工作；
- continuation 是否足够长、足够稳；
- importance weight 过滤和降 LR 是否真的压住了 distribution shift。

### 2.2 为什么新增通道置 0 仍然是一个强默认

用户担心：

- 置 0 太保守；
- Oracle 通道启动慢；
- 可能让模型“看不见”新信息。

这个担心有道理，但对当前 actor Oracle 直连 bridge 结构来说，我仍然认为：

- **置 0 是更好的默认起点**。

原因有四个：

1. 它最大限度保住原 visible-only 策略函数。
2. 它不会在 step 0 就因为随机 oracle slice 把 policy 推偏。
3. 它天然抑制 shortcut learning 的突发启动。
4. 它和后续 `gamma 1 -> 0` 的 schedule 逻辑是同方向的。

“置 0 会不会让新增通道完全学不到”这个担心，需要精确一点看：

- 不会永远学不到；
- 只是**初始贡献为 0**；
- 只要 loss 对这些通道对应的第一层权重有梯度，它们还是会被更新；
- 代价主要是启动更保守，而不是永久失明。

对于 actor Oracle 这种高风险 shortcut 源来说：

- 我更愿意接受“慢一点但稳一点”，
- 而不是“更快吃到 oracle，但一开始就把 deployable visible policy 搅坏”。

### 2.3 为什么“直接随机初始化新增 slice”不是更优默认

如果把新增 oracle slice 用标准 `Kaiming/Xavier` 随机初始化，会发生什么：

- 好处：
  - Oracle 信息更快进入网络；
  - 学习启动更积极。
- 坏处：
  - 一上来就改变原函数；
  - 一上来就把 oracle 噪声/捷径注进 policy；
  - 更容易让 actor 走向依赖 privileged info 的短路解。

对于 `critic-only` 路线，这个风险还相对可控。

对于 `actor Oracle` 路线，这恰恰是最该防的风险。

所以我的当前结论是：

- **对当前直连 actor bridge 主线，更合理的默认是：visible slice 直拷 + oracle slice 用 `small kaiming * tiny_scale`，而不是满幅随机初始化，也不是继续坚持全 0**。

## 3. 可考虑的初始化改良，但不建议一上来替代 0-init

如果后续确认“0-init 启动过慢”确实是瓶颈，那么更合理的探索不是“直接把新增 slice 换成普通 Kaiming/Xavier”，而是以下几类更克制的方法。

### 3.1 小幅随机初始化，而不是满幅随机初始化

可以考虑：

- visible slice 继续完全拷贝；
- oracle slice 用 `Kaiming` 或 `Xavier` 初始化；
- 但再乘一个很小的系数，例如 `0.01 ~ 0.05`。

这样做的含义是：

- 给 oracle 通道一个非零起点；
- 但不允许它一开始就大幅改写原策略函数。

这是一个比“满幅随机初始化”更合理的中间方案。

### 3.2 给 oracle 分支加可学习 gain，而不是直接裸拼接

可以把逻辑改成：

- 先做 visible 编码；
- oracle 新信息先走一条小分支；
- 最后通过一个很小的 gain 融入主干。

比如概念上：

```text
phi_vis = E_vis(obs)
phi_orc = E_orc(obs, oracle_obs)
phi = phi_vis + alpha * F(phi_vis, phi_orc)
```

其中：

- `alpha` 初始很小，甚至接近 0；
- 后续训练自己决定 oracle 分支介入多少。

这类思路比“直接随机初始化新增输入 slice”更符合当前目标：

- 保住 visible trunk；
- 允许 oracle 有渐进影响；
- 更方便做 dropout 和 continuation。

### 3.3 LSUV/数据驱动校准，只适合随机化方案的后处理

如果真的做了带随机初始化的新分支，可以再考虑：

- 用类似 `LSUV` 的 activation variance 校准做一次启动后校准。

但这类方法更像是：

- 给随机化方案做稳态修正；
- 不是对当前 zero-bridge 方案的首选替代。

### 3.4 Fixup 风格的小尺度/零尺度残差分支

如果后续转向“额外 sidecar oracle branch + 残差融合”，那么：

- 用很小的 residual scale 起步，
- 或者用近似 0 的初始融合强度，

会比直接把 oracle 通道粗暴拼到第一层更自然。

这里它更适合支持的是“**sidecar 分支怎么接入**”，不是“第一层新增输入 slice 该不该直接随机化”。

## 4. Dual-Encoder Fusion 到底是什么

之前“dual-encoder fusion”说得太抽象，这里写得更明确一点。

### 4.1 它的结构含义

它的核心不是“把输入通道加宽”，而是：

- visible-only 路径保留一条完整主干；
- privileged/oracle 信息单独走另一条 encoder；
- 两条路径在中高层做融合。

可以是下面几种样子：

1. `phi = phi_vis + alpha * phi_orc`
2. `phi = MLP([phi_vis, phi_orc])`
3. `phi = phi_vis + alpha * Proj([phi_vis, phi_orc])`

这里最重要的不是具体 fusion 公式，而是结构上的隔离：

- visible trunk 负责最终可部署能力；
- oracle trunk 负责训练期辅助；
- 融合强度是可控的。

### 4.2 它相比当前 direct-input bridge 的优点

它的优点在于：

- shortcut risk 更局部；
- visible trunk 更干净；
- 更容易观察“没有 oracle 时主干到底学会了什么”；
- 更容易把 oracle 的影响逐渐压小；
- 到 `gamma=0` 后继续 normal continuation 时，visible trunk 的连续性更强。

### 4.3 它和 dropout / importance weight 过滤是否兼容

是兼容的，而且我认为兼容度比 direct-input 还更好。

兼容方式很直接：

- dropout 可以作用在：
  - oracle encoder 的输入；
  - oracle encoder 的输出；
  - fusion gate `alpha`；
- importance weight 过滤仍然是 policy 更新层面的稳定器，和 encoder 组织方式正交。

所以如果以后要升级结构，我认为：

- `sidecar privileged encoder + gated fusion`
- 是比“第一层直接加更多 oracle slice 并随机初始化”更值得优先试的方案。

### 4.4 它的代价

代价也很明确：

- 参数量更大；
- 工程复杂度更高；
- 需要额外设计 fusion 点和 gate；
- rollout / export / eval / resume 签名都会更复杂。

因此它更像：

- 下一阶段值得探索的结构增强；
- 而不是现在就该替换当前主线的最低风险方案。

## 5. 对当前仓库主线的倾向性建议

### 5.1 不建议放弃 actor Oracle

用户强调：

- “不做 Oracle actor 是不可接受的，直接放弃了微软最大的创新点。”

我理解这个判断，而且在“研究价值”层面我基本同意：

- `actor Oracle guiding` 的确是 Suphx 最具辨识度的创新点之一；
- 如果完全不做，确实会错过一条非常重要的技术线。

但这里更合适的说法是：

- **不应轻易放弃 actor Oracle 这条研究线**
- 而不是
- **默认它在当前工程里已经被证明是最优主线**。

也就是说：

- 值得做；
- 但要严肃做可证伪的 A/B；
- 不能把“论文里做过”直接等价成“这里一定最优”。

### 5.2 我对当前主线的首选态度

如果只讨论当前这份仓库、当前这版代码，我的偏好是：

1. 继续保留当前 `actor Oracle + Oracle critic + continuation` 主线；
2. 对 direct-input bridge，继续以 `visible slice 直拷 + oracle slice = small kaiming * tiny_scale` 为默认；
3. 把主要怀疑点放在：
   - `gamma` 退火曲线；
   - continuation 长度；
   - importance weight 过滤阈值；
   - critic 是否真的需要任何屏蔽；
   - visible-only `1v3` 评估是否真的站得住；
4. 如果后续要做结构升级，优先试：
   - `sidecar oracle encoder + gated fusion`
   - 而不是
   - “把当前新增 slice 改成普通随机初始化”。

### 5.3 对 look-ahead feature 的策略含义

如果把这轮讨论落回“后续该往哪里加东西”，我的倾向是：

- 牌效率不再是最该额外强化的地方；
- 防守和攻守判断更值得吃额外建模预算；
- privileged info 更应该优先服务：
  - critic；
  - opponent-state estimation；
  - danger estimation；
  - higher-level attack/defense switching。

这与当前仓库已经在推进的：

- `Oracle critic`
- `OpponentStateAuxNet`
- `DangerAuxNet`

方向是相容的。

## 6. 这轮讨论后的实验优先级建议

如果后续要做最小代价验证，我建议按下面顺序：

1. 先把当前主线真正跑出一轮完整 `gamma 1 -> 0 -> continuation` 的 visible-only 结果；
2. 对比：
   - `critic-only Oracle`
   - `actor+critic Oracle`（critic 保持 full-oracle）
3. 在 actor Oracle 主线内，只做一个初始化小对照：
   - `bridge slice = zero`
   - `bridge slice = small kaiming * tiny_scale`
4. 只有在确认“0-init 确实拖慢且最终效果受损”后，再立项：
   - `dual-encoder / sidecar fusion`

## 7. 参考材料

### 7.1 本地 Suphx LaTeX 源码

- `C:\Users\numbe\Desktop\suphx\flow.tex`
  - look-ahead feature 的定义与强简化
- `C:\Users\numbe\Desktop\suphx\rl.tex`
  - oracle guiding、perfect-feature dropout、continuation tricks
- `C:\Users\numbe\Desktop\suphx\remark.tex`
  - distillation 与 oracle critic 的讨论
- `C:\Users\numbe\Desktop\suphx\Offline.tex`
  - `RL-2` 相对 `RL-1` 的公开实验结论

### 7.2 当前仓库实现

- `mortal/core/checkpoint_utils.py`
  - input bridge 与 normal export
- `mortal/online/train_online.py`
  - actor Oracle、continuation、visible-only eval/export
- `mortal/core/model.py`
  - oracle input 拼接和 keep-prob 处理

### 7.3 初始化相关外部参考

- [Net2Net: Accelerating Learning via Knowledge Transfer](https://arxiv.org/abs/1511.05641)
  - function-preserving transformation 的代表性工作
- [Network Morphism](https://arxiv.org/abs/1603.01670)
  - 保持原函数不变地扩网络的代表性工作
- [PyTorch `torch.nn.init` 文档](https://docs.pytorch.org/docs/stable/nn.init.html)
  - `dirac_` / `xavier_*` / `kaiming_*` 的官方说明
- [Fixup Initialization: Residual Learning Without Normalization](https://openreview.net/forum?id=H1gsz30cKX)
  - 残差分支小尺度初始化的代表性思路
- [All you need is a good init (LSUV)](https://arxiv.org/abs/1511.06422)
  - 数据驱动 activation 校准思路
