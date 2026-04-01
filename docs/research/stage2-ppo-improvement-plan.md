# Stage 2 PPO 改进方案：LuckyJ/ACH 技术融合与局部后悔度设计

> 写于 2026-04-01，用于在进入 Stage 2（PPO 在线自对弈）前规划所有可引入的改进。  
> 本文档是自包含的，不依赖对话上下文。几个月后无需任何背景即可依据本文档执行修改。

---

## 一、本文档的定位

当前项目处于 Stage 0.5 / P1 监督训练阶段。Stage 2（PPO 自对弈）的入口是 `mortal/train_online.py`。本文档针对 Stage 2 的所有已知可改进点进行分析和方案设计，包括：

1. 从 LuckyJ/腾讯 AI Lab ACH 框架中可借鉴的技术
2. 从 RVR（Reward Variance Reduction）论文中可借鉴的技术
3. 项目自身提出的"局部后悔度"辅助头方案

---

## 二、背景：LuckyJ 和 ACH 是什么

### 2.1 LuckyJ

LuckyJ 是社区公认最强的日本四人麻将 AI 之一。其核心团队成员来自腾讯 AI Lab（Haobo Fu 等人）。LuckyJ 的牌风诡异、防守激进，社区普遍认为其实力超过 Suphx 和 Mortal。

### 2.2 ACH（Actor-Critic Hedge）

ACH 是 NW-CFR（Neural-based Weighted CFR）的工程实现，发表于 ICLR 2022。论文：《Actor-Critic Policy Optimization in a Large-Scale Imperfect-Information Game》（Haobo Fu et al.）。

**核心思想：**
- 传统 CFR 需要遍历博弈树维护表格化遗憾值，这对大规模博弈不可行
- Deep CFR / DREAM 用神经网络近似遗憾值，但操作的是 sampled counterfactual regret $\tilde{r}^c_k(s,a)$，方差很大
- ACH/NW-CFR 改为操作 sampled advantage $\tilde{A}^{\pi_k}(s,a)$，方差更小，因为避免了除以到达概率 $[f_p^{\mu_k}(s)]^{-1}$
- 策略更新使用 Hedge（指数加权）而非 RM+（regret matching plus）
- 用 IMPALA 风格的解耦 actor-learner 架构实现高效扩展
- 仅在 **两人零和** IIG 中有理论保证收敛到 NE

**代码仓库：** `ACH_poker/`（桌面已有）。技术栈为 C++ + TensorFlow 1.15 + Open Spiel，与本项目的 Rust + PyTorch 完全不同。

### 2.3 RVR（Reward Variance Reduction）

论文：《Speedup Training Artificial Intelligence for Mahjong via Reward Variance Reduction》（Jinqiu Li, Shuang Wu, Haobo Fu et al.）

**针对中国标准麻将四人博弈的两项改进：**
1. **Relative Value Network**：训练时输入全局信息（含对手手牌和牌山），输出四个玩家的价值。损失函数确保四人价值和为 0
2. **Expected Reward Network**：在终局阶段用网络预测期望奖励，替代高方差的实际奖励

---

## 三、当前项目 PPO 架构分析

以下是 Stage 2 入口 `mortal/train_online.py` 的关键结构（截至 2026-04-01）。

### 3.1 网络组件

| 组件 | 代码位置 | 功能 |
|------|----------|------|
| `Brain` | `model.py` L112-190 | 1D-ResNet 40 blocks × 192ch + SE attention + GroupNorm + Mish → 1024 维特征 |
| `CategoricalPolicy` | `model.py` L271-280 | Linear(1024→256) + tanh + Linear(256→46)，softmax 输出策略 |
| `AuxNet` | `model.py` L200-206 | Linear(1024→4, bias=False)，顺位预测辅助头 |
| `GRP` | `model.py` L340-380 | GRU(7, 384, 3 层) → FC(1152→24)，全局排位预测 |

### 3.2 Advantage 计算

当前 advantage 在 `dataloader.py` L153 计算：

```python
advantage = self.reward_calc.calc_delta_pt(player_id, grp_feature, rank_by_player)
```

`calc_delta_pt()`（`reward_calculator.py`）的逻辑：
1. GRP 模型对每个小局（kyoku）开始时的特征预测四人排位分布矩阵
2. 每人的期望得点 = 排位概率 × 得点向量 `[3, 1, -1, -3]`
3. 相邻小局之间的期望得点差值即为 advantage：`reward = exp_pts[k+1] - exp_pts[k]`
4. 全局 Welford's online standardization 标准化为零均值单位方差

**关键特性：这是 kyoku 级别的 advantage，不是 step 级别的。** 同一小局内所有动作共享同一个 advantage 值。

### 3.3 PPO 损失

```python
ratio = exp(new_log_prob - old_log_prob)
loss1 = ratio * advantage
loss2 = clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * advantage
min_loss = min(loss1, loss2)
clip_loss = where(advantage < 0, max(min_loss, dual_clip * advantage), min_loss)
entropy_loss = entropy * dynamic_entropy_weight
loss = -(clip_loss + entropy_loss).mean()
```

特性：
- 支持 dual-clip PPO（负 advantage 方向有额外下限）
- Suphx-style 动态熵正则化：`dynamic_entropy_weight += rate * (target - entropy.mean())`
- Old policy 每 `old_update_every` 步更新一次

### 3.4 已有的辅助头（监督阶段）

在 Stage 0.5 / Stage 1 的监督训练中，已有三类辅助头：

1. **Rank 预测**（`AuxNet`）：预测当前局面下最终四人排位的交叉熵
2. **对手状态**（`OpponentStateAuxNet`）：预测三个对手各自的向听数和是否听牌
3. **危险度**（`DangerAuxNet`）：预测手牌 37 张可打牌各自的放铳概率、放铳牌点和放哪家
   - `danger_any`：每张牌是否会放铳（二分类，BCE）
   - `danger_value`：放铳牌点大小（回归，smooth L1）
   - `danger_player_mask`：放给哪个对手（三分类 × 37，BCE）
   - 三者通过 `danger_mix_weights` 混合（默认 ~9% / ~82% / ~9%）

---

## 四、改进方案总览

按优先级从高到低排列。每个方案标注了修改文件、代码位置和精确的改法。

### P0 — 零成本稳定性改进（Stage 1 或 Stage 2 均可立即实施）

#### 4.1 Logits 阈值裁剪（Thresholded Gradient）

**来源：** ACH `deep_cfr_model.py` L364-401, NeuRD `neurd.py` L38-45

**问题：** `CategoricalPolicy` 的 softmax 虽然在概率空间做了归一化，但 logit 空间可以无界增长。当某个动作的 logit 已经很大时，正 advantage 继续推高它没有意义，反而可能造成策略过度极化和训练不稳定。

**原理：** 当 `logit[a] > thres` 且 `advantage > 0` 时，停止对该动作的梯度（不让正 advantage 继续推高已经很极端的 logit）。反方向同理。

**参考实现（ACH 的 NeuRD 风格）：**

```python
# neurd.py L38-45 的核心思想
# 正 advantage → 只在 logit < thres 时才有梯度
# 负 advantage → 只在 logit > -thres 时才有梯度
```

**在 `train_online.py` 中的修改方案：**

修改位置：`train_online.py` 的 `train_batch()` 函数，PPO loss 计算部分。

```python
# 在 config.toml [policy] 中新增（建议默认值）：
# logit_thres = 2.0

# 在 train_batch() 中：
with torch.autocast(device.type, enabled=enable_amp):
    phi = mortal(obs)
    logits = policy_net.logits(phi, masks)   # ← 拿到 raw logits
    dist = Categorical(logits=logits)
    new_log_prob = dist.log_prob(actions)
    ratio = (new_log_prob - old_log_prob).exp()

    # --- 新增：thresholded gradient mask ---
    if logit_thres > 0:
        with torch.no_grad():
            can_increase = (logits < logit_thres).float()
            can_decrease = (logits > -logit_thres).float()
            adv_sign = (advantage >= 0).float().unsqueeze(-1)
            grad_mask = adv_sign * can_increase + (1 - adv_sign) * can_decrease
        # 对 policy loss 应用 mask（只影响梯度流向 logits）
        # 具体实现：在 loss backward 之前对 logits 做 stop-gradient masking
        # 一种简单实现是将 clip_loss 乘以对应动作的 grad_mask
    # --- 结束 ---

    loss1 = ratio * advantage
    # ... 后续不变 ...
```

**更简单的替代实现（推荐先用这个）：**

```python
# 直接在 logits 上做 clamp，限制梯度回传
logits = policy_net.logits(phi, masks)
logits = logits.clamp(-logit_thres, logit_thres)  # 简单粗暴但有效
dist = Categorical(logits=logits)
```

**风险评估：** 极低。这是纯稳定性改进，不改变策略表达能力。  
**预期效果：** 防止后期训练中策略过度极化，尤其在长时间自对弈后。

---

#### 4.2 自适应熵正则化改为乘法更新

**当前实现：** `train_online.py` L370-372

```python
dynamic_entropy_weight += entropy_adjust_rate * (entropy_target - entropy.mean().item())
dynamic_entropy_weight = max(1e-4, min(1e-2, dynamic_entropy_weight))
```

**问题：** 线性加法更新在 `dynamic_entropy_weight` 接近边界时响应不对称，且不稳定。

**改进方案：** 在对数空间做乘法更新。

```python
# 初始化（放在 train() 函数的参数初始化区域）：
log_entropy_alpha = math.log(entropy_weight)

# 在 train_batch() 中替换现有的自适应更新：
if entropy_target > 0:
    log_entropy_alpha += entropy_adjust_rate * (entropy_target - entropy.mean().item())
    dynamic_entropy_weight = math.exp(log_entropy_alpha)
    dynamic_entropy_weight = max(1e-4, min(1e-2, dynamic_entropy_weight))
    log_entropy_alpha = math.log(dynamic_entropy_weight)  # 保持同步
```

**风险评估：** 极低。行为差别很小，但收敛更平滑。

---

### P1 — 中等工程量的核心改进（Stage 2 启动后首批实施）

#### 4.3 Centralized Value Function（Oracle Critic / 全局信息辅助 Critic）

**来源：** RVR 论文的 Relative Value Network + CTDE（Centralized Training Decentralized Execution）范式

**当前瓶颈：** advantage 是 kyoku 级别的（由 GRP 驱动），缺乏 step 级别的信号。同一小局内所有动作共享相同 advantage 值，无法区分同一小局内的好动作和坏动作。

**方案概述：**

在 PPO 训练时引入一个 Value Head。训练时用 oracle obs（全局信息）辅助 critic，推理/自对弈时只用普通 obs 做 policy。

**涉及修改：**

1. **`model.py` — 新增 ValueHead 类：**

```python
class ValueHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(1024, 256),
            nn.Mish(inplace=True),
            nn.Linear(256, 1),
        )
        # 初始化
        for m in self.fc.modules():
            if isinstance(m, nn.Linear):
                orthogonal_init(m)
    
    def forward(self, phi):
        return self.fc(phi).squeeze(-1)
```

2. **`train_online.py` — 加入 value head 训练：**
   - 初始化 `value_net = ValueHead().to(device)` 和 `oracle_brain = Brain(is_oracle=True, ...).to(device)`
   - 在 `train_batch()` 中：
     - 用 `oracle_brain(obs, oracle_obs)` 提取全局特征，`value_net(oracle_phi)` 预测 V(s)
     - 用 GAE(λ) 计算 step-level advantage
   - Value loss = MSE(V_pred, V_target)

3. **`dataloader.py` — 传递 step-level reward 和 oracle obs：**
   - 当前 `FileDatasetsIter` 只传 kyoku-level advantage，需要额外传 step reward 或 done 信号
   - 也需要传 `oracle_obs` 到 PPO trainer

4. **Advantage 的组合方式：**
   - GRP 的 kyoku-level advantage 作为 **baseline 偏移**（整局层面："这一局偏好还是偏差"）
   - GAE 的 step-level advantage 作为 **局内调制**（步骤层面："这一步的出牌比局内均值好还是差"）
   - 最终 advantage = GRP_delta_pt × GAE_normalized（乘法；或加法：GRP_delta_pt + α * GAE）

**Relative Value 特性（参考 RVR）：**

RVR 论文的核心 insight 是：value network 输出所有四个玩家的 value，损失函数确保和为 0。这在四人零和博弈中能显著降低方差（实验显示方差从 6.36 降至 4.44）。

如果实现，建议 ValueHead 输出 4 维（四个玩家的 value），而非 1 维。

**风险评估：** 中等。需要改 dataloader 管线并调试 GAE 参数。  
**预期效果：** 这是对强度提升最大的单项改进。从 kyoku-level 到 step-level advantage 是质的飞跃。

---

#### 4.4 Reward 方差控制

**当前实现：** `reward_calculator.py` 的 Welford's online standardization 已经做了全局标准化。

**改进方向（由易到难）：**

**4.4.1 终局平滑（简单）**

在 GRP 的 `calc_delta_pt()` 中，最后一个 kyoku 的 reward 跳变最大（从概率分布突变为 one-hot 终局结果）。可在最后一步做 temporal smoothing：

```python
# 在 calc_rank_prob() 中，最终排名改为 soft label
# 而非 one-hot [0, 0, 1, 0] → 改为 [0.05, 0.05, 0.85, 0.05]
final_ranking = torch.ones((1, 4), ...) * label_smoothing / 3
final_ranking[0, rank_by_player[player_id]] = 1.0 - label_smoothing
```

**4.4.2 Expected Reward Network（中等）**

参考 RVR 论文式 (2)。在牌局终盘训练一个小网络预测期望回报：

- 训练数据：自对弈过程中积累的 (终局前一步全局状态, 实际回报) 对
- 网络输入：四人手牌 + 副露 + 牌山余量（约 `(217, 34)` 的 oracle obs）
- 网络输出：四人期望得点
- 训练目标：MSE(predicted_reward, actual_reward)

在 PPO 训练时，终局一步的 reward 用 expected reward network 的预测值替代实际值。

**风险评估：** 低到中。终局平滑几乎零成本；expected reward network 需要额外训练管线。

---

### P2 — 低优先级 / 需谨慎

#### 4.5 Importance Sampling 截断（V-trace 风格）

**当前实现：** PPO 的 `ratio = exp(new_log_prob - old_log_prob)` 已经是 IS 修正。但当 `old_update_every` 较大或数据复用多个 epoch 时，ratio 可能很大。

**改进方案：** 加 V-trace 风格的 IS 截断。

```python
rho = ratio.clamp(max=rho_bar)  # rho_bar 通常取 1.0
```

**风险评估：** 低。  
**时机：** Stage 2 跑出第一批结果后，如果发现 ratio 方差大再加。

---

### P3 — 不建议实施

#### 4.6 CFR 完整遗憾追踪

**原因：**
- ACH 的理论保证仅限两人零和博弈，日本四人麻将不是两人零和
- ACH_poker 代码基于 Open Spiel 的两人扑克实现，Player = 0 / 1
- 四人麻将的 infoset 空间 $10^{48}$，即使用函数近似也需要额外的 regret network
- PPO + GRP 路线是更适合四人博弈的方法论

#### 4.7 Hedge 策略更新替代 Softmax

**原因：**
- Hedge 是 `policy[a] = exp(η * cumulative_regret[a]) / Z`
- 你的 softmax 是 `policy[a] = exp(logit[a]) / Z`
- 二者在形式上等价（当网络学到了正确的 cumulative advantage mapping 时）
- 不需要额外修改 `CategoricalPolicy` 的 softmax

---

## 五、局部后悔度辅助头方案（项目原创设计）

### 5.1 设计思路

传统 CFR 需要全盘搜索博弈树来计算后悔值（regret），这在日麻中不可行。但可以针对特定的局部场景，用辅助头预测"局部后悔度"——即「如果我没这么打，可能会怎样」。

当前项目已有 **防守层面的后悔度** —— `DangerAuxNet` 预测的就是「打出这张牌会不会放铳、放多大、放给谁」。这本质上是在回答：「如果我打了某张牌，事后发现放铳了，我会有多后悔？」

接下来要做的是补充 **进攻层面的后悔度**。

### 5.2 进攻后悔度的三个维度

#### 维度 1：牌效率后悔度（Tile Efficiency Regret）

**语义：** 「我打了 A 牌之后，又摸进了 A 牌周边的牌（使得 A 本来能组成面子/搭子），有多后悔？」

**形式化：**
- 输入：当前手牌状态的编码（已包含在 Brain 的 1024 维 phi 中）
- 输出：37 维向量（对应 37 种可打牌），每个值表示「打出这张牌后，下 N 巡内摸到能与之搭配的牌的期望损失」
- 监督标签的构造（从历史对局提取）：
  1. 记录实际打出的牌 $t_{\text{discard}}$
  2. 查看接下来 1-3 巡摸进的牌 $t_{\text{draw}}$
  3. 如果 $t_{\text{draw}}$ 与 $t_{\text{discard}}$ 能形成面子（顺子/刻子）或搭子，标记为正后悔
  4. 后悔值 = 该搭配可能带来的向听数改善（0/1/2）

**实现方案：**

```python
class TileEfficiencyRegretHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Linear(1024, 37, bias=False)
    
    def forward(self, phi):
        """输出每张可打牌的牌效率后悔度预测（回归值，0-2）"""
        return self.net(phi).sigmoid() * 2.0  # 归到 [0, 2]
```

标签构造需要在 Rust 端 `gameplay.rs` 中新增逻辑：回溯每一步打牌后 1-3 巡的摸牌，计算是否与弃牌相邻。

**训练损失：** Smooth L1 loss，仅在有后续摸牌记录的步骤上计算。

---

#### 维度 2：立直改良后悔度（Riichi Improvement Regret）

**语义：** 「我选择了立直（riichi），之后摸进了本来可以改良听牌的牌，有多后悔？」

**背景：** 日麻中，立直后手牌锁定，不能换听牌。如果立直后摸进改良牌（能从差听变好听的牌），这说明延迟立直（默听再改良后再立直）可能更好。

**形式化：**
- 仅在 action = riichi 的步骤触发
- 输出：标量，表示「此时立直 vs 不立直的期望改良损失」
- 监督标签的构造：
  1. 当玩家选择 riichi 时，记录当前听牌集合 $W_{\text{now}}$
  2. 模拟不立直的情况下，检查后续摸到的牌是否能改善听牌集合
  3. 后悔值 = 改良后的等效听牌枚数提升（0 表示没有改良机会）

**实现方案：**

```python
class RiichiRegretHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1024, 64),
            nn.Mish(inplace=True),
            nn.Linear(64, 1),
        )
    
    def forward(self, phi):
        """输出立直决策的改良后悔度（标量，>= 0）"""
        return self.net(phi).squeeze(-1).sigmoid() * 10.0  # 归到 [0, 10] 枚
```

标签构造较复杂，需在 Rust 端：
1. 在 riichi 动作时，计算当前听牌数
2. 向后扫描 3-5 巡的摸牌
3. 对每张摸牌模拟「不立直、改打该牌」后的新听牌数
4. 后悔值 = max(0, best_simulated_waiting_count - current_waiting_count)

**训练损失：** MSE loss，仅在 riichi 动作上计算。样本量较少，需要较高的 loss weight。

---

#### 维度 3：默听后悔度（Damaten Regret）

**语义：** 「我选择了默听（不立直但已听牌），之后别家放铳给我 / 我自摸了，这个选择有多好或多坏？」以及反面：「我默听了但最终没有和牌，别人先和了，有多后悔？」

**背景：** 默听是日麻中非常重要的策略选择。默听不宣言立直，可以换听但不享受立直的加番。选择默听的后悔来自两方面：
- **默听成功**（正面）：别家不知道你听牌，放铳给你 → 后悔值为负（即这是好决策）
- **默听失败**（负面）：别家先和 / 流局 / 你没摸到和牌 → 后悔值为正（此时立直可能更好，因为有一发/里宝牌/立直棒收益）

**形式化：**
- 仅在听牌但未立直（选择 pass riichi 或在具有立直机会时选择打牌）的步骤触发
- 输出：标量，表示「此时默听 vs 立直的期望得点差异」
- 监督标签的构造：
  1. 当玩家处于听牌状态但选择不立直时
  2. 记录最终结果：和牌方式（自摸/荣和/流局/被荣和）和得点
  3. 后悔值 = 估算的「如果选择立直可能的得点差」
     - 如果是荣和：立直能额外获得立直棒 + 可能的里宝牌 + 一发收益，后悔值 = 这些额外收益的期望值
     - 如果是自摸：同上
     - 如果是流局/被荣和：立直会暴露意图 + 损失 1000 点立直棒，后悔值为负（默听更好）

**实现方案：**

```python
class DamatenRegretHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1024, 64),
            nn.Mish(inplace=True),
            nn.Linear(64, 1),
        )
    
    def forward(self, phi):
        """输出默听 vs 立直的得点差预测（正 = 立直更好，负 = 默听更好）"""
        return self.net(phi).squeeze(-1)
```

标签构造需在 Rust 端做复杂的终局回溯分析。由于涉及里宝牌等随机因素，建议使用 expected value（多次采样取平均）。

**训练损失：** Smooth L1 loss，仅在听牌未立直的步骤上计算。

---

#### 维度 4：副露后悔度（Furo/Meld Call Regret）

**语义：** 「我选择了吃/碰/明杠，之后副露破坏了手牌结构、暴露了意图、降低了打点，有多后悔？」以及反面：「我放弃了吃/碰的机会，之后发现那个搭子再也组不成了，有多后悔？」

**背景：** 副露（鸣き）是日麻中与立直同等重要的战略决策。副露的不可逆性非常强：
- 一旦吃/碰，手牌从门前变为副露状态，失去立直资格（门前清限定役全部消失）
- 碰/杠会改变场上的巡目顺序，影响其他玩家
- 副露暴露手牌信息，对手可以据此推理你的待牌
- 但副露也能加速和牌

目前 `DangerAuxNet` 只覆盖了「打牌」动作的防守后悔，而 action space 中的 `chi_low/mid/high`(38-40)、`pon`(41)、`kan`(42)、`pass`(45) 这些副露相关动作完全没有后悔度信号。

**形式化：**
- **触发条件**：当 `can_chi()` 或 `can_pon` 或 `can_daiminkan` 为 true 时
- **输出**：2 维向量 `[call_regret, pass_regret]`
  - `call_regret`：如果选择副露，最终结果相比全局期望差了多少
  - `pass_regret`：如果选择不副露，最终结果相比全局期望差了多少
- **监督标签的构造**：
  1. 在有副露选择权的步骤，记录玩家实际选择（call 或 pass）
  2. 记录该局最终结果（得点变化）
  3. **call_regret 标签**（仅当玩家实际选择了副露时有效）：
     - 如果最终该局结果为负（被荣和/被自摸/流局），且副露后向听数未减少 → 高后悔
     - 如果最终该局结果为正且和牌 → 低后悔（好决策）
     - 后悔值 = -delta_pt_this_kyoku（负得点变化 = 正后悔）
  4. **pass_regret 标签**（仅当玩家实际选择了 pass 时有效）：
     - 如果后续 3 巡内再也没有机会组成该搭子 → 高后悔
     - 如果后续成功门前听牌 → 低后悔（好决策）
     - 后悔值 = 基于向听数变化的量化指标

**实现方案：**

```python
class FuroRegretHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1024, 64),
            nn.Mish(inplace=True),
            nn.Linear(64, 2),  # [call_regret, pass_regret]
        )

    def forward(self, phi):
        return self.net(phi)
```

**Rust 端标签构造**：在 `extend_from_event_window()` 中，当检测到 `can_chi() || can_pon || can_daiminkan` 时，记录快照。在局结束时回溯计算标签。

**Gameplay struct 新增字段**：
```rust
pub furo_regret: Vec<f32>,        // 2 维 [call, pass]，每个副露决策点
pub furo_regret_valid: Vec<bool>, // 是否在有副露选择的步骤
```

**训练损失：** Smooth L1 loss。样本量中等（每局约 3-8 次副露机会）。

**这是文档更新时新增的维度，优先级高**——副露是日麻中与打牌同等频繁的决策类型，没有后悔度覆盖是一个明显的缺口。

---

#### 维度 5：见逃后悔度（Pass-Win Regret / 放弃和牌后悔度）

**语义：** 「我本可以荣和（ron）/ 自摸（tsumo），但选择了放弃（pass-hu），结果这局我没和成，有多后悔？」

**背景：** 在日麻中，玩家有时会主动放弃和牌机会：
- 和牌打点太低（例如只值 1000 点的荣和），为了追求更大的手牌选择见逃
- 已经有人立直，自己做好了防守准备，选择放弃一个低打点的和牌继续防守
- 故意见逃后进入振听状态（不能再荣和同一张牌），但可以继续追求自摸和更大的手牌
- 这在 action space 中对应 `can_ron_agari = true` 但实际动作为 `pass`(45)，或 `can_tsumo_agari = true` 但实际选择不和

见逃的后果可能非常极端：放弃了稳定的小和牌 → 之后被别家大和，损失远超放弃的小和牌。

**形式化：**
- **触发条件**：`can_ron_agari = true` 或 `can_tsumo_agari = true`，但玩家选择了 pass
- **输出**：标量，表示「放弃这次和牌机会之后的期望得点损失」
- **监督标签的构造**：
  1. 在见逃发生时，记录当前可和的打点 $P_{\text{pass}}$
  2. 记录该局最终实际得点 $P_{\text{actual}}$
  3. 后悔值 = $P_{\text{pass}} - P_{\text{actual}}$（如果放弃了 3000 点但最终被荣和 8000 点，后悔值 = 3000 - (-8000) = 11000）
  4. 如果最终和了更大的牌，后悔值可以为负（说明见逃是**好决策**）

**实现方案：**

```python
class PassWinRegretHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1024, 64),
            nn.Mish(inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, phi):
        """正值 = 见逃后悔（应该和的），负值 = 见逃正确（和了更大的）"""
        return self.net(phi).squeeze(-1)
```

**Rust 端**：在 `extend_from_event_window()` 中已有 `can_ron_agari` / `can_tsumo_agari` 的检测，见逃对应 `label = 45`（pass）。需要额外记录当时的 `ron_point_on_tile()` 值，以及局结束时的最终得点。

**Gameplay struct 新增字段**：
```rust
pub pass_win_regret: Vec<f32>,
pub pass_win_regret_valid: Vec<bool>,
```

**训练损失：** Smooth L1 loss。样本非常稀疏（每百局可能只有几次见逃），需要高 loss weight。

**这个维度的独特价值**：这是唯一一个直接将「放弃的具体得点」和「最终实际得点」联系起来的后悔信号。它能教会模型量化「鸟在手 vs 林中二鸟」的权衡。

---

#### 维度 6：押引后悔度（Push/Fold Stance Regret）

**语义：** 「在有人立直或多家听牌的危险局面下，我继续进攻（push），结果放铳了，有多后悔？」或者「我选择了全面防守（fold/オリ），但其实我很接近和牌，有多后悔？」

**背景：** 与 `DangerAuxNet` 的区别：
- `DangerAuxNet` 预测的是**单张牌**的放铳概率（微观：「打这张 3m 会不会放铳」）
- 押引后悔度关注的是**宏观姿态**（「在当前这个局面，我整体应该进攻还是防守」）
- 一个典型场景：有人立直了，我 2 向听，继续进攻摸好牌的概率 vs 在进攻过程中放铳的概率。这不是单张牌的问题，而是「接下来 5-10 巡的战略方向」的问题

**形式化：**
- **触发条件**：当场上存在至少一个对手的听牌信号时（如有人立直、有人明显副露进攻）。具体可用 `opponent_tenpai`（已有的辅助头预测值）或明确的 `riichi_accepted` 信号
- **输出**：标量，表示「进攻 vs 防守的期望得点差」
  - 正值 = 应该进攻（fold 是后悔的），负值 = 应该防守（push 是后悔的）
- **监督标签的构造**：
  1. 在触发条件满足的步骤，定义「push」为：该步打出的牌使向听数不变或减少，「fold」为：该步打出的牌使向听数增加或打出安全牌
  2. 记录该局的实际结果
  3. **push 时的后悔**：如果 push 后放铳 → 后悔值 = 放铳点数；如果 push 成功和牌 → 后悔值为负（好决策）
  4. **fold 时的后悔**：如果 fold 后安全度过危机并和牌 → 后悔值为负（好决策）；如果 fold 后虽然没放铳但被自摸 → 后悔值 = 被自摸点数（本来 push 可能赢的）

**实现方案：**

```python
class PushFoldRegretHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1024, 64),
            nn.Mish(inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, phi):
        """正值 = 应该 push，负值 = 应该 fold"""
        return self.net(phi).squeeze(-1)
```

**Rust 端**：触发条件可用 `opponent_states[i].riichi_accepted[0]` 或 `opponent_states[i].shanten() == 0` 来判断场上有对手听牌。push/fold 的分类可以基于 `keep_shanten_discards` / `next_shanten_discards` 来判断打出的牌是否在进攻。

**Gameplay struct 新增字段**：
```rust
pub push_fold_regret: Vec<f32>,
pub push_fold_regret_valid: Vec<bool>,
```

**训练损失：** Smooth L1 loss。样本量适中（有人立直后的所有巡目都触发）。

**这个维度与 danger 的关系**：danger 是微观（单张牌的放铳概率），push/fold 是宏观（整体攻防姿态的选择）。二者互补而非重叠。danger 帮助回答「打哪张安全」，push/fold 帮助回答「到底该不该继续打」。

---

#### 维度 7：杠后悔度（Kan Regret）

**语义：** 「我选择了开杠（暗杠/加杠/大明杠），结果翻出的新宝牌指示牌给了对手大量宝牌 / 岭上摸牌没有用 / 槍杠被和，有多后悔？」

**背景：** 杠是日麻中风险最高的副露操作之一：
- **暗杠**：不破坏门前清但翻新宝牌指示牌（可能给对手加番）
- **加杠**：在已有碰的刻子上加第四张。有被**槍杠**（别家国士无双荣和）的风险
- **大明杠**：从别家弃牌杠。翻新宝牌 + 破坏门前清 + 暴露手牌
- 正面：杠后进行岭上摸牌，有不错的和牌概率（岭上开花）
- 所有杠都会翻新宝牌指示牌，这是一个高度随机的事件，新宝牌可能极大改变其他玩家手牌的价值

**形式化：**
- **触发条件**：`can_ankan || can_kakan || can_daiminkan` 为 true
- **输出**：2 维 `[kan_regret, pass_regret]`
  - `kan_regret`：选择杠后的期望得点损失（考虑新宝牌是否利好对手）
  - `pass_regret`：放弃杠后的期望得点损失（考虑失去的岭上摸牌 + 杠宝牌收益）
- **监督标签的构造**：
  1. **如果选择了杠**：记录新翻的宝牌指示牌 → 统计对手手牌中该宝牌的数量 → 如果对手获得大量宝牌加点且最终和牌，后悔值高
  2. **如果放弃了杠**：记录假设杠了会翻出什么宝牌（需要牌山信息，oracle 模式下可用） → 评估错失收益

**实现方案**：

```python
class KanRegretHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1024, 64),
            nn.Mish(inplace=True),
            nn.Linear(64, 2),  # [kan_regret, pass_regret]
        )

    def forward(self, phi):
        return self.net(phi)
```

**Rust 端**：需要在 oracle 模式下访问牌山信息。牌山 idx 已由 `LoaderContext` 的 `yama_idx` / `rinshan_idx` 追踪。新宝牌指示牌的影响需要回溯到局结束时各家的宝牌数变化。

**Gameplay struct 新增字段**：
```rust
pub kan_regret: Vec<f32>,        // 2 维 [kan, pass]
pub kan_regret_valid: Vec<bool>,
```

**训练损失：** Smooth L1 loss。样本较稀疏（每局 0-2 次杠的机会）。

---

### 5.3 后悔度维度总览与覆盖分析

| # | 维度 | 决策类型 | 攻/守 | 触发频率 | 已有覆盖 | 信号独特性 |
|---|------|---------|-------|---------|---------|----------|
| 0 | **危险度**（已有） | 打牌 | 守 | 每巡 | `DangerAuxNet` ✅ | 单张放铳概率 |
| 1 | **牌效率后悔度** | 打牌 | 攻 | 每巡 | ❌ | 弃牌与后续摸牌的搭配损失 |
| 2 | **立直改良后悔度** | 立直 | 攻 | 每局 0-1 | ❌ | 立直后的改良损失 |
| 3 | **默听后悔度** | 立直/默听 | 攻守 | 每局 0-3 | ❌ | 立直 vs 默听的得点差 |
| 4 | **副露后悔度** | 吃/碰/杠 or pass | 攻 | 每局 3-8 | ❌ | 副露决策的长期影响 |
| 5 | **见逃后悔度** | 和牌 or pass | 攻 | 每百局几次 | ❌ | 放弃得点 vs 实际结果 |
| 6 | **押引后悔度** | 宏观攻防 | 攻守 | 有人立直后每巡 | 部分（danger 覆盖微观面） | 整体攻防姿态的选择 |
| 7 | **杠后悔度** | 杠 or pass | 攻守 | 每局 0-2 | ❌ | 宝牌变化 + 岭上摸牌 |

**关键观察：**
- 维度 0（danger）+ 维度 6（push/fold）构成**完整的防守后悔链条**：先判断「该不该防」（push/fold），再判断「防的话打哪张」（danger）
- 维度 1（牌效率）+ 维度 4（副露）构成**完整的进攻后悔链条**：先判断「要不要鸣牌加速」（furo），再判断「自己摸牌打什么」（tile eff）
- 维度 2（立直改良）+ 维度 3（默听）+ 维度 5（见逃）覆盖了**和牌路径上的关键分叉点**
- 维度 7（杠）是一个**独立的高方差决策点**，因为杠会引入新的随机变量（宝牌）

**效果估计更新：** 加上新增的 4 个维度后，局部后悔度覆盖的日麻决策信号大约从 60-75% 提升到 **85-90%**。未覆盖的 10-15% 主要是多步序列后悔（连续 2-3 巡的打牌路线选择）和对手模型相关后悔（如果我能预测到下家的手牌我就不会打这张），这些在 step-level value head（改进 4.3）和 `OpponentStateAuxNet` 中会自然捕获。

### 5.4 实施优先级排序

所有 7 个后悔度维度（含已有 danger）的实施建议：

| 优先级 | 维度 | 理由 |
|--------|------|------|
| **P0** | 1. 牌效率后悔度 | 每巡触发、样本最多、信号最密集 |
| **P0** | 4. 副露后悔度 | 重大决策缺口、副露是未覆盖的最大类别 |
| **P1** | 6. 押引后悔度 | 与 danger 互补形成完整防守链 |
| **P1** | 3. 默听后悔度 | 日麻特色策略、高杠杆 |
| **P2** | 2. 立直改良后悔度 | 有价值但样本少 |
| **P2** | 5. 见逃后悔度 | 极端稀疏但单样本信号极强 |
| **P2** | 7. 杠后悔度 | 需要 oracle 牌山信息、标签构造复杂 |

---

### 5.6 局部后悔度在 PPO 中的使用方式

这些后悔度辅助头有两种使用方式：

**方式 A：纯辅助头（Stage 0.5 / Stage 1）**

与 `DangerAuxNet` 一样，作为监督训练的辅助 loss。其作用是让 Brain encoder 学到更丰富的表示（知道哪些牌打出去后会后悔）。

```python
total_loss = policy_loss + rank_aux_loss + opp_aux_loss + danger_aux_loss
           + tile_eff_regret_loss + riichi_regret_loss + damaten_regret_loss
           + furo_regret_loss + pass_win_regret_loss + push_fold_regret_loss
           + kan_regret_loss
```

**方式 B：作为 PPO advantage 的辅助信号（Stage 2）**

在 PPO 训练中，将后悔度预测作为 advantage 的补充信号：

```python
# 在 train_batch() 中：
with torch.no_grad():
    tile_regret = tile_eff_regret_head(phi)  # (batch, 37)
    # 取实际打出牌的后悔度作为 penalty
    played_regret = tile_regret.gather(1, actions_37.unsqueeze(1)).squeeze(1)

# 将后悔度作为 advantage 的负向修正
adjusted_advantage = advantage - regret_penalty_weight * played_regret
```

**推荐路径：** 先做方式 A（在 Stage 0.5 / Stage 1 中验证辅助头的预测质量），确认预测准确后再在 Stage 2 中做方式 B。注意：方式 B 中不同后悔度维度的 penalty weight 需要单独调参，建议先只用维度 1（牌效率）做 POC。

---

### 5.7 局部后悔度 vs 完整 CFR 的对比

| 维度 | 完整 CFR | 局部后悔度（本方案） |
|------|---------|-------------------|
| 搜索范围 | 全博弈树（不可行） | 针对 7 个关键决策类别 |
| 理论保证 | 两人零和 → NE | 无（是辅助训练信号） |
| 计算成本 | 极高 | 几乎为零（额外 linear head） |
| 标签构造 | 需要遍历 | 从历史对局回溯（离线） |
| 适用博弈 | 两人零和 | 任意，包括四人麻将 |
| 信息级别 | 全局最优 | 局部次优但实用 |
| 决策覆盖率 | 100%（理论上） | ~85-90%（覆盖所有主要决策类型） |

**结论：** 局部后悔度不是完整 CFR 的替代品，而是一种面向四人麻将实际场景的工程化近似。它回答的不是「全局最优策略是什么」，而是「这几个关键决策点，事后来看有多后悔」。7 个后悔度维度（含已有 danger）覆盖了日麻中的所有主要决策类型：打牌（维度 0+1）、副露（维度 4）、杠（维度 7）、立直/默听（维度 2+3）、和牌/见逃（维度 5）、攻防姿态（维度 6）。

---

### 5.8 标签构造的 Rust 端实现要点

所有后悔度标签都需要在 `libriichi/src/dataset/gameplay.rs` 中构造，与 danger 标签的模式相同。

**新增字段（在 `Gameplay` struct 中）：**

```rust
// 维度 1: 牌效率后悔度
pub tile_eff_regret: Vec<f32>,        // 37 维，每步每张牌的后悔值
pub tile_eff_regret_valid: Vec<bool>, // 是否有有效标签

// 维度 2: 立直改良后悔度
pub riichi_regret: Vec<f32>,          // 标量，仅在 riichi 动作时有效
pub riichi_regret_valid: Vec<bool>,

// 维度 3: 默听后悔度
pub damaten_regret: Vec<f32>,         // 标量，仅在听牌未立直时有效
pub damaten_regret_valid: Vec<bool>,

// 维度 4: 副露后悔度
pub furo_regret: Vec<f32>,            // 2 维 [call, pass]
pub furo_regret_valid: Vec<bool>,

// 维度 5: 见逃后悔度
pub pass_win_regret: Vec<f32>,        // 标量
pub pass_win_regret_valid: Vec<bool>,

// 维度 6: 押引后悔度
pub push_fold_regret: Vec<f32>,       // 标量
pub push_fold_regret_valid: Vec<bool>,

// 维度 7: 杠后悔度
pub kan_regret: Vec<f32>,             // 2 维 [kan, pass]
pub kan_regret_valid: Vec<bool>,
```

**构造时机：** 在 `LoaderContext::load_game()` 中，在解析完整个对局后，回溯计算各步的后悔度标签。这需要在正向解析时记录关键状态快照。

---

## 六、实施时间线

| 阶段 | 改进项 | 前置条件 |
|------|--------|---------|
| **Stage 0.5 / 1（现在就可以做）** | 4.1 Logits 阈值裁剪 | 无 |
| **Stage 0.5 / 1（现在就可以做）** | 4.2 熵正则化乘法更新 | 无 |
| **Stage 0.5 / 1（规划中）** | 5.x 局部后悔度辅助头（方式 A） | 需 Rust 端标签构造 |
| **Stage 2 启动时首批** | 4.3 Centralized Value Function | 需改 dataloader |
| **Stage 2 启动时首批** | 4.4.1 终局平滑 | 无 |
| **Stage 2 稳定后** | 4.4.2 Expected Reward Network | 需额外训练管线 |
| **Stage 2 稳定后** | 4.5 V-trace IS 截断 | 观察 ratio 方差后决定 |
| **Stage 2 稳定后** | 5.x 局部后悔度（方式 B，作为 PPO 信号） | 方式 A 验证质量后 |

---

## 七、关键结论

1. **不要移植 ACH 的完整 CFR/Hedge 框架到四人麻将**。理论不支持（仅限两人零和），工程量大，收益不确定。

2. **LuckyJ 的"秘方"大概率不是 ACH 本身**，而是：高质量 value estimation（全局信息辅助）+ 大规模自对弈 + 奖励方差控制 + 可能的对手建模。

3. **当前架构方向是正确的**。Mortal 基底 + Suphx Oracle Dropout + PPO 自对弈是社区验证过的有效路径。

4. **最高优先级的改进**是在 Stage 2 的 PPO 训练中加入 centralized value function（用全局信息训练 critic）+ step-level GAE，这与 RVR 的核心思想一致。

5. **局部后悔度是本项目的原创方向**。它不替代 CFR，而是用辅助头的方式让 encoder 学到"事后来看哪些决策会后悔"的表示。这是低成本、高信息量的改进，尤其适合日麻这种有明确可枚举后悔场景的博弈。

---

## 附录 A：相关文件索引

| 文件 | 用途 |
|------|------|
| `mortal/train_online.py` | Stage 2 PPO 训练入口，所有 P0/P1 改进在此实施 |
| `mortal/model.py` | 网络定义，新 head 在此添加 |
| `mortal/reward_calculator.py` | GRP 驱动的 advantage 计算，终局平滑在此改 |
| `mortal/dataloader.py` | 数据管线，GAE 需要改此文件传递 step reward |
| `mortal/train_supervised.py` | 监督训练入口，后悔度辅助头方式 A 在此实施 |
| `libriichi/src/dataset/gameplay.rs` | Rust 端标签构造，后悔度标签在此实现 |
| `libriichi/src/state/obs_repr.rs` | 特征编码，如需新 obs 通道在此改 |
| `ACH_poker/algorithms/deep_cfr_model.py` | ACH 参考：thresholded gradient 实现 |
| `ACH_poker/third_party/open_spiel/python/algorithms/neurd.py` | NeuRD 参考：阈值裁剪原始思路 |

## 附录 B：相关论文

1. **ACH**：Haobo Fu et al., "Actor-Critic Policy Optimization in a Large-Scale Imperfect-Information Game", ICLR 2022
2. **RVR**：Jinqiu Li, Shuang Wu, Haobo Fu et al., "Speedup Training Artificial Intelligence for Mahjong via Reward Variance Reduction"
3. **Suphx**：Junjie Li et al., "Suphx: Mastering Mahjong with Deep Reinforcement Learning", arXiv 2020
4. **IMPALA**：Lasse Espeholt et al., "IMPALA: Scalable Distributed Deep-RL with Importance Weighted Actor-Learner Architectures", ICML 2018
5. **PPO**：John Schulman et al., "Proximal Policy Optimization Algorithms", arXiv 2017
