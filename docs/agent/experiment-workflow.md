# 当前实验流程

这份文档只讲“现在怎么跑”，不复述所有历史原因。

## 总体原则

- 每个阶段结束后停下来，让人确认方向
- 不使用一键跑到底的默认工作流
- 自动摘要只记 run 产物，不替代人工判断
- 新 run 必须使用新输出目录，不复用旧目录判断新实验

## 双机并行运行纪律

- 台式机与笔记本默认各跑各的完整实验，不共享同一个运行目录
- 两台机器同时跑同一 stage 时，run 名称必须带机器标记，例如 `desktop_*` / `laptop_*`
- 不要让两台机器共写同一路径下的 `state_file`、`best_*_state_file`、TensorBoard 目录或日志目录
- 如果实验依赖最新源码，先同步 `origin/main`，再启动笔记本 run；不要默认相信笔记本上的旧工作树
- 笔记本当前更适合承担并行 A/B、loader benchmark、validation benchmark、短 probe，而不是和台式机拼成单次分布式训练

## Stage 0.5 的当前流程

1. `P0`
   - 目标：从监督协议里选出值得进入 `P1` 的 `top3`
   - 当前已冻结，无需重跑才能继续后续主线
2. `P1 calibration`
   - 目标：把三头预算映射到统一量纲，并估计组合耦合
   - 当前默认是瘦版 `A2y-only + combo-only`
   - 这里的 `combo_only` 明确表示沿用 `2026-03-25` 那轮旧 `single-head cali` 数值，只补当前 run 的组合因子
   - 这一步不能直接宣布 winner
3. `P1 protocol_decide`
   - 目标：在三头同时开启的统一脚手架下，尽早选出协议 winner
   - 当前运行就停在这里
4. `P1 winner_refine`
   - 目标：只在 winner 协议内部细调三头全开配比
   - 需要人工确认后再启动
5. `P1 ablation`
   - 目标：验证 `all_three / drop_* / ce_only` 的边际贡献
   - 也需要人工确认后再启动

## 当前人工停点

- `protocol_decide` 结束后必须停
- 不自动进入 `winner_refine`
- `winner_refine` 结束后再次停
- `ablation` 结束后再决定是否写入更下游默认

## 当前运行纪律

- 训练入口只用 `scripts/` 下现行入口
- 默认不设置 `MORTAL_CPU_AFFINITY`
- 训练快路径默认：
  - train：`4 / 10 / 3`
  - val：`8 / 5`
- 若 run 明确在笔记本节点上跑，当前优先用：
  - train：`6 / 7 / 3`
  - val：`7 / 6`
  - 若 `7 / 6` 出现仅限笔记本的偶发抖动，可先回退到非常接近的 `7 / 5`
- 如果验证出问题，优先修共享映射生命周期，不降级成安全慢路径

## P1 结果应该怎么解释

- 是否入围、谁是 winner、谁只是诊断信息，统一按 `docs/status/p1-selection-canonical.md`
- `calibration` 只能解释预算映射和组合耦合
- `protocol_decide` 才回答“哪个协议该继续往下走”
- `winner_refine` 才回答“winner 协议下三头全开应该怎么配”
- `ablation` 才回答“三个头是否都还有边际贡献”

## 不再使用的旧流程

- 旧 `solo -> pairwise -> joint refine` 仍保留诊断价值
- 但它们已经不是当前默认主线，不应用来覆盖新流程的 winner 判断

## 推荐配套文档

- 当前默认与冻结结论：`docs/agent/mainline.md`
- 当前真实状态：`docs/status/stage05-verified-status.md`
- 工程方法与排障经验：`docs/research/stage05/engineering-playbook.md`
