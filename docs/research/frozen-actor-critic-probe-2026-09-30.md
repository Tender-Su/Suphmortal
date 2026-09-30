# 冻结 actor 的 fixed-imputed critic pilot

## 要回答的问题

用固定 C50k / Bbest actor 的新采样分布比较已有 warm0、warm40k、clean40k。三者评分完全相同的完整 trainee 决策轨迹，保留 all_players 相对座位头与 score_rank return-to-go。此阶段不训练，不创建 optimizer，不更新 checkpoint，不接入 PPO；256 局是诊断 pilot，不能证明牌力或 RL readiness。clean40k − warm0 是候选差异，不是 clean 初始化臂的净训练收益。

## 运行接口

从已固定 commit 的 repo 根运行（Windows Python 路径按 runner 环境配置）。专用最小 config 不包含旧训练起点/对手/optimizer/data 路径：

```text
python -m mortal.research.frozen_actor_critic_probe \
  --config mortal/research/frozen_actor_critic_probe.toml \
  --actor <C50k-or-identical-Bbest.pth> --opponent <canonical.pth> \
  --warm0 <warm-step0.pth> --warm40k <warm-40000.pth> --clean40k <clean-40000.pth> \
  --output-dir <never-existing-run-dir> --games 8 \
  --seed-start <reserved-smoke-start> --seed-key <reserved-key> --sampling-seed <explicit-seed> \
  --imputation-seed 20260905 --device cuda --batch-size 32 --torch-threads 1 --rayon-threads 4
```

小 smoke 成功后另建新目录、使用不重叠 seed 范围，`--games 256` 得到64个四座 seed groups。smoke 不能并入正式 pilot。输出目录即使为空也必须不存在；不支持 resume。外部独立 deadline supervisor 负责停止边界，不允许绕过监督入口。

actor 按 production TrainPlayer 的 categorical 分布采样，explore_rate=1，search/guiding off；canonical 为 explore_rate=0、agari guard=true 的训练式对手。trainee agari guard=false，quick-eval 采用 production 默认。该分布与旧 formal 1v3 的 guard 设置不同，旧1v3分数不是这次实验的配对基线。actor 权重自身旧配置中的 pt 不定义本次 reward；本次固定 `[2,1,0,-3]`、gamma=1，并严格验证三 critic 的标签协议，分别使用它们自己的 fusion/head 结构，strict state_dict 加载；角色内部 steps 必须为0/40000/40000，ScheduleFree checkpoint 不得处于 train_mode。

## Oracle 输入含义

本阶段名称为 **FIXED-IMPUTED Oracle**。native GameplayLoader 使用 `trust_seed=false`；日志里实际记录的起手隐藏牌和已抽到的牌，与尚未观测到的牌山部分是不同信息。后者用 seed 20260905 的确定性补全，不宣称重建真实完整牌山。每个游戏一次解码结果同时交给三 critic，首场额外 A/A 解码验证确定性。true Oracle 重建一致性暂缓验证，不阻塞 like-for-like fixed-imputed 对照。

`FileDatasetsIter.iter_game_trajectories(..., oracle_imputation_seed=None)` 新参数仅作用于完整轨迹 API；默认不调用 seed setter、不改变线上输入协议。请求固定 seed 而 native 扩展不支持时直接失败，不退回随机补全。

## 产物与统计

- `request.json`、`effective_config.json`、`provenance.json` 记录显式输入、实际覆盖、各 checkpoint SHA256/steps/契约、源码 commit/status/关键源码 hash、native 版本路径、开始结束时间；输入文件在前后重算 hash
- `outcomes.json` 为完整游戏日志、完整 seed/key、trainee 座位、最终排名及 hash；每组必须恰好四座，日志必须 end_game、局数完整
- `predictions.jsonl.gz` 每状态保留 game/seed/key/seat/index、当前 rank/all_last、四头 target 与三个 critic predictions，无大型观测 feature cache
- `metrics.json` 包含全状态等权 p0/all_players MSE、bias、constant-zero 基线；按预测值分箱的 p0 校准；当前 rank（0=第一）×all_last 输入分层；production GAE lambda=1 完整 return 恒等式误差与 lambda=.95 p0 advantage 分布
- 三组候选差异及各 critic − constant-zero 使用配对四座 seed groups 的 cluster bootstrap，报告 count、SE、95% interval。估计量为各 seed group 内状态等权 MSE，再对独立组等权；与全样本状态等权 MSE 明确分列。CI 不包含训练 seed 变异或候选筛选校正
- 核心结果原子写入；失败留下 failure.json，未完整写完的预测文件不对外标成完成。没有按已实现 outcome 分组的硬护栏，也没有任意 readiness 阈值

## 验证边界

轻量测试 `python -m unittest mortal.tests.test_frozen_actor_critic_probe mortal.tests.test_value_coordinate_contracts` 覆盖路径保护、参数校验、标签契约、分箱/误差算术、完整日志检查、可选 seed API 与 production reward/GAE 算术。无 torch 的云端仅运行这些测试及语法检查；真实 torch/native 加载、arena、显存与固定补全 A/A 必须在 runner 的 smoke 确认，不将轻量测试称为集成通过。
