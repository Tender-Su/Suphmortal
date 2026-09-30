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

### 分阶段耗时

`timing.json` 为可选、附加的单调 wall-clock 报告，`provenance.json.timing_report` 指向它；完成 stdout 也附带相同 `timing` 字段。原有 schema、指标和 reuse 核验不变。报告在核心产物完整发布后才写入，不进入 provenance 的 artifact hash 列表；在这个间隙中断时，报告可能缺失，不能据此否定已经完整核验的核心产物。

`total_seconds` 从写 `request.json` 前开始，到最终输入/产物 hash 检查和 metrics/provenance 发布结束。不含解释器启动、CLI 解析、输出目录预留、计时报告自身落盘和最终 stdout，不能直接视为外部 supervisor 的进程总耗时。`phase_seconds` 是互斥区间累计，不嵌套、不重复计数，完成时总和等于 total（允许浮点舍入误差）：

- `preflight_input_fingerprint_model_load`：导入、输入/源码/native 指纹、模型 CPU 加载、契约验证与初始配置/provenance 写入
- `arena_generation_validation`：新 arena 的初始化、actor/opponent 设备放置、完整对局、日志检查及 outcomes/provenance 写入；并非仅纯打牌时间
- `rollout_reuse_validation`：只读原 rollout 核验和新 outcomes/provenance 写入；复用时只有这一项，不伪造 arena 为 0 秒
- `scoring_setup`：actor 转回 CPU、critic 设备放置及评分容器/输出流初始化
- `replay_decode_validation`：轨迹解码、首场固定补全 A/A、reward/target 构建、形状/有限值检查，以及可选隐藏置换映射
- `critic_inference`：三个 critic 的全部 normal/可选 shuffled forward、输入设备传输及已有的 CPU 结果取回
- `aggregation_statistics`：逐游戏指标与 GAE 核验、汇总、分层、bootstrap 及进度输出
- `prediction_write`：逐状态 JSON/gzip、最终关闭与原子发布
- `final_integrity_write`：末尾输入/reuse 完整性复核、metrics 写入、产物 hash 与完整 provenance 发布

arena 区间包含 TrainPlayer 构造、其内部设备准备、采样种子设置及完整性检查，不能直接当作稳态纯对局吞吐。

CUDA 仅在新 arena 完成、critic 设备放置完成这两个粗边界同步（reuse 只有后者）；forward 时间依赖原有阻塞 `.cpu()` 已完成结果，不新增逐 batch 同步，也不将 CPU 提交耗时称为 GPU kernel 时间。核心流程中途失败时 `failure.json.timing` 标记 `incomplete`，只把已结束区间放进 `phase_seconds`；当前区间单独记为 `incomplete_phase`，不把失败阶段或未运行阶段标成完成；若仅后续计时报告发布失败，已完成的核心区间仍如实保留 `complete`。

## 验证边界

轻量测试 `python -m unittest mortal.tests.test_frozen_actor_critic_probe mortal.tests.test_frozen_actor_critic_probe_timing mortal.tests.test_value_coordinate_contracts` 覆盖路径保护、参数校验、标签契约、分箱/误差算术、完整日志检查、可选 seed API 与 production reward/GAE 算术；新增计时测试用确定性时钟核对互斥累计/失败区间，并以本地 host stub 执行八游戏控制流，核对 smoke/reuse 分支、原有产物 hash、可选 shuffle 与固定次数 CUDA 边界调用。该 host 测试不执行真实模型、native 对局或 CUDA。无 torch 的云端仅运行这些测试及语法检查；真实 torch/native 加载、arena、显存与固定补全 A/A 必须在 runner 的 smoke 确认，不将轻量测试称为集成通过，也不为新增计时单独重跑八局。

## 只读复用已生成的 rollout

新建输出目录，并增加 `--reuse-rollout-dir <original-probe-output>`。actor/opponent、games、seed-start、seed-key、sampling-seed、device、Torch/Rayon线程数仍须显式传入并与原 rollout 一致。只接受这个入口原始生成的输出，不接受任意牌谱目录，不接受链式复用输出。新输出不得位于原输出目录内。

复用前核对 actor/opponent checkpoint hash、native 扩展与包 hash、Torch/NumPy版本、采样/guard/search/guiding设置、engine/player/model/checkpoint_utils源码hash；旧版缺失model/checkpoint_utils哈希时，仅允许从已记录的干净source commit读取本地Git blob，与当前HEAD对应blob及当前实际文件hash一致才接受，禁止lazy fetch、缺失或不匹配即拒绝；逐个检查原 outcomes 注册的日志hash、终局完整性，并由 native 重新解析 seat/seed/key/rank，要求恰好完整的请求四座 seed groups。额外、缺失、重复、目录外或被改写的日志都会失败。评分后再次检查原日志和 outcomes 未变。原有 provenance 即使仍为 running，只要已原子发布完整 outcomes 且以上证据都成立，也可复用；一个 running 状态本身绝不能证明 arena 完成。原 provenance、日志、outcomes 不会被修改，新 provenance 记录原文件hash和核验结果。

此路径直接跳过 TrainPlayer/arena，不增加对局；仍重新构造模型、解码并评分，不是训练 resume，也不复用旧预测替代新核验。

## 可选隐藏输入置换（默认关闭）

`--shuffle-hidden --shuffle-seed 20260930` 才启用单一对照。当前研究核心是 privileged critic 如何通过 advantage 正确帮助 visible policy；此可选对照不自动安排实验，不作为“Oracle 信息是否有价值”的裁决。

对每场完整 trainee 轨迹，用 `[game seed, seed key, trainee seat, shuffle seed]` 的 SHA256 前128位（little-endian）初始化 PCG64；随机排列全部状态后形成一个无固定点循环，给每个可见状态配另一个时间点的隐藏输入。三个 critic 共用完全相同的索引；不在 physical batch 内重新抽样，也不跨 independent seed groups 交换输入。记录固定算法、seed、batch布局、每场置换hash及逐状态 `hidden_source_index`。单状态轨迹无法置换时失败，不静默退回 identity。

每 critic 每状态增加一次 forward，三模型评分计算量约为正常模式的两倍，不增加对局，不新增zero-input模型模式或feature cache；constant-zero数值基线保留。额外输出三组 shuffle − normal 的 p0/all_players MSE 和四座seed-group bootstrap区间。逐状态记录可选的 shuffled predictions，因此轻量预测记录变大；模型数、单批GPU输入规模不变。

置换使可见/隐藏信息不一致，属于 OOD 依赖诊断，不能称为合格 visible-only 对照或 Oracle 的因果提升。warm40k相对warm0可描述响应变化；clean40k缺少clean0，不作其训练学习归因。没有任意硬阈值。
