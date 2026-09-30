# 有截止时间的进程所有权 supervisor

`scripts/run_deadline_supervisor.py` 是无第三方依赖的本机 lease launcher，适用于独立 runner checkout。它不连接其他机器、不修改系统服务、不设置 cron、不接管旧进程，也不按名称杀 Python / Codex。先在两台目标 Windows 机器运行下面的 CPU smoke tests，再允许启动真实训练。云端 Linux 测试通过不等于 Windows 已验收。

## 范围与安全边界

- 每个 lease 使用唯一输出目录、run ID、UTC deadline；记录 supervisor PID / 开始时间、每个 role 的 bootstrap PID、实际 workload PID、精确 argv、cwd、Git commit / dirty 标记、配置 SHA256、stdout / stderr / result 路径
- 默认拒绝 dirty Git checkout；spec 可设全局 `expected_commit`，role 可覆盖，必须精确等于实际完整 HEAD。每次打开 GO gate 之前和 bootstrap 创建 workload 之前重新核对 commit、clean 状态和配置 SHA256。只有开发 smoke 可显式设置全局 `allow_dirty: true`，该豁免记入 manifest；真实训练不可使用。spec / config / lease 输出放工作树外或已忽略目录，避免运行产物使源码变 dirty
- `argv` 必须是数组，首元素为现存可执行文件的绝对路径；`shell=False`。只运行可信且已审核的训练入口，不用 `cmd /c`、PowerShell 包装、后台任务、跨机器 launcher 或会脱离所有权的命令
- Windows 使用独立、不继承的 `KILL_ON_JOB_CLOSE` Job Object，禁止 breakaway。bootstrap 先等待私有管道，成功加入 job 并记录身份后才收到 GO，随后创建 workload；后代继承 job。Job 分配失败就不打开该 role 的 GO gate，并清理本 lease。关闭 job handle 只终止此 job；不使用 `taskkill` 或存档 PID 恢复 ownership
- POSIX 仅支持可信、不调用 `setsid` / `setpgid` 脱离进程组的命令，必须显式设置 `posix_process_group_acknowledged: true`。每个 role 的 bootstrap 是独立 session/process-group leader，在清理前保持未 reap，避免 PGID 重用误杀。它监控 owner pipe EOF；owner 崩溃则杀自身组。不能为任意不受信命令提供不可逃逸隔离，不满足这个前提时不要启用 POSIX 模式
- `--detach` 创建与发起 shell / chat 连接分离的 supervisor，并把 stdio 重定向到文件。聊天连接断开、发起 shell 退出不会续租或改变 deadline；supervisor 正常退出或崩溃会清理其拥有的任务
- 操作系统须正常运行、调度进程和响应终止请求。这不是实时系统保证：OS suspend / hibernate / 内核故障、不可中断 I/O、supervisor 被暂停或整机挂起可能导致截止执行延迟。不要依赖它抵御管理员主动更改进程/系统。重启不会恢复 lease 或自动续训；异常关机留下的 manifest 可能停在 running，不能据此接管或杀 PID
- UTC 时钟和启动时 monotonic ceiling 取较早者，回拨系统时间不会延长 lease。OS suspend 后的第一个检查按 UTC 立即截止。截止前 `grace_seconds` 开始要求保存；到截止仍未退出则强制终止本 lease。进程创建、文件系统或内核 API 本身卡死不在用户态计时保证范围内

## 配置与启动

使用显式 UTC 时间：本轮授权截止为 **2026-10-08T11:00:00+00:00**，即北京时间 19:00。5070 Ti 的 RL / Oracle 与 4060 的 SL 分别在各自 Git 同步的不可变 checkout 建立独立 lease；不得复用输出目录/训练 checkpoint 路径。supervisor 不自动验证机器身份和模型目录碰撞，启动者必须核对。

示意 JSON（路径必须替换为本机已验证路径）：

```json
{
  "deadline_utc": "2026-10-08T11:00:00+00:00",
  "grace_seconds": 180,
  "expected_commit": "替换为两台 runner 已核验的完整 Git commit",
  "roles": [
    {
      "name": "trainer",
      "argv": ["C:\\ProgramData\\anaconda3\\envs\\mortal\\python.exe", "-m", "mortal.online.train_online"],
      "cwd": "D:\\verified-checkout",
      "config": "D:\\verified-checkout\\runtime-config.toml"
    }
  ]
}
```

三角色在线训练应将已核对的 server / trainer / client 三个直接命令都加入 `roles`；它们共享一个 lease 且使用不同日志文件。每个 role 的 config 可不同。role 启动没有 readiness dependency 编排，连接重试/端口 readiness 必须由现有入口保障；本工具不猜测训练命令。任一 role 自然完成不会立即杀其他 role，所有 role 自然退出 0 才是 `completed`；若服务需要一直存活，最终将由 deadline 清理。任一非零意外退出请求其他 role 停止。

```powershell
$py = 'C:\ProgramData\anaconda3\envs\mortal\python.exe'
& $py scripts/run_deadline_supervisor.py run --spec lease.json --output logs/lease-machine-unique --dry-run
& $py scripts/run_deadline_supervisor.py run --spec lease.json --output logs/lease-machine-unique --detach
```

`--dry-run` 只验证/打印 manifest，不创建 lease、不执行命令。真实启动拒绝已过截止或已进入保存余量的任务；GO 前也再次检查绝对截止。`--detach` 返回 `launch_requested`，必须读取输出目录 `manifest.json` 验证启动成功，并检查每个 `<role>.started.json`、日志、有效吞吐与模型指标；PID 存在不等于训练有效。不要修改运行中的脚本、checkout、config 或 checkpoint。

## 停止、结果与恢复语义

所有 workload 获得以下环境变量：

- `MORTAL_STOP_FILE` 与现有 `MORTAL_ORACLE_PAUSE_FILE` 指向同一个 lease `STOP` 文件
- `MORTAL_RUN_ID`、`MORTAL_RUN_DEADLINE_UTC`
- role 配置存在时，`MORTAL_CFG` 指向其绝对路径

正常到保存余量时 supervisor 创建 STOP。SL 已有 external pause：训练在 optimizer 边界保存，validation 阶段保留之前 checkpoint，退出 75。只有本 lease 已发出 stop 后，75（SL）/ 87（直接 RL 子进程）才视为预期的暂停退出。RL 入口支持 `MORTAL_STOP_FILE`，保存后子进程退出 87，外层 wrapper 将其视为终态返回 0，不会重新启动；不支持的进程仍会在截止被硬停止。不要假设 KeyboardInterrupt 就会保存。

也可以由操作者创建该 lease 的 STOP 请求提前停止，supervisor 将其记录为 interrupted；最终结果仍以各 role 的实际 exit code / 日志为准。若希望 supervisor 记录为 `interrupted`，在前台按 Ctrl+C，或用已验证的进程对象请求 supervisor 终止。不要从旧 manifest 拿 PID 批量 kill。

manifest 终态区分 `completed`、`deadline`、`external_failure`、`interrupted`、`refused_late_start`、`supervisor_failure`；`forced_stop` 和 `role_results` 标记硬截止与不完整 role。exit code 为 0（完成）、124（deadline）、1（失败/中断）、2（拒绝启动/配置错误）。JSON 使用同目录临时文件 + fsync + atomic replace；崩溃时仍可能保留最后一个非终态。实际 checkpoint 是否存在/完整须单独核验。

**保存不等于 exact resume**：supervisor 不读取或证明 optimizer、scaler、scheduler、data cursor、rollout / replay / behavior-policy 身份。它始终将 checkpoint 状态标为未验证；中途终止产物只能在读取实际 checkpoint 合约后决定是否恢复，不能宣传完整训练轨迹无损续训。

## CPU 验收

```powershell
& $py -m unittest mortal.tests.test_deadline_supervisor -v
```

测试使用秒级 deadline、普通 Python sleep / 文件写入，不需要 GPU：覆盖晚启动拒绝、dry-run、多个 role、SL 75 保存契约、硬截止、不杀无关进程、角色失败联动、发起进程退出后的 detached 截止。测试还通过实际 Popen owner handle 强制终止 supervisor，验证 owned grandchild 心跳停止；在 Windows 执行该测试将验证 Job Object 的 owner 异常退出清理路径。仍须人工核对目标 shell / Codex 与独立 Python 探针保持正常。若 Job Object 受现有 runner job 限制而分配失败，保持 fail-closed，不能退回全局 taskkill。
