# 雀魂网页接入

> 核验：2026-09-05 · 依据本目录代码。当前具备浏览器观测和 UI 适配框架，尚未形成从真实雀魂牌局到模型动作的完整链路。

## 能力边界

| 模式 | 当前行为 | 依赖 |
| --- | --- | --- |
| `observe` | 打开独立浏览器 profile，记录 WebSocket 生命周期与帧元数据 | Playwright 与浏览器运行时 |
| `assist` | 框架可接 MJAI 事件、记录建议动作 | 仍需接入事件适配器 |
| `autoplay` | 框架可将 MJAI 动作变为 UI 计划，默认 dry-run | 事件适配器、坐标校准与显式点击开关 |

**当前没有 `majsoul protobuf → mjai` 完整解码表。** `--integration-mode assist/autoplay` 只是声明模式；切换该参数不会自动补齐协议转换。点击通过可见 UI 执行，不伪造服务器协议动作。

源码：[capture.py](capture.py) 管理 CLI，[browser.py](browser.py) 管理浏览器，[records.py](records.py) 定义记录，[actions.py](actions.py) / [ui.py](ui.py) 生成和执行 UI 计划。

## 观测流程

以下从仓库根运行。环境只需安装一次：

```powershell
$mortalPython = 'C:\ProgramData\anaconda3\envs\mortal\python.exe'
& $mortalPython -m pip install playwright
& $mortalPython -m playwright install chromium
```

使用平台提供的测试账号，在独立 profile 中手动登录；账号密码不写入命令或日志：

```powershell
.\scripts\run_majsoul_login.bat --channel chrome --profile-dir .\logs\majsoul\browser-profile
```

登录完成后关闭该窗口，再用同一 profile 观测；不要让两个浏览器进程同时占用同一 profile：

```powershell
& $mortalPython -m integrations.majsoul.capture `
  --channel chrome --integration-mode observe `
  --out .\logs\majsoul\capture.jsonl `
  --profile-dir .\logs\majsoul\browser-profile
```

不传 `--channel` 时使用安装的 Chromium。默认去掉 URL query，payload 只记长度和 hash；需要原始帧时显式使用 `--payload-mode full`，该日志与 profile 均视为本地敏感产物。

## 补齐链路与验证

接入协议样本/说明后，先实现事件转换，验证座位、牌 ID、动作合法性和局终止；再把模型接到 `assist`，检查 MJAI 输入与建议记录。最后在已授权的测试场景验证 UI 校准和 dry-run 计划。

真正点击必须同时有可用适配器、校准和 `--enable-ui-clicks`；传开关本身不证明端到端接入成功。当前缺口应在实际补齐并完成整局 smoke 后再从本页移除。

项目状态见 [接手摘要](../../docs/agent/handoff.md)，模型与宿主边界见 [代码地图](../../docs/agent/code-map.md#平台与宿主)。
