# RTK 命令输出约定

长输出外部命令使用 `rtk`；PowerShell 内建、短探针和管道直接运行。原始错误或完整证据需要保留时用 `rtk proxy`。

```powershell
rtk git status --short
rtk cargo test -p libriichi
rtk proxy C:\ProgramData\anaconda3\envs\mortal\python.exe -m unittest mortal.tests.test_greedy
```

这些命令直接运行：

```powershell
Get-Content .\AGENTS.md
Test-Path .\mortal\config.toml
$env:PYO3_PYTHON
rg --files mortal | Select-String 'test_'
```

检查可用性和压缩统计：

```powershell
Get-Command rtk
rtk --version
rtk gain
rtk gain --history
```

没有 `rtk` 时保留任务本身，直接执行并限制输出。代码整理继续遵守 [重构与验证](docs/agent/code-health.md)：缩小改动范围，保持可读性和运行效率。
