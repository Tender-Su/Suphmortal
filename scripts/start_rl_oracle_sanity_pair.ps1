param(
    [string]$PairName = '',
    [string]$DesktopProfile = 'ms_rl2_smoke_10k',
    [string]$LaptopProfile = 'ms_rl1_smoke_10k',
    [string]$DesktopOpponentPoolPreset = 'default',
    [string]$LaptopOpponentPoolPreset = 'default',
    [string]$DesktopArm = 'current_config',
    [string]$LaptopArm = 'current_config',
    [string]$LaptopHost = 'mahjong-laptop',
    [string]$LaptopRepo = (Join-Path $env:USERPROFILE 'Desktop\MahjongAI'),
    [string]$LaptopPython = (Join-Path $env:USERPROFILE 'miniconda3\envs\mortal\python.exe'),
    [string]$DesktopPython = 'C:\ProgramData\anaconda3\envs\mortal\python.exe',
    [string]$DesktopClientDevice = 'cuda:0',
    [string]$DesktopClientBaselineTrainDevice = '',
    [string]$LaptopClientDevice = 'cuda:0',
    [string]$LaptopClientBaselineTrainDevice = '',
    [Nullable[int]]$DesktopReproSeed = $null,
    [Nullable[int]]$LaptopReproSeed = $null,
    [Nullable[UInt64]]$TrainKey = $null,
    [Nullable[int]]$TrainSeedStart = $null,
    [switch]$SkipLaptopGitSync
)

$ErrorActionPreference = 'Stop'
if (Get-Variable PSNativeCommandUseErrorActionPreference -ErrorAction SilentlyContinue) {
    $PSNativeCommandUseErrorActionPreference = $false
}

function Quote-Single {
    param([string]$Value)
    return $Value.Replace("'", "''")
}

function Stop-LocalRuntimeProcesses {
    param(
        [Parameter(Mandatory = $true)]
        [string]$MatchToken
    )

    $pattern = [regex]::Escape($MatchToken)
    $processes = @(Get-CimInstance Win32_Process)
    $targets = @{}
    foreach ($proc in $processes) {
        if (
            $proc.ProcessId -ne $PID -and
            $proc.CommandLine -and
            $proc.Name -in @('python.exe', 'powershell.exe', 'pwsh.exe', 'cmd.exe') -and
            ($proc.CommandLine -match $pattern)
        ) {
            $targets[[int]$proc.ProcessId] = $proc
        }
    }
    $pending = @($targets.Keys)
    while ($pending.Count -gt 0) {
        $parentPid = [int]$pending[0]
        if ($pending.Count -eq 1) {
            $pending = @()
        }
        else {
            $pending = $pending[1..($pending.Count - 1)]
        }
        foreach ($child in $processes) {
            if ($child.ParentProcessId -eq $parentPid -and -not $targets.ContainsKey([int]$child.ProcessId)) {
                $targets[[int]$child.ProcessId] = $child
                $pending += [int]$child.ProcessId
            }
        }
    }
    foreach ($proc in ($targets.Values | Sort-Object ProcessId -Descending)) {
        try {
            Stop-Process -Id $proc.ProcessId -Force -ErrorAction Stop
        }
        catch {
        }
    }
}

function Stop-RemoteRuntimeProcesses {
    param(
        [Parameter(Mandatory = $true)]
        [string]$HostName,
        [Parameter(Mandatory = $true)]
        [string]$MatchToken
    )

    $remoteScript = @"
`$pattern = [regex]::Escape('$(Quote-Single $MatchToken)')
`$processes = @(Get-CimInstance Win32_Process)
`$targets = @{}
foreach (`$proc in `$processes) {
  if (
    `$proc.ProcessId -ne `$PID -and
    `$proc.CommandLine -and
    `$proc.Name -in @('python.exe', 'powershell.exe', 'pwsh.exe', 'cmd.exe') -and
    (`$proc.CommandLine -match `$pattern)
  ) {
    `$targets[[int]`$proc.ProcessId] = `$proc
  }
}
`$pending = @(`$targets.Keys)
while (`$pending.Count -gt 0) {
  `$parentPid = [int]`$pending[0]
  if (`$pending.Count -eq 1) {
    `$pending = @()
  }
  else {
    `$pending = `$pending[1..(`$pending.Count - 1)]
  }
  foreach (`$child in `$processes) {
    if (`$child.ParentProcessId -eq `$parentPid -and -not `$targets.ContainsKey([int]`$child.ProcessId)) {
      `$targets[[int]`$child.ProcessId] = `$child
      `$pending += [int]`$child.ProcessId
    }
  }
}
foreach (`$proc in (`$targets.Values | Sort-Object ProcessId -Descending)) {
  try {
    Stop-Process -Id `$proc.ProcessId -Force -ErrorAction Stop
  }
  catch {
  }
}
"@
    Invoke-RemoteEncodedPowerShell -HostName $HostName -Script $remoteScript | Out-Null
}

function Invoke-RemoteEncodedPowerShell {
    param(
        [Parameter(Mandatory = $true)]
        [string]$HostName,
        [Parameter(Mandatory = $true)]
        [string]$Script
    )
    $encoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($Script))
    & ssh -q $HostName powershell -NoProfile -EncodedCommand $encoded
}

function Copy-FileToRemote {
    param(
        [Parameter(Mandatory = $true)]
        [string]$LocalPath,
        [Parameter(Mandatory = $true)]
        [string]$HostName,
        [Parameter(Mandatory = $true)]
        [string]$RemotePath
    )
    & scp -q $LocalPath ($HostName + ':' + $RemotePath.Replace('\', '/'))
}

function Get-RemoteFileLength {
    param(
        [Parameter(Mandatory = $true)]
        [string]$HostName,
        [Parameter(Mandatory = $true)]
        [string]$RemotePath
    )

    $script = @"
if (Test-Path '$(Quote-Single $RemotePath)') {
    (Get-Item '$(Quote-Single $RemotePath)').Length
}
else {
    -1
}
"@
    $encoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($script))
    $output = & ssh -q $HostName powershell -NoProfile -EncodedCommand $encoded 2>$null
    $lastLine = $output | Select-Object -Last 1
    if ($null -eq $lastLine) {
        return -1
    }
    $value = ([string]$lastLine).Trim()
    if (-not $value) {
        return -1
    }
    return [int64]$value
}

function Sync-LocalRuntimePythonSourcesToRemote {
    param(
        [Parameter(Mandatory = $true)]
        [string]$RepoRoot,
        [Parameter(Mandatory = $true)]
        [string]$HostName,
        [Parameter(Mandatory = $true)]
        [string]$RemoteRepoRoot
    )

    $mortalRoot = Join-Path $RepoRoot 'mortal'
    Get-ChildItem $mortalRoot -Filter '*.py' -File -Recurse |
        ForEach-Object {
            $relativePath = $_.FullName.Substring($mortalRoot.Length).TrimStart('\', '/')
            $remotePath = Join-Path $RemoteRepoRoot (Join-Path 'mortal' $relativePath)
            $remoteDir = Split-Path -Parent $remotePath
            Invoke-RemoteEncodedPowerShell `
                -HostName $HostName `
                -Script "New-Item -ItemType Directory -Force -Path '$(Quote-Single $remoteDir)' | Out-Null" | Out-Null
            Copy-FileToRemote `
                -LocalPath $_.FullName `
                -HostName $HostName `
                -RemotePath $remotePath
        }

    Copy-FileToRemote `
        -LocalPath (Join-Path $RepoRoot 'scripts\start_interactive_remote_python.ps1') `
        -HostName $HostName `
        -RemotePath (Join-Path $RemoteRepoRoot 'scripts\start_interactive_remote_python.ps1')
}

function Sync-CommonRuntimeAssetsToRemote {
    param(
        [Parameter(Mandatory = $true)]
        [string]$RepoRoot,
        [Parameter(Mandatory = $true)]
        [string]$HostName,
        [Parameter(Mandatory = $true)]
        [string]$RemoteRepoRoot
    )

    $assetNames = @(
        'grp.pth',
        'baseline.pth',
        'sl_canonical.pth',
        'sl_best_acc.pth'
    )
    foreach ($assetName in $assetNames) {
        $localPath = Join-Path $RepoRoot ('mortal\checkpoints\' + $assetName)
        if (-not (Test-Path $localPath)) {
            continue
        }
        $remotePath = Join-Path $RemoteRepoRoot ('mortal\checkpoints\' + $assetName)
        $localLength = (Get-Item $localPath).Length
        $remoteLength = Get-RemoteFileLength -HostName $HostName -RemotePath $remotePath
        if ($remoteLength -eq $localLength -and $localLength -gt 0) {
            continue
        }
        Copy-FileToRemote `
            -LocalPath $localPath `
            -HostName $HostName `
            -RemotePath $remotePath
    }
}

function Start-LocalRoleWindow {
    param(
        [Parameter(Mandatory = $true)]
        [string]$RepoRoot,
        [Parameter(Mandatory = $true)]
        [string]$PythonExe,
        [Parameter(Mandatory = $true)]
        [string]$ConfigPath,
        [Parameter(Mandatory = $true)]
        [string]$Arm,
        [Parameter(Mandatory = $true)]
        [string]$Role,
        [Parameter(Mandatory = $true)]
        [string]$WindowTitle,
        [string[]]$ExtraArgs = @()
    )

    $mortalRoot = Join-Path $RepoRoot 'mortal'
    $extraArgsPayloadBase64 = [Convert]::ToBase64String(
        [Text.Encoding]::UTF8.GetBytes([string]::Join("`0", $ExtraArgs))
    )
    $roleScript = @"
`$env:MORTAL_CFG = '$(Quote-Single $ConfigPath)'
`$env:MORTAL_ORACLE_ARM = '$(Quote-Single $Arm)'
Set-Location '$(Quote-Single $RepoRoot)'
`$extraArgsPayload = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('$(Quote-Single $extraArgsPayloadBase64)'))
`$extraArgs = @()
if (`$extraArgsPayload.Length -gt 0) {
  foreach (`$item in (`$extraArgsPayload -split "`0", 0, 'SimpleMatch')) {
    `$extraArgs += [string]`$item
  }
}
& '$(Quote-Single $PythonExe)' '-m' 'mortal.online.online_role_runner' '$(Quote-Single $Role)' '--config' '$(Quote-Single $ConfigPath)' '--arm' '$(Quote-Single $Arm)' @extraArgs
"@

    Start-Process `
        -FilePath 'powershell.exe' `
        -ArgumentList @(
            '-NoLogo',
            '-NoProfile',
            '-ExecutionPolicy',
            'Bypass',
            '-NoExit',
            '-Command',
            $roleScript
        ) `
        -WindowStyle Normal | Out-Null
}

$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

if (-not $PairName) {
    $PairName = Get-Date -Format 'yyyyMMdd_HHmmss'
}

$desktopRuntimeName = $PairName + '__desktop_rl2'
$laptopRuntimeName = $PairName + '__laptop_rl1'

$desktopRuntimeRoot = Join-Path $repoRoot ('logs\online_modes\independent_arm\' + $desktopRuntimeName)
$desktopConfigPath = Join-Path $desktopRuntimeRoot 'config.toml'

$remoteRuntimeRoot = Join-Path $LaptopRepo ('logs\online_modes\independent_arm\' + $laptopRuntimeName)
$remoteConfigLocalPath = Join-Path $repoRoot ('logs\online_modes\independent_arm\' + $laptopRuntimeName + '\config.toml')
$remoteConfigRemotePath = Join-Path $remoteRuntimeRoot 'config.toml'

$machineModesPath = Join-Path $repoRoot 'mortal\online\online_machine_modes.py'
$baseConfigPath = Join-Path $repoRoot 'mortal\config.toml'

$desktopReproArgs = @()
if ($null -ne $DesktopReproSeed) {
    $desktopReproArgs += @('--repro-seed', [string]$DesktopReproSeed)
}
if ($null -ne $TrainKey) {
    $desktopReproArgs += @('--train-key', [string]$TrainKey)
}
if ($null -ne $TrainSeedStart) {
    $desktopReproArgs += @('--train-seed-start', [string]$TrainSeedStart)
}
$laptopReproArgs = @()
if ($null -ne $LaptopReproSeed) {
    $laptopReproArgs += @('--repro-seed', [string]$LaptopReproSeed)
}
if ($null -ne $TrainKey) {
    $laptopReproArgs += @('--train-key', [string]$TrainKey)
}
if ($null -ne $TrainSeedStart) {
    $laptopReproArgs += @('--train-seed-start', [string]$TrainSeedStart)
}

& $DesktopPython $machineModesPath `
    --mode independent_arm `
    --base-config $baseConfigPath `
    --experiment-profile $DesktopProfile `
    --opponent-pool-preset $DesktopOpponentPoolPreset `
    --output $desktopConfigPath `
    --runtime-root $desktopRuntimeRoot `
    @desktopReproArgs | Out-Null

& $DesktopPython $machineModesPath `
    --mode independent_arm `
    --base-config $baseConfigPath `
    --experiment-profile $LaptopProfile `
    --opponent-pool-preset $LaptopOpponentPoolPreset `
    --output $remoteConfigLocalPath `
    --runtime-root $remoteRuntimeRoot `
    @laptopReproArgs | Out-Null

Stop-LocalRuntimeProcesses -MatchToken $desktopRuntimeName

Start-LocalRoleWindow `
    -RepoRoot $repoRoot `
    -PythonExe $DesktopPython `
    -ConfigPath $desktopConfigPath `
    -Arm $DesktopArm `
    -Role 'server' `
    -WindowTitle ('MahjongAI Desktop Server ' + $desktopRuntimeName)

Start-LocalRoleWindow `
    -RepoRoot $repoRoot `
    -PythonExe $DesktopPython `
    -ConfigPath $desktopConfigPath `
    -Arm $DesktopArm `
    -Role 'trainer' `
    -WindowTitle ('MahjongAI Desktop Trainer ' + $desktopRuntimeName)

if ($DesktopClientDevice) {
    $desktopClientExtraArgs = @(
        '--control-device', $DesktopClientDevice
    )
}
else {
    $desktopClientExtraArgs = @()
}
$resolvedDesktopClientBaselineTrainDevice = $DesktopClientBaselineTrainDevice
if (-not $resolvedDesktopClientBaselineTrainDevice) {
    $resolvedDesktopClientBaselineTrainDevice = $DesktopClientDevice
}
if ($resolvedDesktopClientBaselineTrainDevice) {
    $desktopClientExtraArgs += @(
        '--baseline-train-device', $resolvedDesktopClientBaselineTrainDevice
    )
}
Start-LocalRoleWindow `
    -RepoRoot $repoRoot `
    -PythonExe $DesktopPython `
    -ConfigPath $desktopConfigPath `
    -Arm $DesktopArm `
    -Role 'client' `
    -WindowTitle ('MahjongAI Desktop Worker ' + $desktopRuntimeName) `
    -ExtraArgs $desktopClientExtraArgs

if (-not $SkipLaptopGitSync) {
    & "$PSScriptRoot\sync_laptop_repo.ps1" -LaptopHost $LaptopHost
}

$remotePrepScript = @"
`$ErrorActionPreference = 'Stop'
New-Item -ItemType Directory -Path '$(Quote-Single $remoteRuntimeRoot)' -Force | Out-Null
New-Item -ItemType Directory -Path '$(Quote-Single (Join-Path $remoteRuntimeRoot 'checkpoints'))' -Force | Out-Null
New-Item -ItemType Directory -Path '$(Quote-Single (Join-Path $remoteRuntimeRoot 'tb_log'))' -Force | Out-Null
New-Item -ItemType Directory -Path '$(Quote-Single (Join-Path $remoteRuntimeRoot 'logs\test_play'))' -Force | Out-Null
New-Item -ItemType Directory -Path '$(Quote-Single (Join-Path $remoteRuntimeRoot 'logs\1v3'))' -Force | Out-Null
New-Item -ItemType Directory -Path '$(Quote-Single (Join-Path $remoteRuntimeRoot 'logs\oracle_dependency'))' -Force | Out-Null
New-Item -ItemType Directory -Path '$(Quote-Single (Join-Path $remoteRuntimeRoot 'logs\train_play'))' -Force | Out-Null
New-Item -ItemType Directory -Path '$(Quote-Single (Join-Path $remoteRuntimeRoot 'server\buffer'))' -Force | Out-Null
New-Item -ItemType Directory -Path '$(Quote-Single (Join-Path $remoteRuntimeRoot 'server\drain'))' -Force | Out-Null
New-Item -ItemType Directory -Path '$(Quote-Single (Join-Path $remoteRuntimeRoot 'server_task'))' -Force | Out-Null
New-Item -ItemType Directory -Path '$(Quote-Single (Join-Path $remoteRuntimeRoot 'trainer_task'))' -Force | Out-Null
New-Item -ItemType Directory -Path '$(Quote-Single (Join-Path $remoteRuntimeRoot 'client_task'))' -Force | Out-Null
"@
Invoke-RemoteEncodedPowerShell -HostName $LaptopHost -Script $remotePrepScript | Out-Null

Copy-FileToRemote -LocalPath $remoteConfigLocalPath -HostName $LaptopHost -RemotePath $remoteConfigRemotePath
Sync-LocalRuntimePythonSourcesToRemote `
    -RepoRoot $repoRoot `
    -HostName $LaptopHost `
    -RemoteRepoRoot $LaptopRepo
Sync-CommonRuntimeAssetsToRemote `
    -RepoRoot $repoRoot `
    -HostName $LaptopHost `
    -RemoteRepoRoot $LaptopRepo

Stop-RemoteRuntimeProcesses -HostName $LaptopHost -MatchToken $laptopRuntimeName

$remoteClientExtra = ''
if ($LaptopClientDevice) {
    $remoteClientExtra += ", '--control-device', '$(Quote-Single $LaptopClientDevice)'"
}
$resolvedLaptopClientBaselineTrainDevice = $LaptopClientBaselineTrainDevice
if (-not $resolvedLaptopClientBaselineTrainDevice) {
    $resolvedLaptopClientBaselineTrainDevice = $LaptopClientDevice
}
if ($resolvedLaptopClientBaselineTrainDevice) {
    $remoteClientExtra += ", '--baseline-train-device', '$(Quote-Single $resolvedLaptopClientBaselineTrainDevice)'"
}

$remoteLaunchScript = @"
`$ErrorActionPreference = 'Stop'
`$repo = '$(Quote-Single $LaptopRepo)'
`$py = '$(Quote-Single $LaptopPython)'
`$runtimeRoot = '$(Quote-Single $remoteRuntimeRoot)'
`$configPath = Join-Path `$runtimeRoot 'config.toml'
`$entry = Join-Path `$repo 'mortal\online\online_role_runner.py'
`$helper = Join-Path `$repo 'scripts\start_interactive_remote_python.ps1'
function Encode-Args([string[]]`$Items) {
  return [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes([string]::Join("`0", `$Items)))
}
`$serverArgs = Encode-Args @('server', '--config', `$configPath, '--arm', '$(Quote-Single $LaptopArm)')
`$trainerArgs = Encode-Args @('trainer', '--config', `$configPath, '--arm', '$(Quote-Single $LaptopArm)')
`$clientArgs = Encode-Args @('client', '--config', `$configPath, '--arm', '$(Quote-Single $LaptopArm)'$remoteClientExtra)
& `$helper -RepoRoot `$repo -PythonExe `$py -PythonScript `$entry -PythonArgsBase64 `$serverArgs -TaskId 'online_server_$(Quote-Single $laptopRuntimeName)' -RuntimeRoot (Join-Path `$runtimeRoot 'server_task') -WindowTitle 'MahjongAI Laptop Server $(Quote-Single $laptopRuntimeName)' -WaitForStartOnly
& `$helper -RepoRoot `$repo -PythonExe `$py -PythonScript `$entry -PythonArgsBase64 `$trainerArgs -TaskId 'online_trainer_$(Quote-Single $laptopRuntimeName)' -RuntimeRoot (Join-Path `$runtimeRoot 'trainer_task') -WindowTitle 'MahjongAI Laptop Trainer $(Quote-Single $laptopRuntimeName)' -WaitForStartOnly
& `$helper -RepoRoot `$repo -PythonExe `$py -PythonScript `$entry -PythonArgsBase64 `$clientArgs -TaskId 'online_client_$(Quote-Single $laptopRuntimeName)' -RuntimeRoot (Join-Path `$runtimeRoot 'client_task') -WindowTitle 'MahjongAI Laptop Worker $(Quote-Single $laptopRuntimeName)' -WaitForStartOnly
Write-Output ('REMOTE_RUNTIME_ROOT ' + `$runtimeRoot)
Write-Output ('REMOTE_CONFIG ' + `$configPath)
"@
Invoke-RemoteEncodedPowerShell -HostName $LaptopHost -Script $remoteLaunchScript

Write-Output ('DESKTOP_RUNTIME_ROOT ' + $desktopRuntimeRoot)
Write-Output ('DESKTOP_CONFIG ' + $desktopConfigPath)
Write-Output ('LAPTOP_RUNTIME_ROOT ' + $remoteRuntimeRoot)
Write-Output ('LAPTOP_CONFIG ' + $remoteConfigRemotePath)
