param(
    [Parameter(Mandatory = $true)]
    [string]$Arm,
    [string]$ExperimentProfile = 'default',
    [string]$OpponentPoolPreset = 'default',
    [string]$LaptopHost = 'mahjong-laptop',
    [string]$LaptopRepo = 'C:\Users\numbe\Desktop\MahjongAI',
    [string]$LaptopPython = 'C:\Users\numbe\miniconda3\envs\mortal\python.exe',
    [int]$RemotePort = 5000,
    [string]$RemoteHost = '127.0.0.1',
    [string]$RuntimeName = '',
    [string]$ClientDevice = 'cuda:0',
    [string]$ClientBaselineTrainDevice = '',
    [switch]$SkipSync
)

$ErrorActionPreference = 'Stop'
if (Get-Variable PSNativeCommandUseErrorActionPreference -ErrorAction SilentlyContinue) {
    $PSNativeCommandUseErrorActionPreference = $false
}

function Quote-Single {
    param([string]$Value)
    return $Value.Replace("'", "''")
}

function Join-PwshArgs {
    param([string[]]$Args)
    return (($Args | ForEach-Object { "'" + (Quote-Single $_) + "'" }) -join ' ')
}

function Invoke-RemotePwsh {
    param(
        [Parameter(Mandatory = $true)]
        [string]$HostName,
        [Parameter(Mandatory = $true)]
        [string]$Script
    )
    $encoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($Script))
    & ssh -q $HostName powershell -NoProfile -EncodedCommand $encoded
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
    Invoke-RemotePwsh -HostName $HostName -Script $remoteScript | Out-Null
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
            Invoke-RemotePwsh `
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

function Start-RemotePythonWindow {
    param(
        [Parameter(Mandatory = $true)]
        [string]$HostName,
        [Parameter(Mandatory = $true)]
        [string]$RepoRoot,
        [Parameter(Mandatory = $true)]
        [string]$PythonExe,
        [Parameter(Mandatory = $true)]
        [string]$PythonScript,
        [Parameter(Mandatory = $true)]
        [string[]]$PythonArgs,
        [Parameter(Mandatory = $true)]
        [string]$TaskId,
        [Parameter(Mandatory = $true)]
        [string]$RuntimeRoot,
        [Parameter(Mandatory = $true)]
        [string]$WindowTitle
    )

    $argsPayload = [string]::Join("`0", $PythonArgs)
    $argsBase64 = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($argsPayload))
    $remoteScriptPath = Join-Path $RepoRoot 'scripts\start_interactive_remote_python.ps1'

    $remoteScript = @"
`$ErrorActionPreference = 'Stop'
& '$(Quote-Single $remoteScriptPath)' `
  -RepoRoot '$(Quote-Single $RepoRoot)' `
  -PythonExe '$(Quote-Single $PythonExe)' `
  -PythonScript '$(Quote-Single $PythonScript)' `
  -PythonArgsBase64 '$(Quote-Single $argsBase64)' `
  -TaskId '$(Quote-Single $TaskId)' `
  -RuntimeRoot '$(Quote-Single $RuntimeRoot)' `
  -WindowTitle '$(Quote-Single $WindowTitle)' `
  -WaitForStartOnly
"@
    Invoke-RemotePwsh -HostName $HostName -Script $remoteScript | Write-Output
}

if (-not $SkipSync) {
    & "$PSScriptRoot\sync_laptop_repo.ps1" -LaptopHost $LaptopHost
}

$repoRoot = Split-Path -Parent $PSScriptRoot
Sync-LocalRuntimePythonSourcesToRemote `
    -RepoRoot $repoRoot `
    -HostName $LaptopHost `
    -RemoteRepoRoot $LaptopRepo
Sync-CommonRuntimeAssetsToRemote `
    -RepoRoot $repoRoot `
    -HostName $LaptopHost `
    -RemoteRepoRoot $LaptopRepo

$timestamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$resolvedRuntimeName = if ($RuntimeName) { $RuntimeName } else { "$timestamp" + "__" + $Arm }
$runtimeRoot = Join-Path $LaptopRepo ("logs\online_modes\independent_arm\" + $resolvedRuntimeName)
$configPath = Join-Path $runtimeRoot 'config.toml'
$builderPath = Join-Path $LaptopRepo 'mortal\online\online_machine_modes.py'
$entryPath = Join-Path $LaptopRepo 'mortal\online\online_role_runner.py'
$baseConfigPath = Join-Path $LaptopRepo 'mortal\config.toml'
$builderArgs = @(
    '--mode', 'independent_arm',
    '--base-config', $baseConfigPath,
    '--output', $configPath,
    '--runtime-root', $runtimeRoot,
    '--experiment-profile', $ExperimentProfile,
    '--opponent-pool-preset', $OpponentPoolPreset,
    '--remote-host', $RemoteHost,
    '--remote-port', [string]$RemotePort
)

$buildScript = @"
`$ErrorActionPreference = 'Stop'
Set-Location '$(Quote-Single $LaptopRepo)'
& '$(Quote-Single $LaptopPython)' '$(Quote-Single $builderPath)' $(Join-PwshArgs $builderArgs)
"@
Invoke-RemotePwsh -HostName $LaptopHost -Script $buildScript | Write-Output

Stop-RemoteRuntimeProcesses -HostName $LaptopHost -MatchToken $resolvedRuntimeName

$commonArgs = @('--config', $configPath, '--arm', $Arm)
$serverArgs = @('server') + $commonArgs
$trainerArgs = @('trainer') + $commonArgs
$clientArgs = @('client') + $commonArgs
if ($ClientDevice) {
    $clientArgs += '--control-device'
    $clientArgs += $ClientDevice
}
$resolvedClientBaselineTrainDevice = $ClientBaselineTrainDevice
if (-not $resolvedClientBaselineTrainDevice) {
    $resolvedClientBaselineTrainDevice = $ClientDevice
}
if ($resolvedClientBaselineTrainDevice) {
    $clientArgs += '--baseline-train-device'
    $clientArgs += $resolvedClientBaselineTrainDevice
}
$serverRuntimeRoot = Join-Path $runtimeRoot 'server_task'
$trainerRuntimeRoot = Join-Path $runtimeRoot 'trainer_task'
$clientRuntimeRoot = Join-Path $runtimeRoot 'client_task'

Start-RemotePythonWindow `
    -HostName $LaptopHost `
    -RepoRoot $LaptopRepo `
    -PythonExe $LaptopPython `
    -PythonScript $entryPath `
    -PythonArgs $serverArgs `
    -TaskId ("online_server_" + $resolvedRuntimeName) `
    -RuntimeRoot $serverRuntimeRoot `
    -WindowTitle ("MahjongAI Laptop Server " + $resolvedRuntimeName)

Start-RemotePythonWindow `
    -HostName $LaptopHost `
    -RepoRoot $LaptopRepo `
    -PythonExe $LaptopPython `
    -PythonScript $entryPath `
    -PythonArgs $trainerArgs `
    -TaskId ("online_trainer_" + $resolvedRuntimeName) `
    -RuntimeRoot $trainerRuntimeRoot `
    -WindowTitle ("MahjongAI Laptop Trainer " + $resolvedRuntimeName)

Start-RemotePythonWindow `
    -HostName $LaptopHost `
    -RepoRoot $LaptopRepo `
    -PythonExe $LaptopPython `
    -PythonScript $entryPath `
    -PythonArgs $clientArgs `
    -TaskId ("online_client_" + $resolvedRuntimeName) `
    -RuntimeRoot $clientRuntimeRoot `
    -WindowTitle ("MahjongAI Laptop Worker " + $resolvedRuntimeName)

Write-Output ("REMOTE_RUNTIME_ROOT " + $runtimeRoot)
Write-Output ("REMOTE_CONFIG " + $configPath)
