[CmdletBinding()]
param(
    [ValidateRange(0, [int]::MaxValue)]
    [int]$WaitForProcessId = 0,

    [Parameter(Mandatory = $true)]
    [string]$RunDir,

    [Parameter(Mandatory = $true)]
    [string]$Checkpoint,

    [int]$TargetGames = 2000,
    [double]$StopRatingAt = 0,
    [int]$RatingBotId = 0,
    [int]$RatingActivationTotalGames = -1,
    [string]$RatingApiBase = "https://api.riichi.dev",
    [string]$StopPolicyJson,
    [string]$Device = "cpu",
    [string]$BotName = "MahjongAI-S70",
    [int]$RestartDelaySeconds = 60,
    [int]$MaxConsecutiveErrors = 100000,
    [string]$ProxyUrl,
    [switch]$NoProxy,
    [string]$PhysicalDirectIp,
    [string]$PhysicalSourceAddress,
    [int]$PhysicalInterfaceIndex = 0
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($env:RIICHILAB_BOT_TOKEN)) {
    throw "RIICHILAB_BOT_TOKEN is not set"
}
if ($TargetGames -le 0) {
    throw "TargetGames must be positive"
}
if ($MaxConsecutiveErrors -le 0) {
    throw "MaxConsecutiveErrors must be positive"
}
$physicalEnabled = -not [string]::IsNullOrWhiteSpace($PhysicalDirectIp) -or `
    -not [string]::IsNullOrWhiteSpace($PhysicalSourceAddress) -or `
    $PhysicalInterfaceIndex -gt 0
if ($physicalEnabled -and (
    [string]::IsNullOrWhiteSpace($PhysicalDirectIp) -or $PhysicalInterfaceIndex -le 0
)) {
    throw "Physical direct mode requires IP and interface index"
}
$networkModeCount = @(
    -not [string]::IsNullOrWhiteSpace($ProxyUrl),
    [bool]$NoProxy,
    $physicalEnabled
) | Where-Object { $_ }
if ($networkModeCount.Count -gt 1) {
    throw "ProxyUrl, NoProxy, and physical direct mode are mutually exclusive"
}
$networkPath = if ($physicalEnabled) {
    "physical_direct"
} elseif (-not [string]::IsNullOrWhiteSpace($ProxyUrl)) {
    "explicit_proxy"
} elseif ($NoProxy) {
    "no_explicit_proxy"
} else {
    "auto_proxy"
}

$repoRoot = Split-Path -Parent $PSScriptRoot
$resolvedRunDir = if ([System.IO.Path]::IsPathRooted($RunDir)) {
    [System.IO.Path]::GetFullPath($RunDir)
} else {
    [System.IO.Path]::GetFullPath((Join-Path $repoRoot $RunDir))
}
$resolvedCheckpoint = if ([System.IO.Path]::IsPathRooted($Checkpoint)) {
    [System.IO.Path]::GetFullPath($Checkpoint)
} else {
    [System.IO.Path]::GetFullPath((Join-Path $repoRoot $Checkpoint))
}
$pythonExe = "C:\ProgramData\anaconda3\envs\mortal\python.exe"
$progressPath = Join-Path $resolvedRunDir "progress.json"
$defaultStopPolicyPath = Join-Path $resolvedRunDir "stop_policy.json"
$resolvedStopPolicyPath = if (-not [string]::IsNullOrWhiteSpace($StopPolicyJson)) {
    if ([System.IO.Path]::IsPathRooted($StopPolicyJson)) {
        [System.IO.Path]::GetFullPath($StopPolicyJson)
    } else {
        [System.IO.Path]::GetFullPath((Join-Path $repoRoot $StopPolicyJson))
    }
} elseif (Test-Path -LiteralPath $defaultStopPolicyPath) {
    $defaultStopPolicyPath
} else {
    $null
}
$policyStopRatingAt = if ($null -ne $resolvedStopPolicyPath) {
    if (-not (Test-Path -LiteralPath $resolvedStopPolicyPath)) {
        throw "Stop policy does not exist: $resolvedStopPolicyPath"
    }
    [double](
        Get-Content -LiteralPath $resolvedStopPolicyPath -Raw |
            ConvertFrom-Json |
            Select-Object -ExpandProperty stop_rating_at
    )
} else {
    0
}
$effectiveStopRatingAt = if ($policyStopRatingAt -gt 0) {
    $policyStopRatingAt
} else {
    $StopRatingAt
}
$ratingStopEnabled = $effectiveStopRatingAt -gt 0
if ($StopRatingAt -gt 0 -and $RatingBotId -le 0) {
    throw "StopRatingAt requires a positive RatingBotId"
}
$statusPath = Join-Path $resolvedRunDir "continuation_status.json"
$supervisorLogPath = Join-Path $resolvedRunDir "continuation_supervisor.log"

New-Item -ItemType Directory -Force -Path $resolvedRunDir | Out-Null

function Write-SupervisorLog {
    param([string]$Message)

    $line = "{0} {1}{2}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message, [Environment]::NewLine
    [System.IO.File]::AppendAllText($supervisorLogPath, $line, [System.Text.Encoding]::UTF8)
}

function Write-ContinuationStatus {
    param(
        [string]$State,
        [int]$CompletedGames,
        [Nullable[int]]$ChildProcessId = $null,
        [string]$Message = $null
    )

    $payload = [ordered]@{
        state = $State
        target_games = if ($ratingStopEnabled) { $null } else { $TargetGames }
        stop_rating_at = if ($ratingStopEnabled) { $effectiveStopRatingAt } else { $null }
        completed_games = $CompletedGames
        wait_for_process_id = $WaitForProcessId
        child_process_id = $ChildProcessId
        network_path = $networkPath
        updated_at = (Get-Date).ToString("o")
        message = $Message
    }
    $tempPath = "$statusPath.tmp"
    [System.IO.File]::WriteAllText(
        $tempPath,
        ($payload | ConvertTo-Json -Depth 4),
        [System.Text.Encoding]::UTF8
    )
    Move-Item -LiteralPath $tempPath -Destination $statusPath -Force
}

function Get-Progress {
    if (-not (Test-Path -LiteralPath $progressPath)) {
        return $null
    }
    return Get-Content -LiteralPath $progressPath -Raw | ConvertFrom-Json
}

function Get-CompletedGames {
    $progress = Get-Progress
    if ($null -eq $progress) {
        return 0
    }
    return [int]$progress.completed_games
}

function Test-RunComplete {
    param($Progress)

    if ($null -ne $Progress -and `
        $Progress.status -eq "complete" -and `
        $Progress.stop_reason -eq "rating_target_reached") {
        return $true
    }
    return -not $ratingStopEnabled -and `
        $null -ne $Progress -and `
        [int]$Progress.completed_games -ge $TargetGames
}

$existingProcess = if ($WaitForProcessId -gt 0) {
    Get-Process -Id $WaitForProcessId -ErrorAction SilentlyContinue
} else {
    $null
}
if ($null -ne $existingProcess) {
    $completed = Get-CompletedGames
    Write-ContinuationStatus -State "waiting_for_existing_batch" -CompletedGames $completed
    Write-SupervisorLog "Waiting for existing batch PID $WaitForProcessId at $completed/$TargetGames"
    Wait-Process -Id $WaitForProcessId
}

while ($true) {
    $progress = Get-Progress
    $completed = if ($null -eq $progress) { 0 } else { [int]$progress.completed_games }
    if (Test-RunComplete -Progress $progress) {
        Write-ContinuationStatus -State "complete" -CompletedGames $completed
        Write-SupervisorLog "Stop condition reached at completed_games=$completed"
        break
    }

    $stamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $stdoutPath = Join-Path $resolvedRunDir "stdout.resume_$stamp.log"
    $stderrPath = Join-Path $resolvedRunDir "stderr.resume_$stamp.log"
    $arguments = @(
        "-u",
        "-m",
        "integrations.riichilab.run_mortal_bot",
        "--checkpoint", $resolvedCheckpoint,
        "--mode", "ranked",
        "--device", $Device,
        "--name", $BotName,
        "--games", [string]$TargetGames,
        "--output-dir", $resolvedRunDir,
        "--max-consecutive-errors", [string]$MaxConsecutiveErrors,
        "--retry-min-seconds", "5",
        "--retry-max-seconds", "300",
        "--resume"
    )
    if (-not [string]::IsNullOrWhiteSpace($StopPolicyJson)) {
        $arguments += @("--stop-policy-json", $resolvedStopPolicyPath)
    } elseif ($StopRatingAt -gt 0) {
        $arguments += @(
            "--stop-rating-at", [string]$StopRatingAt,
            "--rating-bot-id", [string]$RatingBotId,
            "--rating-api-base", $RatingApiBase
        )
        if ($RatingActivationTotalGames -ge 0) {
            $arguments += @(
                "--rating-activation-total-games",
                [string]$RatingActivationTotalGames
            )
        }
    }
    if ($physicalEnabled) {
        $arguments += @(
            "--physical-direct-ip", $PhysicalDirectIp,
            "--physical-interface-index", [string]$PhysicalInterfaceIndex
        )
        if (-not [string]::IsNullOrWhiteSpace($PhysicalSourceAddress)) {
            $arguments += @("--physical-source-address", $PhysicalSourceAddress)
        }
    } elseif (-not [string]::IsNullOrWhiteSpace($ProxyUrl)) {
        $arguments += @("--proxy-url", $ProxyUrl)
    } elseif ($NoProxy) {
        $arguments += "--no-proxy"
    }

    $startParams = @{
        FilePath = $pythonExe
        ArgumentList = $arguments
        WorkingDirectory = $repoRoot
        WindowStyle = "Hidden"
        RedirectStandardOutput = $stdoutPath
        RedirectStandardError = $stderrPath
        PassThru = $true
    }
    $child = Start-Process @startParams
    $statusParams = @{
        State = "running_continuation"
        CompletedGames = $completed
        ChildProcessId = $child.Id
    }
    Write-ContinuationStatus @statusParams
    Write-SupervisorLog "Started continuation PID $($child.Id) at $completed/$TargetGames"
    Wait-Process -Id $child.Id

    $progress = Get-Progress
    $completed = if ($null -eq $progress) { 0 } else { [int]$progress.completed_games }
    if (Test-RunComplete -Progress $progress) {
        continue
    }
    $statusParams = @{
        State = "waiting_to_restart"
        CompletedGames = $completed
        Message = "Continuation exited before reaching target"
    }
    Write-ContinuationStatus @statusParams
    Write-SupervisorLog "Continuation exited at $completed/$TargetGames; restarting after $RestartDelaySeconds seconds"
    Start-Sleep -Seconds $RestartDelaySeconds
}
