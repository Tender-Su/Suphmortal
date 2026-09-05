[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$SpecPath,

    [string]$WatchedProcessName = "r5apex_dx12",
    [int]$PollSeconds = 2,
    [int]$ResumeDelaySeconds = 5,
    [int]$PauseTimeoutSeconds = 90,
    [int]$RestartDelaySeconds = 20,
    [int]$MaxUnexpectedRestarts = 3
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$PauseEnvironmentVariable = "MORTAL_ORACLE_PAUSE_FILE"
$PauseExitCode = 75

function Resolve-SpecPath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Value,
        [Parameter(Mandatory = $true)]
        [string]$BasePath
    )

    if ([IO.Path]::IsPathRooted($Value)) {
        return [IO.Path]::GetFullPath($Value)
    }
    return [IO.Path]::GetFullPath((Join-Path $BasePath $Value))
}

function Test-PathWithinRoot {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Candidate,
        [Parameter(Mandatory = $true)]
        [string]$Root
    )

    $rootPrefix = [IO.Path]::GetFullPath($Root).TrimEnd("\") + "\"
    return [IO.Path]::GetFullPath($Candidate).StartsWith(
        $rootPrefix,
        [StringComparison]::OrdinalIgnoreCase
    )
}

$resolvedSpec = (Resolve-Path -LiteralPath $SpecPath).Path
$specBase = Split-Path -Parent $resolvedSpec
$spec = Get-Content -LiteralPath $resolvedSpec -Raw | ConvertFrom-Json

$repoRoot = Resolve-SpecPath -Value ([string]$spec.repo_root) -BasePath $specBase
$searchRoot = Resolve-SpecPath -Value ([string]$spec.search_root) -BasePath $specBase
$pythonExe = Resolve-SpecPath -Value ([string]$spec.python_executable) -BasePath $specBase
$pauseFile = Resolve-SpecPath -Value ([string]$spec.pause_file) -BasePath $specBase
$statusFile = Resolve-SpecPath -Value ([string]$spec.status_file) -BasePath $specBase
$supervisorLog = Resolve-SpecPath -Value ([string]$spec.log_file) -BasePath $specBase
$runnerArguments = @($spec.runner_arguments | ForEach-Object { [string]$_ })

if (-not (Test-Path -LiteralPath $repoRoot -PathType Container)) {
    throw "repo_root does not exist: $repoRoot"
}
if (-not (Test-Path -LiteralPath $searchRoot -PathType Container)) {
    throw "search_root does not exist: $searchRoot"
}
if (-not (Test-Path -LiteralPath $pythonExe -PathType Leaf)) {
    throw "python_executable does not exist: $pythonExe"
}
foreach ($managedPath in @($pauseFile, $statusFile, $supervisorLog)) {
    if (-not (Test-PathWithinRoot -Candidate $managedPath -Root $searchRoot)) {
        throw "managed path escapes search_root: $managedPath"
    }
}
if ($runnerArguments.Count -eq 0) {
    throw "runner_arguments must not be empty"
}

foreach ($directory in @(
    (Split-Path -Parent $pauseFile),
    (Split-Path -Parent $statusFile),
    (Split-Path -Parent $supervisorLog)
)) {
    [IO.Directory]::CreateDirectory($directory) | Out-Null
}

$lockPath = Join-Path $searchRoot "apex_supervisor.lock"
try {
    $lockStream = [IO.File]::Open(
        $lockPath,
        [IO.FileMode]::OpenOrCreate,
        [IO.FileAccess]::ReadWrite,
        [IO.FileShare]::None
    )
}
catch {
    throw "another Apex supervisor already holds $lockPath"
}
$lockBytes = [Text.Encoding]::UTF8.GetBytes("$PID`n")
$lockStream.SetLength(0)
$lockStream.Write($lockBytes, 0, $lockBytes.Length)
$lockStream.Flush()

function Write-SupervisorLog {
    param([Parameter(Mandatory = $true)][string]$Message)

    $line = "{0:o} {1}{2}" -f (Get-Date), $Message, [Environment]::NewLine
    [IO.File]::AppendAllText(
        $supervisorLog,
        $line,
        [Text.UTF8Encoding]::new($false)
    )
}

function Move-AtomicFile {
    param(
        [Parameter(Mandatory = $true)][string]$Temporary,
        [Parameter(Mandatory = $true)][string]$Target
    )

    if (Test-Path -LiteralPath $Target) {
        $backup = "$Target.$PID.replace.bak"
        try {
            [IO.File]::Replace($Temporary, $Target, $backup)
        }
        finally {
            if (Test-Path -LiteralPath $backup) {
                Remove-Item -LiteralPath $backup -Force
            }
        }
    }
    else {
        [IO.File]::Move($Temporary, $Target)
    }
}

function Write-SupervisorStatus {
    param(
        [Parameter(Mandatory = $true)][string]$State,
        [bool]$ApexRunning,
        [Nullable[int]]$RunnerPid = $null,
        [Nullable[int]]$RunnerExitCode = $null,
        [string]$RunnerOut = "",
        [string]$RunnerErr = "",
        [string]$Detail = ""
    )

    $payload = [ordered]@{
        format = "oracle_critic_apex_supervisor_status_v1"
        updated_at = (Get-Date).ToString("o")
        supervisor_pid = $PID
        state = $State
        watched_process = $WatchedProcessName
        apex_running = $ApexRunning
        runner_pid = $RunnerPid
        runner_exit_code = $RunnerExitCode
        pause_file = $pauseFile
        runner_out = $RunnerOut
        runner_err = $RunnerErr
        detail = $Detail
    }
    $temporary = "$statusFile.$PID.tmp"
    [IO.File]::WriteAllText(
        $temporary,
        ($payload | ConvertTo-Json -Depth 4),
        [Text.UTF8Encoding]::new($false)
    )
    Move-AtomicFile -Temporary $temporary -Target $statusFile
}

function Test-ApexRunning {
    return $null -ne (Get-Process -Name $WatchedProcessName -ErrorAction SilentlyContinue)
}

function Write-PauseRequest {
    $payload = [ordered]@{
        format = "oracle_critic_external_pause_v1"
        requested_at = (Get-Date).ToString("o")
        requested_by_pid = $PID
        watched_process = $WatchedProcessName
    } | ConvertTo-Json -Compress
    $temporary = "$pauseFile.$PID.tmp"
    [IO.File]::WriteAllText(
        $temporary,
        $payload,
        [Text.UTF8Encoding]::new($false)
    )
    Move-AtomicFile -Temporary $temporary -Target $pauseFile
}

function Remove-PauseRequest {
    if (Test-Path -LiteralPath $pauseFile) {
        Remove-Item -LiteralPath $pauseFile -Force
    }
}

function Start-ManagedRunner {
    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $stdoutPath = Join-Path $searchRoot "launcher_apex_$timestamp.out.log"
    $stderrPath = Join-Path $searchRoot "launcher_apex_$timestamp.err.log"
    $previousPauseFile = [Environment]::GetEnvironmentVariable(
        $PauseEnvironmentVariable,
        "Process"
    )
    [Environment]::SetEnvironmentVariable(
        $PauseEnvironmentVariable,
        $pauseFile,
        "Process"
    )
    try {
        $process = Start-Process `
            -FilePath $pythonExe `
            -ArgumentList $runnerArguments `
            -WorkingDirectory $repoRoot `
            -RedirectStandardOutput $stdoutPath `
            -RedirectStandardError $stderrPath `
            -WindowStyle Hidden `
            -PassThru
        # Keep the process handle so ExitCode survives short-lived runners.
        $null = $process.Handle
    }
    finally {
        [Environment]::SetEnvironmentVariable(
            $PauseEnvironmentVariable,
            $previousPauseFile,
            "Process"
        )
    }
    return [ordered]@{
        Process = $process
        Stdout = $stdoutPath
        Stderr = $stderrPath
    }
}

function Stop-ManagedRunnerTree {
    param([Parameter(Mandatory = $true)][Diagnostics.Process]$Process)

    if ($Process.HasExited) {
        return
    }
    & "$env:SystemRoot\System32\taskkill.exe" /PID $Process.Id /T /F | Out-Null
}

$runner = $null
$runnerStdout = ""
$runnerStderr = ""
$pauseRequestedAt = $null
$nextStartAt = Get-Date
$unexpectedRestarts = 0
$lastApexState = $null
$completed = $false

Write-SupervisorLog "supervisor started pid=$PID spec=$resolvedSpec"

try {
    while (-not $completed) {
        $apexRunning = Test-ApexRunning
        if ($null -eq $lastApexState -or $apexRunning -ne $lastApexState) {
            Write-SupervisorLog "Apex running=$apexRunning"
            $lastApexState = $apexRunning
        }

        if ($null -ne $runner -and $runner.HasExited) {
            $runner.WaitForExit()
            $exitCode = $runner.ExitCode
            Write-SupervisorLog "runner pid=$($runner.Id) exited code=$exitCode"
            $runner.Dispose()
            $runner = $null
            $pauseRequestedAt = $null

            if ($exitCode -eq 0) {
                Write-SupervisorStatus `
                    -State "completed" `
                    -ApexRunning $apexRunning `
                    -RunnerExitCode $exitCode `
                    -RunnerOut $runnerStdout `
                    -RunnerErr $runnerStderr `
                    -Detail "search runner completed"
                $completed = $true
                continue
            }
            if ($exitCode -eq $PauseExitCode) {
                Write-SupervisorStatus `
                    -State "paused" `
                    -ApexRunning $apexRunning `
                    -RunnerExitCode $exitCode `
                    -RunnerOut $runnerStdout `
                    -RunnerErr $runnerStderr `
                    -Detail "exact checkpoint saved after Apex launch"
                $nextStartAt = (Get-Date).AddSeconds($ResumeDelaySeconds)
            }
            else {
                $unexpectedRestarts += 1
                Write-SupervisorStatus `
                    -State "runner_error" `
                    -ApexRunning $apexRunning `
                    -RunnerExitCode $exitCode `
                    -RunnerOut $runnerStdout `
                    -RunnerErr $runnerStderr `
                    -Detail "unexpected runner exit $unexpectedRestarts/$MaxUnexpectedRestarts"
                if ($unexpectedRestarts -ge $MaxUnexpectedRestarts) {
                    throw "runner exceeded unexpected restart limit"
                }
                $nextStartAt = (Get-Date).AddSeconds($RestartDelaySeconds)
            }
        }

        if ($apexRunning) {
            if ($null -ne $runner -and $null -eq $pauseRequestedAt) {
                Write-PauseRequest
                $pauseRequestedAt = Get-Date
                Write-SupervisorLog "pause requested for runner pid=$($runner.Id)"
                Write-SupervisorStatus `
                    -State "pause_requested" `
                    -ApexRunning $true `
                    -RunnerPid $runner.Id `
                    -RunnerOut $runnerStdout `
                    -RunnerErr $runnerStderr `
                    -Detail "waiting for an atomic exact checkpoint"
            }
            elseif ($null -eq $runner) {
                Write-SupervisorStatus `
                    -State "waiting_for_apex_exit" `
                    -ApexRunning $true `
                    -RunnerOut $runnerStdout `
                    -RunnerErr $runnerStderr `
                    -Detail "training is not using GPU memory"
            }

            if (
                $null -ne $runner `
                -and $null -ne $pauseRequestedAt `
                -and -not $runner.HasExited `
                -and (Get-Date) -ge $pauseRequestedAt.AddSeconds($PauseTimeoutSeconds)
            ) {
                Write-SupervisorLog "pause timeout; terminating owned runner tree pid=$($runner.Id)"
                Stop-ManagedRunnerTree -Process $runner
            }
        }
        elseif ($null -eq $runner -and (Get-Date) -ge $nextStartAt) {
            Remove-PauseRequest
            Start-Sleep -Seconds $ResumeDelaySeconds
            if (-not (Test-ApexRunning)) {
                $started = Start-ManagedRunner
                $runner = $started.Process
                $runnerStdout = $started.Stdout
                $runnerStderr = $started.Stderr
                $pauseRequestedAt = $null
                Write-SupervisorLog "runner started pid=$($runner.Id)"
                Write-SupervisorStatus `
                    -State "running" `
                    -ApexRunning $false `
                    -RunnerPid $runner.Id `
                    -RunnerOut $runnerStdout `
                    -RunnerErr $runnerStderr `
                    -Detail "training or paired evaluation is active"
            }
        }

        Start-Sleep -Seconds $PollSeconds
    }
}
catch {
    Write-SupervisorLog "supervisor error: $($_.Exception.Message)"
    Write-SupervisorStatus `
        -State "supervisor_error" `
        -ApexRunning (Test-ApexRunning) `
        -RunnerPid $(if ($null -ne $runner -and -not $runner.HasExited) { $runner.Id } else { $null }) `
        -RunnerOut $runnerStdout `
        -RunnerErr $runnerStderr `
        -Detail $_.Exception.Message
    throw
}
finally {
    if ($null -ne $runner -and -not $runner.HasExited) {
        Write-PauseRequest
        $deadline = (Get-Date).AddSeconds($PauseTimeoutSeconds)
        while (-not $runner.HasExited -and (Get-Date) -lt $deadline) {
            Start-Sleep -Seconds 1
        }
        if (-not $runner.HasExited) {
            Stop-ManagedRunnerTree -Process $runner
        }
    }
    $lockStream.Dispose()
}
