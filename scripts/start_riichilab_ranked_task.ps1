[CmdletBinding()]
param(
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
    [string]$TaskName = "MahjongAI-RiichiLab-S70-2000",
    [int]$ConnectTimeoutSeconds = 120,
    [int]$WaitForProcessId = 0,
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
if ($StopRatingAt -gt 0 -and $RatingBotId -le 0) {
    throw "StopRatingAt requires a positive RatingBotId"
}

$repoRoot = Split-Path -Parent $PSScriptRoot
$resolvedStopPolicyJson = $null
$effectiveStopRatingAt = if ($StopRatingAt -gt 0) { $StopRatingAt } else { $null }
if (-not [string]::IsNullOrWhiteSpace($StopPolicyJson)) {
    $resolvedStopPolicyJson = if ([System.IO.Path]::IsPathRooted($StopPolicyJson)) {
        [System.IO.Path]::GetFullPath($StopPolicyJson)
    } else {
        [System.IO.Path]::GetFullPath((Join-Path $repoRoot $StopPolicyJson))
    }
    if (-not (Test-Path -LiteralPath $resolvedStopPolicyJson)) {
        throw "Stop policy does not exist: $resolvedStopPolicyJson"
    }
    $effectiveStopRatingAt = [double](
        Get-Content -LiteralPath $resolvedStopPolicyJson -Raw |
            ConvertFrom-Json |
            Select-Object -ExpandProperty stop_rating_at
    )
}
$workerScript = Join-Path $PSScriptRoot "run_riichilab_ranked_from_pipe.ps1"
$powerShellExe = (Get-Process -Id $PID).Path
$pipeName = "MahjongAI_RiichiLab_{0}" -f ([guid]::NewGuid().ToString("N"))
$pipeOptions = [System.IO.Pipes.PipeOptions]::Asynchronous -bor `
    [System.IO.Pipes.PipeOptions]::CurrentUserOnly
$pipe = [System.IO.Pipes.NamedPipeServerStream]::new(
    $pipeName,
    [System.IO.Pipes.PipeDirection]::Out,
    1,
    [System.IO.Pipes.PipeTransmissionMode]::Byte,
    $pipeOptions
)
$writer = $null
$registered = $false
$tokenHandedOff = $false

function ConvertTo-CommandLineArgument {
    param([string]$Value)

    return '"{0}"' -f $Value.Replace('"', '\"')
}

try {
    $existingTask = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    if ($null -ne $existingTask -and $existingTask.State -eq "Running") {
        throw "Scheduled task '$TaskName' is already running"
    }

    $argumentParts = @(
        "-NoProfile",
        "-NonInteractive",
        "-ExecutionPolicy", "Bypass",
        "-File", (ConvertTo-CommandLineArgument $workerScript),
        "-PipeName", (ConvertTo-CommandLineArgument $pipeName),
        "-RunDir", (ConvertTo-CommandLineArgument $RunDir),
        "-Checkpoint", (ConvertTo-CommandLineArgument $Checkpoint),
        "-TargetGames", [string]$TargetGames,
        "-Device", (ConvertTo-CommandLineArgument $Device),
        "-BotName", (ConvertTo-CommandLineArgument $BotName),
        "-WaitForProcessId", [string]$WaitForProcessId
    )
    if (-not [string]::IsNullOrWhiteSpace($StopPolicyJson)) {
        $argumentParts += @(
            "-StopPolicyJson",
            (ConvertTo-CommandLineArgument $resolvedStopPolicyJson)
        )
    } elseif ($StopRatingAt -gt 0) {
        $argumentParts += @(
            "-StopRatingAt", [string]$StopRatingAt,
            "-RatingBotId", [string]$RatingBotId,
            "-RatingActivationTotalGames", [string]$RatingActivationTotalGames,
            "-RatingApiBase", (ConvertTo-CommandLineArgument $RatingApiBase)
        )
    }
    if (-not [string]::IsNullOrWhiteSpace($PhysicalDirectIp)) {
        $argumentParts += @(
            "-PhysicalDirectIp", (ConvertTo-CommandLineArgument $PhysicalDirectIp),
            "-PhysicalInterfaceIndex", [string]$PhysicalInterfaceIndex
        )
        if (-not [string]::IsNullOrWhiteSpace($PhysicalSourceAddress)) {
            $argumentParts += @(
                "-PhysicalSourceAddress",
                (ConvertTo-CommandLineArgument $PhysicalSourceAddress)
            )
        }
    } elseif (-not [string]::IsNullOrWhiteSpace($ProxyUrl)) {
        $argumentParts += @("-ProxyUrl", (ConvertTo-CommandLineArgument $ProxyUrl))
    } elseif ($NoProxy) {
        $argumentParts += "-NoProxy"
    }
    $action = New-ScheduledTaskAction `
        -Execute $powerShellExe `
        -Argument ($argumentParts -join " ") `
        -WorkingDirectory $repoRoot
    $trigger = New-ScheduledTaskTrigger -Once -At ([datetime]"2099-01-01T00:00:00")
    $settings = New-ScheduledTaskSettingsSet `
        -ExecutionTimeLimit ([TimeSpan]::Zero) `
        -AllowStartIfOnBatteries `
        -DontStopIfGoingOnBatteries `
        -StartWhenAvailable `
        -MultipleInstances IgnoreNew
    $principal = New-ScheduledTaskPrincipal `
        -UserId ([System.Security.Principal.WindowsIdentity]::GetCurrent().Name) `
        -LogonType Interactive `
        -RunLevel Limited
    $task = New-ScheduledTask `
        -Action $action `
        -Trigger $trigger `
        -Settings $settings `
        -Principal $principal
    Register-ScheduledTask -TaskName $TaskName -InputObject $task -Force | Out-Null
    $registered = $true
    Start-ScheduledTask -TaskName $TaskName

    $connection = $pipe.WaitForConnectionAsync()
    if (-not $connection.Wait($ConnectTimeoutSeconds * 1000)) {
        throw "Scheduled runner did not connect to the token handoff pipe"
    }
    $connection.GetAwaiter().GetResult()

    $writer = [System.IO.StreamWriter]::new(
        $pipe,
        [System.Text.Encoding]::UTF8,
        4096,
        $true
    )
    $writer.AutoFlush = $true
    $writer.WriteLine($env:RIICHILAB_BOT_TOKEN)
    $tokenHandedOff = $true

    [pscustomobject]@{
        task_name = $TaskName
        task_state = (Get-ScheduledTask -TaskName $TaskName).State.ToString()
        target_games = if ($StopRatingAt -gt 0 -or `
            -not [string]::IsNullOrWhiteSpace($StopPolicyJson)) { $null } else { $TargetGames }
        stop_rating_at = $effectiveStopRatingAt
        token_persisted = $false
    }
} finally {
    Remove-Item Env:RIICHILAB_BOT_TOKEN -ErrorAction SilentlyContinue
    if ($null -ne $writer) {
        $writer.Dispose()
    }
    $pipe.Dispose()

    if ($registered -and -not $tokenHandedOff) {
        Stop-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
    }
}
