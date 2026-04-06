param(
    [string]$RunName = "sl_fidelity_main",
    [int]$PollSeconds = 1
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$runDir = Join-Path $repoRoot ("logs\sl_fidelity\" + $RunName)
$targetJson = Join-Path $runDir "p0_round2.json"
$logPath = Join-Path $runDir "stop_after_round2.log"
$stopHelper = Join-Path $PSScriptRoot "stop_sl_fidelity.py"
$pythonExe = "C:\ProgramData\anaconda3\envs\mortal\python.exe"
if (-not (Test-Path $pythonExe)) {
    $pythonExe = "python"
}

function Write-Log {
    param([string]$Message)
    $line = "[{0}] {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $Message
    Add-Content -Path $logPath -Value $line -Encoding UTF8
}

function Stop-TargetProcesses {
    if (-not (Test-Path $stopHelper)) {
        throw "missing stop helper: $stopHelper"
    }
    $output = & $pythonExe $stopHelper --run-name $RunName 2>&1
    $exitCode = $LASTEXITCODE
    foreach ($line in @($output)) {
        Write-Log ("stop_sl_fidelity.py: {0}" -f $line)
    }
    if ($exitCode -ne 0) {
        throw "stop helper failed with exit code $exitCode"
    }
}

New-Item -ItemType Directory -Force -Path $runDir | Out-Null
$watchStart = Get-Date
if (Test-Path $targetJson) {
    $existingItem = Get-Item $targetJson
    Write-Log ("Ignoring stale existing p0_round2.json from {0}" -f $existingItem.LastWriteTime.ToString("yyyy-MM-dd HH:mm:ss"))
}
Write-Log ("Watcher started. target_json={0}; watch_start={1}" -f $targetJson, $watchStart.ToString("yyyy-MM-dd HH:mm:ss"))

while ($true) {
    if (Test-Path $targetJson) {
        $targetItem = Get-Item $targetJson
        if ($targetItem.LastWriteTime -lt $watchStart) {
            Start-Sleep -Seconds $PollSeconds
            continue
        }
        Write-Log "Detected p0_round2.json. Stopping fidelity runner before round3."
        Stop-TargetProcesses
        Write-Log "Watcher exiting."
        exit 0
    }
    Start-Sleep -Seconds $PollSeconds
}
