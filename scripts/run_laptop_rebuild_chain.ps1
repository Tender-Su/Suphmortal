param(
    [string]$HostIp = '192.168.1.8',
    [string]$User = 'numbe',
    [string]$RepoRoot = 'C:\Users\numbe\Desktop\MahjongAI',
    [string]$RemoteDataRoot = 'C:\Users\numbe\mahjong_data_root',
    [string]$CondaPython = 'C:\Users\numbe\miniconda3\envs\mortal\python.exe',
    [string]$LogPath = '',
    [int]$ExtractWorkers = 10,
    [int]$DecompressWorkers = 0,
    [switch]$ResumeExisting
)

$ErrorActionPreference = 'Stop'

if (-not $LogPath) {
    $stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
    $LogPath = Join-Path $RepoRoot "logs\laptop_rebuild_chain_$stamp.log"
}

$sshKey = Join-Path $HOME '.ssh\mahjong_laptop_ed25519'
$remoteRepo = Join-Path $RepoRoot ''
$remoteDataset = Join-Path $RemoteDataRoot 'dataset_rebuilt'
$remoteDatasetJson = Join-Path $RemoteDataRoot 'dataset_json_rebuilt'
$remoteStamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$remoteBase = Join-Path $remoteRepo "logs\laptop_rebuild_remote_$remoteStamp.log"
$remoteRunner = "$remoteBase.runner.ps1"
$remoteOutLog = "$remoteBase.out.log"
$remoteErrLog = "$remoteBase.err.log"

function Write-Log {
    param([string]$Message)
    $line = "[{0}] {1}" -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $Message
    Add-Content -Path $LogPath -Value $line
    Write-Output $line
}

function Invoke-RemoteScript {
    param([string]$ScriptText)
    $encoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($ScriptText))
    & ssh -i $sshKey "$User@$HostIp" "powershell -NoProfile -EncodedCommand $encoded" 2>&1 | Tee-Object -FilePath $LogPath -Append
    if ($LASTEXITCODE -ne 0) {
        throw "remote command failed"
    }
}

Write-Log 'REBUILD_CHAIN_START'
Write-Log "LOG_PATH=$LogPath"
Write-Log "REMOTE_DATASET=$remoteDataset"
Write-Log "REMOTE_DATASET_JSON=$remoteDatasetJson"
Write-Log "REMOTE_RUNNER=$remoteRunner"
Write-Log "REMOTE_OUT_LOG=$remoteOutLog"
Write-Log "REMOTE_ERR_LOG=$remoteErrLog"
Write-Log "EXTRACT_WORKERS=$ExtractWorkers"
Write-Log "RESUME_EXISTING=$ResumeExisting"

$resolvedDecompressWorkers = $DecompressWorkers
if ($resolvedDecompressWorkers -le 0) {
    $resolvedDecompressWorkers = (Get-ChildItem -Path $RemoteDataRoot -Filter '*.zip' -File -ErrorAction SilentlyContinue | Measure-Object).Count
    if ($resolvedDecompressWorkers -le 0) {
        $resolvedDecompressWorkers = 18
    }
}
Write-Log "DECOMPRESS_WORKERS=$resolvedDecompressWorkers"

$cleanupScript = @"
`$ErrorActionPreference = 'Stop'
`$allProcesses = @(Get-CimInstance Win32_Process)
`$rootProcessIds = @(`$allProcesses | Where-Object {
    (`$_.Name -in @('python.exe', 'powershell.exe')) -and (
        (`$_.CommandLine -like '*extract_data.py*') -or
        (`$_.CommandLine -like '*decompress_dataset_json.py*') -or
        (`$_.CommandLine -like '*laptop_rebuild_remote_*')
    )
} | Select-Object -ExpandProperty ProcessId)
`$pending = New-Object System.Collections.Generic.Queue[int]
`$toStop = New-Object System.Collections.Generic.HashSet[int]
foreach (`$rootId in `$rootProcessIds) {
    [void]`$pending.Enqueue([int]`$rootId)
}
while (`$pending.Count -gt 0) {
    `$current = `$pending.Dequeue()
    if (-not `$toStop.Add(`$current)) {
        continue
    }
    foreach (`$child in `$allProcesses | Where-Object { `$_.ParentProcessId -eq `$current }) {
        [void]`$pending.Enqueue([int]`$child.ProcessId)
    }
}
`$toStop | Sort-Object -Descending | ForEach-Object {
    Stop-Process -Id `$_ -Force -ErrorAction SilentlyContinue
}
function Quarantine-IfExists([string]`$targetPath) {
    if (-not (Test-Path `$targetPath)) { return }
    `$parent = Split-Path -Parent `$targetPath
    `$leaf = Split-Path -Leaf `$targetPath
    `$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
    `$quarantine = Join-Path `$parent ("`$leaf" + "_stale_" + `$stamp)
    Move-Item -LiteralPath `$targetPath -Destination `$quarantine
}
`$resumeExisting = [System.Convert]::ToBoolean('$ResumeExisting')
if (-not `$resumeExisting) {
    Quarantine-IfExists '$remoteDataset'
    Quarantine-IfExists '$remoteDatasetJson'
}
New-Item -ItemType Directory -Path '$remoteDataset' -Force | Out-Null
New-Item -ItemType Directory -Path '$remoteDatasetJson' -Force | Out-Null
Write-Host 'REMOTE_REBUILT_PATHS_READY'
"@
Write-Log 'REMOTE_CLEANUP_BEGIN'
Invoke-RemoteScript $cleanupScript
Write-Log 'REMOTE_CLEANUP_DONE'

$runnerBody = @"
`$ErrorActionPreference = 'Stop'
& '$CondaPython' '$remoteRepo\scripts\extract_data.py' --src-root '$RemoteDataRoot' --dst-root '$remoteDataset' --workers $ExtractWorkers --report-every 1
& '$CondaPython' '$remoteRepo\scripts\decompress_dataset_json.py' --src-root '$remoteDataset' --dst-root '$remoteDatasetJson' --workers $resolvedDecompressWorkers --report-every 1000
"@

$launchScript = @"
`$ErrorActionPreference = 'Stop'
`$runnerBody = @'
$runnerBody
'@
Set-Content -Path '$remoteRunner' -Value `$runnerBody -Encoding UTF8
if (Test-Path '$remoteOutLog') { Remove-Item -LiteralPath '$remoteOutLog' -Force }
if (Test-Path '$remoteErrLog') { Remove-Item -LiteralPath '$remoteErrLog' -Force }
`$runnerEsc = '$remoteRunner'.Replace('"','""')
`$outEsc = '$remoteOutLog'.Replace('"','""')
`$errEsc = '$remoteErrLog'.Replace('"','""')
`$cmd = 'cmd /c start "" /b powershell -NoProfile -ExecutionPolicy Bypass -File "' + `$runnerEsc + '" 1>"' + `$outEsc + '" 2>"' + `$errEsc + '"'
Invoke-Expression `$cmd
Start-Sleep -Seconds 1
`$p = Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" | Where-Object { `$_.CommandLine -like ('*' + '$remoteRunner' + '*') } | Sort-Object CreationDate -Descending | Select-Object -First 1
if (`$null -eq `$p) { throw 'failed to locate remote rebuild runner process' }
Write-Host ('REMOTE_CHAIN_PID=' + `$p.ProcessId)
Write-Host ('REMOTE_CHAIN_RUNNER=' + '$remoteRunner')
Write-Host ('REMOTE_CHAIN_OUT_LOG=' + '$remoteOutLog')
Write-Host ('REMOTE_CHAIN_ERR_LOG=' + '$remoteErrLog')
"@
Write-Log 'REMOTE_CHAIN_LAUNCH_BEGIN'
Invoke-RemoteScript $launchScript
Write-Log 'REMOTE_CHAIN_LAUNCH_DONE'

$probeScript = @"
`$ErrorActionPreference = 'Stop'
Start-Sleep -Seconds 8
if (Test-Path '$remoteOutLog') {
    Get-Content '$remoteOutLog' -Tail 20
} else {
    Write-Host 'REMOTE_OUT_LOG_NOT_CREATED'
}
"@
Write-Log 'REMOTE_CHAIN_PROBE_BEGIN'
Invoke-RemoteScript $probeScript
Write-Log 'REMOTE_CHAIN_PROBE_DONE'

Write-Log 'REBUILD_CHAIN_STARTED'
