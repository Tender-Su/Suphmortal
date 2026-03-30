param(
    [string]$HostIp = '192.168.1.8',
    [string]$User = 'numbe',
    [string]$RepoRoot = 'C:\Users\numbe\Desktop\MahjongAI',
    [string]$RemoteDataRoot = 'C:\Users\numbe\mahjong_data_root',
    [string]$CondaPython = 'C:\Users\numbe\miniconda3\envs\mortal\python.exe',
    [int]$Workers = 18
)

$ErrorActionPreference = 'Stop'

$sshKey = Join-Path $HOME '.ssh\mahjong_laptop_ed25519'

function Invoke-RemoteScript {
    param([string]$ScriptText)
    $ScriptText | & ssh -i $sshKey "$User@$HostIp" "powershell -NoProfile -Command -"
    if ($LASTEXITCODE -ne 0) {
        throw 'remote command failed'
    }
}

$remoteScript = @"
`$ErrorActionPreference = 'Stop'
`$repo = '$RepoRoot'
`$root = '$RemoteDataRoot'
`$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
`$cleanupRunner = Join-Path `$repo ("logs\laptop_cleanup_`$stamp.runner.ps1")
`$cleanupOut = `$cleanupRunner.Replace('.runner.ps1', '.out.log')
`$cleanupErr = `$cleanupRunner.Replace('.runner.ps1', '.err.log')
`$decompRunner = Join-Path `$repo ("logs\laptop_decompress_`$stamp.runner.ps1")
`$decompOut = `$decompRunner.Replace('.runner.ps1', '.out.log')
`$decompErr = `$decompRunner.Replace('.runner.ps1', '.err.log')

`$cleanupBody = @'
`$ErrorActionPreference = 'Stop'
`$root = '$RemoteDataRoot'
`$keep = @(
  (Join-Path `$root 'dataset_rebuilt'),
  (Join-Path `$root 'dataset_json_rebuilt')
)
function Resolve-SafePath([string]`$path) {
  `$resolved = (Resolve-Path -LiteralPath `$path).ProviderPath
  if (-not `$resolved.StartsWith(`$root, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "refusing to touch path outside root: `$resolved"
  }
  return `$resolved
}
Write-Host 'CLEANUP_BEGIN'
`$candidates = @(
  (Join-Path `$root 'dataset'),
  (Join-Path `$root 'dataset_bad_20260330'),
  (Join-Path `$root 'dataset_json')
)
`$candidates += @(Get-ChildItem `$root -Directory -ErrorAction SilentlyContinue | Where-Object {
  `$_.Name -like 'dataset_rebuilt_stale_*' -or `$_.Name -like 'dataset_json_rebuilt_stale_*'
} | Select-Object -ExpandProperty FullName)
foreach (`$candidate in `$candidates) {
  if (-not (Test-Path -LiteralPath `$candidate)) { continue }
  `$resolved = Resolve-SafePath `$candidate
  if (`$keep -contains `$resolved) { throw "refusing to delete keep dir: `$resolved" }
  Write-Host ('DELETE ' + `$resolved)
  Remove-Item -LiteralPath `$resolved -Recurse -Force
  Write-Host ('DELETED ' + `$resolved)
}
Write-Host 'CLEANUP_DONE'
'@

`$decompBody = @'
`$ErrorActionPreference = 'Stop'
Get-CimInstance Win32_Process | Where-Object {
  (`$_.Name -in @('python.exe', 'powershell.exe')) -and (
    (`$_.CommandLine -like '*extract_data.py*') -or (`$_.CommandLine -like '*decompress_dataset_json.py*') -or (`$_.CommandLine -like '*laptop_rebuild_remote_*')
  )
} | ForEach-Object {
  Stop-Process -Id `$_.ProcessId -Force -ErrorAction SilentlyContinue
}
Write-Host 'DECOMPRESS_BEGIN'
& '$CondaPython' '$RepoRoot\scripts\decompress_dataset_json.py' --src-root '$RemoteDataRoot\dataset_rebuilt' --dst-root '$RemoteDataRoot\dataset_json_rebuilt' --workers $Workers --report-every 1000
'@

Set-Content -Path `$cleanupRunner -Value `$cleanupBody -Encoding UTF8
Set-Content -Path `$decompRunner -Value `$decompBody -Encoding UTF8
foreach (`$p in @(`$cleanupOut, `$cleanupErr, `$decompOut, `$decompErr)) {
  if (Test-Path `$p) { Remove-Item -LiteralPath `$p -Force }
}
`$cleanupProc = Start-Process powershell -ArgumentList @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', `$cleanupRunner) -RedirectStandardOutput `$cleanupOut -RedirectStandardError `$cleanupErr -WindowStyle Hidden -PassThru
`$decompProc = Start-Process powershell -ArgumentList @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', `$decompRunner) -RedirectStandardOutput `$decompOut -RedirectStandardError `$decompErr -WindowStyle Hidden -PassThru
Start-Sleep -Seconds 2
Write-Output ('CLEANUP_RUNNER=' + `$cleanupRunner)
Write-Output ('CLEANUP_OUT=' + `$cleanupOut)
Write-Output ('CLEANUP_ERR=' + `$cleanupErr)
Write-Output ('CLEANUP_PID=' + `$cleanupProc.Id)
Write-Output ('DECOMP_RUNNER=' + `$decompRunner)
Write-Output ('DECOMP_OUT=' + `$decompOut)
Write-Output ('DECOMP_ERR=' + `$decompErr)
Write-Output ('DECOMP_PID=' + `$decompProc.Id)
if (Test-Path `$cleanupOut) { Write-Output '---CLEANUP_OUT---'; Get-Content `$cleanupOut -Tail 20 }
if (Test-Path `$decompOut) { Write-Output '---DECOMP_OUT---'; Get-Content `$decompOut -Tail 20 }
"@

Invoke-RemoteScript $remoteScript
