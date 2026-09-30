param([Parameter(Mandatory = $true)][string]$Spec)
$ErrorActionPreference = 'Stop'
$settings = [IO.File]::ReadAllText((Resolve-Path -LiteralPath $Spec)) | ConvertFrom-Json
$repoRoot = [IO.Path]::GetFullPath([string]$settings.repo_root)
$logRoot = [IO.Path]::GetFullPath([string]$settings.log_root)
if (-not $logRoot.StartsWith($repoRoot + '\', [StringComparison]::OrdinalIgnoreCase)) {
    throw 'Diagnostic logs must be inside the isolated checkout'
}
if ($settings.timeout_seconds -lt 60 -or $settings.timeout_seconds -gt 1800) {
    throw 'Diagnostic timeout must be between 60 and 1800 seconds'
}
if ($settings.minimum_free_ram_gib -lt 4) { throw 'Insufficient system RAM guard' }
New-Item -ItemType Directory -Path $logRoot -Force | Out-Null
$receiptPath = Join-Path $logRoot ($settings.name + '.resources.json')
$startPath = Join-Path $logRoot ($settings.name + '.started.json')
if ((Test-Path -LiteralPath $receiptPath) -or (Test-Path -LiteralPath $startPath)) {
    throw 'Diagnostic name is already used; previous evidence is immutable'
}
$utf8 = New-Object Text.UTF8Encoding($false)
function Write-Json($path, $value) {
    $temporary = $path + '.tmp'
    [IO.File]::WriteAllText($temporary, ($value | ConvertTo-Json -Depth 4), $utf8)
    Move-Item -LiteralPath $temporary -Destination $path -Force
}
Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
public static class PreparationMemory {
    [StructLayout(LayoutKind.Sequential)] public class Status {
        public uint length = (uint)Marshal.SizeOf(typeof(Status));
        public uint load;
        public ulong total, available, totalPage, availablePage, totalVirtual, availableVirtual, extended;
    }
    [DllImport("kernel32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    public static extern bool GlobalMemoryStatusEx([In, Out] Status status);
}
'@
$pausePath = Join-Path $logRoot ($settings.name + '.pause.request')
$env:MORTAL_ORACLE_PAUSE_FILE = $pausePath
$arguments = @([string]$settings.script) + @($settings.arguments | ForEach-Object { [string]$_ })
if (@($arguments | Where-Object { $_ -match '[\s"]' }).Count) {
    throw 'This benchmark launcher requires already validated arguments without spaces or quotes'
}
$process = Start-Process -FilePath ([string]$settings.python) -ArgumentList $arguments `
    -WorkingDirectory $repoRoot -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput (Join-Path $logRoot ($settings.name + '.stdout.log')) `
    -RedirectStandardError (Join-Path $logRoot ($settings.name + '.stderr.log'))
# Retain the handle before waiting; PowerShell 5 otherwise sometimes loses
# ExitCode when a short-lived Start-Process child exits before property access.
$processHandle = $process.Handle
$created = $process.StartTime
$started = Get-Date
Write-Json $startPath ([ordered]@{at=$started.ToString('o');pid=$process.Id;created=$created.ToString('o');spec=$Spec})
$samples = New-Object System.Collections.Generic.List[object]
$guardReason = $null
$guardAt = $null
$killed = $false
while (-not $process.WaitForExit(1000)) {
    $memory = New-Object PreparationMemory+Status
    if (-not [PreparationMemory]::GlobalMemoryStatusEx($memory)) { throw 'System memory sample failed' }
    $elapsed = ((Get-Date) - $started).TotalSeconds
    $pythonRows = @(Get-Process python, pythonw -ErrorAction SilentlyContinue | ForEach-Object {
        [ordered]@{pid=$_.Id;cpu_s=$_.CPU;rss_bytes=$_.WorkingSet64;private_bytes=$_.PrivateMemorySize64;
            peak_rss_bytes=$_.PeakWorkingSet64;session=$_.SessionId}
    })
    $gpu = [string](& nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader,nounits)
    $samples.Add([ordered]@{elapsed_s=$elapsed;available_ram_bytes=$memory.available;
        available_commit_bytes=$memory.availablePage;gpu=$gpu;python=$pythonRows})
    if (-not $guardReason) {
        if ($memory.available -lt $settings.minimum_free_ram_gib * 1GB) { $guardReason='system_ram_floor' }
        elseif ($elapsed -gt $settings.timeout_seconds) { $guardReason='bounded_diagnostic_timeout' }
        if ($guardReason) {
            $guardAt = Get-Date
            [IO.File]::WriteAllText($pausePath, $guardReason, $utf8)
        }
    }
    if ($guardReason -and ((Get-Date) - $guardAt).TotalSeconds -gt 45) {
        $current = Get-Process -Id $process.Id -ErrorAction SilentlyContinue
        if ($current -and $current.StartTime -eq $created -and $current.Path -eq [string]$settings.python) {
            & taskkill /PID $process.Id /T /F | Out-Null
            $killed = $true
        }
        $process.WaitForExit()
        break
    }
}
$process.WaitForExit()
$code = $process.ExitCode
$summary = [ordered]@{at=(Get-Date).ToString('o');spec=$Spec;pid=$process.Id;
    started=$created.ToString('o');exit_code=$code;guard_reason=$guardReason;forced_owned_tree_stop=$killed;
    wall_seconds=((Get-Date)-$started).TotalSeconds;sample_count=$samples.Count;samples=$samples.ToArray();
    note='System free RAM includes all co-load and shared mappings; summed process RSS would double count shared buffers.'}
Write-Json $receiptPath $summary
Write-Output ($summary | Select-Object at,pid,exit_code,guard_reason,wall_seconds,sample_count | ConvertTo-Json -Compress)
if ($null -eq $code -or $code -ne 0) { exit 1 }
