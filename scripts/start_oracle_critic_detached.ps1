[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$SpecPath,
    [switch]$Probe,
    [switch]$HostRun,
    [string]$LaunchId = ([guid]::NewGuid().ToString("N"))
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
if ($PSVersionTable.PSVersion.Major -lt 7) {
    throw "Use PowerShell 7 (pwsh.exe)."
}
if ($LaunchId -notmatch '^[a-f0-9]{32}$') {
    throw "Invalid launch ID."
}
$resolvedSpec = (Resolve-Path -LiteralPath $SpecPath).Path
$spec = Get-Content -LiteralPath $resolvedSpec -Raw | ConvertFrom-Json
$specBase = Split-Path -Parent $resolvedSpec
function Resolve-SpecValue([string]$Value) {
    return [IO.Path]::GetFullPath($Value, $specBase)
}
$runRoot = Resolve-SpecValue $spec.search_root
$repoRoot = Resolve-SpecValue $spec.repo_root
$supervisor = Join-Path $repoRoot "scripts/supervise_oracle_critic_around_apex.ps1"
if (-not (Test-Path -LiteralPath $runRoot -PathType Container)) {
    throw "search_root does not exist: $runRoot"
}
if (-not (Test-Path -LiteralPath $supervisor -PathType Leaf)) {
    throw "supervisor does not exist: $supervisor"
}
$receiptPath = Join-Path $runRoot "detached_launch_$LaunchId.json"
$hostLog = Join-Path $runRoot "detached_launch_$LaunchId.log"

if ($HostRun) {
    function Write-Receipt([string]$State, [string]$Detail = "") {
        $payload = [ordered]@{
            format = "oracle_critic_detached_launch_v1"
            launch_id = $LaunchId
            updated_at = (Get-Date).ToString("o")
            supervisor_pid = $PID
            session_id = [Diagnostics.Process]::GetCurrentProcess().SessionId
            in_job = $inJob
            probe = [bool]$Probe
            state = $State
            detail = $Detail
            spec = $resolvedSpec
            host_log = $hostLog
        }
        $temporary = "$receiptPath.$PID.tmp"
        [IO.File]::WriteAllText($temporary, ($payload | ConvertTo-Json), [Text.UTF8Encoding]::new($false))
        [IO.File]::Move($temporary, $receiptPath, $true)
    }
    $inJob = $null
    try {
        Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
public static class DetachedHostJob {
    [DllImport("kernel32.dll", SetLastError = true)]
    [return: MarshalAs(UnmanagedType.Bool)]
    public static extern bool IsProcessInJob(IntPtr process, IntPtr job,
        [MarshalAs(UnmanagedType.Bool)] out bool result);
}
'@
        $inJob = $false
        if (-not [DetachedHostJob]::IsProcessInJob(
            [Diagnostics.Process]::GetCurrentProcess().Handle, [IntPtr]::Zero, [ref]$inJob)) {
            throw "Cannot verify Windows Job isolation."
        }
        if ($inJob) { throw "Host still belongs to a Windows Job; refusing to start training." }
        if (-not $Probe) {
            # The supervisor takes this same exclusive lock for its whole lifetime.
            $lock = [IO.File]::Open((Join-Path $runRoot "apex_supervisor.lock"), 'OpenOrCreate', 'ReadWrite', 'None')
            $lock.Dispose()
        }
        Write-Receipt "isolated"
        if ($Probe) {
            Start-Sleep -Seconds 12
            Write-Receipt "probe_completed"
        }
        else {
            & $supervisor -SpecPath $resolvedSpec *>> $hostLog
            Write-Receipt "completed"
        }
    }
    catch {
        Write-Receipt "error" $_.Exception.Message
        throw
    }
    exit
}

# Invoke the desktop's automation object, not a new in-process Shell.Application.
# Explorer creates the host; no caller handles, console or Job are inherited.
$windows = (New-Object -ComObject Shell.Application).Windows()
$hwnd = 0
$desktop = $windows.FindWindowSW(0, 0, 8, [ref]$hwnd, 1)
if ($null -eq $desktop) { throw "No interactive Explorer desktop available." }
function Quote-Literal([string]$Value) { return "'" + $Value.Replace("'", "''") + "'" }
$command = "& $(Quote-Literal $PSCommandPath) -HostRun -SpecPath $(Quote-Literal $resolvedSpec) -LaunchId '$LaunchId'"
if ($Probe) { $command += " -Probe" }
$encoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($command))
$desktop.Document.Application.ShellExecute(
    (Join-Path $PSHOME "pwsh.exe"),
    "-NoProfile -NonInteractive -WindowStyle Hidden -EncodedCommand $encoded",
    $repoRoot, "open", 0
)
$deadline = (Get-Date).AddSeconds(30)
do {
    if (Test-Path -LiteralPath $receiptPath) {
        $receipt = Get-Content -LiteralPath $receiptPath -Raw | ConvertFrom-Json
        if ($receipt.state -eq "error") { throw $receipt.detail }
        if ($Probe) { $receipt | ConvertTo-Json; return }
        $statusPath = Resolve-SpecValue $spec.status_file
        if (Test-Path -LiteralPath $statusPath) {
            $status = Get-Content -LiteralPath $statusPath -Raw | ConvertFrom-Json
            if ($status.supervisor_pid -eq $receipt.supervisor_pid) {
                if ($status.state -eq "supervisor_error") { throw $status.detail }
                $receipt | ConvertTo-Json
                return
            }
        }
    }
    Start-Sleep -Milliseconds 200
} while ((Get-Date) -lt $deadline)
throw "Launch acknowledgement timed out. Inspect $receiptPath and $hostLog before retrying."
