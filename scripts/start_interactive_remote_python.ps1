param(
    [Parameter(Mandatory = $true)]
    [string]$RepoRoot,
    [Parameter(Mandatory = $true)]
    [string]$PythonExe,
    [Parameter(Mandatory = $true)]
    [string]$PythonScript,
    [string]$PythonArgsJson = '',
    [string]$PythonArgsBase64 = '',
    [Parameter(Mandatory = $true)]
    [string]$TaskId,
    [Parameter(Mandatory = $true)]
    [string]$RuntimeRoot,
    [string]$WindowTitle = 'MahjongAI Remote Task',
    [switch]$WaitForStartOnly
)

$ErrorActionPreference = 'Stop'

function Convert-ArgsPayloadToList {
    param(
        [string]$ArgsJson,
        [string]$ArgsBase64
    )

    if ($ArgsBase64) {
        $decoded = [System.Text.Encoding]::UTF8.GetString([System.Convert]::FromBase64String($ArgsBase64))
        if ($decoded.TrimStart().StartsWith('[')) {
            return @(
                ConvertFrom-Json -InputObject $decoded | ForEach-Object { [string]$_ }
            )
        }
        if ($decoded.Length -eq 0) {
            return @()
        }
        return @($decoded -split "`0", 0, 'SimpleMatch')
    }

    if ($ArgsJson) {
        return @(
            ConvertFrom-Json -InputObject $ArgsJson | ForEach-Object { [string]$_ }
        )
    }

    return @()
}

$pythonArgsList = Convert-ArgsPayloadToList -ArgsJson $PythonArgsJson -ArgsBase64 $PythonArgsBase64
if ($pythonArgsList.Count -eq 0) {
    throw 'PythonArgsJson/PythonArgsBase64 must provide at least one argument'
}

function Write-Utf8NoBomFile {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,
        [Parameter(Mandatory = $true)]
        [string]$Content
    )
    $dir = Split-Path -Parent $Path
    if ($dir -and -not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }
    $encoding = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText($Path, $Content, $encoding)
}

Set-Location $RepoRoot
New-Item -ItemType Directory -Path $RuntimeRoot -Force | Out-Null

$launcherPath = Join-Path $RuntimeRoot 'interactive_launcher.ps1'
$startedPath = Join-Path $RuntimeRoot 'started.json'
$donePath = Join-Path $RuntimeRoot 'done.json'
$stdoutPath = Join-Path $RuntimeRoot 'stdout.log'
$stderrPath = Join-Path $RuntimeRoot 'stderr.log'
$taskName = 'MahjongAI-WinnerRefine-' + $TaskId
$userId = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name

foreach ($path in @($startedPath, $donePath, $stdoutPath, $stderrPath)) {
    if (Test-Path $path) {
        Remove-Item -LiteralPath $path -Force
    }
}

$repoEscaped = $RepoRoot.Replace("'", "''")
$pythonEscaped = $PythonExe.Replace("'", "''")
$scriptEscaped = $PythonScript.Replace("'", "''")
$argsPayloadBase64Escaped = [Convert]::ToBase64String(
    [Text.Encoding]::UTF8.GetBytes([string]::Join("`0", $pythonArgsList))
).Replace("'", "''")
$startedEscaped = $startedPath.Replace("'", "''")
$doneEscaped = $donePath.Replace("'", "''")
$stdoutEscaped = $stdoutPath.Replace("'", "''")
$stderrEscaped = $stderrPath.Replace("'", "''")
$windowEscaped = $WindowTitle.Replace("'", "''")

$launcher = @"
`$ErrorActionPreference = 'Stop'
if (Get-Variable PSNativeCommandUseErrorActionPreference -ErrorAction SilentlyContinue) {
    `$PSNativeCommandUseErrorActionPreference = `$false
}
Set-Location '$repoEscaped'
`$existingPythonPath = [string]`$env:PYTHONPATH
if (`$existingPythonPath) {
    `$env:PYTHONPATH = '$repoEscaped' + [System.IO.Path]::PathSeparator + `$existingPythonPath
}
else {
    `$env:PYTHONPATH = '$repoEscaped'
}
Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
public static class ConsoleModeNative {
    [DllImport("kernel32.dll", SetLastError=true)]
    public static extern IntPtr GetStdHandle(int nStdHandle);
    [DllImport("kernel32.dll", SetLastError=true)]
    public static extern bool GetConsoleMode(IntPtr hConsoleHandle, out uint lpMode);
    [DllImport("kernel32.dll", SetLastError=true)]
    public static extern bool SetConsoleMode(IntPtr hConsoleHandle, uint dwMode);
}
'@ -ErrorAction SilentlyContinue | Out-Null
function Disable-ConsoleQuickEdit {
    try {
        `$STD_INPUT_HANDLE = -10
        `$ENABLE_QUICK_EDIT_MODE = 0x40
        `$ENABLE_EXTENDED_FLAGS = 0x80
        `$handle = [ConsoleModeNative]::GetStdHandle(`$STD_INPUT_HANDLE)
        if (`$handle -eq [IntPtr]::Zero -or `$handle.ToInt64() -eq -1) {
            return
        }
        [uint32]`$mode = 0
        if (-not [ConsoleModeNative]::GetConsoleMode(`$handle, [ref]`$mode)) {
            return
        }
        `$mode = (`$mode -bor `$ENABLE_EXTENDED_FLAGS) -band (-bnot `$ENABLE_QUICK_EDIT_MODE)
        [ConsoleModeNative]::SetConsoleMode(`$handle, `$mode) | Out-Null
    }
    catch {
    }
}
try {
    `$startedPayload = @{
        task_id = '$TaskId'
        started_at = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
        stdout_path = '$stdoutEscaped'
        stderr_path = '$stderrEscaped'
    }
    (`$startedPayload | ConvertTo-Json -Compress) | Set-Content -LiteralPath '$startedEscaped' -Encoding UTF8
    `$Host.UI.RawUI.WindowTitle = '$windowEscaped'
    Disable-ConsoleQuickEdit
    New-Item -ItemType File -Path '$stdoutEscaped' -Force | Out-Null
    New-Item -ItemType File -Path '$stderrEscaped' -Force | Out-Null
    `$pythonArgsPayload = [Text.Encoding]::UTF8.GetString([Convert]::FromBase64String('$argsPayloadBase64Escaped'))
    `$pythonArgs = @()
    if (`$pythonArgsPayload.Length -gt 0) {
        foreach (`$item in (`$pythonArgsPayload -split "`0", 0, 'SimpleMatch')) {
            `$pythonArgs += [string]`$item
        }
    }
    `$nativeExitCode = 0
    `$previousErrorActionPreference = `$ErrorActionPreference
    try {
        # Direct native invocation is more reliable under ScheduledTask than
        # Start-Process -NoNewWindow on the laptop host. Keep stderr as plain
        # logs during the native run and redirect both streams to task files.
        `$ErrorActionPreference = 'Continue'
        & '$pythonEscaped' '$scriptEscaped' @pythonArgs 1>> '$stdoutEscaped' 2>> '$stderrEscaped'
        if (`$LASTEXITCODE -is [int]) {
            `$nativeExitCode = [int]`$LASTEXITCODE
        }
    }
    finally {
        `$ErrorActionPreference = `$previousErrorActionPreference
    }
    `$exitCode = `$nativeExitCode
    `$errorText = if (`$nativeExitCode -eq 0) { `$null } else { "python exited with code `$nativeExitCode" }
}
catch {
    `$exitCode = 1
    `$errorText = [string]`$_
    Write-Error `$_
}
finally {
    `$donePayload = @{
        task_id = '$TaskId'
        finished_at = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
        exit_code = `$exitCode
        error = `$errorText
        stdout_path = '$stdoutEscaped'
        stderr_path = '$stderrEscaped'
    }
    (`$donePayload | ConvertTo-Json -Compress) | Set-Content -LiteralPath '$doneEscaped' -Encoding UTF8
}
exit `$exitCode
"@

Write-Utf8NoBomFile -Path $launcherPath -Content $launcher

try {
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction Stop | Out-Null
}
catch {
}

$launcherCmdPath = $launcherPath.Replace('"', '""')
$shellExe = if (Get-Command 'pwsh.exe' -ErrorAction SilentlyContinue) {
    'pwsh.exe'
}
elseif (Get-Command 'powershell.exe' -ErrorAction SilentlyContinue) {
    'powershell.exe'
}
else {
    throw 'neither pwsh.exe nor powershell.exe is available on the remote machine'
}
$actionArgs = '-NoLogo -NoProfile -ExecutionPolicy Bypass -File "' + $launcherCmdPath + '"'
$action = New-ScheduledTaskAction -Execute $shellExe -Argument $actionArgs
$principal = New-ScheduledTaskPrincipal -UserId $userId -LogonType Interactive -RunLevel Highest
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -MultipleInstances IgnoreNew

Register-ScheduledTask -TaskName $taskName -Action $action -Principal $principal -Settings $settings -Force | Out-Null
Start-ScheduledTask -TaskName $taskName

$startupDeadline = (Get-Date).AddSeconds(60)
while (-not (Test-Path $startedPath) -and -not (Test-Path $donePath)) {
    if ((Get-Date) -gt $startupDeadline) {
        throw "interactive task `$taskName did not start within 60 seconds"
    }
    Start-Sleep -Seconds 2
}

if ($WaitForStartOnly) {
    $startedPayload = if (Test-Path $startedPath) {
        Get-Content -LiteralPath $startedPath -Raw | ConvertFrom-Json
    }
    else {
        [pscustomobject]@{
            task_id = $TaskId
            started_at = $null
        }
    }
    # Keep the scheduled task registered in wait-for-start mode. Reusing the
    # same task name on the next launch will clean it up, and unregistering
    # immediately can tear down the just-started interactive process on some
    # machines.
    Write-Output ($startedPayload | ConvertTo-Json -Compress)
    exit 0
}

while (-not (Test-Path $donePath)) {
    Start-Sleep -Seconds 2
}

$done = Get-Content -LiteralPath $donePath -Raw | ConvertFrom-Json
try {
    Unregister-ScheduledTask -TaskName $taskName -Confirm:$false -ErrorAction Stop | Out-Null
}
catch {
}

Write-Output ($done | ConvertTo-Json -Compress)
exit [int]$done.exit_code
