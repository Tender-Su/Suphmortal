param(
    [Parameter(Mandatory = $true)]
    [string]$RepoRoot,
    [Parameter(Mandatory = $true)]
    [string]$PythonExe,
    [Parameter(Mandatory = $true)]
    [string]$PythonScript,
    [Parameter(Mandatory = $true)]
    [string]$PythonArgsJson,
    [Parameter(Mandatory = $true)]
    [string]$TaskId,
    [Parameter(Mandatory = $true)]
    [string]$RuntimeRoot,
    [string]$WindowTitle = 'MahjongAI Remote Task'
)

$ErrorActionPreference = 'Stop'

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
$taskName = 'MahjongAI-WinnerRefine-' + $TaskId
$userId = if ($env:USERDOMAIN) { $env:USERDOMAIN + '\' + $env:USERNAME } else { $env:USERNAME }

foreach ($path in @($startedPath, $donePath)) {
    if (Test-Path $path) {
        Remove-Item -LiteralPath $path -Force
    }
}

$repoEscaped = $RepoRoot.Replace("'", "''")
$pythonEscaped = $PythonExe.Replace("'", "''")
$scriptEscaped = $PythonScript.Replace("'", "''")
$argsEscaped = $PythonArgsJson.Replace("'", "''")
$startedEscaped = $startedPath.Replace("'", "''")
$doneEscaped = $donePath.Replace("'", "''")
$windowEscaped = $WindowTitle.Replace("'", "''")

$launcher = @"
`$ErrorActionPreference = 'Stop'
Set-Location '$repoEscaped'
try {
    `$startedPayload = @{
        task_id = '$TaskId'
        started_at = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
    }
    (`$startedPayload | ConvertTo-Json -Compress) | Set-Content -LiteralPath '$startedEscaped' -Encoding UTF8
    `$Host.UI.RawUI.WindowTitle = '$windowEscaped'
    `$pythonArgs = @()
    foreach (`$item in (ConvertFrom-Json -InputObject '$argsEscaped')) {
        `$pythonArgs += [string]`$item
    }
    & '$pythonEscaped' '$scriptEscaped' @pythonArgs
    `$exitCode = if (`$LASTEXITCODE -is [int]) { [int]`$LASTEXITCODE } else { 0 }
    `$errorText = `$null
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
$windowCmdTitle = $WindowTitle.Replace('"', '""')
$actionArgs = '/c start "' + $windowCmdTitle + '" /max powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -WindowStyle Normal -File "' + $launcherCmdPath + '"'
$action = New-ScheduledTaskAction -Execute 'cmd.exe' -Argument $actionArgs
$principal = New-ScheduledTaskPrincipal -UserId $userId -LogonType InteractiveToken -RunLevel Highest
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
