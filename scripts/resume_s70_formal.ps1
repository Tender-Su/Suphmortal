param(
    [Parameter(Mandatory = $true)]
    [string]$RepoRoot,
    [string]$PythonExe = 'C:\Users\numbe\miniconda3\envs\mortal\python.exe',
    [string]$RunName = 'sl_anchor_longabc_s70_20260609_r1',
    [string]$CandidateArm = 'C_A2x_cosine_broad_to_recent_strong_24m_12m__W_r00516_o000135_d000804',
    [ValidateSet('phase_a', 'phase_b', 'phase_c')]
    [string]$ResumePhase = 'phase_a',
    [string]$PhaseAStorageRoot = 'C:\Users\numbe\AppData\Local\Temp\mahjongai_sl_ab\b5d80ad4f515ab48\phase_a',
    [string]$PhaseBStorageRoot = '',
    [string]$PhaseCStorageRoot = '',
    [int]$LatestStepsBeforeResume = -1,
    [int]$LatestOptimizerStepsBeforeResume = -1
)

$ErrorActionPreference = 'Stop'

Set-Location -LiteralPath $RepoRoot

$phaseDir = Join-Path $RepoRoot "logs\sl_ab\sl_anchor_longabc_s70_20260609_r1__W_r00516_o000135_d000804_formal\checkpoint_compare\$ResumePhase"
$trainLog = Join-Path $phaseDir 'train.log'
$storageRoots = [ordered]@{
    phase_a = $PhaseAStorageRoot
    phase_b = $PhaseBStorageRoot
    phase_c = $PhaseCStorageRoot
}
$phaseStorageRoot = [string]$storageRoots[$ResumePhase]
if ([string]::IsNullOrWhiteSpace($phaseStorageRoot)) {
    $phaseStorageRoot = $phaseDir
}
$latestCheckpoint = Join-Path $phaseStorageRoot 'checkpoints\latest.pth'
if (-not (Test-Path -LiteralPath $phaseStorageRoot)) {
    throw "missing $ResumePhase storage root: $phaseStorageRoot"
}
if (-not (Test-Path -LiteralPath $latestCheckpoint)) {
    throw "missing latest checkpoint: $latestCheckpoint"
}
if (-not (Test-Path -LiteralPath $trainLog)) {
    throw "missing $ResumePhase train log: $trainLog"
}

if (($LatestStepsBeforeResume -lt 0) -or ($LatestOptimizerStepsBeforeResume -lt 0)) {
    $checkpointSummary = & $PythonExe -c "import json, sys, torch; p=sys.argv[1]; s=torch.load(p, map_location='cpu', weights_only=False); print(json.dumps({'steps': int(s.get('steps', -1) or -1), 'optimizer_steps': int(s.get('optimizer_steps', -1) or -1)}))" $latestCheckpoint
    $checkpointSummary = $checkpointSummary | ConvertFrom-Json
    if ($LatestStepsBeforeResume -lt 0) {
        $LatestStepsBeforeResume = [int]$checkpointSummary.steps
    }
    if ($LatestOptimizerStepsBeforeResume -lt 0) {
        $LatestOptimizerStepsBeforeResume = [int]$checkpointSummary.optimizer_steps
    }
}

$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$resumeRoot = Join-Path $RepoRoot "logs\sl_fidelity\$RunName\distributed\formal_dispatch\resume_runtime\formal_resume_file__$stamp"
$resultJson = Join-Path $RepoRoot "logs\sl_fidelity\$RunName\distributed\formal_dispatch\remote_results\formal__$CandidateArm.json"
$stdoutPath = Join-Path $resumeRoot 'stdout.log'
$stderrPath = Join-Path $resumeRoot 'stderr.log'
$donePath = Join-Path $resumeRoot 'done.json'
$launcherPath = Join-Path $resumeRoot 'resume_launcher.ps1'

New-Item -ItemType Directory -Force -Path $resumeRoot | Out-Null
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $resultJson) | Out-Null
if (Test-Path -LiteralPath $resultJson) {
    Remove-Item -LiteralPath $resultJson -Force
}

$head = (git rev-parse --short HEAD).Trim()
$manifest = [ordered]@{
    launched_at = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
    code = $head
    run_name = $RunName
    candidate_arm = $CandidateArm
    resume_phase = $ResumePhase
    latest_steps_before_resume = $LatestStepsBeforeResume
    latest_optimizer_steps_before_resume = $LatestOptimizerStepsBeforeResume
    phase_storage_root = $phaseStorageRoot
    train_log = (Resolve-Path -LiteralPath $trainLog).Path
    resume_root = $resumeRoot
    result_json = $resultJson
}
$manifest | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $resumeRoot 'resume_manifest.json') -Encoding UTF8

$scriptPath = Join-Path $RepoRoot 'mortal\supervised\run_sl_formal_distributed.py'
$launcher = @'
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath '__REPO_ROOT__'
$env:PYTHONPATH = '__REPO_ROOT__'
$env:MORTAL_SL_AB_PHASE_A_STORAGE_ROOT = '__PHASE_A_STORAGE_ROOT__'
$env:MORTAL_SL_AB_PHASE_B_STORAGE_ROOT = '__PHASE_B_STORAGE_ROOT__'
$env:MORTAL_SL_AB_PHASE_C_STORAGE_ROOT = '__PHASE_C_STORAGE_ROOT__'
$argsList = @(
    'run-task',
    '--run-name', '__RUN_NAME__',
    '--candidate-arm', '__CANDIDATE_ARM__',
    '--machine-label', 'laptop',
    '--num-workers', '4',
    '--file-batch-size', '10',
    '--prefetch-factor', '4',
    '--val-file-batch-size', '7',
    '--val-prefetch-factor', '5',
    '--resume-existing',
    '--result-json', '__RESULT_JSON__'
)
$exitCode = 0
$errorText = $null
try {
    & '__PYTHON_EXE__' '__SCRIPT_PATH__' @argsList 1>> '__STDOUT_PATH__' 2>> '__STDERR_PATH__'
    if ($LASTEXITCODE -is [int]) {
        $exitCode = [int]$LASTEXITCODE
    }
    if ($exitCode -ne 0) {
        $errorText = "python exited with code $exitCode"
    }
}
catch {
    $exitCode = 1
    $errorText = [string]$_
    [string]$_ | Add-Content -LiteralPath '__STDERR_PATH__'
}
finally {
    [ordered]@{
        finished_at = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
        exit_code = $exitCode
        error = $errorText
    } | ConvertTo-Json -Compress | Set-Content -LiteralPath '__DONE_PATH__' -Encoding UTF8
}
exit $exitCode
'@

foreach ($replacement in @(
    @('__REPO_ROOT__', $RepoRoot),
    @('__PHASE_A_STORAGE_ROOT__', $PhaseAStorageRoot),
    @('__PHASE_B_STORAGE_ROOT__', $PhaseBStorageRoot),
    @('__PHASE_C_STORAGE_ROOT__', $PhaseCStorageRoot),
    @('__RUN_NAME__', $RunName),
    @('__CANDIDATE_ARM__', $CandidateArm),
    @('__RESULT_JSON__', $resultJson),
    @('__PYTHON_EXE__', $PythonExe),
    @('__SCRIPT_PATH__', $scriptPath),
    @('__STDOUT_PATH__', $stdoutPath),
    @('__STDERR_PATH__', $stderrPath),
    @('__DONE_PATH__', $donePath)
)) {
    $launcher = $launcher.Replace($replacement[0], $replacement[1].Replace("'", "''"))
}

$encoding = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllText($launcherPath, $launcher, $encoding)

$process = Start-Process `
    -FilePath 'powershell.exe' `
    -ArgumentList @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $launcherPath) `
    -WorkingDirectory $RepoRoot `
    -WindowStyle Hidden `
    -PassThru

Start-Sleep -Seconds 3
$process.Refresh()
if ($process.HasExited) {
    $stdoutText = if (Test-Path -LiteralPath $stdoutPath) { Get-Content -LiteralPath $stdoutPath -Raw } else { '' }
    $stderrText = if (Test-Path -LiteralPath $stderrPath) { Get-Content -LiteralPath $stderrPath -Raw } else { '' }
    throw "resume launcher exited early code=$($process.ExitCode)`nSTDOUT:`n$stdoutText`nSTDERR:`n$stderrText"
}

$marker = "`n=== manual resume launch @ $($manifest.launched_at) phase=$ResumePhase latest_steps=$LatestStepsBeforeResume latest_optimizer_steps=$LatestOptimizerStepsBeforeResume code=$head storage_root=$phaseStorageRoot runtime_root=$resumeRoot pid=$($process.Id) ===`n"
Add-Content -LiteralPath $trainLog -Value $marker -Encoding UTF8

[ordered]@{
    pid = $process.Id
    code = $head
    resume_root = $resumeRoot
    result_json = $resultJson
} | ConvertTo-Json -Compress
