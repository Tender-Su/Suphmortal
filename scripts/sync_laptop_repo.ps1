param(
    [string]$Branch = 'main',
    [string]$LaptopHost = 'mahjong-laptop',
    [string]$LaptopRepo = 'C:\Users\numbe\Desktop\MahjongAI',
    [switch]$SkipWorktreeUpdate
)

$ErrorActionPreference = 'Stop'

Write-Output ('PUSH_BEGIN branch=' + $Branch)
git push laptop-sync ($Branch + ':refs/heads/' + $Branch)

if ($SkipWorktreeUpdate) {
    Write-Output 'SKIP_WORKTREE_UPDATE'
    exit 0
}

$remoteScript = @"
`$ErrorActionPreference = 'Stop'
Set-Location '$LaptopRepo'
git fetch origin
git checkout $Branch
git pull --ff-only origin $Branch
git status --short --branch
"@

Write-Output ('WORKTREE_UPDATE_BEGIN host=' + $LaptopHost)
$remoteScript | ssh $LaptopHost powershell -NoProfile -Command -
