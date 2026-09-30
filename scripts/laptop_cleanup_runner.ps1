$ErrorActionPreference = 'Stop'

$root = 'C:\Users\numbe\mahjong_data_root'
$keep = @(
    (Join-Path $root 'dataset_rebuilt'),
    (Join-Path $root 'dataset_json_rebuilt')
)

function Resolve-SafePath([string]$path) {
    $resolved = (Resolve-Path -LiteralPath $path).ProviderPath
    if (-not $resolved.StartsWith($root, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "refusing to touch path outside root: $resolved"
    }
    return $resolved
}

Write-Host 'CLEANUP_BEGIN'

$candidates = @(
    (Join-Path $root 'dataset'),
    (Join-Path $root 'dataset_bad_20260330'),
    (Join-Path $root 'dataset_json')
)
$candidates += @(Get-ChildItem $root -Directory -ErrorAction SilentlyContinue | Where-Object {
    $_.Name -like 'dataset_rebuilt_stale_*' -or $_.Name -like 'dataset_json_rebuilt_stale_*'
} | Select-Object -ExpandProperty FullName)

foreach ($candidate in $candidates) {
    if (-not (Test-Path -LiteralPath $candidate)) {
        continue
    }
    $resolved = Resolve-SafePath $candidate
    if ($keep -contains $resolved) {
        throw "refusing to delete keep dir: $resolved"
    }
    Write-Host ('DELETE ' + $resolved)
    Remove-Item -LiteralPath $resolved -Recurse -Force
    Write-Host ('DELETED ' + $resolved)
}

Write-Host 'CLEANUP_DONE'
