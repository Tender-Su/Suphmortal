$ErrorActionPreference = 'Stop'

Get-CimInstance Win32_Process | Where-Object {
    ($_.Name -in @('python.exe', 'powershell.exe', 'pwsh.exe')) -and (
        ($_.CommandLine -like '*extract_data.py*') -or
        ($_.CommandLine -like '*decompress_dataset_json.py*') -or
        ($_.CommandLine -like '*laptop_rebuild_remote_*')
    )
} | ForEach-Object {
    Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
}

Write-Host 'DECOMPRESS_BEGIN'

& (Join-Path $env:USERPROFILE 'miniconda3\envs\mortal\python.exe') `
    (Join-Path $env:USERPROFILE 'Desktop\MahjongAI\scripts\decompress_dataset_json.py') `
    --src-root (Join-Path $env:USERPROFILE 'mahjong_data_root\dataset_rebuilt') `
    --dst-root (Join-Path $env:USERPROFILE 'mahjong_data_root\dataset_json_rebuilt') `
    --workers 18 `
    --report-every 1000
