$ErrorActionPreference = 'Stop'

Get-CimInstance Win32_Process | Where-Object {
    ($_.Name -in @('python.exe', 'powershell.exe')) -and (
        ($_.CommandLine -like '*extract_data.py*') -or
        ($_.CommandLine -like '*decompress_dataset_json.py*') -or
        ($_.CommandLine -like '*laptop_rebuild_remote_*')
    )
} | ForEach-Object {
    Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
}

Write-Host 'DECOMPRESS_BEGIN'

& 'C:\Users\numbe\miniconda3\envs\mortal\python.exe' `
    'C:\Users\numbe\Desktop\MahjongAI\scripts\decompress_dataset_json.py' `
    --src-root 'C:\Users\numbe\mahjong_data_root\dataset_rebuilt' `
    --dst-root 'C:\Users\numbe\mahjong_data_root\dataset_json_rebuilt' `
    --workers 18 `
    --report-every 1000
