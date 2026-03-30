$ErrorActionPreference = 'Stop'

$python = 'C:\Users\numbe\miniconda3\envs\mortal\python.exe'
$script = 'C:\Users\numbe\Desktop\MahjongAI\scripts\decompress_dataset_json.py'
$srcRoot = 'C:\Users\numbe\mahjong_data_root\dataset_rebuilt'
$dstRoot = 'C:\Users\numbe\mahjong_data_root\dataset_json_rebuilt'
$pollSeconds = 15

Write-Output 'WAIT_EXTRACT_BEGIN'

while ($true) {
    $extract = @(Get-CimInstance Win32_Process | Where-Object {
        $_.Name -eq 'python.exe' -and $_.CommandLine -like '*extract_data.py*'
    })
    if ($extract.Count -eq 0) {
        break
    }
    Write-Output ('WAIT_EXTRACT_STILL_RUNNING=' + $extract.Count)
    Start-Sleep -Seconds $pollSeconds
}

$decompress = @(Get-CimInstance Win32_Process | Where-Object {
    $_.Name -eq 'python.exe' -and $_.CommandLine -like '*decompress_dataset_json.py*'
})
if ($decompress.Count -gt 0) {
    Write-Output 'DECOMPRESS_ALREADY_RUNNING'
    exit 0
}

Get-CimInstance Win32_Process | Where-Object {
    $_.Name -eq 'powershell.exe' -and (
        $_.CommandLine -like '*laptop_rebuild_remote_*' -or
        $_.CommandLine -like '*laptop_decompress_active.runner.ps1*'
    )
} | ForEach-Object {
    Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
}

Write-Output 'DECOMPRESS_BEGIN'
& $python $script --src-root $srcRoot --dst-root $dstRoot --workers 18 --report-every 1000
