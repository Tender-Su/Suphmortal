[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$PipeName,

    [Parameter(Mandatory = $true)]
    [string]$RunDir,

    [Parameter(Mandatory = $true)]
    [string]$Checkpoint,

    [int]$TargetGames = 2000,
    [double]$StopRatingAt = 0,
    [int]$RatingBotId = 0,
    [int]$RatingActivationTotalGames = -1,
    [string]$RatingApiBase = "https://api.riichi.dev",
    [string]$StopPolicyJson,
    [string]$Device = "cpu",
    [string]$BotName = "MahjongAI-S70",
    [int]$ConnectTimeoutSeconds = 120,
    [int]$WaitForProcessId = 0,
    [string]$ProxyUrl,
    [switch]$NoProxy,
    [string]$PhysicalDirectIp,
    [string]$PhysicalSourceAddress,
    [int]$PhysicalInterfaceIndex = 0
)

$ErrorActionPreference = "Stop"

$pipe = [System.IO.Pipes.NamedPipeClientStream]::new(
    ".",
    $PipeName,
    [System.IO.Pipes.PipeDirection]::In
)
$reader = $null

try {
    $pipe.Connect($ConnectTimeoutSeconds * 1000)
    $reader = [System.IO.StreamReader]::new(
        $pipe,
        [System.Text.Encoding]::UTF8,
        $false,
        4096,
        $true
    )
    $token = $reader.ReadLine()
    if ([string]::IsNullOrWhiteSpace($token)) {
        throw "RiichiLab token handoff returned no data"
    }

    $env:RIICHILAB_BOT_TOKEN = $token
    $token = $null
    $continuationParams = @{
        WaitForProcessId = $WaitForProcessId
        RunDir = $RunDir
        Checkpoint = $Checkpoint
        TargetGames = $TargetGames
        Device = $Device
        BotName = $BotName
        MaxConsecutiveErrors = 100000
    }
    if (-not [string]::IsNullOrWhiteSpace($StopPolicyJson)) {
        $continuationParams.StopPolicyJson = $StopPolicyJson
    } elseif ($StopRatingAt -gt 0) {
        $continuationParams.StopRatingAt = $StopRatingAt
        $continuationParams.RatingBotId = $RatingBotId
        $continuationParams.RatingActivationTotalGames = $RatingActivationTotalGames
        $continuationParams.RatingApiBase = $RatingApiBase
    }
    if (-not [string]::IsNullOrWhiteSpace($PhysicalDirectIp)) {
        $continuationParams.PhysicalDirectIp = $PhysicalDirectIp
        $continuationParams.PhysicalInterfaceIndex = $PhysicalInterfaceIndex
        if (-not [string]::IsNullOrWhiteSpace($PhysicalSourceAddress)) {
            $continuationParams.PhysicalSourceAddress = $PhysicalSourceAddress
        }
    } elseif (-not [string]::IsNullOrWhiteSpace($ProxyUrl)) {
        $continuationParams.ProxyUrl = $ProxyUrl
    } elseif ($NoProxy) {
        $continuationParams.NoProxy = $true
    }
    & (Join-Path $PSScriptRoot "continue_riichilab_ranked_after_process.ps1") `
        @continuationParams
} finally {
    Remove-Item Env:RIICHILAB_BOT_TOKEN -ErrorAction SilentlyContinue
    if ($null -ne $reader) {
        $reader.Dispose()
    }
    $pipe.Dispose()
}
