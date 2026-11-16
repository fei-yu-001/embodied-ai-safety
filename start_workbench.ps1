param(
    [switch]$NoBrowser,
    [string]$Token = "dev-token",
    [string]$Backend = "robocasa",
    [ValidateRange(1024, 65535)][int]$Port = 8766,
    [string]$DataRoot = "",
    [string]$FrontendDir = ""
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = if ($env:PRISM_PYTHON) { $env:PRISM_PYTHON } elseif (Test-Path "$root/.venv/Scripts/python.exe") { "$root/.venv/Scripts/python.exe" } else { (Get-Command python -ErrorAction Stop).Source }
$workbenchPort = $Port
if (-not $DataRoot) { $DataRoot = Join-Path $root "data/sessions" }
if (-not $FrontendDir) { $FrontendDir = Join-Path $root "web/dist" }

$existing = Get-NetTCPConnection -LocalPort $workbenchPort -State Listen -ErrorAction SilentlyContinue
if ($existing) {
    try {
        $health = Invoke-RestMethod -Uri "http://127.0.0.1:$workbenchPort/api/v1/health" -Headers @{ Host = "127.0.0.1" } -TimeoutSec 3
        if ($health.api -ne "ok") { throw "服务健康状态不正确" }
    } catch {
        throw "端口 $workbenchPort 已被占用，但工作台健康检查失败。请检查占用进程，或使用 -Port 指定其他端口。"
    }
    Write-Host "控制面已在运行：http://127.0.0.1:$workbenchPort"
} else {
    if (-not (Test-Path "$root/data")) { New-Item -ItemType Directory -Path "$root/data" | Out-Null }
    $env:PRISM_TOKEN = $Token
    $env:PRISM_BACKEND = $Backend
    $env:PRISM_PORT = [string]$workbenchPort
    $env:PRISM_DATA_ROOT = [System.IO.Path]::GetFullPath($DataRoot)
    $env:PRISM_FRONTEND_DIR = [System.IO.Path]::GetFullPath($FrontendDir)
    Start-Process -FilePath $python `
        -ArgumentList "-X", "utf8", "-m", "server.main" `
        -WorkingDirectory $root `
        -RedirectStandardOutput "$root/data/control-plane-$workbenchPort.log" `
        -RedirectStandardError "$root/data/control-plane-$workbenchPort.err.log" `
        -WindowStyle Hidden
    Write-Host "启动中，等待服务就绪…"
    $ready = $false
    for ($i = 0; $i -lt 40; $i++) {
        Start-Sleep -Milliseconds 500
        try {
            $health = Invoke-RestMethod -Uri "http://127.0.0.1:$workbenchPort/api/v1/health" -Headers @{ Host = "127.0.0.1" } -TimeoutSec 2
            if ($health.api -eq "ok") { $ready = $true; break }
        } catch { }
    }
    if (-not $ready) {
        throw "工作台未能就绪。请检查 $root/data/control-plane-$workbenchPort.err.log"
    }
    Write-Host "控制面已就绪：http://127.0.0.1:$workbenchPort"
}

Write-Host "工作台：http://127.0.0.1:$workbenchPort/#token=$Token"
Write-Host "接口文档：http://127.0.0.1:$workbenchPort/docs"
Write-Host "日志：$root/data/control-plane-$workbenchPort.log"

if (-not $NoBrowser) {
    Start-Process "http://127.0.0.1:$workbenchPort/#token=$Token"
}
