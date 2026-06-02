# daily_trade.ps1 — 每个交易日 14:30 运行：读 target JSON → QMT 调仓执行
#
# 由 Windows 任务计划程序调用（14:25 触发，留 5 分钟缓冲）
# 依赖：MiniQMT 客户端已启动并登录
#       C:\Python312\python.exe (xtquant 环境)
#       C:\Users\Administrator\Documents\stock\qmt\targets\target_YYYYMMDD.json (由 daily_signal.ps1 写入)

$QMT_DIR  = "C:\Users\Administrator\Documents\stock\qmt"
$QMT_PY   = "C:\Python312\python.exe"
$LOG_DIR  = "$QMT_DIR\scheduler\logs"

$today    = Get-Date -Format "yyyyMMdd"
New-Item -ItemType Directory -Force $LOG_DIR | Out-Null
$LOG_FILE = "$LOG_DIR\trade_$today.log"
$ERR_FILE = "$LOG_DIR\trade_$today.err"

function Log($msg) {
    $line = "[$( Get-Date -Format 'HH:mm:ss' )] $msg"
    Write-Host $line
    Add-Content -Path $LOG_FILE -Value $line -Encoding UTF8
}

function Fail($msg) {
    $line = "[FAIL] $msg"
    Write-Host $line
    Add-Content -Path $LOG_FILE -Value $line -Encoding UTF8
    Add-Content -Path $ERR_FILE -Value "[$( Get-Date -Format 'yyyy-MM-dd HH:mm:ss' )] $line" -Encoding UTF8
    exit 1
}

# 检查 target 文件是否已生成
$targetFile = "$QMT_DIR\targets\target_$today.json"
if (-not (Test-Path $targetFile)) {
    Fail "target 文件不存在: $targetFile  请确认 daily_signal.ps1 已成功运行"
}

Log "=== daily_trade 开始 ==="
Log "target 文件: $targetFile"

Set-Location $QMT_DIR
$result = & $QMT_PY live_trade.py --live 2>&1
if ($LASTEXITCODE -ne 0) {
    Log "live_trade 失败（exit $LASTEXITCODE）：$result"
    Fail "live_trade.py 执行失败"
}
Log "[OK] 调仓执行完成"
Log "=== daily_trade 完成 ==="
