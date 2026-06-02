# daily_signal.ps1 — 每个交易日上午运行：Tushare 更新 + qlib 打分 + 写 target JSON
#
# 由 Windows 任务计划程序调用（建议 08:30，T-1 收盘数据 18:00 后已就绪）
# 脚本自己不检查交易日——qlib updater 和 live_trade 各自有交易日守卫
# 失败时写 scheduler/logs/signal_YYYYMMDD.err，可按需配置告警
#
# 依赖：
#   C:\Users\Administrator\Documents\stock\qlib_quant\.venv  (pyqlib 环境)
#   C:\Users\Administrator\Documents\stock\qlib_quant\.env   (TUSHARE_TOKEN)
#   C:\Python312\python.exe                                  (xtquant / live_trade 环境，不在本脚本里用)

# ── 路径常量 ────────────────────────────────────────────────
$QLIB_DIR   = "C:\Users\Administrator\Documents\stock\qlib_quant"
$VENV_PY    = "$QLIB_DIR\.venv\Scripts\python.exe"
$ENV_FILE   = "$QLIB_DIR\.env"
$CONFIG     = "$QLIB_DIR\config\models\alpha158_momentum_volume_k6_dd10_overlay.yaml"
$TARGET_DIR = "C:\Users\Administrator\Documents\stock\qmt\targets"
$LOG_DIR    = "C:\Users\Administrator\Documents\stock\qmt\scheduler\logs"

# ── 日志 ────────────────────────────────────────────────────
$today   = Get-Date -Format "yyyyMMdd"
$ts      = Get-Date -Format "yyyy-MM-dd HH:mm:ss"

New-Item -ItemType Directory -Force $LOG_DIR | Out-Null
$LOG_FILE = "$LOG_DIR\signal_$today.log"
$ERR_FILE = "$LOG_DIR\signal_$today.err"

function Log($msg) {
    $line = "[$( Get-Date -Format 'HH:mm:ss' )] $msg"
    Write-Host $line
    Add-Content -Path $LOG_FILE -Value $line -Encoding UTF8
}

function Fail($msg) {
    $line = "[FAIL] $msg"
    Write-Host $line
    Add-Content -Path $LOG_FILE -Value $line -Encoding UTF8
    Add-Content -Path $ERR_FILE -Value "[$ts] $line" -Encoding UTF8
    exit 1
}

# ── 加载 .env 里的 TUSHARE_TOKEN ────────────────────────────
if (Test-Path $ENV_FILE) {
    Get-Content $ENV_FILE | ForEach-Object {
        if ($_ -match "^\s*([^#=]+)=(.+)\s*$") {
            $k = $Matches[1].Trim()
            $v = $Matches[2].Trim()
            [System.Environment]::SetEnvironmentVariable($k, $v, "Process")
        }
    }
}

if (-not $env:TUSHARE_TOKEN) {
    Fail ".env 里没有 TUSHARE_TOKEN"
}

# ── Step 1: 更新 Tushare 数据 ────────────────────────────────
Log "=== Step 1: Tushare 数据更新 ==="
Set-Location $QLIB_DIR
$result = & $VENV_PY -X utf8 main.py update 2>&1
if ($LASTEXITCODE -ne 0) {
    Log "数据更新失败（exit $LASTEXITCODE）：$result"
    Fail "Step 1 failed"
}
# main.py update 在预检/数据源失败时可能仍返回 exit 0（软失败），
# 退出码守卫抓不到，需额外校验输出内容。失败时会打印 "数据预检失败"
# 或 "[WARN] ... 预检: False"。命中任一即视为失败。
$out = $result | Out-String
if ($out -match "数据预检失败" -or $out -match "预检:\s*False" -or $out -match "Tushare API 不可用") {
    Log "数据更新软失败（exit 0 但预检/数据源未通过）：$out"
    Fail "Step 1 soft-failed (precheck/datasource)"
}
Log "[OK] 数据更新完成"

# ── Step 2: 打分 + 写 target JSON ───────────────────────────
Log "=== Step 2: qlib 打分 + 写 target JSON ==="
$result = & $VENV_PY -X utf8 scripts\export_target.py --config $CONFIG --target-dir $TARGET_DIR 2>&1
if ($LASTEXITCODE -ne 0) {
    Log "export_target 失败（exit $LASTEXITCODE）：$result"
    Fail "Step 2 failed"
}
Log "[OK] target JSON 已写入 $TARGET_DIR\target_$today.json"

Log "=== daily_signal 完成 ==="
