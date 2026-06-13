# switch_data_source.ps1 — Tushare/QMT 双数据源一键切换（NTFS junction 翻转）
#
#   .\switch_data_source.ps1            # 查看当前指向
#   .\switch_data_source.ps1 tushare    # 生产默认：原生 Tushare 源
#   .\switch_data_source.ps1 qmt        # QMT 源（qmt_data curated 供数）
#
# 两棵源树各自带独立 cn_data（factor_data + bins），切换零重建成本。
# 注意：08:30 QlibDailySignal 会向当前指向的源写入——非验证窗口请保持 tushare。

param([ValidateSet("tushare", "qmt", "")] [string]$Target = "")

$ROOT = "C:\Users\Administrator\Documents\stock\qlib_quant"
$LINK = "$ROOT\data"
$MAP = @{ tushare = "$ROOT\data_src_tushare"; qmt = "$ROOT\data_src_qmt" }

$cur = (Get-Item $LINK -ErrorAction SilentlyContinue)
if (-not $cur -or -not $cur.LinkType) { Write-Host "[FAIL] $LINK 不是 junction"; exit 1 }
Write-Host "当前: data -> $($cur.Target)"

if ($Target -eq "") { exit 0 }
$dest = $MAP[$Target]
if (-not (Test-Path $dest)) { Write-Host "[FAIL] 源树不存在: $dest"; exit 1 }
if ("$($cur.Target)" -eq $dest) { Write-Host "[OK] 已指向 $Target，无需切换"; exit 0 }

cmd /c "rmdir `"$LINK`""
cmd /c "mklink /J `"$LINK`" `"$dest`"" | Out-Null
Write-Host "[OK] 已切换: data -> $dest"
