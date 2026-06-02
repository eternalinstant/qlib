# register_tasks.ps1 — 通过 XML 定义注册 Windows 任务计划（兼容 Windows Server 2022）

$SCHED_DIR = "C:\Users\Administrator\Documents\stock\qmt\scheduler"

function Register-TaskXml {
    param([string]$Name, [string]$Script, [string]$Hour, [string]$Minute)

    $xml = @"
<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.2" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <Triggers>
    <CalendarTrigger>
      <StartBoundary>2026-01-01T${Hour}:${Minute}:00</StartBoundary>
      <Enabled>true</Enabled>
      <ScheduleByWeek>
        <WeeksInterval>1</WeeksInterval>
        <DaysOfWeek>
          <Monday />
          <Tuesday />
          <Wednesday />
          <Thursday />
          <Friday />
        </DaysOfWeek>
      </ScheduleByWeek>
    </CalendarTrigger>
  </Triggers>
  <Principals>
    <Principal id="Author">
      <LogonType>InteractiveToken</LogonType>
      <RunLevel>HighestAvailable</RunLevel>
    </Principal>
  </Principals>
  <Settings>
    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
    <ExecutionTimeLimit>PT2H</ExecutionTimeLimit>
    <StartWhenAvailable>true</StartWhenAvailable>
  </Settings>
  <Actions>
    <Exec>
      <Command>powershell.exe</Command>
      <Arguments>-NonInteractive -ExecutionPolicy Bypass -File "$Script"</Arguments>
    </Exec>
  </Actions>
</Task>
"@

    $xmlFile = "$env:TEMP\${Name}.xml"
    # XML 需要 UTF-16 编码
    [System.IO.File]::WriteAllText($xmlFile, $xml, [System.Text.Encoding]::Unicode)

    schtasks /Delete /TN $Name /F 2>$null | Out-Null
    schtasks /Create /TN $Name /XML $xmlFile /F
    Remove-Item $xmlFile -ErrorAction SilentlyContinue

    if ($LASTEXITCODE -eq 0) {
        Write-Host "[OK] 注册: $Name  时间: ${Hour}:${Minute} (周一~五)"
    } else {
        Write-Host "[FAIL] 注册失败: $Name"
    }
}

Register-TaskXml -Name "QlibDailySignal" -Script "$SCHED_DIR\daily_signal.ps1" -Hour "08" -Minute "30"
Register-TaskXml -Name "QmtDailyTrade"   -Script "$SCHED_DIR\daily_trade.ps1"  -Hour "14" -Minute "25"

Write-Host ""
schtasks /Query /TN "QlibDailySignal" /FO LIST 2>$null | Select-String "TaskName|Next Run|Status"
schtasks /Query /TN "QmtDailyTrade"   /FO LIST 2>$null | Select-String "TaskName|Next Run|Status"
