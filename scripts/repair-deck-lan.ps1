# Keeps the Shrimp deck reachable on the LAN. Two chains exist:
#   8080 (live, no-admin path): portproxy 0.0.0.0:8080 -> WSL-IP:8080 ->
#        deck-lan-8080 socat container -> deck. Works regardless of the WSL
#        localhost relay.
#   8000 (original): portproxy 0.0.0.0:8000 -> WSL-IP:8000 -> deck directly.
# Both portproxy targets go stale when the WSL IP changes (WSL restart) —
# this script re-resolves and rewrites them, and starts the containers if
# stopped. netsh needs elevation, so run elevated; -Register installs a
# scheduled task (logon + every 15 min, highest privileges, no UAC at
# trigger time) that keeps everything pointed correctly.

param([switch]$Register)

$ports = @(8000, 8080)
$log = Join-Path $env:LOCALAPPDATA "shrimp-deck-lan.log"

function Log($msg) {
    $line = "{0} {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $msg
    Add-Content -Path $log -Value $line -Encoding utf8
}

# No-ops when already running; brings the chain back after a WSL/docker restart.
try { wsl docker start docker-deck-1 deck-lan-8080 | Out-Null } catch {}

$wslIp = ""
try { $wslIp = (wsl hostname -I).Trim().Split()[0] } catch {}
if (-not $wslIp) { Log "no WSL IP; WSL not running?"; exit 1 }

foreach ($port in $ports) {
    $healthy = $false
    try {
        $r = Invoke-WebRequest -Uri "http://$($wslIp):$port/api/runs" -UseBasicParsing -TimeoutSec 5
        $healthy = ($r.StatusCode -eq 200)
    } catch { Log ("deck not answering at {0}:{1} - {2}" -f $wslIp, $port, $_.Exception.Message) }
    if (-not $healthy) { continue }

    $existing = ""
    $m = netsh interface portproxy show v4tov4 | Select-String ("0\.0\.0\.0\s+{0}\s+(\S+)\s+{0}" -f $port)
    if ($m) { $existing = $m.Matches[0].Groups[1].Value }
    if ($existing -ne $wslIp) {
        netsh interface portproxy delete v4tov4 listenport=$port listenaddress=0.0.0.0 | Out-Null
        netsh interface portproxy add v4tov4 listenaddress=0.0.0.0 listenport=$port connectaddress=$wslIp connectport=$port | Out-Null
        Log ("portproxy 0.0.0.0:{0} -> {1}:{0} (was '{2}')" -f $port, $wslIp, $existing)
    }
}

if ($Register) {
    $action = New-ScheduledTaskAction -Execute "powershell.exe" `
        -Argument "-NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$PSCommandPath`""
    $t1 = New-ScheduledTaskTrigger -AtLogOn
    $t2 = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(2) `
        -RepetitionInterval (New-TimeSpan -Minutes 15) -RepetitionDuration (New-TimeSpan -Days 3650)
    $principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" `
        -RunLevel Highest -LogonType Interactive
    $settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
        -StartWhenAvailable -MultipleInstances IgnoreNew
    Register-ScheduledTask -TaskName "ShrimpDeckLanRepair" -Action $action -Trigger $t1, $t2 `
        -Principal $principal -Settings $settings -Force | Out-Null
    Log "scheduled task ShrimpDeckLanRepair registered"
}
