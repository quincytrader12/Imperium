@echo off
REM ======================================================================
REM  IMPERIUM - continuous operation
REM
REM  Runs the terminal and restarts it if it ever exits, for as long as this
REM  window is open. Use this one when you want the bot running all day and
REM  overnight; use Start-IMPERIUM.bat when you want a single run you watch.
REM
REM  What this does and does not do:
REM    - It restarts the process if it crashes or is killed.
REM    - It backs off after repeated immediate failures, so a machine that
REM      cannot run it at all does not spin restarting forever.
REM    - It does NOT survive a reboot, a sign-out, or closing this window.
REM    - It does NOT stop the laptop sleeping when the lid is closed. The
REM      terminal asks Windows to keep the system awake while a session is
REM      running, but a closed lid overrides that on most machines.
REM
REM  Stop it with Ctrl+C, twice: once for the terminal, once for this loop.
REM ======================================================================

title IMPERIUM - running continuously
cd /d "%~dp0"

set /a RUNS=0
set /a FASTFAILS=0

:loop
set /a RUNS+=1
echo.
echo ======================================================================
echo  Run #%RUNS%  -  %DATE% %TIME%
echo ======================================================================

REM Seconds since midnight, to tell a crash-on-startup from a long clean run.
for /f "tokens=1-3 delims=:." %%a in ("%TIME: =0%") do set /a T0=(1%%a-100)*3600+(1%%b-100)*60+(1%%c-100)

"%~dp0IMPERIUM.exe" --no-browser %*
set EXITCODE=%errorlevel%

for /f "tokens=1-3 delims=:." %%a in ("%TIME: =0%") do set /a T1=(1%%a-100)*3600+(1%%b-100)*60+(1%%c-100)
set /a RAN=T1-T0
if %RAN% LSS 0 set /a RAN+=86400

if %EXITCODE% EQU 0 (
  echo IMPERIUM stopped cleanly ^(exit 0^). Not restarting.
  goto done
)

REM A run that lasted a while and then died is worth restarting immediately.
REM A run that died in seconds will almost certainly die again, so slow down
REM rather than hammering the machine with a broken binary.
if %RAN% GEQ 60 (
  set /a FASTFAILS=0
) else (
  set /a FASTFAILS+=1
)

if %FASTFAILS% GEQ 5 (
  echo.
  echo ----------------------------------------------------------------------
  echo IMPERIUM has failed to stay up 5 times in a row ^(exit %EXITCODE%^).
  echo Something is wrong that restarting will not fix.
  echo Read imperium-startup.log in this folder, then run Diagnose:
  echo     IMPERIUM.exe --no-browser
  echo and open http://127.0.0.1:8787/diagnose
  echo ----------------------------------------------------------------------
  goto done
)

set /a WAIT=5
if %FASTFAILS% GEQ 2 set /a WAIT=15
if %FASTFAILS% GEQ 3 set /a WAIT=60

echo.
echo IMPERIUM exited with code %EXITCODE% after %RAN%s. Restarting in %WAIT%s...
echo (Ctrl+C now to stop.)
timeout /t %WAIT% /nobreak >nul
goto loop

:done
echo.
pause
