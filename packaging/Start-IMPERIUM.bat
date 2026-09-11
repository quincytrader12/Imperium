@echo off
REM Start IMPERIUM and keep this window open afterwards.
REM
REM Double-clicking IMPERIUM.exe directly works too, but if it exits for any
REM reason the console closes with it and takes the error message along. This
REM wrapper always pauses, so there is something to read.

title IMPERIUM - trading terminal
cd /d "%~dp0"

echo Starting IMPERIUM...
echo.
"%~dp0IMPERIUM.exe" %*

echo.
echo ----------------------------------------------------------------------
if errorlevel 1 (
  echo IMPERIUM exited with an error ^(code %errorlevel%^).
  echo See imperium-startup.log in this folder for the full reason.
) else (
  echo IMPERIUM has stopped.
)
echo ----------------------------------------------------------------------
pause
