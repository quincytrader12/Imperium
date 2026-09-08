@echo off
REM Start GODALGO and keep this window open afterwards.
REM
REM Double-clicking GODALGO.exe directly works too, but if it exits for any
REM reason the console closes with it and takes the error message along. This
REM wrapper always pauses, so there is something to read.

title GODALGO - trading terminal
cd /d "%~dp0"

echo Starting GODALGO...
echo.
"%~dp0GODALGO.exe" %*

echo.
echo ----------------------------------------------------------------------
if errorlevel 1 (
  echo GODALGO exited with an error ^(code %errorlevel%^).
  echo See godalgo-startup.log in this folder for the full reason.
) else (
  echo GODALGO has stopped.
)
echo ----------------------------------------------------------------------
pause
