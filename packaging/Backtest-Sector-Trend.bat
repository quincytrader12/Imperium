@echo off
REM ======================================================================
REM  IMPERIUM - backtest the Sector Trend sleeve
REM
REM  Reads twenty years of adjusted daily bars with the Alpaca key already
REM  stored in the terminal, walks the strategy through them, and prints the
REM  report. It places no orders, arms nothing, and changes no settings: this
REM  only reads.
REM
REM  It takes a few minutes, most of it downloading bars.
REM
REM  It reads the same settings the live sleeve does, so it measures the
REM  strategy you are about to run rather than the defaults:
REM      "%USERPROFILE%\.imperium\settings.txt"
REM
REM  Run it before you let the sleeve arm. The report ends with the published
REM  figures from the paper the strategy comes from, and a list of anything in
REM  your own result that does not look like them -- a backtest that comes out
REM  far better than the paper is a bug until you have found out why.
REM
REM  Other ways to run it:
REM    IMPERIUM.exe --backtest --start 2010-01-01   a shorter history
REM    IMPERIUM.exe --backtest --csv .\bars         your own bars instead
REM    IMPERIUM.exe --backtest --help               everything it takes
REM ======================================================================

title IMPERIUM - Sector Trend backtest
cd /d "%~dp0"

echo Backtesting the Sector Trend sleeve. Nothing is ordered and nothing is armed.
echo Fetching bars takes a few minutes.
echo.
"%~dp0IMPERIUM.exe" --backtest %*

echo.
echo ----------------------------------------------------------------------
if errorlevel 1 (
  echo The backtest did not finish ^(code %errorlevel%^).
  echo The most common reason is no Alpaca key: start IMPERIUM, open the
  echo Connections panel, add one, then run this again.
) else (
  echo Backtest finished. Read the warnings before believing the numbers.
)
echo ----------------------------------------------------------------------
pause
