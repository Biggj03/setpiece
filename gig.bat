@echo off
REM One-command launcher for the Resolume control rig (Windows).
REM Double-click this, or run it from a terminal. Close the window to stop.
cd /d "%~dp0"
python gig.py %*
REM Keep the window open on error so you can read what went wrong.
if errorlevel 1 pause
