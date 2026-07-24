@echo off
setlocal

if not defined STS_PYTHON set "STS_PYTHON=%USERPROFILE%\.conda\envs\pytorch_env\python.exe"
if not exist "%STS_PYTHON%" (
    echo Python executable was not found at "%STS_PYTHON%".
    echo Set STS_PYTHON before running this script to select another interpreter.
    exit /b 1
)

"%STS_PYTHON%" "%~dp0check-lightspeed.py" %*
exit /b %errorlevel%
