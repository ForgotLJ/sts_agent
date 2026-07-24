@echo off
setlocal

set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" (
    echo Visual Studio Installer vswhere.exe was not found.
    exit /b 1
)

for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSINSTALL=%%i"
if not defined VSINSTALL (
    echo A Visual Studio installation with the x64/x86 C++ toolset was not found.
    exit /b 1
)

call "%VSINSTALL%\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64
if errorlevel 1 exit /b %errorlevel%
set "VSLANG=1033"

where cl
where cmake
where ninja

if not defined STS_PYTHON set "STS_PYTHON=%USERPROFILE%\.conda\envs\pytorch_env\python.exe"
if not exist "%STS_PYTHON%" (
    echo Python executable was not found at "%STS_PYTHON%".
    echo Set STS_PYTHON before running this script to select another interpreter.
    exit /b 1
)

echo Using Python: %STS_PYTHON%
cmake -S "%~dp0..\vendor\sts_lightspeed" -B "%~dp0..\build\sts_lightspeed-py311" -G Ninja -DCMAKE_BUILD_TYPE=Release -DPYTHON_EXECUTABLE="%STS_PYTHON%"
exit /b %errorlevel%
