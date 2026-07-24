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

set "BUILD_JOBS=%~1"
if not defined BUILD_JOBS set "BUILD_JOBS=4"

cmake --build "%~dp0..\build\sts_lightspeed-py311" --parallel %BUILD_JOBS%
exit /b %errorlevel%
