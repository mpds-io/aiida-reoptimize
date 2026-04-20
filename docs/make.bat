@ECHO OFF

pushd %~dp0

set SOURCEDIR=source
set BUILDDIR=build

if "%1" == "" goto help
if "%1" == "help" (
    :help
    echo.Please use `make ^<target^>` where ^<target^> is one of
    echo.  html       to make standalone HTML files
    echo.  clean      to remove the build directory
    goto end
)
if "%1" == "clean" (
    rmdir /s /q %BUILDDIR%
    goto end
)
if "%1" == "html" (
    sphinx-build -M html %SOURCEDIR% %BUILDDIR%
    goto end
)

:end
popd
