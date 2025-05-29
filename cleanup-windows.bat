@echo off
echo 🧹 Cleaning up problematic files for Windows Docker build...
echo ============================================================

REM Remove macOS resource fork files that cause UTF-8 encoding errors
echo Removing macOS resource fork files...
for /r %%i in (._*) do (
    if exist "%%i" (
        echo Deleting: %%i
        del /q "%%i" 2>nul
    )
)

REM Remove .DS_Store files
echo Removing .DS_Store files...
for /r %%i in (.DS_Store) do (
    if exist "%%i" (
        echo Deleting: %%i
        del /q "%%i" 2>nul
    )
)

REM Remove other problematic files
echo Removing other OS-generated files...
for /r %%i in (Thumbs.db) do (
    if exist "%%i" (
        echo Deleting: %%i
        del /q "%%i" 2>nul
    )
)

for /r %%i in (Desktop.ini) do (
    if exist "%%i" (
        echo Deleting: %%i
        del /q "%%i" 2>nul
    )
)

echo ✅ Cleanup complete!
echo. 