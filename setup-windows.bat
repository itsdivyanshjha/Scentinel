@echo off
echo 🌸 Setting up Scentinel for Windows...
echo ======================================================

REM Check if Docker is running
docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Docker is not running. Please start Docker Desktop first.
    pause
    exit /b 1
)

echo ✅ Docker is running

echo.
echo 🧹 Cleaning up problematic files...
call cleanup-windows.bat

echo.
echo 🧹 Stopping any existing containers...
docker compose down -v

echo.
echo 🗑️ Removing any existing images to force rebuild...
docker rmi scentinel-backend scentinel-frontend 2>nul

echo.
echo 🧹 Cleaning up Docker system...
docker system prune -f

echo.
echo 📁 Creating necessary directories...
if not exist "data\db" mkdir data\db
if not exist "logs" mkdir logs

echo.
echo 🔧 Creating backend .env file if it doesn't exist...
if not exist "backend\.env" (
    if exist "backend\env.example" (
        copy backend\env.example backend\.env
        echo ⚠️  Please edit backend/.env file with your configuration before running the application
    ) else (
        echo ⚠️  No backend/env.example found. You'll need to create a backend/.env file manually
    )
) else (
    echo ✅ backend/.env file already exists
)

echo.
echo 🚀 Building and starting containers...
docker compose up --build

echo.
echo 🎉 Setup complete!
echo.
echo Frontend: http://localhost:3000
echo Backend API: http://localhost:5001
echo.
echo 📖 For detailed instructions, see README.md
pause 