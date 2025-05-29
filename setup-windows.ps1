#!/usr/bin/env powershell
param(
    [switch]$Background
)

Write-Host "🌸 Setting up Scentinel for Windows..." -ForegroundColor Green
Write-Host "======================================================" -ForegroundColor Green

# Check if Docker is running
try {
    $dockerInfo = docker info 2>$null
    if ($LASTEXITCODE -ne 0) {
        throw "Docker not running"
    }
    Write-Host "✅ Docker is running" -ForegroundColor Green
}
catch {
    Write-Host "❌ Docker is not running. Please start Docker Desktop first." -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "`n🧹 Cleaning up problematic files..." -ForegroundColor Yellow
# Remove macOS resource fork files that cause UTF-8 encoding errors
Get-ChildItem -Path . -Name "._*" -Recurse -Force | Remove-Item -Force -ErrorAction SilentlyContinue
Get-ChildItem -Path . -Name ".DS_Store" -Recurse -Force | Remove-Item -Force -ErrorAction SilentlyContinue
Get-ChildItem -Path . -Name "Thumbs.db" -Recurse -Force | Remove-Item -Force -ErrorAction SilentlyContinue
Get-ChildItem -Path . -Name "Desktop.ini" -Recurse -Force | Remove-Item -Force -ErrorAction SilentlyContinue
Write-Host "✅ Cleanup complete" -ForegroundColor Green

Write-Host "`n🧹 Stopping any existing containers..." -ForegroundColor Yellow
docker compose down -v

Write-Host "`n🗑️ Removing any existing images to force rebuild..." -ForegroundColor Yellow
try {
    docker rmi scentinel-backend scentinel-frontend 2>$null
    Write-Host "✅ Existing images removed" -ForegroundColor Green
} catch {
    Write-Host "ℹ️ No existing images to remove" -ForegroundColor Gray
}

Write-Host "`n🧹 Cleaning up Docker system..." -ForegroundColor Yellow
docker system prune -f

Write-Host "`n📁 Creating necessary directories..." -ForegroundColor Yellow
if (!(Test-Path "data\db")) {
    New-Item -ItemType Directory -Path "data\db" -Force | Out-Null
}
if (!(Test-Path "logs")) {
    New-Item -ItemType Directory -Path "logs" -Force | Out-Null
}
Write-Host "✅ Directories created" -ForegroundColor Green

Write-Host "`n🔧 Setting up backend environment file..." -ForegroundColor Yellow
if (!(Test-Path "backend\.env")) {
    if (Test-Path "backend\env.example") {
        Copy-Item "backend\env.example" "backend\.env"
        Write-Host "✅ Created backend/.env file from template" -ForegroundColor Green
        Write-Host "⚠️  Please edit backend/.env file with your configuration before running the application" -ForegroundColor Yellow
    } else {
        Write-Host "⚠️  No backend/env.example found. You'll need to create a backend/.env file manually" -ForegroundColor Yellow
    }
} else {
    Write-Host "✅ backend/.env file already exists" -ForegroundColor Green
}

Write-Host "`n🚀 Building and starting containers..." -ForegroundColor Yellow
if ($Background) {
    docker compose up --build -d
    Write-Host "✅ Containers started in background" -ForegroundColor Green
} else {
    docker compose up --build
}

Write-Host "`n🎉 Setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "🌐 Access your application:" -ForegroundColor Cyan
Write-Host "  Frontend: http://localhost:3000" -ForegroundColor White
Write-Host "  Backend API: http://localhost:5001" -ForegroundColor White
Write-Host ""
Write-Host "🔧 Optional: Pre-train ML models for better recommendations:" -ForegroundColor Cyan
Write-Host "  .\pretraining\pretrain.bat" -ForegroundColor White
Write-Host ""
Write-Host "📖 For detailed instructions, see README.md" -ForegroundColor Cyan

if (!$Background) {
    Read-Host "`nPress Enter to exit"
} 