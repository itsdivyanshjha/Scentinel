#!/bin/bash

echo "🌸 Setting up Scentinel - Fragrance Recommendation System"
echo "======================================================="

# Check if Docker and Docker Compose are installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

echo "✅ Docker and Docker Compose are installed"

# Create backend .env file if it doesn't exist
if [ ! -f "backend/.env" ]; then
    echo "📝 Creating backend/.env file from template..."
    if [ -f "backend/env.example" ]; then
        cp backend/env.example backend/.env
        echo "⚠️  Please edit backend/.env file with your configuration before running the application"
    else
        echo "⚠️  No backend/env.example found. You'll need to create a backend/.env file manually"
    fi
else
    echo "✅ backend/.env file already exists"
fi

# Create necessary directories
echo "📁 Creating necessary directories..."
mkdir -p data/db
mkdir -p logs

# Set permissions
chmod +x pretraining/pretrain.sh
chmod +x pretraining/pretrain.bat
chmod +x backend/entrypoint.sh

echo ""
echo "🎉 Setup complete! Next steps:"
echo ""
echo "1. Edit the backend/.env file with your configuration (if created):"
echo "   nano backend/.env"
echo ""
echo "2. Start the application:"
echo "   docker-compose up -d"
echo ""
echo "3. Wait for the services to be ready, then open:"
echo "   Frontend: http://localhost:3000"
echo "   Backend API: http://localhost:5000"
echo ""
echo "4. (Optional) Pre-train the ML models:"
echo "   ./pretraining/pretrain.sh    # On Linux/macOS"
echo "   ./pretraining/pretrain.bat   # On Windows"
echo ""
echo "📖 For detailed instructions, see README.md or INSTRUCTION_MANUAL.md" 