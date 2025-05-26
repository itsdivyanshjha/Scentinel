# Scentinel Setup Guide

This comprehensive guide provides step-by-step instructions to set up and run the Scentinel perfume recommendation system on any platform.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Quick Start (Recommended)](#quick-start-recommended)
3. [Development Setup](#development-setup)
4. [Manual Setup](#manual-setup)
5. [Verification](#verification)
6. [Troubleshooting](#troubleshooting)

## Prerequisites

Before setting up Scentinel, ensure you have the following installed:

### Required Software
- **Docker**: [Download Docker Desktop](https://docs.docker.com/get-docker/)
- **Docker Compose**: Included with Docker Desktop (or install separately for Linux)
- **Git**: [Download Git](https://git-scm.com/downloads)

### System Requirements
- **RAM**: Minimum 4GB, Recommended 8GB
- **Storage**: At least 2GB free space
- **OS**: Windows 10+, macOS 10.14+, or Linux (Ubuntu 18.04+)

## Quick Start (Recommended)

The fastest way to get Scentinel running is using Docker Compose:

### 1. Clone the Repository
```bash
git clone <your-repository-url>
cd Scentinel
```

### 2. Start the Application
```bash
docker compose up
```

This single command will:
- Build all Docker images (Frontend, Backend, Database)
- Start MongoDB database
- Initialize the database with perfume data
- Start the Flask backend API (port 5001)
- Start the Next.js frontend (port 3000)
- Set up networking between all services

### 3. Access the Application
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:5001
- **MongoDB**: localhost:27017 (for database tools)

### 4. Stop the Application
```bash
# Stop all services
docker compose down

# Stop and remove all data (fresh start)
docker compose down -v
```

## Development Setup

For active development with hot reloading:

### Backend Development
```bash
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export FLASK_ENV=development
export MONGO_URI=mongodb://localhost:27017/scentinel

# Start MongoDB (if not using Docker)
# Option 1: Use Docker for just MongoDB
docker run -d -p 27017:27017 --name scentinel-mongo mongo:latest

# Option 2: Install MongoDB locally
# Follow MongoDB installation guide for your OS

# Initialize database
python init_db.py

# Run pre-training (optional, for better recommendations)
python pretrain.py

# Start backend server
python run.py
```

### Frontend Development
```bash
cd frontend

# Install dependencies
npm install

# Set environment variables
export NEXT_PUBLIC_API_URL=http://localhost:5000

# Start development server
npm run dev
```

## Manual Setup

### Environment Configuration

Create a `.env` file in the project root (optional, defaults are provided):

```env
# Backend Configuration
FLASK_ENV=development
SECRET_KEY=your-secret-key-here
JWT_SECRET_KEY=your-jwt-secret-key-here
MONGO_URI=mongodb://db:27017/scentinel

# Frontend Configuration
NEXT_PUBLIC_API_URL=http://localhost:5001
```

### Database Initialization

The database is automatically initialized when using Docker Compose. For manual setup:

```bash
# Ensure MongoDB is running
# Then run the initialization script
cd backend
python init_db.py
```

This script will:
- Create the `scentinel` database
- Load perfume data from `perfume_data.csv`
- Create necessary collections: `users`, `perfumes`, `rankings`, `recommendations`

### Model Pre-training (Optional)

For better recommendation quality, pre-train the ML models:

```bash
# From project root
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install pandas numpy torch scikit-learn gensim python-dotenv

# Run pre-training
python standalone_pretrain.py
```

This will train three ML models:
- **RankNet**: Pairwise learning-to-rank model
- **DPL**: Deep Preference Learning model  
- **BPR**: Bayesian Personalized Ranking model

## Verification

### Health Checks

1. **Backend Health Check**:
   ```bash
   curl http://localhost:5001/health
   # Expected: {"status": "healthy"}
   ```

2. **Database Connection**:
   ```bash
   # Check if perfumes are loaded
   curl http://localhost:5001/api/perfumes/all?limit=1
   ```

3. **Frontend Access**:
   - Navigate to http://localhost:3000
   - You should see the Scentinel homepage

### Test the Complete Flow

1. **Register a new account** at http://localhost:3000/register
2. **Login** with your credentials
3. **Rank perfumes** - you'll be presented with 10 random perfumes to rank
4. **View recommendations** - get personalized recommendations based on your rankings

## Troubleshooting

### Common Issues

#### Port Conflicts
If ports 3000 or 5001 are already in use:

```bash
# Check what's using the ports
# On Windows:
netstat -ano | findstr :3000
netstat -ano | findstr :5001

# On macOS/Linux:
lsof -i :3000
lsof -i :5001

# Kill the process or change ports in docker-compose.yml
```

#### Docker Issues
```bash
# Clean up Docker resources
docker system prune -a

# Rebuild containers
docker compose build --no-cache
docker compose up
```

#### Database Connection Issues
```bash
# Check MongoDB container logs
docker compose logs db

# Reset database
docker compose down -v
docker compose up
```

#### Permission Issues (Linux/macOS)
```bash
# Fix file permissions
sudo chown -R $USER:$USER .
chmod +x backend/entrypoint.sh
```

### Platform-Specific Notes

#### Windows
- Use PowerShell or Command Prompt
- Ensure Docker Desktop is running
- Use `venv\Scripts\activate` for virtual environments

#### macOS
- Use Terminal
- May need to allow Docker in Security & Privacy settings
- Use `source venv/bin/activate` for virtual environments

#### Linux
- Install Docker and Docker Compose separately
- May need to add user to docker group: `sudo usermod -aG docker $USER`
- Use `source venv/bin/activate` for virtual environments

### Getting Help

If you encounter issues:

1. Check the logs: `docker compose logs [service-name]`
2. Verify all prerequisites are installed
3. Ensure no other services are using the required ports
4. Try a clean restart: `docker compose down -v && docker compose up`

### Performance Optimization

For better performance:

1. **Allocate more memory to Docker** (4GB+ recommended)
2. **Run pre-training** for better recommendation quality
3. **Use SSD storage** for faster database operations
4. **Close unnecessary applications** to free up resources

## Next Steps

After successful setup:

1. **Explore the API** - Check `documentation/RECOMMENDATION_SYSTEM.md`
2. **Understand the architecture** - Read `documentation/architecture.md`
3. **Learn about the ML models** - See `documentation/PRETRAIN_INSTRUCTIONS.md`
4. **Review the workflow** - Check `documentation/workflow.md` 