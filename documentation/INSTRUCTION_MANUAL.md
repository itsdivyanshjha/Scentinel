# SCENTINEL - Instruction Manual
## Personalized Fragrance Recommendation System

---

## Table of Contents
1. [Introduction](#introduction)
2. [System Overview](#system-overview)
3. [Prerequisites](#prerequisites)
4. [Installation Guide](#installation-guide)
5. [Component Architecture](#component-architecture)
6. [Running the Application](#running-the-application)
7. [User Guide](#user-guide)
8. [Developer Guide](#developer-guide)
9. [Troubleshooting](#troubleshooting)
10. [Maintenance](#maintenance)
11. [FAQ](#faq)

---

## Introduction

### What is Scentinel?
Scentinel is an AI-powered fragrance recommendation system that learns your personal scent preferences through an intuitive ranking interface. Using advanced machine learning algorithms, it provides personalized perfume recommendations from a database of 2000+ fragrances.

### Why This Manual?
This instruction manual will help you:
- **Successfully deploy** the Scentinel application in any environment
- **Understand** how each component works and interacts
- **Troubleshoot** common issues and maintain the system
- **Customize** the application for your specific needs

### Who Should Use This Manual?
- **System Administrators** deploying the application
- **Developers** contributing to or customizing the codebase
- **End Users** wanting to understand the application functionality
- **Data Scientists** working with the recommendation algorithms

---

## System Overview

### Core Functionality
Scentinel provides three main services:
1. **Matching Service**: Finds perfumes similar to user preferences
2. **Recommendation Service**: Suggests new fragrances based on learned preferences
3. **Ranking Interface**: Allows users to train the system with their preferences

### Technology Stack
- **Frontend**: Next.js 13+ with TypeScript and Tailwind CSS
- **Backend**: Flask with Python 3.8+
- **Database**: MongoDB for flexible document storage
- **ML Framework**: PyTorch with scikit-learn
- **Deployment**: Docker and Docker Compose

### Data Source
The application uses the [Perfume Recommendation Dataset](https://www.kaggle.com/datasets/nandini1999/perfume-recommendation-dataset) containing comprehensive perfume information including notes, brands, and characteristics.

---

## Prerequisites

### System Requirements
- **Operating System**: Windows 10+, macOS 10.15+, or Linux (Ubuntu 18.04+)
- **RAM**: Minimum 8GB (16GB recommended for ML training)
- **Storage**: 10GB free space
- **Network**: Internet connection for initial setup

### Required Software
- **Docker**: Version 20.10+ ([Download Docker](https://docs.docker.com/get-docker/))
- **Docker Compose**: Version 2.0+ (included with Docker Desktop)
- **Git**: For cloning the repository ([Download Git](https://git-scm.com/downloads))

### Optional Tools
- **Python 3.8+**: For standalone development
- **Node.js 16+**: For frontend development
- **MongoDB Compass**: For database management

---

## Installation Guide

### Step 1: Clone the Repository
```bash
# Clone the Scentinel repository
git clone https://github.com/your-username/scentinel.git

# Navigate to the project directory
cd scentinel
```

### Step 2: Verify Docker Installation
```bash
# Check Docker version
docker --version

# Check Docker Compose version
docker-compose --version

# Ensure Docker daemon is running
docker info
```

### Step 3: Environment Setup
```bash
# Create necessary directories (if not present)
mkdir -p data/mongo-init

# Verify project structure
ls -la
# You should see: backend/, frontend/, data/, docker-compose.yml
```

### Step 4: Build and Start Services
```bash
# Build and start all services
docker-compose up --build

# Or run in detached mode
docker-compose up --build -d
```

### Step 5: Initialize Database
```bash
# Wait for services to start (about 2-3 minutes)
# Then initialize the database with perfume data
docker-compose exec backend python init_db.py
```

### Step 6: Verify Installation
Open your browser and navigate to:
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:5001/health
- **MongoDB**: mongodb://localhost:27017 (if using MongoDB Compass)

---

## Component Architecture

### 1. Frontend Component (Next.js)
**Location**: `./frontend/`
**Port**: 3000
**Purpose**: User interface for perfume ranking and recommendations

#### Key Files:
- `pages/`: Next.js page components
- `components/`: Reusable React components
- `contexts/`: React context for state management
- `styles/`: Tailwind CSS styling

#### Functionality:
- User authentication and registration
- Drag-and-drop perfume ranking interface
- Recommendation display and filtering
- Responsive design for mobile and desktop

### 2. Backend Component (Flask)
**Location**: `./backend/`
**Port**: 5001
**Purpose**: API server and machine learning pipeline

#### Key Files:
- `app/`: Flask application modules
- `run.py`: Application entry point
- `requirements.txt`: Python dependencies
- `pretrain.py`: ML model pretraining script

#### Functionality:
- RESTful API endpoints
- User authentication with JWT
- Machine learning model training and inference
- Database operations and data processing

### 3. Database Component (MongoDB)
**Location**: Containerized service
**Port**: 27017
**Purpose**: Data storage for users, perfumes, and rankings

#### Collections:
- `users`: User accounts and authentication data
- `perfumes`: Perfume database with attributes and embeddings
- `rankings`: User preference rankings
- `models`: Trained ML model parameters

### 4. Machine Learning Pipeline
**Components**: RankNet, Deep Preference Learning (DPL), Bayesian Personalized Ranking (BPR)
**Purpose**: Generate personalized recommendations

#### Process Flow:
1. **Feature Engineering**: Convert perfume attributes to 300D embeddings
2. **Model Training**: Train ensemble of three ML models
3. **Prediction**: Generate preference scores for all perfumes
4. **Diversity Enhancement**: Apply bonuses for brand and note diversity
5. **Ranking**: Return top-N recommendations

### 4. Model Pre-training (Optional but Recommended)

For better recommendation quality, pre-train the machine learning models:

```bash
# From project root
./pretraining/pretrain.sh    # Linux/macOS
./pretraining/pretrain.bat   # Windows

# Or run directly
python pretraining/standalone_pretrain.py
```

---

## Running the Application

### Standard Startup
```bash
# Start all services
docker-compose up

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Development Mode
```bash
# Start with file watching for development
docker-compose up --build

# Restart specific service
docker-compose restart backend
docker-compose restart frontend
```

### Production Deployment
```bash
# Start in detached mode
docker-compose up -d --build

# Check service status
docker-compose ps

# View resource usage
docker stats
```

### Service Management
```bash
# Stop all services
docker-compose down

# Stop and remove volumes (WARNING: This deletes data)
docker-compose down -v

# Rebuild specific service
docker-compose build backend
docker-compose up backend
```

---

## User Guide

### Getting Started

#### 1. Create an Account
1. Navigate to http://localhost:3000
2. Click "Sign Up" 
3. Enter email and password
4. Verify email (if email service is configured)

#### 2. Complete Initial Ranking
1. After login, you'll see 10 random perfumes
2. Drag and drop perfumes to rank them from 1 (least preferred) to 10 (most preferred)
3. Click "Submit Rankings" to train your personal model

#### 3. View Recommendations
1. After ranking, navigate to "Recommendations"
2. View your personalized perfume suggestions
3. Filter by brand, notes, or price range
4. Click on perfumes for detailed information

#### 4. Improve Recommendations
1. Rank additional perfumes to improve accuracy
2. Rate recommended perfumes you've tried
3. Update your preferences over time

### Advanced Features

#### Preference Analysis
- View your preference patterns in the "Profile" section
- See which notes and brands you prefer
- Track your ranking history

#### Recommendation Explanations
- Each recommendation includes similarity scores
- See which attributes influenced the recommendation
- Understand why certain perfumes were suggested

---

## Developer Guide

### Local Development Setup

#### Backend Development
```bash
# Navigate to backend directory
cd backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export FLASK_ENV=development
export MONGO_URI=mongodb://localhost:27017/scentinel

# Run Flask development server
python run.py
```

#### Frontend Development
```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build
```

### API Endpoints

#### Authentication
- `POST /api/auth/register` - User registration
- `POST /api/auth/login` - User login
- `POST /api/auth/logout` - User logout

#### Perfumes
- `GET /api/perfumes/random` - Get random perfumes for ranking
- `GET /api/perfumes/search` - Search perfumes by criteria
- `GET /api/perfumes/{id}` - Get specific perfume details

#### Rankings
- `POST /api/rankings` - Submit user rankings
- `GET /api/rankings/user` - Get user's ranking history

#### Recommendations
- `GET /api/recommendations` - Get personalized recommendations
- `POST /api/recommendations/feedback` - Submit recommendation feedback

### Database Schema

#### Users Collection
```json
{
  "_id": "ObjectId",
  "email": "string",
  "password_hash": "string",
  "created_at": "datetime",
  "preferences": {
    "favorite_notes": ["array"],
    "favorite_brands": ["array"]
  }
}
```

#### Perfumes Collection
```json
{
  "_id": "ObjectId",
  "name": "string",
  "brand": "string",
  "notes": {
    "top": ["array"],
    "middle": ["array"],
    "base": ["array"]
  },
  "embeddings": [300],
  "price_range": "string"
}
```