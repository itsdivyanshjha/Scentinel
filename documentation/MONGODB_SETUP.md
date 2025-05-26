# MongoDB Setup and Configuration Guide

## Table of Contents
1. [Overview](#overview)
2. [Docker Setup (Recommended)](#docker-setup-recommended)
3. [Local MongoDB Installation](#local-mongodb-installation)
4. [Database Schema](#database-schema)
5. [Data Initialization](#data-initialization)
6. [Performance Optimization](#performance-optimization)
7. [Backup and Recovery](#backup-and-recovery)
8. [Monitoring and Maintenance](#monitoring-and-maintenance)
9. [Troubleshooting](#troubleshooting)

## Overview

MongoDB serves as the primary database for the Scentinel perfume recommendation system, storing user accounts, perfume data, user rankings, and recommendation results. This guide covers setup, configuration, and maintenance of the MongoDB instance.

### Database Requirements

- **MongoDB Version**: 4.4 or higher
- **Storage**: Minimum 1GB, Recommended 5GB+
- **Memory**: Minimum 2GB RAM allocated
- **Collections**: users, perfumes, rankings, recommendations
- **Indexes**: Optimized for recommendation queries

### Architecture Role

```
┌─────────────────────────────────────────────────────────────────┐
│                        Scentinel System                        │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐      │
│  │  Frontend   │────▶│   Backend   │────▶│  MongoDB    │      │
│  │  (Next.js)  │     │   (Flask)   │     │ (Database)  │      │
│  └─────────────┘     └─────────────┘     └─────────────┘      │
│                                                 │              │
│                                                 ▼              │
│                                    ┌─────────────────────┐     │
│                                    │   Collections:      │     │
│                                    │   • users           │     │
│                                    │   • perfumes        │     │
│                                    │   • rankings        │     │
│                                    │   • recommendations │     │
│                                    └─────────────────────┘     │
└─────────────────────────────────────────────────────────────────┘
```

## Docker Setup (Recommended)

### 1. Using Docker Compose (Automatic)

The easiest way to set up MongoDB is through the provided Docker Compose configuration:

```bash
# Start the entire stack (includes MongoDB)
docker compose up

# Start only MongoDB
docker compose up db
```

**Docker Compose Configuration**:
```yaml
services:
  db:
    image: mongo:latest
    container_name: scentinel-mongo
    restart: unless-stopped
    ports:
      - "27017:27017"
    volumes:
      - mongodb_data:/data/db
    environment:
      - MONGO_INITDB_DATABASE=scentinel
    networks:
      - scentinel-network

volumes:
  mongodb_data:
    driver: local

networks:
  scentinel-network:
    driver: bridge
```

### 2. Manual Docker Setup

If you prefer to run MongoDB separately:

```bash
# Pull MongoDB image
docker pull mongo:latest

# Run MongoDB container
docker run -d \
  --name scentinel-mongo \
  -p 27017:27017 \
  -v mongodb_data:/data/db \
  -e MONGO_INITDB_DATABASE=scentinel \
  mongo:latest

# Verify container is running
docker ps | grep scentinel-mongo
```

### 3. Docker MongoDB Management

```bash
# Access MongoDB shell
docker exec -it scentinel-mongo mongosh

# View MongoDB logs
docker logs scentinel-mongo

# Stop MongoDB
docker stop scentinel-mongo

# Start MongoDB
docker start scentinel-mongo

# Remove MongoDB container (data preserved in volume)
docker rm scentinel-mongo

# Remove MongoDB data (WARNING: This deletes all data)
docker volume rm mongodb_data
```

## Local MongoDB Installation

### 1. Windows Installation

```bash
# Download MongoDB Community Server
# Visit: https://www.mongodb.com/try/download/community

# Install using MSI installer
# Default installation path: C:\Program Files\MongoDB\Server\{version}\bin

# Add to PATH environment variable
# Add: C:\Program Files\MongoDB\Server\{version}\bin

# Start MongoDB service
net start MongoDB

# Connect to MongoDB
mongosh
```

### 2. macOS Installation

```bash
# Using Homebrew
brew tap mongodb/brew
brew install mongodb-community

# Start MongoDB service
brew services start mongodb/brew/mongodb-community

# Connect to MongoDB
mongosh

# Stop MongoDB service
brew services stop mongodb/brew/mongodb-community
```

### 3. Linux Installation (Ubuntu/Debian)

```bash
# Import MongoDB public GPG key
wget -qO - https://www.mongodb.org/static/pgp/server-6.0.asc | sudo apt-key add -

# Add MongoDB repository
echo "deb [ arch=amd64,arm64 ] https://repo.mongodb.org/apt/ubuntu focal/mongodb-org/6.0 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-6.0.list

# Update package database
sudo apt-get update

# Install MongoDB
sudo apt-get install -y mongodb-org

# Start MongoDB service
sudo systemctl start mongod
sudo systemctl enable mongod

# Connect to MongoDB
mongosh
```

## Database Schema

### 1. Users Collection

```javascript
// Collection: users
{
  _id: ObjectId("507f1f77bcf86cd799439011"),
  email: "user@example.com",
  password: "$2b$12$hashedpassword...",  // Werkzeug hashed
  created_at: ISODate("2023-01-01T00:00:00Z"),
  updated_at: ISODate("2023-01-01T00:00:00Z"),
  // Optional fields
  first_name: "John",
  last_name: "Doe",
  preferences: {
    diversity_weight: 0.3,
    notification_enabled: true
  }
}

// Indexes
db.users.createIndex({ "email": 1 }, { unique: true })
db.users.createIndex({ "created_at": 1 })
```

### 2. Perfumes Collection

```javascript
// Collection: perfumes
{
  _id: ObjectId("507f1f77bcf86cd799439012"),
  Name: "Chanel No. 5",
  Brand: "Chanel",
  Notes: "Aldehydes, Ylang-ylang, Rose, Lily of the Valley",
  Description: "Iconic floral fragrance with timeless elegance",
  "Image URL": "https://example.com/chanel-no5.jpg",
  Gender: "Women",
  // Additional attributes
  "Fragrance Family": "Floral",
  "Top Notes": "Aldehydes, Ylang-ylang",
  "Middle Notes": "Rose, Lily of the Valley",
  "Base Notes": "Sandalwood, Vetiver",
  "Launch Year": 1921,
  Price: 120.00,
  Volume: "100ml"
}

// Indexes
db.perfumes.createIndex({ "Brand": 1 })
db.perfumes.createIndex({ "Gender": 1 })
db.perfumes.createIndex({ "Name": "text", "Brand": "text", "Notes": "text" })
db.perfumes.createIndex({ "Fragrance Family": 1 })
```

### 3. Rankings Collection

```javascript
// Collection: rankings
{
  _id: ObjectId("507f1f77bcf86cd799439013"),
  user_id: "507f1f77bcf86cd799439011",  // String reference to user._id
  perfume_id: ObjectId("507f1f77bcf86cd799439012"),
  rank: 1,  // 1-10 scale (1 = most preferred)
  created_at: ISODate("2023-01-01T00:00:00Z"),
  updated_at: ISODate("2023-01-01T00:00:00Z"),
  // Optional metadata
  session_id: "ranking_session_123",
  confidence: 0.85,  // User confidence in ranking
  context: "evening_wear"  // Ranking context
}

// Indexes
db.rankings.createIndex({ "user_id": 1, "perfume_id": 1 }, { unique: true })
db.rankings.createIndex({ "user_id": 1, "updated_at": -1 })
db.rankings.createIndex({ "perfume_id": 1 })
db.rankings.createIndex({ "rank": 1 })
```

### 4. Recommendations Collection

```javascript
// Collection: recommendations
{
  _id: ObjectId("507f1f77bcf86cd799439014"),
  user_id: "507f1f77bcf86cd799439011",
  perfume_id: ObjectId("507f1f77bcf86cd799439012"),
  score: 0.87,  // Final recommendation score
  model_scores: {
    ranknet: 0.85,
    dpl: 0.82,
    bpr: 0.91
  },
  diversity_bonus: 0.05,
  rank_position: 1,  // Position in recommendation list
  created_at: ISODate("2023-01-01T00:00:00Z"),
  // Metadata
  model_version: "v1.0",
  diversity_weight: 0.3,
  explanation: "Recommended based on your preference for floral fragrances"
}

// Indexes
db.recommendations.createIndex({ "user_id": 1, "score": -1 })
db.recommendations.createIndex({ "user_id": 1, "created_at": -1 })
db.recommendations.createIndex({ "perfume_id": 1 })
```

## Data Initialization

### 1. Automatic Initialization (Docker Compose)

When using Docker Compose, the database is automatically initialized:

```bash
# Start the system - initialization happens automatically
docker compose up

# Check initialization status
docker compose logs backend | grep "Database initialized"
```

### 2. Manual Initialization

```bash
# Navigate to backend directory
cd backend

# Ensure MongoDB is running
# Then run initialization script
python init_db.py
```

### 3. Initialization Script Details

```python
# backend/init_db.py
import pandas as pd
from pymongo import MongoClient
from bson import ObjectId
import os

def initialize_database():
    """Initialize MongoDB with perfume data"""
    
    # Connect to MongoDB
    client = MongoClient('mongodb://localhost:27017/')
    db = client['scentinel']
    
    # Clear existing data (optional)
    db.perfumes.delete_many({})
    print("Cleared existing perfume data")
    
    # Load perfume data from CSV
    csv_path = 'perfume_data.csv'
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found")
        return False
    
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} perfumes from CSV")
    
    # Convert DataFrame to documents
    perfumes = []
    for _, row in df.iterrows():
        perfume = {
            '_id': ObjectId(),
            'Name': row.get('Name', ''),
            'Brand': row.get('Brand', ''),
            'Notes': row.get('Notes', ''),
            'Description': row.get('Description', ''),
            'Image URL': row.get('Image URL', ''),
            'Gender': row.get('Gender', '')
        }
        
        # Add optional fields if present
        optional_fields = [
            'Fragrance Family', 'Top Notes', 'Middle Notes', 
            'Base Notes', 'Launch Year', 'Price', 'Volume'
        ]
        for field in optional_fields:
            if field in row and pd.notna(row[field]):
                perfume[field] = row[field]
        
        perfumes.append(perfume)
    
    # Insert perfumes into database
    if perfumes:
        result = db.perfumes.insert_many(perfumes)
        print(f"Inserted {len(result.inserted_ids)} perfumes")
    
    # Create indexes
    create_indexes(db)
    
    print("Database initialization completed successfully")
    return True

def create_indexes(db):
    """Create database indexes for performance"""
    
    # Users collection indexes
    db.users.create_index([("email", 1)], unique=True)
    db.users.create_index([("created_at", 1)])
    
    # Perfumes collection indexes
    db.perfumes.create_index([("Brand", 1)])
    db.perfumes.create_index([("Gender", 1)])
    db.perfumes.create_index([
        ("Name", "text"), 
        ("Brand", "text"), 
        ("Notes", "text")
    ])
    
    # Rankings collection indexes
    db.rankings.create_index([
        ("user_id", 1), 
        ("perfume_id", 1)
    ], unique=True)
    db.rankings.create_index([("user_id", 1), ("updated_at", -1)])
    
    # Recommendations collection indexes
    db.recommendations.create_index([("user_id", 1), ("score", -1)])
    db.recommendations.create_index([("user_id", 1), ("created_at", -1)])
    
    print("Database indexes created successfully")

if __name__ == "__main__":
    initialize_database()
```

### 4. Data Validation

```bash
# Connect to MongoDB and verify data
mongosh

# Switch to scentinel database
use scentinel

# Check collections
show collections

# Verify perfume data
db.perfumes.countDocuments()
db.perfumes.findOne()

# Check indexes
db.perfumes.getIndexes()
db.users.getIndexes()
db.rankings.getIndexes()
db.recommendations.getIndexes()
```

## Performance Optimization

### 1. Index Optimization

```javascript
// Compound indexes for common queries
db.rankings.createIndex({ "user_id": 1, "updated_at": -1 })
db.recommendations.createIndex({ "user_id": 1, "score": -1 })

// Text search index for perfume search
db.perfumes.createIndex({
  "Name": "text",
  "Brand": "text", 
  "Notes": "text",
  "Description": "text"
})

// Sparse indexes for optional fields
db.perfumes.createIndex({ "Launch Year": 1 }, { sparse: true })
db.perfumes.createIndex({ "Price": 1 }, { sparse: true })
```

### 2. Query Optimization

```javascript
// Efficient random sampling
db.perfumes.aggregate([
  { $sample: { size: 10 } },
  { $project: { Name: 1, Brand: 1, Notes: 1, "Image URL": 1 } }
])

// Optimized user rankings retrieval
db.rankings.find({ "user_id": "user123" })
  .sort({ "updated_at": -1 })
  .limit(50)

// Efficient recommendation queries
db.recommendations.find({ "user_id": "user123" })
  .sort({ "score": -1 })
  .limit(10)
```

### 3. Connection Optimization

```python
# Backend connection configuration
from pymongo import MongoClient

# Optimized connection settings
client = MongoClient(
    'mongodb://localhost:27017/',
    maxPoolSize=50,
    minPoolSize=10,
    maxIdleTimeMS=30000,
    serverSelectionTimeoutMS=5000,
    socketTimeoutMS=20000
)
```

## Backup and Recovery

### 1. Docker Volume Backup

```bash
# Create backup of MongoDB data volume
docker run --rm \
  -v mongodb_data:/data \
  -v $(pwd):/backup \
  ubuntu tar czf /backup/mongodb_backup_$(date +%Y%m%d_%H%M%S).tar.gz -C /data .

# Restore from backup
docker run --rm \
  -v mongodb_data:/data \
  -v $(pwd):/backup \
  ubuntu tar xzf /backup/mongodb_backup_YYYYMMDD_HHMMSS.tar.gz -C /data
```

### 2. MongoDB Dump/Restore

```bash
# Create database dump
mongodump --db scentinel --out ./backup/

# Restore database
mongorestore --db scentinel ./backup/scentinel/

# Export specific collection
mongoexport --db scentinel --collection perfumes --out perfumes.json

# Import collection
mongoimport --db scentinel --collection perfumes --file perfumes.json
```

### 3. Automated Backup Script

```bash
#!/bin/bash
# backup_mongodb.sh

BACKUP_DIR="/backup/mongodb"
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_NAME="scentinel_backup_$DATE"

# Create backup directory
mkdir -p $BACKUP_DIR

# Create MongoDB dump
docker exec scentinel-mongo mongodump --db scentinel --out /tmp/backup

# Copy backup from container
docker cp scentinel-mongo:/tmp/backup $BACKUP_DIR/$BACKUP_NAME

# Compress backup
tar -czf $BACKUP_DIR/$BACKUP_NAME.tar.gz -C $BACKUP_DIR $BACKUP_NAME

# Remove uncompressed backup
rm -rf $BACKUP_DIR/$BACKUP_NAME

# Keep only last 7 backups
find $BACKUP_DIR -name "*.tar.gz" -mtime +7 -delete

echo "Backup completed: $BACKUP_DIR/$BACKUP_NAME.tar.gz"
```

## Monitoring and Maintenance

### 1. Database Statistics

```javascript
// Database statistics
db.stats()

// Collection statistics
db.perfumes.stats()
db.users.stats()
db.rankings.stats()
db.recommendations.stats()

// Index usage statistics
db.perfumes.aggregate([{ $indexStats: {} }])
```

### 2. Performance Monitoring

```javascript
// Current operations
db.currentOp()

// Slow query profiling
db.setProfilingLevel(2, { slowms: 100 })
db.system.profile.find().sort({ ts: -1 }).limit(5)

// Connection statistics
db.serverStatus().connections
```

### 3. Maintenance Tasks

```javascript
// Compact collections (reclaim space)
db.runCommand({ compact: "perfumes" })

// Rebuild indexes
db.perfumes.reIndex()

// Validate collection integrity
db.perfumes.validate()

// Clean up old recommendations (older than 30 days)
db.recommendations.deleteMany({
  "created_at": { 
    $lt: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000) 
  }
})
```

## Troubleshooting

### 1. Connection Issues

```bash
# Check if MongoDB is running
docker ps | grep mongo

# Check MongoDB logs
docker logs scentinel-mongo

# Test connection
mongosh --eval "db.adminCommand('ping')"

# Check port availability
netstat -an | grep 27017
```

### 2. Performance Issues

```javascript
// Check slow queries
db.setProfilingLevel(2, { slowms: 100 })
db.system.profile.find({ "millis": { $gt: 100 } }).sort({ ts: -1 })

// Analyze query performance
db.perfumes.find({ "Brand": "Chanel" }).explain("executionStats")

// Check index usage
db.perfumes.aggregate([{ $indexStats: {} }])
```

### 3. Data Issues

```javascript
// Check data consistency
db.rankings.find({ "user_id": { $exists: false } })
db.rankings.find({ "rank": { $not: { $gte: 1, $lte: 10 } } })

// Find orphaned records
db.rankings.find({
  "perfume_id": { 
    $nin: db.perfumes.distinct("_id") 
  }
})

// Repair data inconsistencies
db.rankings.deleteMany({ "rank": { $not: { $gte: 1, $lte: 10 } } })
```

### 4. Common Error Solutions

#### "Connection Refused"
```bash
# Check if MongoDB container is running
docker ps | grep mongo

# Restart MongoDB container
docker restart scentinel-mongo

# Check Docker network
docker network ls
docker network inspect scentinel_scentinel-network
```

#### "Database Not Found"
```bash
# Initialize database
cd backend && python init_db.py

# Verify database creation
mongosh --eval "show dbs"
```

#### "Index Build Failed"
```javascript
// Drop and recreate problematic index
db.perfumes.dropIndex("Name_text_Brand_text_Notes_text")
db.perfumes.createIndex({
  "Name": "text",
  "Brand": "text", 
  "Notes": "text"
})
```

### 5. Recovery Procedures

```bash
# Complete system recovery
docker compose down
docker volume rm mongodb_data  # WARNING: Deletes all data
docker compose up

# Restore from backup
docker compose down
# Restore volume from backup (see backup section)
docker compose up

# Reinitialize database
cd backend
python init_db.py
```

This comprehensive MongoDB setup guide ensures proper configuration, optimization, and maintenance of the database component in the Scentinel system. 