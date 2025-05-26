# Scentinel Architecture

## Table of Contents
1. [System Overview](#system-overview)
2. [Technology Stack](#technology-stack)
3. [Component Architecture](#component-architecture)
4. [Machine Learning Pipeline](#machine-learning-pipeline)
5. [Data Flow](#data-flow)
6. [API Architecture](#api-architecture)
7. [Database Schema](#database-schema)
8. [Security Architecture](#security-architecture)
9. [Deployment Architecture](#deployment-architecture)

## System Overview

Scentinel is a sophisticated perfume recommendation system that leverages machine learning to provide personalized fragrance recommendations. The system employs a hybrid approach combining multiple ML models to learn user preferences from ranking data and generate high-quality recommendations with diversity enhancement.

### Core Principles
- **Pure Recommendation System**: No filtering mechanisms, only ML-driven recommendations
- **Ranking-Based Learning**: Models learn from user rankings (1-10 scale)
- **Diversity Enhancement**: Score-based diversity bonuses (not filtering)
- **Hybrid ML Approach**: Ensemble of three specialized models

## Technology Stack

### Frontend
- **Framework**: Next.js 13+ (React-based)
- **Language**: TypeScript
- **Styling**: Tailwind CSS
- **State Management**: React Hooks
- **HTTP Client**: Fetch API
- **Authentication**: JWT tokens

### Backend
- **Framework**: Flask (Python)
- **Language**: Python 3.8+
- **ML Libraries**: PyTorch, scikit-learn, gensim
- **Authentication**: Flask-JWT-Extended
- **Data Processing**: NumPy, Pandas
- **API**: RESTful endpoints

### Database
- **Primary Database**: MongoDB
- **Data Format**: BSON documents
- **Collections**: users, perfumes, rankings, recommendations
- **Indexing**: Optimized for recommendation queries

### Infrastructure
- **Containerization**: Docker & Docker Compose
- **Orchestration**: Docker Compose for multi-service deployment
- **Networking**: Internal Docker networks
- **Storage**: Docker volumes for persistent data

## Component Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Scentinel System                        │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐      │
│  │             │     │             │     │             │      │
│  │  Frontend   │────▶│   Backend   │────▶│  Database   │      │
│  │  (Next.js)  │◀────│   (Flask)   │◀────│  (MongoDB)  │      │
│  │             │     │             │     │             │      │
│  └─────────────┘     └─────────────┘     └─────────────┘      │
│       │                      │                     │          │
│       │                      │                     │          │
│       ▼                      ▼                     ▼          │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐      │
│  │    User     │     │     ML      │     │    Data     │      │
│  │ Interface   │     │  Pipeline   │     │  Storage    │      │
│  └─────────────┘     └─────────────┘     └─────────────┘      │
└─────────────────────────────────────────────────────────────────┘
```

### Frontend Layer (Next.js)
- **Pages**: Authentication, ranking interface, recommendations display
- **Components**: Reusable UI components for perfume cards, ranking interface
- **Services**: API communication, authentication management
- **Routing**: File-based routing with dynamic pages

### Backend Layer (Flask)
- **Routes**: API endpoints for auth, perfumes, recommendations
- **Services**: Business logic, recommendation generation
- **Models**: ML model implementations and management
- **Utils**: Database utilities, data processing helpers

### Data Layer (MongoDB)
- **Collections**: Structured data storage for all entities
- **Indexes**: Performance optimization for queries
- **Aggregation**: Complex data processing pipelines

## Machine Learning Pipeline

### Architecture Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                    ML Recommendation Pipeline                    │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐          │
│  │   User      │    │  Perfume    │    │ Historical  │          │
│  │  Rankings   │    │  Features   │    │   Data      │          │
│  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘          │
│         │                  │                  │                 │
│         └──────────────────┼──────────────────┘                 │
│                            │                                    │
│                            ▼                                    │
│         ┌─────────────────────────────────────┐                 │
│         │        Feature Engineering          │                 │
│         │   • Word2Vec Embeddings (300D)     │                 │
│         │   • Perfume Attribute Vectors      │                 │
│         │   • User Preference Profiles       │                 │
│         └─────────────────┬───────────────────┘                 │
│                           │                                     │
│                           ▼                                     │
│    ┌──────────────────────────────────────────────────────────┐ │
│    │                 Model Ensemble                           │ │
│    │                                                          │ │
│    │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │ │
│    │  │   RankNet   │  │     DPL     │  │     BPR     │     │ │
│    │  │  (Pairwise  │  │ (Preference │  │ (Bayesian   │     │ │
│    │  │  Learning)  │  │ Prediction) │  │  Ranking)   │     │ │
│    │  └─────────────┘  └─────────────┘  └─────────────┘     │ │
│    │         │               │               │              │ │
│    │         └───────────────┼───────────────┘              │ │
│    │                         │                              │ │
│    │                         ▼                              │ │
│    │              ┌─────────────────┐                       │ │
│    │              │ Score Ensemble  │                       │ │
│    │              │   & Weighting   │                       │ │
│    │              └─────────────────┘                       │ │
│    └──────────────────────┬───────────────────────────────────┘ │
│                           │                                     │
│                           ▼                                     │
│         ┌─────────────────────────────────────┐                 │
│         │      Diversity Enhancement         │                 │
│         │   • Brand Diversity Bonus         │                 │
│         │   • Note Group Diversity Bonus    │                 │
│         │   • Score-based Enhancement       │                 │
│         └─────────────────┬───────────────────┘                 │
│                           │                                     │
│                           ▼                                     │
│         ┌─────────────────────────────────────┐                 │
│         │      Final Recommendations         │                 │
│         │    (Ranked by Enhanced Scores)     │                 │
│         └─────────────────────────────────────┘                 │
└──────────────────────────────────────────────────────────────────┘
```

### Model Implementations

#### 1. RankNet Model
```python
class RankNetModel:
    """Pairwise learning-to-rank using neural networks"""
    - Architecture: 300 → 128 → 128 → 1
    - Loss Function: BCEWithLogitsLoss
    - Training: Pairwise comparisons from rankings
    - Strength: Excellent at learning relative preferences
```

#### 2. Deep Preference Learning (DPL)
```python
class DPLModel:
    """Direct preference prediction using MLP"""
    - Architecture: 300 → 128 → 64 → 1
    - Loss Function: MSELoss
    - Training: Direct score prediction
    - Strength: Captures non-linear preference patterns
```

#### 3. Bayesian Personalized Ranking (BPR)
```python
class BayesianPersonalizedRanking:
    """Probabilistic ranking optimization"""
    - Embeddings: 50-dimensional item embeddings
    - Optimization: Pairwise ranking loss
    - Training: Positive/negative item pairs
    - Strength: Handles sparse data effectively
```

### Feature Engineering

#### Word2Vec Embeddings
- **Dimension**: 300D vectors
- **Training Data**: Perfume notes, descriptions, brands
- **Vocabulary**: Built from perfume attributes
- **Purpose**: Semantic similarity between perfumes

#### Diversity Enhancement
- **Brand Diversity**: Bonus for different brands
- **Note Group Diversity**: Bonus for varied fragrance families
- **Mathematical Formula**: 
  ```
  enhanced_score = relevance_score + diversity_weight * diversity_bonus
  ```

## Data Flow

### User Journey Flow

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│             │     │             │     │             │     │             │
│  Register/  │────▶│    Rank     │────▶│   Generate  │────▶│    View     │
│   Login     │     │  Perfumes   │     │Recommendations│     │   Results   │
│             │     │             │     │             │     │             │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
        │                   │                   │                   │
        ▼                   ▼                   ▼                   ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Create    │     │   Store     │     │   Process   │     │   Display   │
│   Account   │     │  Rankings   │     │  Through ML │     │  Perfumes   │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
```

### API Request Flow

```
Frontend Request → Authentication → Route Handler → Service Layer → ML Pipeline → Database → Response
```

### Data Processing Flow

1. **Input**: User rankings (1-10 scale)
2. **Feature Extraction**: Convert to 300D embeddings
3. **Model Training**: Update user-specific models
4. **Prediction**: Generate scores for all perfumes
5. **Diversity Enhancement**: Apply diversity bonuses
6. **Ranking**: Sort by enhanced scores
7. **Output**: Top-N recommendations

## API Architecture

### Authentication Endpoints
```
POST /api/auth/register    - User registration
POST /api/auth/login       - User authentication
GET  /api/auth/profile     - User profile retrieval
```

### Perfume Endpoints
```
GET  /api/perfumes/list    - Get perfumes to rank (authenticated)
POST /api/perfumes/rank    - Submit user rankings (authenticated)
GET  /api/perfumes/all     - Get all perfumes (paginated)
```

### Recommendation Endpoints
```
GET  /api/recommend                - Personalized recommendations (authenticated)
GET  /api/recommend/cold-start     - Cold-start recommendations (public)
GET  /api/recommend/popular        - Popular perfumes (public)
```

### Request/Response Format

#### Authentication Request
```json
{
  "email": "user@example.com",
  "password": "securepassword"
}
```

#### Ranking Submission
```json
{
  "rankings": [
    {"perfume_id": "507f1f77bcf86cd799439011", "rank": 1},
    {"perfume_id": "507f1f77bcf86cd799439012", "rank": 2}
  ]
}
```

#### Recommendation Response
```json
[
  {
    "_id": "507f1f77bcf86cd799439011",
    "name": "Chanel No. 5",
    "brand": "Chanel",
    "notes": "Aldehydes, Ylang-ylang, Rose",
    "description": "Iconic floral fragrance",
    "image_url": "https://example.com/image.jpg",
    "gender": "Women"
  }
]
```

## Database Schema

### Collections Structure

#### Users Collection
```javascript
{
  _id: ObjectId,
  email: String (unique),
  password: String (hashed),
  created_at: Date,
  updated_at: Date
}
```

#### Perfumes Collection
```javascript
{
  _id: ObjectId,
  Name: String,
  Brand: String,
  Notes: String,
  Description: String,
  "Image URL": String,
  Gender: String,
  // Additional attributes...
}
```

#### Rankings Collection
```javascript
{
  _id: ObjectId,
  user_id: String,
  perfume_id: ObjectId,
  rank: Number (1-10),
  created_at: Date,
  updated_at: Date
}
```

#### Recommendations Collection
```javascript
{
  _id: ObjectId,
  user_id: String,
  perfume_id: ObjectId,
  score: Number,
  model_scores: {
    ranknet: Number,
    dpl: Number,
    bpr: Number
  },
  diversity_bonus: Number,
  created_at: Date
}
```

### Indexing Strategy
```javascript
// Performance indexes
db.rankings.createIndex({"user_id": 1, "perfume_id": 1})
db.perfumes.createIndex({"Brand": 1})
db.perfumes.createIndex({"Gender": 1})
db.recommendations.createIndex({"user_id": 1, "score": -1})
```

## Security Architecture

### Authentication & Authorization
- **JWT Tokens**: Stateless authentication
- **Password Hashing**: Werkzeug security with salt
- **Token Expiration**: Configurable token lifetime
- **Route Protection**: Decorator-based authentication

### Data Security
- **Input Validation**: Request data validation
- **SQL Injection Prevention**: MongoDB's BSON protection
- **CORS Configuration**: Controlled cross-origin requests
- **Environment Variables**: Sensitive data protection

### API Security
```python
# Example security implementation
@jwt_required()
def protected_endpoint():
    user_id = get_jwt_identity()
    # Endpoint logic
```

## Deployment Architecture

### Docker Compose Setup

```yaml
services:
  backend:
    build: ./backend
    ports: ["5001:5000"]
    environment:
      - MONGO_URI=mongodb://db:27017/scentinel
    depends_on: [db]
    
  frontend:
    build: ./frontend
    ports: ["3000:3000"]
    environment:
      - NEXT_PUBLIC_API_URL=http://localhost:5001
    depends_on: [backend]
    
  db:
    image: mongo:latest
    ports: ["27017:27017"]
    volumes: [mongodb_data:/data/db]
```

### Service Dependencies
```
Database (MongoDB) → Backend (Flask) → Frontend (Next.js)
```

### Network Architecture
- **Internal Network**: Docker Compose network for service communication
- **External Access**: Exposed ports for frontend (3000) and backend (5001)
- **Database Access**: Internal MongoDB access on port 27017

### Scalability Considerations
- **Horizontal Scaling**: Multiple backend instances with load balancer
- **Database Scaling**: MongoDB replica sets or sharding
- **Caching**: Redis for recommendation caching
- **CDN**: Static asset delivery optimization

## Performance Characteristics

### Response Times
- **Authentication**: < 100ms
- **Perfume Listing**: < 200ms
- **Recommendation Generation**: < 500ms
- **Database Queries**: < 50ms (indexed)

### Throughput
- **Concurrent Users**: 100+ (single instance)
- **Requests per Second**: 1000+ (with caching)
- **Database Operations**: 10,000+ ops/sec

### Resource Usage
- **Memory**: 512MB - 2GB (depending on model size)
- **CPU**: 1-2 cores for typical workload
- **Storage**: 1GB+ (including models and data)
- **Network**: Minimal bandwidth requirements 