# Scentinel System Workflow

## Table of Contents
1. [Overview](#overview)
2. [User Journey Workflows](#user-journey-workflows)
3. [Data Flow Architecture](#data-flow-architecture)
4. [Machine Learning Pipeline](#machine-learning-pipeline)
5. [API Request Workflows](#api-request-workflows)
6. [Database Operations](#database-operations)
7. [Frontend State Management](#frontend-state-management)
8. [Error Handling Workflows](#error-handling-workflows)
9. [Performance Optimization](#performance-optimization)

## Overview

The Scentinel system orchestrates complex workflows involving user interactions, machine learning processing, and data management. This document provides a comprehensive view of how different components interact to deliver personalized perfume recommendations.

### Core Workflow Principles

1. **User-Centric Design**: All workflows prioritize user experience
2. **ML-Driven Decisions**: Machine learning models guide recommendation logic
3. **Graceful Degradation**: System provides fallbacks for edge cases
4. **Real-Time Processing**: Immediate responses to user actions
5. **Data Consistency**: Maintains data integrity across all operations

## User Journey Workflows

### 1. New User Onboarding Workflow

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Landing   │───▶│  Register   │───▶│   Verify    │───▶│  Welcome    │
│    Page     │    │   Account   │    │   Email     │    │   Screen    │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
       │                   │                   │                   │
       ▼                   ▼                   ▼                   ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Browse    │    │   Create    │    │   Send      │    │   Start     │
│Cold-Start   │    │   User      │    │Verification │    │  Ranking    │
│Recommendations│   │  Record     │    │   Email     │    │  Process    │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

**Detailed Steps**:

1. **Landing Page Access**:
   - User visits application
   - System loads cold-start recommendations
   - No authentication required

2. **Account Registration**:
   - User provides email and password
   - System validates input format
   - Password hashed using Werkzeug security
   - User record created in MongoDB

3. **Email Verification** (Optional):
   - Verification email sent
   - User clicks verification link
   - Account status updated

4. **Welcome Experience**:
   - User guided to ranking interface
   - Introduction to recommendation system
   - First-time user tutorial

### 2. Returning User Workflow

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│    Login    │───▶│  Validate   │───▶│   Load      │───▶│   Display   │
│   Screen    │    │Credentials  │    │  Profile    │    │Personalized │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
       │                   │                   │                   │
       ▼                   ▼                   ▼                   ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Enter     │    │   Check     │    │  Retrieve   │    │   Show      │
│Credentials  │    │  Password   │    │  Rankings   │    │Recommendations│
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

**Detailed Steps**:

1. **Authentication**:
   - User enters email/password
   - System validates credentials
   - JWT token generated and stored

2. **Profile Loading**:
   - User ranking history retrieved
   - Preference patterns analyzed
   - Model weights calculated

3. **Personalized Experience**:
   - Custom recommendations generated
   - User-specific interface adaptations
   - Historical data displayed

### 3. Perfume Ranking Workflow

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Request   │───▶│   Sample    │───▶│   Display   │───▶│   Collect   │
│  Perfumes   │    │  Random     │    │  Perfumes   │    │  Rankings   │
│  to Rank    │    │  Perfumes   │    │             │    │             │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
       │                   │                   │                   │
       ▼                   ▼                   ▼                   ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Fetch     │    │   Select    │    │   Render    │    │   Submit    │
│   10 Items  │    │   Diverse   │    │   Ranking   │    │  Rankings   │
│             │    │   Sample    │    │ Interface   │    │             │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

**Detailed Steps**:

1. **Perfume Sampling**:
   ```python
   # MongoDB aggregation pipeline for random sampling
   pipeline = [{"$sample": {"size": 10}}]
   perfumes = mongo.db.perfumes.aggregate(pipeline)
   ```

2. **Interface Rendering**:
   - Drag-and-drop ranking interface
   - Perfume cards with images and details
   - Real-time ranking validation

3. **Data Submission**:
   - Rankings validated (1-10 scale)
   - Stored in rankings collection
   - Triggers model retraining

### 4. Recommendation Generation Workflow

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Request   │───▶│   Load      │───▶│   Process   │───▶│   Apply     │
│Recommendations│   │   User      │    │  Through    │    │  Diversity  │
│             │    │   Models    │    │   ML        │    │Enhancement  │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
       │                   │                   │                   │
       ▼                   ▼                   ▼                   ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Check     │    │  Retrieve   │    │  Ensemble   │    │   Rank      │
│   Auth      │    │  RankNet    │    │   Model     │    │   Final     │
│   Status    │    │  DPL, BPR   │    │ Predictions │    │   Results   │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

**Detailed Steps**:

1. **Authentication Check**:
   - Validate JWT token
   - Determine user type (new/returning)
   - Route to appropriate recommendation path

2. **Model Processing**:
   - Load user-specific models
   - Generate predictions from ensemble
   - Apply dynamic model weighting

3. **Diversity Enhancement**:
   - Calculate brand diversity bonuses
   - Apply note group diversity bonuses
   - Enhance scores mathematically

4. **Result Ranking**:
   - Sort by enhanced scores
   - Select top-N recommendations
   - Format for frontend display

## Data Flow Architecture

### 1. Frontend to Backend Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                        Frontend (Next.js)                      │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐        │
│  │    User     │───▶│   React     │───▶│    API      │        │
│  │ Interaction │    │ Components  │    │   Calls     │        │
│  └─────────────┘    └─────────────┘    └─────────────┘        │
└─────────────────────────┬───────────────────────────────────────┘
                          │ HTTP Requests (JSON)
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Backend (Flask)                         │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐        │
│  │   Route     │───▶│  Service    │───▶│    ML       │        │
│  │  Handlers   │    │   Layer     │    │  Pipeline   │        │
│  └─────────────┘    └─────────────┘    └─────────────┘        │
└─────────────────────────┬───────────────────────────────────────┘
                          │ Database Queries
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                       Database (MongoDB)                       │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐        │
│  │    Users    │    │  Perfumes   │    │  Rankings   │        │
│  │ Collection  │    │ Collection  │    │ Collection  │        │
│  └─────────────┘    └─────────────┘    └─────────────┘        │
└─────────────────────────────────────────────────────────────────┘
```

### 2. Request Processing Flow

```
HTTP Request → Authentication → Route Handler → Service Layer → ML Processing → Database Query → Response Formation → HTTP Response
```

**Detailed Processing**:

1. **Request Reception**:
   - Flask receives HTTP request
   - CORS headers validated
   - Request body parsed

2. **Authentication Layer**:
   ```python
   @jwt_required()
   def protected_endpoint():
       user_id = get_jwt_identity()
       # Process authenticated request
   ```

3. **Service Layer Processing**:
   - Business logic execution
   - Data validation
   - Error handling

4. **ML Pipeline Execution**:
   - Model loading
   - Feature extraction
   - Prediction generation

5. **Database Operations**:
   - Query execution
   - Data retrieval/storage
   - Index utilization

6. **Response Formation**:
   - Data serialization
   - JSON formatting
   - HTTP status codes

## Machine Learning Pipeline

### 1. Training Pipeline Workflow

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   User      │───▶│  Feature    │───▶│   Model     │───▶│   Model     │
│  Rankings   │    │Extraction   │    │  Training   │    │  Storage    │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
       │                   │                   │                   │
       ▼                   ▼                   ▼                   ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Validate   │    │  Generate   │    │  Train      │    │   Save      │
│  Rankings   │    │  300D       │    │  RankNet    │    │  User       │
│   (1-10)    │    │ Embeddings  │    │  DPL, BPR   │    │  Models     │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

### 2. Prediction Pipeline Workflow

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Load      │───▶│  Extract    │───▶│  Generate   │───▶│  Ensemble   │
│   User      │    │  Perfume    │    │   Model     │    │ Predictions │
│  Models     │    │ Features    │    │Predictions  │    │             │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
       │                   │                   │                   │
       ▼                   ▼                   ▼                   ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Retrieve   │    │  Word2Vec   │    │  RankNet    │    │  Weighted   │
│  RankNet    │    │ Embeddings  │    │  DPL, BPR   │    │  Average    │
│  DPL, BPR   │    │   (300D)    │    │  Scores     │    │  Ensemble   │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

### 3. Model Weight Calculation

```python
def calculate_dynamic_weights(user_ranking_count):
    """Dynamic model weighting based on data availability"""
    
    if user_ranking_count < 5:
        # Sparse data: favor BPR
        return {
            'ranknet': 0.2,
            'dpl': 0.3,
            'bpr': 0.5
        }
    elif user_ranking_count < 15:
        # Medium data: balanced approach
        return {
            'ranknet': 0.4,
            'dpl': 0.4,
            'bpr': 0.2
        }
    else:
        # Rich data: favor RankNet and DPL
        return {
            'ranknet': 0.5,
            'dpl': 0.4,
            'bpr': 0.1
        }
```

## API Request Workflows

### 1. Authentication Workflow

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   POST      │───▶│  Validate   │───▶│   Hash      │───▶│  Generate   │
│/api/auth/   │    │   Input     │    │  Password   │    │    JWT      │
│  register   │    │             │    │             │    │   Token     │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
       │                   │                   │                   │
       ▼                   ▼                   ▼                   ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Check     │    │  Email      │    │  Werkzeug   │    │   Return    │
│  Required   │    │  Format     │    │  Security   │    │   Token     │
│   Fields    │    │ Validation  │    │             │    │             │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

### 2. Recommendation Request Workflow

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│    GET      │───▶│   Verify    │───▶│   Check     │───▶│  Generate   │
│/api/        │    │    JWT      │    │  Rankings   │    │Recommendations│
│ recommend   │    │   Token     │    │   History   │    │             │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
       │                   │                   │                   │
       ▼                   ▼                   ▼                   ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Extract    │    │  Decode     │    │  Query      │    │   Apply     │
│ Parameters  │    │   User      │    │  Database   │    │  Diversity  │
│             │    │    ID       │    │             │    │Enhancement  │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

### 3. Ranking Submission Workflow

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   POST      │───▶│  Validate   │───▶│   Store     │───▶│  Trigger    │
│/api/perfumes│    │  Rankings   │    │  Rankings   │    │   Model     │
│   /rank     │    │   Format    │    │             │    │ Retraining  │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
       │                   │                   │                   │
       ▼                   ▼                   ▼                   ▼
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Check     │    │  Ensure     │    │  Update     │    │  Schedule   │
│   Auth      │    │  1-10       │    │  Database   │    │  Training   │
│  Headers    │    │   Range     │    │             │    │    Task     │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

## Database Operations

### 1. User Management Operations

```python
# User Registration
def create_user(email, password):
    """Create new user account"""
    
    # 1. Validate input
    if not email or not password:
        raise ValueError("Email and password required")
    
    # 2. Check existing user
    existing = mongo.db.users.find_one({'email': email})
    if existing:
        raise ValueError("User already exists")
    
    # 3. Hash password
    password_hash = generate_password_hash(password)
    
    # 4. Insert user
    user_id = mongo.db.users.insert_one({
        'email': email,
        'password': password_hash,
        'created_at': datetime.utcnow()
    }).inserted_id
    
    return str(user_id)
```

### 2. Ranking Storage Operations

```python
# Store User Rankings
def store_rankings(user_id, rankings):
    """Store user perfume rankings"""
    
    for ranking in rankings:
        mongo.db.rankings.update_one(
            {
                'user_id': user_id,
                'perfume_id': ranking['perfume_id']
            },
            {
                '$set': {
                    'rank': ranking['rank'],
                    'updated_at': datetime.utcnow()
                }
            },
            upsert=True
        )
```

### 3. Recommendation Retrieval Operations

```python
# Get Perfume Details
def get_perfume_details(perfume_ids):
    """Retrieve detailed perfume information"""
    
    perfumes = []
    for perfume_id in perfume_ids:
        perfume = mongo.db.perfumes.find_one({'_id': perfume_id})
        if perfume:
            # Transform field names for frontend compatibility
            transformed = {
                '_id': str(perfume['_id']),
                'name': perfume.get('Name', ''),
                'brand': perfume.get('Brand', ''),
                'notes': perfume.get('Notes', ''),
                'description': perfume.get('Description', ''),
                'image_url': perfume.get('Image URL', ''),
                'gender': perfume.get('Gender', '')
            }
            perfumes.append(transformed)
    
    return perfumes
```

## Frontend State Management

### 1. Authentication State Flow

```javascript
// Authentication state management
const AuthContext = createContext();

function AuthProvider({ children }) {
    const [user, setUser] = useState(null);
    const [token, setToken] = useState(localStorage.getItem('token'));
    
    const login = async (email, password) => {
        try {
            const response = await fetch('/api/auth/login', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ email, password })
            });
            
            const data = await response.json();
            
            if (response.ok) {
                setToken(data.access_token);
                setUser({ id: data.user_id, email });
                localStorage.setItem('token', data.access_token);
                return true;
            }
        } catch (error) {
            console.error('Login failed:', error);
        }
        return false;
    };
    
    return (
        <AuthContext.Provider value={{ user, token, login }}>
            {children}
        </AuthContext.Provider>
    );
}
```

### 2. Recommendation State Flow

```javascript
// Recommendation state management
function useRecommendations() {
    const [recommendations, setRecommendations] = useState([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const { token } = useAuth();
    
    const fetchRecommendations = async (params = {}) => {
        setLoading(true);
        setError(null);
        
        try {
            const url = new URL('/api/recommend', window.location.origin);
            Object.keys(params).forEach(key => 
                url.searchParams.append(key, params[key])
            );
            
            const response = await fetch(url, {
                headers: token ? { 'Authorization': `Bearer ${token}` } : {}
            });
            
            if (response.ok) {
                const data = await response.json();
                setRecommendations(data);
            } else {
                throw new Error('Failed to fetch recommendations');
            }
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };
    
    return { recommendations, loading, error, fetchRecommendations };
}
```

### 3. Ranking Interface State Flow

```javascript
// Ranking interface state management
function RankingInterface() {
    const [perfumes, setPerfumes] = useState([]);
    const [rankings, setRankings] = useState([]);
    const [draggedItem, setDraggedItem] = useState(null);
    
    const handleDragStart = (e, perfume) => {
        setDraggedItem(perfume);
    };
    
    const handleDrop = (e, targetIndex) => {
        e.preventDefault();
        
        if (draggedItem) {
            const newRankings = [...rankings];
            const draggedIndex = newRankings.findIndex(
                item => item.perfume_id === draggedItem._id
            );
            
            // Reorder rankings
            newRankings.splice(draggedIndex, 1);
            newRankings.splice(targetIndex, 0, {
                perfume_id: draggedItem._id,
                rank: targetIndex + 1
            });
            
            // Update ranks
            newRankings.forEach((item, index) => {
                item.rank = index + 1;
            });
            
            setRankings(newRankings);
        }
        
        setDraggedItem(null);
    };
    
    return (
        <div className="ranking-interface">
            {perfumes.map((perfume, index) => (
                <PerfumeCard
                    key={perfume._id}
                    perfume={perfume}
                    rank={rankings.find(r => r.perfume_id === perfume._id)?.rank}
                    onDragStart={(e) => handleDragStart(e, perfume)}
                    onDrop={(e) => handleDrop(e, index)}
                />
            ))}
        </div>
    );
}
```

## Error Handling Workflows

### 1. Backend Error Handling

```python
# Global error handler
@app.errorhandler(Exception)
def handle_exception(e):
    """Global exception handler"""
    
    # Log the error
    app.logger.error(f"Unhandled exception: {str(e)}")
    
    # Return appropriate error response
    if isinstance(e, ValidationError):
        return jsonify({"error": "Invalid input data"}), 400
    elif isinstance(e, AuthenticationError):
        return jsonify({"error": "Authentication required"}), 401
    elif isinstance(e, NotFoundError):
        return jsonify({"error": "Resource not found"}), 404
    else:
        return jsonify({"error": "Internal server error"}), 500

# Route-specific error handling
@recommend_bp.route('', methods=['GET'])
@jwt_required()
def get_recommendations():
    try:
        user_id = get_jwt_identity()
        recommendations = generate_recommendations(user_id)
        return jsonify(recommendations), 200
    except NoRankingsError:
        return jsonify({
            "error": "No rankings found. Please rank perfumes first.",
            "fallback_url": "/api/recommend/cold-start"
        }), 400
    except ModelNotFoundError:
        return jsonify({
            "error": "User models not available. Using default models.",
            "recommendations": get_cold_start_recommendations()
        }), 200
```

### 2. Frontend Error Handling

```javascript
// Error boundary component
class ErrorBoundary extends React.Component {
    constructor(props) {
        super(props);
        this.state = { hasError: false, error: null };
    }
    
    static getDerivedStateFromError(error) {
        return { hasError: true, error };
    }
    
    componentDidCatch(error, errorInfo) {
        console.error('Error caught by boundary:', error, errorInfo);
    }
    
    render() {
        if (this.state.hasError) {
            return (
                <div className="error-fallback">
                    <h2>Something went wrong</h2>
                    <p>Please refresh the page or try again later.</p>
                    <button onClick={() => window.location.reload()}>
                        Refresh Page
                    </button>
                </div>
            );
        }
        
        return this.props.children;
    }
}

// API error handling
async function apiCall(url, options = {}) {
    try {
        const response = await fetch(url, options);
        
        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || 'API request failed');
        }
        
        return await response.json();
    } catch (error) {
        console.error('API call failed:', error);
        throw error;
    }
}
```

## Performance Optimization

### 1. Database Query Optimization

```python
# Optimized perfume sampling
def get_random_perfumes(limit=10):
    """Efficiently sample random perfumes"""
    
    # Use MongoDB aggregation for better performance
    pipeline = [
        {"$sample": {"size": limit}},
        {"$project": {
            "Name": 1,
            "Brand": 1,
            "Notes": 1,
            "Description": 1,
            "Image URL": 1,
            "Gender": 1
        }}
    ]
    
    return list(mongo.db.perfumes.aggregate(pipeline))

# Indexed queries for recommendations
def get_user_rankings(user_id):
    """Get user rankings with optimized query"""
    
    # Uses compound index on (user_id, updated_at)
    return mongo.db.rankings.find(
        {"user_id": user_id}
    ).sort("updated_at", -1)
```

### 2. Caching Strategies

```python
# Model caching
from functools import lru_cache

@lru_cache(maxsize=100)
def load_user_models(user_id):
    """Cache user models in memory"""
    
    # Load models from disk only once
    models = {}
    model_dir = f"app/data/models/user_{user_id}"
    
    for model_name in ['ranknet', 'dpl', 'bpr']:
        model_path = f"{model_dir}/{model_name}.pkl"
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                models[model_name] = pickle.load(f)
    
    return models

# Embedding caching
@lru_cache(maxsize=1000)
def get_perfume_embedding(perfume_id):
    """Cache perfume embeddings"""
    
    # Load embedding from pre-computed cache
    return embedding_cache.get(perfume_id)
```

### 3. Frontend Performance Optimization

```javascript
// Component memoization
const PerfumeCard = React.memo(({ perfume, rank, onDragStart, onDrop }) => {
    return (
        <div 
            className="perfume-card"
            draggable
            onDragStart={onDragStart}
            onDrop={onDrop}
        >
            <img src={perfume.image_url} alt={perfume.name} loading="lazy" />
            <h3>{perfume.name}</h3>
            <p>{perfume.brand}</p>
            <span className="rank">{rank}</span>
        </div>
    );
});

// Lazy loading for recommendations
const RecommendationsList = () => {
    const [visibleCount, setVisibleCount] = useState(5);
    const { recommendations } = useRecommendations();
    
    const loadMore = useCallback(() => {
        setVisibleCount(prev => Math.min(prev + 5, recommendations.length));
    }, [recommendations.length]);
    
    return (
        <div>
            {recommendations.slice(0, visibleCount).map(perfume => (
                <PerfumeCard key={perfume._id} perfume={perfume} />
            ))}
            {visibleCount < recommendations.length && (
                <button onClick={loadMore}>Load More</button>
            )}
        </div>
    );
};
```

This comprehensive workflow documentation provides a complete understanding of how the Scentinel system operates, from user interactions to machine learning processing and data management. Each workflow is designed to ensure optimal performance, user experience, and system reliability.