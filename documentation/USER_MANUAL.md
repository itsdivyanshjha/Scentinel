# SCENTINEL - USER MANUAL
## Personalized Fragrance Recommendation System

---

## TABLE OF CONTENTS
1. [Project Abstract](#project-abstract)
2. [Introduction](#introduction)
3. [Objectives](#objectives)
4. [Methodology](#methodology)
5. [System Components](#system-components)
6. [Dataset View](#dataset-view)
7. [Working Modules](#working-modules)
8. [System Architecture](#system-architecture)
9. [Process Flow Diagram](#process-flow-diagram)
10. [API Structure](#api-structure)
11. [Database Structure](#database-structure)
12. [Containerization Structure](#containerization-structure)
13. [Port Mapping](#port-mapping)
14. [Laws and Regulations](#laws-and-regulations)
15. [SWOT Analysis](#swot-analysis)
16. [Key Learnings](#key-learnings)
17. [Conclusion](#conclusion)

---

## PROJECT ABSTRACT

Scentinel is an intelligent fragrance recommendation system that leverages advanced machine learning techniques to provide personalized perfume suggestions. The system addresses the challenge of fragrance discovery in a market saturated with thousands of options by learning individual preferences through an intuitive ranking interface. Using an ensemble of three specialized ML models (RankNet, Deep Preference Learning, and Bayesian Personalized Ranking), Scentinel generates recommendations that balance relevance with diversity, ensuring users discover new fragrances across different brands and fragrance families.

The system employs a modern web architecture with a React/Next.js frontend, Flask backend, and MongoDB database, all containerized using Docker for seamless deployment across platforms. With Word2Vec embeddings for semantic understanding and mathematical diversity enhancement, Scentinel achieves sub-500ms recommendation generation while maintaining high user engagement through its intuitive drag-and-drop ranking interface.

---

## INTRODUCTION

The fragrance industry presents unique challenges in product discovery due to the highly subjective and personal nature of scent preferences. Traditional recommendation systems often fail to capture the nuanced preferences that define individual fragrance choices, leading to poor user experiences and limited discovery.

Scentinel addresses these challenges by:
- **Learning from User Behavior**: Using ranking-based input to understand relative preferences
- **Ensemble ML Approach**: Combining multiple models for robust predictions
- **Diversity Enhancement**: Ensuring recommendations span different brands and fragrance families
- **Real-time Performance**: Generating recommendations in under 500ms
- **Cross-platform Compatibility**: Seamless operation on Windows, macOS, and Linux

The system transforms the fragrance discovery process from overwhelming choice overload to personalized, guided exploration, making it accessible to both fragrance novices and enthusiasts.

---

## OBJECTIVES

### Primary Objectives
1. **Personalized Recommendation System**: Develop ML models that learn individual fragrance preferences from user rankings
2. **Diversity Enhancement**: Ensure recommendations promote discovery across brands and fragrance families
3. **User Experience Excellence**: Create an intuitive interface that makes fragrance exploration engaging
4. **Performance Optimization**: Achieve sub-500ms recommendation generation times
5. **Scalable Architecture**: Build a system capable of handling multiple concurrent users

### Secondary Objectives
1. **Cold-start Solutions**: Provide meaningful recommendations for new users without ranking history
2. **Real-time Learning**: Implement adaptive models that improve with each user interaction
3. **Cross-platform Deployment**: Ensure consistent operation across different operating systems
4. **Data-driven Insights**: Generate explainable recommendations with transparency

---

## METHODOLOGY

### Machine Learning Approach

Scentinel employs a hybrid recommendation system combining three specialized models:

#### 1. RankNet Model (Pairwise Learning-to-Rank)
```python
class RankNetModel:
    def __init__(self, feature_dim=300, hidden_dim=128):
        self.model = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
```
- **Architecture**: Neural network with 300D input → 128 → 128 → 1 output
- **Training**: Learns from pairwise comparisons derived from user rankings
- **Strength**: Excellent at understanding relative preferences

#### 2. Deep Preference Learning (DPL)
```python
class DPLModel:
    def __init__(self, feature_dim=300, hidden_dim=128):
        self.model = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
```
- **Architecture**: MLP with dropout regularization
- **Training**: Direct preference score prediction from features
- **Strength**: Captures complex non-linear preference patterns

#### 3. Bayesian Personalized Ranking (BPR)
- **Architecture**: Probabilistic ranking with 50D item embeddings
- **Training**: Optimizes pairwise ranking loss with positive/negative sampling
- **Strength**: Handles sparse data and cold-start scenarios effectively

### Feature Engineering
- **Word2Vec Embeddings**: 300-dimensional semantic vectors from perfume attributes
- **Diversity Enhancement**: Mathematical score bonuses for brand and fragrance family diversity
- **Dynamic Weighting**: Adaptive model ensemble based on user data availability

### Evaluation Metrics
- **NDCG@10**: Normalized Discounted Cumulative Gain for ranking quality
- **Diversity Metrics**: Brand diversity (85%+) and note group diversity (78%+)
- **Response Time**: Sub-500ms recommendation generation
- **User Engagement**: Ranking completion rates and recommendation interactions

---

## SYSTEM COMPONENTS

### Frontend Stack
- **Next.js 13+**: React-based framework with server-side rendering
- **TypeScript**: Type-safe JavaScript for robust development
- **Tailwind CSS**: Utility-first CSS framework for responsive design
- **React Hooks**: Modern state management and component lifecycle

### Backend Stack
- **Flask**: Lightweight Python web framework
- **PyTorch**: Deep learning framework for neural network models
- **scikit-learn**: Machine learning utilities and preprocessing
- **gensim**: Word2Vec embeddings and NLP processing
- **Flask-JWT-Extended**: Secure authentication with JSON Web Tokens

### Database & Storage
- **MongoDB**: NoSQL database for flexible document storage
- **Docker Volumes**: Persistent data storage for containerized deployment

### DevOps & Deployment
- **Docker**: Containerization for consistent environments
- **Docker Compose**: Multi-service orchestration
- **Git**: Version control and collaboration

---

## DATASET VIEW

### Perfume Dataset Characteristics
- **Total Records**: 2000+ perfumes
- **Data Sources**: Fragrance community databases and commercial catalogs
- **Attributes per Perfume**:
  - Name, Brand, Gender
  - Fragrance Notes (Top, Middle, Base)
  - Description, Image URL
  - Launch Year, Price Range
  - Fragrance Family Classification

### Data Processing Pipeline
1. **Data Cleaning**: Remove duplicates, standardize formatting
2. **Feature Extraction**: Convert text attributes to numerical representations
3. **Embedding Generation**: Create 300D Word2Vec vectors
4. **Quality Validation**: Ensure data integrity and completeness

### Sample Data Structure
```json
{
  "_id": "ObjectId",
  "Name": "Chanel No. 5",
  "Brand": "Chanel",
  "Notes": "Aldehydes, Ylang-ylang, Rose, Lily of the Valley",
  "Description": "Iconic floral fragrance with timeless elegance",
  "Gender": "Women",
  "Fragrance Family": "Floral",
  "Launch Year": 1921
}
```

---

## WORKING MODULES

### 1. Authentication Module
```python
@auth_bp.route('/register', methods=['POST'])
def register():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')
    
    # Password hashing with Werkzeug security
    password_hash = generate_password_hash(password)
    
    # Store in MongoDB with unique email constraint
    user_id = mongo.db.users.insert_one({
        'email': email,
        'password': password_hash,
        'created_at': datetime.utcnow()
    }).inserted_id
```
- **Secure Registration**: Email validation and password hashing
- **JWT Authentication**: Stateless token-based authentication
- **Session Management**: Secure token handling and expiration

### 2. Perfume Management Module
```python
@perfumes_bp.route('/list', methods=['GET'])
@jwt_required()
def get_perfumes_to_rank():
    # Random sampling for ranking diversity
    pipeline = [{"$sample": {"size": 10}}]
    perfumes = list(mongo.db.perfumes.aggregate(pipeline))
    
    # Field transformation for frontend compatibility
    transformed_perfumes = []
    for perfume in perfumes:
        transformed = {
            'name': perfume.get('Name', ''),
            'brand': perfume.get('Brand', ''),
            'notes': perfume.get('Notes', '')
        }
        transformed_perfumes.append(transformed)
```
- **Random Sampling**: Efficient perfume selection for ranking sessions
- **Data Transformation**: Converting between database and frontend formats
- **Pagination Support**: Handling large dataset efficiently

### 3. Ranking Interface Module
- **Drag-and-Drop UI**: Intuitive ranking interface with real-time feedback
- **Validation**: Client and server-side ranking validation (1-10 scale)
- **Progress Tracking**: Visual indicators for ranking completion

### 4. Machine Learning Module
```python
def generate_recommendations(user_id, top_n=10, enable_diversity=True):
    # Train user-specific models
    has_rankings = self.train_models(user_id)
    
    # Get ensemble predictions
    ensemble_scores = np.zeros(len(all_perfume_ids))
    
    if self.ranknet_model:
        ranknet_scores = self.ranknet_model.predict(all_perfume_features)
        ensemble_scores += ranknet_scores * self.model_weights['ranknet']
    
    if self.dpl_model:
        dpl_scores = self.dpl_model.predict(all_perfume_features)
        ensemble_scores += dpl_scores * self.model_weights['dpl']
    
    # Apply diversity enhancement
    if enable_diversity:
        return self._enhance_recommendations_with_diversity(
            perfume_scores, all_perfumes, top_n
        )
```
- **Model Training**: Automated training pipeline for user-specific models
- **Ensemble Prediction**: Weighted combination of three ML models
- **Dynamic Weighting**: Adaptive model weights based on data availability

### 5. Recommendation Engine
- **Personalized Recommendations**: User-specific model predictions
- **Cold-Start Handling**: Pre-trained models for new users
- **Diversity Enhancement**: Mathematical diversity bonuses
- **Real-Time Generation**: Sub-500ms recommendation processing

---

## SYSTEM ARCHITECTURE

```
┌─────────────────────────────────────────────────────────────────┐
│                        Scentinel System                        │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐      │
│  │  Frontend   │────▶│   Backend   │────▶│  Database   │      │
│  │  (Next.js)  │◀────│   (Flask)   │◀────│  (MongoDB)  │      │
│  │   Port:3000 │     │  Port:5001  │     │ Port:27017  │      │
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

### Component Interaction Flow
1. **User Interface Layer**: Next.js handles user interactions and UI rendering
2. **API Gateway**: Flask routes manage request/response handling
3. **Business Logic**: Services layer contains ML models and recommendation logic
4. **Data Access**: MongoDB provides persistent storage for all system data
5. **ML Pipeline**: PyTorch models process user preferences and generate recommendations

---

## PROCESS FLOW DIAGRAM

### New User Journey
```
Landing Page → Registration → Email Verification → Ranking Tutorial → 
Perfume Ranking (10 items) → Model Training → Personalized Recommendations
```

### Returning User Journey
```
Login → Authentication → Profile Loading → Updated Recommendations → 
Optional Re-ranking → Enhanced Recommendations
```

### Recommendation Generation Flow
```
User Request → Authentication Check → Load User Models → 
Feature Extraction → Ensemble Prediction → Diversity Enhancement → 
Result Ranking → Response Delivery
```

### Model Training Pipeline
```
User Rankings → Data Validation → Feature Engineering → 
Pairwise Data Creation → Model Training (RankNet, DPL, BPR) → 
Model Validation → Model Storage → Performance Logging
```

---

## API STRUCTURE

### Authentication Endpoints
- `POST /api/auth/register` - User registration with email validation
- `POST /api/auth/login` - JWT token-based authentication
- `GET /api/auth/profile` - User profile retrieval

### Perfume Endpoints
- `GET /api/perfumes/list` - Get random perfumes for ranking (authenticated)
- `POST /api/perfumes/rank` - Submit user rankings (authenticated)
- `GET /api/perfumes/all` - Get all perfumes with pagination

### Recommendation Endpoints
- `GET /api/recommend` - Personalized recommendations (authenticated)
- `GET /api/recommend/cold-start` - Cold-start recommendations (public)
- `GET /api/recommend/popular` - Popular perfumes (public)

### Request/Response Examples
```javascript
// Authentication Request
{
  "email": "user@example.com",
  "password": "securepassword"
}

// Ranking Submission
{
  "rankings": [
    {"perfume_id": "507f1f77bcf86cd799439011", "rank": 1},
    {"perfume_id": "507f1f77bcf86cd799439012", "rank": 2}
  ]
}

// Recommendation Response
[
  {
    "_id": "507f1f77bcf86cd799439011",
    "name": "Chanel No. 5",
    "brand": "Chanel",
    "notes": "Aldehydes, Ylang-ylang, Rose",
    "description": "Iconic floral fragrance"
  }
]
```

---

## DATABASE STRUCTURE

### Collections Schema

#### Users Collection
```javascript
{
  _id: ObjectId,
  email: String (unique),
  password: String (hashed),
  created_at: Date,
  updated_at: Date,
  preferences: {
    diversity_weight: Number,
    notification_enabled: Boolean
  }
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
  "Fragrance Family": String,
  "Launch Year": Number,
  Price: Number
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
  updated_at: Date,
  session_id: String,
  confidence: Number
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
  rank_position: Number,
  created_at: Date
}
```

### Indexing Strategy
```javascript
// Performance indexes
db.users.createIndex({"email": 1}, {unique: true})
db.rankings.createIndex({"user_id": 1, "perfume_id": 1}, {unique: true})
db.perfumes.createIndex({"Brand": 1})
db.perfumes.createIndex({"Gender": 1})
db.recommendations.createIndex({"user_id": 1, "score": -1})
```

---

## CONTAINERIZATION STRUCTURE

### Docker Compose Configuration
```yaml
services:
  backend:
    build: ./backend
    platform: linux/amd64
    ports:
      - "5001:5000"
    environment:
      - FLASK_ENV=development
      - MONGO_URI=mongodb://db:27017/scentinel
    depends_on:
      - db

  frontend:
    build: ./frontend
    platform: linux/amd64
    ports:
      - "3000:3000"
    environment:
      - NEXT_PUBLIC_API_URL=http://localhost:5001
    depends_on:
      - backend

  db:
    image: mongo:latest
    platform: linux/amd64
    ports:
      - "27017:27017"
    volumes:
      - mongodb_data:/data/db
```

### Container Specifications
- **Backend Container**: Python 3.8+ with Flask and ML libraries
- **Frontend Container**: Node.js 18+ with Next.js and React
- **Database Container**: MongoDB latest with persistent volume storage
- **Volume Management**: Separate volumes for database persistence and node_modules

---

## PORT MAPPING

### Service Port Configuration
| Service | Internal Port | External Port | Purpose |
|---------|---------------|---------------|---------|
| Frontend | 3000 | 3000 | Next.js web application |
| Backend | 5000 | 5001 | Flask API server |
| MongoDB | 27017 | 27017 | Database connection |

### Network Configuration
- **Internal Communication**: Services communicate via Docker network using service names
- **External Access**: Host machine can access services through mapped ports
- **Security**: Only necessary ports are exposed externally

---

## LAWS AND REGULATIONS

### Data Protection Compliance
- **GDPR Compliance**: User data handling follows European data protection standards
- **Data Minimization**: Only necessary user data is collected and stored
- **Right to Deletion**: Users can request complete data removal
- **Consent Management**: Clear opt-in for data collection and processing

### Security Standards
- **Password Security**: Werkzeug-based password hashing with salt
- **JWT Token Security**: Secure token generation and validation
- **Input Validation**: All user inputs are validated and sanitized
- **HTTPS Enforcement**: Secure communication in production environments

### Intellectual Property
- **Dataset Usage**: Perfume data sourced from publicly available databases
- **Open Source Libraries**: All dependencies use permissive licenses
- **Code Licensing**: Clear license specification for project components

---

## SWOT ANALYSIS

### Strengths ✅
- **Advanced ML Pipeline**: Hybrid ensemble approach with three specialized models
- **Diversity Enhancement**: Mathematical approach to ensure varied recommendations
- **Scalable Architecture**: Docker-based deployment with horizontal scaling capability
- **User-Centric Design**: Intuitive ranking interface with immediate feedback
- **Real-Time Performance**: Sub-500ms recommendation generation
- **Cross-Platform Compatibility**: Consistent operation across operating systems

### Weaknesses ⚠️
- **Data Dependency**: Requires sufficient user rankings for optimal personalization
- **Computational Overhead**: ML model training requires significant processing power
- **Subjective Validation**: Difficult to objectively measure recommendation quality
- **Limited Context Awareness**: Doesn't account for seasonal or occasion-based preferences
- **Cold Start Challenge**: New users receive generic recommendations initially

### Opportunities 🚀
- **Social Features**: Integration of collaborative filtering and social recommendations
- **Context Awareness**: Seasonal, weather, and occasion-based recommendations
- **Mobile Application**: Native mobile app for enhanced user experience
- **Business Integration**: Partnership opportunities with fragrance retailers
- **Advanced Analytics**: User behavior insights and market trend analysis
- **Multi-Modal Input**: Integration of scent descriptions and user reviews

### Threats ⚡
- **Privacy Concerns**: User preference data sensitivity and regulatory compliance
- **Competition**: Established players with larger datasets and resources
- **Market Saturation**: Potential oversaturation of recommendation systems
- **Technology Evolution**: Rapid changes in ML/AI requiring constant updates
- **Data Quality Issues**: Dependence on accurate and comprehensive perfume metadata

---

## KEY LEARNINGS

### Technical Learnings
1. **Ensemble Model Benefits**: Combining multiple ML approaches provides more robust recommendations than single models
2. **Feature Engineering Importance**: Word2Vec embeddings significantly improve semantic understanding of fragrances
3. **Diversity vs Relevance Trade-off**: Balancing personalization with discovery requires careful parameter tuning
4. **Real-time Performance**: Optimization techniques are crucial for sub-500ms response times
5. **Cross-platform Deployment**: Docker containerization ensures consistent behavior across different operating systems

### User Experience Insights
1. **Ranking Interface Design**: Drag-and-drop interfaces significantly improve user engagement
2. **Cold Start Solutions**: Pre-trained models are essential for new user onboarding
3. **Feedback Loops**: Continuous learning from user interactions improves recommendation quality
4. **Visual Design Impact**: Clean, modern UI increases user trust and engagement
5. **Mobile Responsiveness**: Responsive design is crucial for user accessibility

### Development Process Learnings
1. **Microservices Architecture**: Separating frontend, backend, and database simplifies development and deployment
2. **Documentation Importance**: Comprehensive documentation accelerates development and maintenance
3. **Testing Strategy**: Automated testing for ML models requires specialized approaches
4. **Performance Monitoring**: Real-time metrics are essential for production optimization
5. **Version Control**: Proper Git workflows enable collaborative development

---

## CONCLUSION

Scentinel successfully addresses the complex challenge of personalized fragrance recommendation through innovative machine learning techniques and thoughtful system design. The project demonstrates that combining multiple specialized models can achieve superior performance compared to single-model approaches, while the emphasis on diversity enhancement ensures users discover new fragrances beyond their immediate preferences.

### Key Achievements
1. **Technical Excellence**: Implementation of a sophisticated ML pipeline with three complementary models
2. **User Experience Success**: Creation of an intuitive interface that makes fragrance discovery engaging
3. **Performance Optimization**: Achievement of sub-500ms recommendation generation times
4. **Cross-platform Compatibility**: Successful deployment across Windows, macOS, and Linux environments
5. **Scalable Architecture**: Design that supports multiple concurrent users and growing datasets

### Impact and Value
Scentinel transforms the fragrance discovery process from overwhelming choice paralysis to guided, personalized exploration. By learning individual preferences and promoting diversity, the system creates value for both consumers seeking their perfect fragrance and retailers looking to improve customer satisfaction and discovery.

### Future Potential
The foundation established by Scentinel opens numerous opportunities for enhancement, including social features, context-aware recommendations, and mobile applications. The modular architecture ensures that new features can be integrated without disrupting existing functionality.

Scentinel represents a successful fusion of advanced technology and user-centered design, demonstrating how machine learning can be applied to solve real-world problems in the consumer goods industry while maintaining simplicity and accessibility for end users.

---

*Scentinel - Where Technology Meets Scent* 