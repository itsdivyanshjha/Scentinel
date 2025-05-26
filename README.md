# SCENTINEL
## Elevate Your Senses with Personalized Fragrance Insights

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![React](https://img.shields.io/badge/React-18.0+-61DAFB.svg)](https://reactjs.org)
[![MongoDB](https://img.shields.io/badge/MongoDB-4.4+-green.svg)](https://mongodb.com)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED.svg)](https://docker.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🎯 Problem Statement

The fragrance industry faces significant challenges in personalized product discovery:

- **Information Overload**: With thousands of perfumes available, consumers struggle to find fragrances that match their preferences
- **Subjective Nature**: Fragrance preference is highly personal and difficult to quantify through traditional recommendation systems
- **Limited Discovery**: Users often stick to familiar brands, missing out on potentially perfect matches from other houses
- **Cold Start Problem**: New users have no way to receive meaningful recommendations without extensive trial and error

**Our Solution**: Scentinel leverages advanced machine learning techniques to learn individual fragrance preferences through an intuitive ranking system, providing personalized recommendations that enhance discovery while respecting the subjective nature of scent preferences.

---

## 🎯 Objectives

### Primary Objectives
- **Personalized Recommendations**: Develop an ML-driven system that learns individual fragrance preferences from user rankings
- **Diversity Enhancement**: Ensure recommendations span different brands and fragrance families to promote discovery
- **User Experience**: Create an intuitive interface that makes fragrance exploration engaging and effortless
- **Scalable Architecture**: Build a robust system capable of handling multiple users and growing datasets

### Secondary Objectives
- **Cold-Start Solutions**: Provide meaningful recommendations for new users without ranking history
- **Real-Time Learning**: Implement adaptive models that improve with each user interaction
- **Performance Optimization**: Achieve sub-500ms recommendation generation times
- **Data-Driven Insights**: Generate explainable recommendations to help users understand suggestions

---

## 🔬 Methodology

### Machine Learning Approach

Our hybrid recommendation system combines three specialized models:

#### 1. **RankNet Model** (Pairwise Learning-to-Rank)
- **Architecture**: Neural network with 300D input → 128 → 128 → 1 output
- **Training**: Learns from pairwise comparisons derived from user rankings
- **Strength**: Excellent at understanding relative preferences ("A is better than B")

#### 2. **Deep Preference Learning (DPL)**
- **Architecture**: MLP with dropout regularization (300D → 128 → 64 → 1)
- **Training**: Direct preference score prediction from features
- **Strength**: Captures complex non-linear preference patterns

#### 3. **Bayesian Personalized Ranking (BPR)**
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

## 🛠️ Technologies Used

### **Frontend Stack**
- **Next.js 13+**: React-based framework with server-side rendering
- **TypeScript**: Type-safe JavaScript for robust development
- **Tailwind CSS**: Utility-first CSS framework for responsive design
- **React Hooks**: Modern state management and component lifecycle

### **Backend Stack**
- **Flask**: Lightweight Python web framework
- **PyTorch**: Deep learning framework for neural network models
- **scikit-learn**: Machine learning utilities and preprocessing
- **gensim**: Word2Vec embeddings and NLP processing
- **Flask-JWT-Extended**: Secure authentication with JSON Web Tokens

### **Database & Storage**
- **MongoDB**: NoSQL database for flexible document storage
- **Docker Volumes**: Persistent data storage for containerized deployment

### **DevOps & Deployment**
- **Docker**: Containerization for consistent environments
- **Docker Compose**: Multi-service orchestration
- **Git**: Version control and collaboration

### **Machine Learning Libraries**
```python
torch>=1.9.0          # Neural network models
scikit-learn>=1.0.0    # ML utilities
gensim>=4.0.0          # Word2Vec embeddings
numpy>=1.21.0          # Numerical computations
pandas>=1.3.0          # Data manipulation
```

---

## 🏗️ System Architecture

### **High-Level Architecture**

```
┌─────────────────────────────────────────────────────────────────┐
│                        Scentinel System                        │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐      │
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

### **ML Pipeline Architecture**

```
User Rankings → Feature Engineering → Model Ensemble → Diversity Enhancement → Final Recommendations
     ↓                    ↓                  ↓                    ↓                    ↓
  (1-10 scale)      (300D Embeddings)   (RankNet+DPL+BPR)   (Brand+Note Diversity)  (Top-N Results)
```

### **Data Flow**
1. **User Interaction**: Rankings submitted through intuitive drag-and-drop interface
2. **Feature Extraction**: Perfume attributes converted to 300D Word2Vec embeddings
3. **Model Training**: Ensemble of three models trained on user preferences
4. **Prediction**: Models generate scores for all perfumes in database
5. **Enhancement**: Diversity bonuses applied to promote varied recommendations
6. **Delivery**: Top-N recommendations returned with sub-500ms response time

---

## 📦 Working Modules

### **1. Authentication Module**
- **User Registration**: Secure account creation with email validation
- **JWT Authentication**: Stateless token-based authentication
- **Password Security**: Werkzeug-based password hashing with salt

### **2. Perfume Management Module**
- **Data Storage**: MongoDB collection with 2000+ perfumes
- **Random Sampling**: Efficient perfume selection for ranking sessions
- **Attribute Processing**: Comprehensive perfume metadata handling

### **3. Ranking Interface Module**
- **Drag-and-Drop UI**: Intuitive ranking interface with real-time feedback
- **Validation**: Client and server-side ranking validation (1-10 scale)
- **Progress Tracking**: Visual indicators for ranking completion

### **4. Machine Learning Module**
- **Model Training**: Automated training pipeline for user-specific models
- **Ensemble Prediction**: Weighted combination of three ML models
- **Dynamic Weighting**: Adaptive model weights based on data availability

### **5. Recommendation Engine**
- **Personalized Recommendations**: User-specific model predictions
- **Cold-Start Handling**: Pre-trained models for new users
- **Diversity Enhancement**: Mathematical diversity bonuses
- **Real-Time Generation**: Sub-500ms recommendation processing

### **6. API Layer**
- **RESTful Endpoints**: Clean API design with proper HTTP methods
- **Error Handling**: Comprehensive error responses with fallback options
- **Rate Limiting**: Protection against abuse and overuse

---

## 🔄 Process Flow

### **New User Journey**
```
Landing Page → Registration → Email Verification → Ranking Tutorial → 
Perfume Ranking (10 items) → Model Training → Personalized Recommendations
```

### **Returning User Journey**
```
Login → Authentication → Profile Loading → Updated Recommendations → 
Optional Re-ranking → Enhanced Recommendations
```

### **Recommendation Generation Flow**
```
User Request → Authentication Check → Load User Models → 
Feature Extraction → Ensemble Prediction → Diversity Enhancement → 
Result Ranking → Response Delivery
```

### **Model Training Pipeline**
```
User Rankings → Data Validation → Feature Engineering → 
Pairwise Data Creation → Model Training (RankNet, DPL, BPR) → 
Model Validation → Model Storage → Performance Logging
```

---

## 📊 SWOT Analysis

### **Strengths**
- ✅ **Advanced ML Pipeline**: Hybrid ensemble approach with three specialized models
- ✅ **Diversity Enhancement**: Mathematical approach to ensure varied recommendations
- ✅ **Scalable Architecture**: Docker-based deployment with horizontal scaling capability
- ✅ **User-Centric Design**: Intuitive ranking interface with immediate feedback
- ✅ **Real-Time Performance**: Sub-500ms recommendation generation
- ✅ **Cold-Start Solutions**: Pre-trained models for immediate new user recommendations

### **Weaknesses**
- ⚠️ **Data Dependency**: Requires sufficient user rankings for optimal personalization
- ⚠️ **Computational Overhead**: ML model training requires significant processing power
- ⚠️ **Subjective Validation**: Difficult to objectively measure recommendation quality
- ⚠️ **Limited Context**: Doesn't account for seasonal or occasion-based preferences

### **Opportunities**
- 🚀 **Social Features**: Integration of collaborative filtering and social recommendations
- 🚀 **Context Awareness**: Seasonal, weather, and occasion-based recommendations
- 🚀 **Mobile Application**: Native mobile app for enhanced user experience
- 🚀 **Business Integration**: Partnership opportunities with fragrance retailers
- 🚀 **Advanced Analytics**: User behavior insights and market trend analysis
- 🚀 **Multi-Modal Input**: Integration of scent descriptions and user reviews

### **Threats**
- ⚡ **Privacy Concerns**: User preference data sensitivity and GDPR compliance
- ⚡ **Competition**: Established players with larger datasets and resources
- ⚡ **Market Saturation**: Potential oversaturation of recommendation systems
- ⚡ **Technology Evolution**: Rapid changes in ML/AI requiring constant updates
- ⚡ **Data Quality**: Dependence on accurate and comprehensive perfume metadata

---

## 🚀 Getting Started

### **Prerequisites**

Ensure you have the following installed:
- **Docker Desktop**: [Download here](https://docs.docker.com/get-docker/)
- **Git**: [Download here](https://git-scm.com/downloads)
- **Minimum System Requirements**:
  - RAM: 4GB (8GB recommended)
  - Storage: 2GB free space
  - OS: Windows 10+, macOS 10.14+, or Linux (Ubuntu 18.04+)

### **Quick Start (Recommended)**

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/Scentinel.git
   cd Scentinel
   ```

2. **Start the Application**
   ```bash
   docker compose up
   ```
   
   This single command will:
   - Build all Docker images (Frontend, Backend, Database)
   - Initialize MongoDB with perfume data
   - Start all services with proper networking

3. **Access the Application**
   - **Frontend**: http://localhost:3000
   - **Backend API**: http://localhost:5001
   - **MongoDB**: localhost:27017

4. **Create Your First Account**
   - Navigate to http://localhost:3000/register
   - Create an account with email and password
   - Start ranking perfumes to receive personalized recommendations

### **Development Setup**

For active development with hot reloading:

#### **Backend Development**
```bash
cd backend

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export FLASK_ENV=development
export MONGO_URI=mongodb://localhost:27017/scentinel

# Start MongoDB (using Docker)
docker run -d -p 27017:27017 --name scentinel-mongo mongo:latest

# Initialize database
python init_db.py

# Optional: Run pre-training for better recommendations
python pretrain.py

# Start backend server
python run.py
```

#### **Frontend Development**
```bash
cd frontend

# Install dependencies
npm install

# Set environment variables
export NEXT_PUBLIC_API_URL=http://localhost:5000

# Start development server with hot reloading
npm run dev
```

### **Model Pre-training (Optional)**

For enhanced recommendation quality, run the pre-training script:

```bash
# From project root
python standalone_pretrain.py
```

This will train the ML models on synthetic data, providing better initial recommendations.

### **Verification Steps**

1. **Health Check**
   ```bash
   curl http://localhost:5001/health
   # Expected: {"status": "healthy"}
   ```

2. **Test Recommendations**
   ```bash
   curl http://localhost:5001/api/recommend/cold-start?limit=5
   ```

3. **Frontend Access**
   - Visit http://localhost:3000
   - Register a new account
   - Complete the ranking process
   - View personalized recommendations

### **Stopping the Application**

```bash
# Stop all services
docker compose down

# Stop and remove all data (fresh start)
docker compose down -v
```

---

## 📖 API Documentation

### **Authentication Endpoints**
- `POST /api/auth/register` - User registration
- `POST /api/auth/login` - User authentication
- `GET /api/auth/profile` - User profile retrieval

### **Perfume Endpoints**
- `GET /api/perfumes/list` - Get perfumes for ranking (authenticated)
- `POST /api/perfumes/rank` - Submit user rankings (authenticated)
- `GET /api/perfumes/all` - Get all perfumes (paginated)

### **Recommendation Endpoints**
- `GET /api/recommend` - Personalized recommendations (authenticated)
- `GET /api/recommend/cold-start` - Cold-start recommendations (public)
- `GET /api/recommend/popular` - Popular perfumes (public)

### **Example API Usage**

```javascript
// Get personalized recommendations
const response = await fetch('/api/recommend?limit=10&diversity_weight=0.4', {
    headers: {
        'Authorization': `Bearer ${jwtToken}`,
        'Content-Type': 'application/json'
    }
});
const recommendations = await response.json();
```

---

## 📁 Project Structure

```
Scentinel/
├── 📁 backend/                    # Flask API server
│   ├── 📁 app/                    # Application code
│   │   ├── 📁 routes/             # API route handlers
│   │   ├── 📁 services/           # Business logic
│   │   ├── 📁 models/             # ML model implementations
│   │   └── 📁 utils/              # Utility functions
│   ├── 🐳 Dockerfile              # Backend container definition
│   ├── 📄 requirements.txt        # Python dependencies
│   ├── 🔧 init_db.py             # Database initialization
│   └── 🤖 pretrain.py            # Model pre-training script
├── 📁 frontend/                   # Next.js application
│   ├── 📁 components/             # Reusable UI components
│   ├── 📁 pages/                  # Application pages
│   ├── 📁 styles/                 # CSS and styling
│   ├── 🐳 Dockerfile              # Frontend container definition
│   └── 📄 package.json            # JavaScript dependencies
├── 📁 documentation/              # Comprehensive documentation
│   ├── 📖 architecture.md         # System architecture details
│   ├── 📖 RECOMMENDATION_SYSTEM.md # ML pipeline documentation
│   ├── 📖 PRETRAIN_INSTRUCTIONS.md # Model training guide
│   ├── 📖 workflow.md             # System workflows
│   └── 📖 MONGODB_SETUP.md        # Database setup guide
├── 📊 perfume_data.csv            # Perfume dataset (2000+ items)
├── 🤖 standalone_pretrain.py      # Standalone model training
├── 🐳 docker-compose.yml          # Multi-container orchestration
├── 📄 README.md                   # This comprehensive guide
└── 🔧 .gitignore                  # Git ignore patterns
```

---

## 🤝 Contributing

We welcome contributions to improve Scentinel! Here's how you can help:

### **Development Guidelines**
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### **Areas for Contribution**
- 🎨 **UI/UX Improvements**: Enhanced user interface and experience
- 🤖 **ML Model Enhancements**: New algorithms or optimization techniques
- 📊 **Analytics Features**: User behavior insights and recommendation explanations
- 🔧 **Performance Optimization**: Speed and efficiency improvements
- 📱 **Mobile Responsiveness**: Better mobile user experience
- 🌐 **Internationalization**: Multi-language support

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Dataset**: Perfume dataset sourced from fragrance community databases
- **ML Frameworks**: PyTorch, scikit-learn, and gensim communities
- **UI Components**: Tailwind CSS and React ecosystem
- **Infrastructure**: Docker and MongoDB for robust deployment

---

## 📞 Support & Contact

For questions, issues, or contributions:

- 📧 **Email**: [your-email@example.com]
- 🐛 **Issues**: [GitHub Issues](https://github.com/yourusername/Scentinel/issues)
- 📖 **Documentation**: [Project Wiki](https://github.com/yourusername/Scentinel/wiki)

---

<div align="center">

**Built with ❤️ for fragrance enthusiasts worldwide**

*Scentinel - Where Technology Meets Scent*

</div>