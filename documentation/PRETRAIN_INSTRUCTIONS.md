# Scentinel Model Pre-training Guide

## Table of Contents
1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Pre-training Process](#pre-training-process)
4. [Model Architecture Details](#model-architecture-details)
5. [Data Processing Pipeline](#data-processing-pipeline)
6. [Training Procedures](#training-procedures)
7. [Model Evaluation](#model-evaluation)
8. [Troubleshooting](#troubleshooting)
9. [Advanced Configuration](#advanced-configuration)

## Overview

The Scentinel pre-training system initializes three machine learning models with synthetic data to provide immediate recommendations for new users. This process creates a foundation that enables cold-start recommendations and provides a baseline for personalized model fine-tuning.

### Why Pre-training is Important

1. **Cold-Start Problem**: Provides recommendations for users without ranking history
2. **Model Initialization**: Creates better starting points for personalized training
3. **System Robustness**: Ensures the system can always provide recommendations
4. **Performance Baseline**: Establishes minimum recommendation quality standards

### Pre-training Strategy

The system uses **synthetic data generation** to create realistic user-perfume interactions:
- Generates diverse user preference profiles
- Creates ranking patterns based on perfume attributes
- Simulates various user behaviors and preferences
- Ensures balanced representation across perfume categories

## Prerequisites

### System Requirements
- **Python**: 3.8 or higher
- **Memory**: Minimum 4GB RAM, Recommended 8GB
- **Storage**: 2GB free space for models and data
- **CPU**: Multi-core processor recommended for faster training

### Required Dependencies

Create a virtual environment and install dependencies:

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install required packages
pip install pandas numpy torch scikit-learn gensim python-dotenv
```

### Dependency Details

```python
# Core ML libraries
torch>=1.9.0          # PyTorch for neural networks
scikit-learn>=1.0.0    # Machine learning utilities
gensim>=4.0.0          # Word2Vec embeddings
numpy>=1.21.0          # Numerical computations
pandas>=1.3.0          # Data manipulation

# Utilities
python-dotenv>=0.19.0  # Environment variable management
```

### Data Requirements

Ensure the following files are present:
- `backend/perfume_data.csv` - Main perfume dataset
- `backend/app/data/` - Directory for storing models and embeddings

## Pre-training Process

### Method 1: Standalone Pre-training (Recommended)

This method uses an isolated environment for better compatibility:

```bash
# From project root directory
python standalone_pretrain.py
```

**Advantages**:
- Independent of Flask application
- Better error handling and logging
- Easier debugging and monitoring
- Cross-platform compatibility

### Method 2: Integrated Pre-training

Using the backend's integrated pre-training:

```bash
cd backend
python pretrain.py
```

**Note**: This method requires the full backend environment and may have dependency conflicts.

### Pre-training Workflow

```
Data Loading → Feature Engineering → Synthetic Data Generation → Model Training → Model Validation → Model Saving
```

## Model Architecture Details

### 1. RankNet Model

**Purpose**: Learn pairwise preferences between perfumes

**Architecture**:
```python
class RankNetModel(nn.Module):
    def __init__(self, feature_dim=300, hidden_dim=128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
```

**Training Details**:
- **Input**: Concatenated feature pairs (600D)
- **Output**: Preference probability (0-1)
- **Loss Function**: Binary Cross-Entropy with Logits
- **Optimizer**: Adam (lr=0.001)
- **Batch Size**: 32
- **Epochs**: 50

### 2. Deep Preference Learning (DPL) Model

**Purpose**: Direct preference score prediction

**Architecture**:
```python
class DPLModel(nn.Module):
    def __init__(self, feature_dim=300, hidden_dim=128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
```

**Training Details**:
- **Input**: Single perfume features (300D)
- **Output**: Preference score (0-1)
- **Loss Function**: Mean Squared Error
- **Optimizer**: Adam (lr=0.001)
- **Batch Size**: 16
- **Epochs**: 50

### 3. Bayesian Personalized Ranking (BPR) Model

**Purpose**: Probabilistic ranking optimization

**Architecture**:
```python
class BayesianPersonalizedRanking:
    def __init__(self, feature_dim=300, embedding_dim=50):
        self.item_embeddings = np.random.normal(size=(n_items, embedding_dim))
        self.item_biases = np.zeros(n_items)
        self.learning_rate = 0.01
        self.regularization = 0.01
```

**Training Details**:
- **Embeddings**: 50D item embeddings
- **Optimization**: Stochastic Gradient Descent
- **Learning Rate**: 0.01
- **Regularization**: L2 (λ=0.01)
- **Iterations**: 100

## Data Processing Pipeline

### 1. Perfume Data Loading

```python
def load_perfume_data():
    """Load and preprocess perfume dataset"""
    df = pd.read_csv('perfume_data.csv')
    
    # Handle missing values
    df['Notes'] = df['Notes'].fillna('')
    df['Description'] = df['Description'].fillna('')
    df['Brand'] = df['Brand'].fillna('Unknown')
    
    # Standardize text fields
    df['Notes'] = df['Notes'].str.lower()
    df['Description'] = df['Description'].str.lower()
    
    return df
```

### 2. Feature Engineering

#### Word2Vec Embedding Generation

```python
def generate_word2vec_embeddings(perfume_data):
    """Generate Word2Vec embeddings for perfumes"""
    
    # Prepare training corpus
    sentences = []
    for _, perfume in perfume_data.iterrows():
        # Combine all text attributes
        text = f"{perfume['Notes']} {perfume['Description']} {perfume['Brand']}"
        tokens = text.lower().split()
        sentences.append(tokens)
    
    # Train Word2Vec model
    model = Word2Vec(
        sentences=sentences,
        vector_size=300,
        window=5,
        min_count=1,
        workers=4,
        epochs=100,
        sg=1  # Skip-gram model
    )
    
    return model
```

#### Perfume Feature Extraction

```python
def extract_perfume_features(perfume, word2vec_model):
    """Extract 300D feature vector for a perfume"""
    
    # Combine text attributes
    text_attributes = [
        perfume.get('Notes', ''),
        perfume.get('Description', ''),
        perfume.get('Brand', '')
    ]
    
    # Tokenize and get embeddings
    all_vectors = []
    for text in text_attributes:
        tokens = text.lower().split()
        for token in tokens:
            if token in word2vec_model.wv:
                all_vectors.append(word2vec_model.wv[token])
    
    # Average embeddings
    if all_vectors:
        feature_vector = np.mean(all_vectors, axis=0)
    else:
        feature_vector = np.zeros(300)
    
    # L2 normalization
    norm = np.linalg.norm(feature_vector)
    if norm > 0:
        feature_vector = feature_vector / norm
    
    return feature_vector
```

### 3. Synthetic Data Generation

#### User Profile Generation

```python
def generate_synthetic_users(n_users=1000):
    """Generate diverse synthetic user profiles"""
    
    user_profiles = []
    
    # Define user archetypes
    archetypes = [
        {'name': 'floral_lover', 'preferences': {'floral': 0.8, 'fresh': 0.6}},
        {'name': 'woody_enthusiast', 'preferences': {'woody': 0.9, 'spicy': 0.7}},
        {'name': 'fresh_seeker', 'preferences': {'fresh': 0.8, 'citrus': 0.7}},
        {'name': 'oriental_fan', 'preferences': {'oriental': 0.9, 'sweet': 0.6}},
        {'name': 'versatile_user', 'preferences': {'balanced': 0.7}}
    ]
    
    for i in range(n_users):
        # Randomly select archetype
        archetype = np.random.choice(archetypes)
        
        # Add noise to preferences
        preferences = {}
        for category, base_pref in archetype['preferences'].items():
            noise = np.random.normal(0, 0.1)
            preferences[category] = np.clip(base_pref + noise, 0, 1)
        
        user_profiles.append({
            'user_id': f'synthetic_user_{i}',
            'archetype': archetype['name'],
            'preferences': preferences
        })
    
    return user_profiles
```

#### Ranking Generation

```python
def generate_user_rankings(user_profile, perfumes, n_rankings=10):
    """Generate realistic rankings for a user"""
    
    # Sample random perfumes
    sampled_perfumes = perfumes.sample(n_rankings)
    
    rankings = []
    for idx, perfume in sampled_perfumes.iterrows():
        # Calculate preference score based on user profile
        score = calculate_preference_score(user_profile, perfume)
        
        # Add noise to make rankings more realistic
        noise = np.random.normal(0, 0.2)
        noisy_score = np.clip(score + noise, 0, 1)
        
        rankings.append({
            'perfume_id': perfume['_id'],
            'score': noisy_score
        })
    
    # Convert scores to ranks (1-10)
    rankings.sort(key=lambda x: x['score'], reverse=True)
    for i, ranking in enumerate(rankings):
        ranking['rank'] = i + 1
    
    return rankings
```

## Training Procedures

### Complete Pre-training Pipeline

```python
def pretrain_models():
    """Complete pre-training pipeline"""
    
    print("Starting Scentinel model pre-training...")
    
    # 1. Load perfume data
    print("Loading perfume data...")
    perfumes = load_perfume_data()
    print(f"Loaded {len(perfumes)} perfumes")
    
    # 2. Generate Word2Vec embeddings
    print("Generating Word2Vec embeddings...")
    word2vec_model = generate_word2vec_embeddings(perfumes)
    
    # 3. Extract perfume features
    print("Extracting perfume features...")
    perfume_features = {}
    for idx, perfume in perfumes.iterrows():
        features = extract_perfume_features(perfume, word2vec_model)
        perfume_features[perfume['_id']] = features
    
    # 4. Generate synthetic training data
    print("Generating synthetic training data...")
    synthetic_users = generate_synthetic_users(n_users=1000)
    
    all_rankings = []
    for user in synthetic_users:
        user_rankings = generate_user_rankings(user, perfumes)
        all_rankings.extend(user_rankings)
    
    # 5. Train models
    print("Training RankNet model...")
    ranknet_model = train_ranknet_model(perfume_features, all_rankings)
    
    print("Training DPL model...")
    dpl_model = train_dpl_model(perfume_features, all_rankings)
    
    print("Training BPR model...")
    bpr_model = train_bpr_model(perfume_features, all_rankings)
    
    # 6. Save models
    print("Saving models...")
    save_pretrained_models(ranknet_model, dpl_model, bpr_model, word2vec_model)
    
    print("Pre-training completed successfully!")
    return True
```

### Individual Model Training

#### RankNet Training

```python
def train_ranknet_model(perfume_features, rankings, epochs=50):
    """Train RankNet model on synthetic data"""
    
    model = RankNetModel(feature_dim=300)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCEWithLogitsLoss()
    
    # Prepare pairwise training data
    pairs_x, pairs_y = create_pairwise_data(perfume_features, rankings)
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        
        for batch_x, batch_y in create_batches(pairs_x, pairs_y, batch_size=32):
            optimizer.zero_grad()
            
            # Forward pass
            scores_i = model(batch_x[:, :300])
            scores_j = model(batch_x[:, 300:])
            diff = scores_i - scores_j
            pred = torch.sigmoid(diff)
            
            # Compute loss
            loss = criterion(pred, batch_y)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss:.4f}")
    
    return model
```

#### DPL Training

```python
def train_dpl_model(perfume_features, rankings, epochs=50):
    """Train DPL model on synthetic data"""
    
    model = DPLModel(feature_dim=300)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # Prepare direct training data
    features_x, scores_y = create_direct_data(perfume_features, rankings)
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        
        for batch_x, batch_y in create_batches(features_x, scores_y, batch_size=16):
            optimizer.zero_grad()
            
            # Forward pass
            pred = model(batch_x)
            
            # Compute loss
            loss = criterion(pred, batch_y)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss:.4f}")
    
    return model
```

## Model Evaluation

### Validation Metrics

```python
def evaluate_models(models, test_data):
    """Evaluate pre-trained models on test data"""
    
    metrics = {}
    
    for model_name, model in models.items():
        # Generate predictions
        predictions = model.predict(test_data['features'])
        true_rankings = test_data['rankings']
        
        # Calculate NDCG@10
        ndcg_10 = calculate_ndcg(predictions, true_rankings, k=10)
        
        # Calculate ranking correlation
        spearman_corr = calculate_spearman_correlation(predictions, true_rankings)
        
        metrics[model_name] = {
            'ndcg_10': ndcg_10,
            'spearman_correlation': spearman_corr
        }
    
    return metrics
```

### Performance Benchmarks

Expected performance on synthetic validation data:

| Model | NDCG@10 | Spearman Correlation | Training Time |
|-------|---------|---------------------|---------------|
| RankNet | 0.82-0.86 | 0.75-0.80 | 2-3 minutes |
| DPL | 0.79-0.83 | 0.72-0.77 | 1-2 minutes |
| BPR | 0.84-0.88 | 0.78-0.82 | 3-4 minutes |

## Troubleshooting

### Common Issues

#### 1. Memory Errors

**Problem**: Out of memory during training

**Solutions**:
```bash
# Reduce batch size
batch_size = 16  # Instead of 32

# Reduce number of synthetic users
n_users = 500  # Instead of 1000

# Use gradient accumulation
accumulation_steps = 2
```

#### 2. Word2Vec Vocabulary Errors

**Problem**: "must first build vocabulary" error

**Solution**:
```python
# Explicit vocabulary building
model = Word2Vec(vector_size=300, window=5, min_count=1, workers=4)
model.build_vocab(sentences)
model.train(sentences, total_examples=len(sentences), epochs=100)
```

#### 3. Feature Dimension Mismatches

**Problem**: Model expects different input dimensions

**Solution**:
```python
# Ensure consistent feature dimensions
def validate_feature_dimensions(features):
    for perfume_id, feature_vector in features.items():
        assert len(feature_vector) == 300, f"Feature dimension mismatch for {perfume_id}"
```

#### 4. Model Convergence Issues

**Problem**: Models not converging or poor performance

**Solutions**:
```python
# Adjust learning rates
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)  # Lower LR

# Add learning rate scheduling
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

# Increase training epochs
epochs = 100  # Instead of 50
```

### Debugging Tips

#### 1. Enable Verbose Logging

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Add progress tracking
from tqdm import tqdm
for epoch in tqdm(range(epochs), desc="Training"):
    # Training code
```

#### 2. Validate Data Quality

```python
def validate_training_data(perfume_features, rankings):
    """Validate training data quality"""
    
    # Check feature completeness
    assert len(perfume_features) > 0, "No perfume features found"
    
    # Check ranking validity
    for ranking in rankings:
        assert 1 <= ranking['rank'] <= 10, f"Invalid rank: {ranking['rank']}"
    
    # Check feature dimensions
    for features in perfume_features.values():
        assert len(features) == 300, "Feature dimension mismatch"
    
    print("Data validation passed!")
```

#### 3. Monitor Training Progress

```python
def plot_training_curves(losses):
    """Plot training loss curves"""
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 6))
    plt.plot(losses['ranknet'], label='RankNet')
    plt.plot(losses['dpl'], label='DPL')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Loss Curves')
    plt.show()
```

## Advanced Configuration

### Custom Training Parameters

```python
# Advanced configuration options
TRAINING_CONFIG = {
    'ranknet': {
        'hidden_dim': 128,
        'learning_rate': 0.001,
        'batch_size': 32,
        'epochs': 50,
        'dropout': 0.0
    },
    'dpl': {
        'hidden_dim': 128,
        'learning_rate': 0.001,
        'batch_size': 16,
        'epochs': 50,
        'dropout': 0.3
    },
    'bpr': {
        'embedding_dim': 50,
        'learning_rate': 0.01,
        'regularization': 0.01,
        'iterations': 100
    },
    'word2vec': {
        'vector_size': 300,
        'window': 5,
        'min_count': 1,
        'epochs': 100,
        'sg': 1  # Skip-gram
    }
}
```

### Distributed Training

```python
def distributed_pretrain():
    """Pre-train models using multiple processes"""
    
    from multiprocessing import Pool
    
    # Split data for parallel processing
    data_chunks = split_training_data(n_chunks=4)
    
    # Train models in parallel
    with Pool(processes=4) as pool:
        results = pool.map(train_model_chunk, data_chunks)
    
    # Combine results
    final_models = combine_model_results(results)
    
    return final_models
```

### Model Versioning

```python
def save_versioned_models(models, version="v1.0"):
    """Save models with version control"""
    
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    model_dir = f"models/{version}_{timestamp}"
    os.makedirs(model_dir, exist_ok=True)
    
    # Save each model with metadata
    for model_name, model in models.items():
        model_path = f"{model_dir}/{model_name}.pkl"
        
        # Save model with metadata
        model_data = {
            'model': model,
            'version': version,
            'timestamp': timestamp,
            'config': TRAINING_CONFIG[model_name]
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
    
    print(f"Models saved to {model_dir}")
```

## Next Steps

After successful pre-training:

1. **Verify Model Quality**: Run evaluation metrics on validation data
2. **Test Integration**: Ensure models work with the main application
3. **Monitor Performance**: Track recommendation quality in production
4. **Schedule Retraining**: Set up periodic model updates with new data

For production deployment, consider:
- **Model Compression**: Reduce model sizes for faster loading
- **Caching**: Cache embeddings and predictions for better performance
- **A/B Testing**: Compare pre-trained vs. randomly initialized models
- **Continuous Learning**: Update models with real user feedback 