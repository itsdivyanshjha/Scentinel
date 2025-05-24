import os
import sys
import pandas as pd
import numpy as np
import torch
import pickle
from sklearn.model_selection import train_test_split
import gensim
from gensim.models import Word2Vec
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import ndcg_score
import time

# Paths for saving pre-trained models
MODELS_DIR = 'backend/app/data/models'
RANKNET_PATH = f'{MODELS_DIR}/ranknet_pretrained.pkl'
DPL_PATH = f'{MODELS_DIR}/dpl_pretrained.pkl'
BPR_PATH = f'{MODELS_DIR}/bpr_pretrained.pkl'

class RankNetModel:
    """Implementation of RankNet for learning to rank"""
    def __init__(self, feature_dim=300, hidden_dim=128):
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.model = None
        self._build_model()
        self.train_losses = []
        
    def _build_model(self):
        """Build RankNet model using PyTorch"""
        self.model = nn.Sequential(
            nn.Linear(self.feature_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 1)
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
    def train(self, features, rankings, epochs=50, batch_size=32):
        """Train RankNet on pairwise comparisons"""
        if len(rankings) < 2:
            # Need at least 2 items for pairwise comparisons
            return False
            
        # Create pairwise training data
        X_pairs = []
        y_pairs = []
        
        for i in range(len(rankings)):
            for j in range(i+1, len(rankings)):
                # Get feature vectors for pair
                X_i = features[i]
                X_j = features[j]
                
                # Get ranks for pair
                rank_i = rankings[i][1]
                rank_j = rankings[j][1]
                
                # Determine label (1 if i > j, 0 if i < j)
                y = 1 if rank_i < rank_j else 0  # Lower rank value means higher preference
                
                # Add pair and swapped pair
                X_pairs.append(np.concatenate([X_i, X_j]))
                y_pairs.append(y)
                
                X_pairs.append(np.concatenate([X_j, X_i]))
                y_pairs.append(1-y)
                
        # Convert to tensors
        X_tensor = torch.tensor(X_pairs, dtype=torch.float32)
        y_tensor = torch.tensor(y_pairs, dtype=torch.float32).view(-1, 1)
        
        # Create dataset and dataloader
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Training loop
        criterion = nn.BCEWithLogitsLoss()
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            for X_batch, y_batch in dataloader:
                # Forward pass
                self.optimizer.zero_grad()
                scores_i = self.model(X_batch[:, :self.feature_dim])
                scores_j = self.model(X_batch[:, self.feature_dim:])
                diff = scores_i - scores_j
                pred = torch.sigmoid(diff)
                
                # Compute loss
                loss = criterion(pred, y_batch)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            # Store loss for visualization
            avg_loss = total_loss / len(dataloader)
            self.train_losses.append(avg_loss)
            
        return True
        
    def predict(self, features):
        """Predict scores for perfumes"""
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(features, dtype=torch.float32)
            scores = self.model(X_tensor).squeeze().numpy()
            
        return scores


class DPLModel:
    """Deep Preference Learning model using MLP"""
    def __init__(self, feature_dim=300, hidden_dim=128):
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.model = None
        self._build_model()
        self.train_losses = []
        
    def _build_model(self):
        """Build MLP model"""
        self.model = nn.Sequential(
            nn.Linear(self.feature_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, 1)
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
    def train(self, features, rankings, epochs=50, batch_size=16):
        """Train model on rankings data"""
        if len(rankings) < 2:
            return False
            
        # Prepare training data
        X = features
        # Convert rankings to scores (reverse the ranking order)
        max_rank = max([r[1] for r in rankings])
        y = np.array([max_rank - r[1] + 1 for r in rankings])  # Higher score for lower rank number
        
        # Normalize targets
        y = y / np.max(y)
        
        # Convert to tensors
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)
        
        # Create dataset and dataloader
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Training loop
        criterion = nn.MSELoss()
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0
            for X_batch, y_batch in dataloader:
                # Forward pass
                self.optimizer.zero_grad()
                pred = self.model(X_batch)
                
                # Compute loss
                loss = criterion(pred, y_batch)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                
            # Store loss for visualization
            avg_loss = total_loss / len(dataloader)
            self.train_losses.append(avg_loss)
                
        return True
        
    def predict(self, features):
        """Predict scores for perfumes"""
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(features, dtype=torch.float32)
            scores = self.model(X_tensor).squeeze().numpy()
            
        return scores


class BayesianPersonalizedRanking:
    """Bayesian Personalized Ranking implementation"""
    def __init__(self, feature_dim=300, embedding_dim=50, learning_rate=0.01, reg=0.01):
        self.feature_dim = feature_dim
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        self.reg = reg
        
        # Initialize item embeddings randomly
        self.item_embeddings = None
        self.item_biases = None
        self.item_map = {}  # Map from perfume_id to index
        self.projection_matrix = None  # Projection matrix to reduce dimensionality
        self.train_losses = []
        
    def train(self, features, rankings, n_iterations=100):
        """Train BPR model on ranking data"""
        if len(rankings) < 2:
            return False
            
        # Create item mapping
        self.item_map = {rankings[i][0]: i for i in range(len(rankings))}
        
        # Create projection matrix for dimensionality reduction
        # This maps from feature_dim to embedding_dim
        self.projection_matrix = np.random.normal(0, 0.1, (features.shape[1], self.embedding_dim))
        
        # Initialize item embeddings
        self.item_embeddings = np.zeros((len(rankings), self.embedding_dim))
        for i in range(len(rankings)):
            # Project features to embedding space
            self.item_embeddings[i] = np.dot(features[i], self.projection_matrix)
        
        self.item_biases = np.zeros(len(rankings))
        
        # Create positive pairs
        # We assume items with lower rank values are preferred
        rankings_sorted = sorted(rankings, key=lambda x: x[1])
        positive_pairs = []
        
        for i in range(len(rankings_sorted) - 1):
            for j in range(i+1, len(rankings_sorted)):
                # Item i is preferred over item j
                pos_idx = self.item_map[rankings_sorted[i][0]]
                neg_idx = self.item_map[rankings_sorted[j][0]]
                positive_pairs.append((pos_idx, neg_idx))
        
        # Train with SGD
        for iteration in range(n_iterations):
            np.random.shuffle(positive_pairs)
            total_loss = 0
            
            for pos_idx, neg_idx in positive_pairs:
                # Get the projected feature vectors
                pos_feature_proj = np.dot(features[pos_idx], self.projection_matrix)
                neg_feature_proj = np.dot(features[neg_idx], self.projection_matrix)
                
                # Calculate scores
                pos_score = np.dot(pos_feature_proj, self.item_embeddings[pos_idx]) + self.item_biases[pos_idx]
                neg_score = np.dot(neg_feature_proj, self.item_embeddings[neg_idx]) + self.item_biases[neg_idx]
                
                # Calculate sigmoid of difference
                diff = pos_score - neg_score
                sigmoid = 1.0 / (1.0 + np.exp(-diff))
                
                # Calculate loss (log likelihood)
                loss = -np.log(sigmoid)
                total_loss += loss
                
                # Calculate gradients
                grad = 1.0 - sigmoid
                
                # Update embeddings with gradient descent
                self.item_embeddings[pos_idx] += self.learning_rate * (grad * pos_feature_proj - self.reg * self.item_embeddings[pos_idx])
                self.item_embeddings[neg_idx] += self.learning_rate * (-grad * neg_feature_proj - self.reg * self.item_embeddings[neg_idx])
                
                # Update biases
                self.item_biases[pos_idx] += self.learning_rate * (grad - self.reg * self.item_biases[pos_idx])
                self.item_biases[neg_idx] += self.learning_rate * (-grad - self.reg * self.item_biases[neg_idx])
            
            # Store average loss for visualization
            if len(positive_pairs) > 0:
                avg_loss = total_loss / len(positive_pairs)
                self.train_losses.append(avg_loss)
                
        return True
        
    def predict(self, features):
        """Predict scores for all items"""
        if self.item_embeddings is None or self.projection_matrix is None:
            return np.zeros(len(features))
            
        # Project features to embedding space
        features_proj = np.dot(features, self.projection_matrix)
        
        # For new items not seen during training, we'll estimate scores
        # based on similarity to the items we've seen
        scores = np.zeros(len(features))
        
        for i in range(len(features)):
            # For each item, calculate similarity with all training items
            similarities = np.dot(features_proj[i], self.item_embeddings.T)
            
            # Get weighted average of item biases
            norm_similarities = similarities / (np.sum(similarities) + 1e-10)
            scores[i] = np.sum(norm_similarities * self.item_biases)
        
        return scores

def preprocess_text(text):
    """Preprocess text data"""
    if not isinstance(text, str):
        return []
    
    # Convert to lowercase and split into words
    text = text.lower()
    words = text.split()
    
    return words

def generate_embeddings(perfumes_df):
    """Generate embeddings for perfumes"""
    print("Generating perfume embeddings...")
    
    # Prepare text corpus for Word2Vec
    corpus = []
    
    # Extract text features
    text_columns = ['Notes', 'Description', 'Brand']
    available_cols = [col for col in text_columns if col in perfumes_df.columns]
    
    for _, row in perfumes_df.iterrows():
        doc = []
        for col in available_cols:
            if isinstance(row[col], str):
                doc.extend(preprocess_text(row[col]))
        corpus.append(doc)
    
    # Check if corpus is empty or contains empty documents
    valid_docs = [doc for doc in corpus if len(doc) > 0]
    if not valid_docs:
        print("Warning: No valid documents for Word2Vec training")
        # Use simple random vectors instead
        perfume_embeddings = {}
        for idx, _ in perfumes_df.iterrows():
            perfume_embeddings[idx] = np.random.normal(size=300)
        return perfume_embeddings
    
    print(f"Training Word2Vec model on {len(valid_docs)} documents...")
    # Train Word2Vec model with explicit build_vocab step
    w2v_model = Word2Vec(vector_size=300, window=5, min_count=1, workers=4)
    w2v_model.build_vocab(valid_docs)
    w2v_model.train(valid_docs, total_examples=len(valid_docs), epochs=10)
    
    # Generate embeddings for each perfume
    perfume_embeddings = {}
    
    for idx, row in perfumes_df.iterrows():
        perfume_id = row.get('id', idx)
        
        # Initialize embedding vector
        embedding = np.zeros(300)
        word_count = 0
        
        # Combine text from all text columns
        for col in available_cols:
            if isinstance(row[col], str):
                words = preprocess_text(row[col])
                for word in words:
                    if word in w2v_model.wv:
                        embedding += w2v_model.wv[word]
                        word_count += 1
        
        # Average word vectors
        if word_count > 0:
            embedding /= word_count
        
        # Add one-hot encoded categorical features for gender if available
        if 'gender' in perfumes_df.columns:
            gender = str(row.get('gender', '')).lower()
            if gender == 'men':
                embedding = np.append(embedding, [1, 0, 0])
            elif gender == 'women':
                embedding = np.append(embedding, [0, 1, 0])
            else:
                embedding = np.append(embedding, [0, 0, 1])  # Unisex or other
                
        perfume_embeddings[perfume_id] = embedding
    
    print(f"Generated embeddings for {len(perfume_embeddings)} perfumes")
    return perfume_embeddings

def generate_synthetic_rankings(perfumes_df, num_users=100, k=10):
    """
    Generate synthetic user rankings to pre-train models
    
    Args:
        perfumes_df: DataFrame with perfume data
        num_users: Number of synthetic users to create
        k: Number of perfumes each user ranks
        
    Returns:
        List of (user_id, perfume_id, rank) tuples
    """
    print(f"Generating synthetic rankings for {num_users} users...")
    synthetic_rankings = []
    
    # Extract key features for clustering similar perfumes
    feature_cols = ['gender', 'brand']
    available_cols = [col for col in feature_cols if col in perfumes_df.columns]
    
    for user_id in range(num_users):
        # For each synthetic user, create a preference profile
        # This could be based on random selection, clustering, or other approaches
        
        # Randomly select k perfumes
        sample_perfumes = perfumes_df.sample(n=k)
        
        # Create preference patterns - e.g., prefer certain brands, genders
        # This is simplified; could be enhanced with more sophisticated preference simulation
        
        # Assign random rankings (simplified)
        rankings = np.random.permutation(k) + 1  # 1-indexed ranks
        
        for i, (idx, perfume) in enumerate(sample_perfumes.iterrows()):
            perfume_id = perfume.get('id', idx)
            synthetic_rankings.append((f"synthetic_user_{user_id}", perfume_id, rankings[i]))
    
    print(f"Generated {len(synthetic_rankings)} synthetic rankings")
    return synthetic_rankings

def load_perfume_data():
    """Load perfume data from CSV"""
    data_path = 'perfume_data.csv'
    
    # Try different encodings
    encodings = ['utf-8', 'latin1', 'ISO-8859-1', 'cp1252']
    
    for encoding in encodings:
        try:
            print(f"Trying to load CSV with {encoding} encoding...")
            df = pd.read_csv(data_path, encoding=encoding)
            print(f"Loaded {len(df)} perfumes from CSV using {encoding} encoding")
            
            # Display dataset sample
            print("\n=== Perfume Dataset Sample ===")
            print(df.head(5))
            
            # Display dataset statistics
            print("\n=== Dataset Statistics ===")
            if 'Notes' in df.columns:
                print(f"Perfumes with notes: {df['Notes'].notna().sum()}")
            if 'Description' in df.columns:
                print(f"Perfumes with descriptions: {df['Description'].notna().sum()}")
            if 'Brand' in df.columns:
                print(f"Number of unique brands: {df['Brand'].nunique()}")
                
            return df
        except Exception as e:
            print(f"Error with {encoding} encoding: {e}")
    
    print("Failed to load CSV with any encoding")
    return None

def evaluate_models(models, test_data, feature_vectors):
    """Evaluate model performance using NDCG"""
    results = {}
    
    # Unpack test data
    true_rankings = {}
    for user_id, rankings in test_data.items():
        true_rankings[user_id] = {item[0]: item[1] for item in rankings}
    
    # Evaluate each model
    for model_name, model in models.items():
        ndcg_scores = []
        
        for user_id, rankings in test_data.items():
            # Get perfume IDs and feature vectors
            perfume_ids = [r[0] for r in rankings]
            
            # Get features for these perfumes
            user_features = np.array([feature_vectors.get(pid, np.zeros(feature_vectors[list(feature_vectors.keys())[0]].shape)) for pid in perfume_ids])
            
            # Get model predictions
            predicted_scores = model.predict(user_features)
            
            # Get true relevance scores (inverted rank)
            max_rank = max(r[1] for r in rankings)
            true_relevance = np.array([max_rank - rankings[i][1] + 1 for i in range(len(rankings))])
            
            # Calculate NDCG
            try:
                true_relevance = true_relevance.reshape(1, -1)
                pred_scores = predicted_scores.reshape(1, -1)
                score = ndcg_score(true_relevance, pred_scores)
                ndcg_scores.append(score)
            except Exception as e:
                print(f"Error calculating NDCG for {model_name}: {e}")
        
        # Store average NDCG
        if ndcg_scores:
            results[model_name] = np.mean(ndcg_scores)
        else:
            results[model_name] = 0.0
    
    return results

def pretrain_models():
    """Pre-train recommendation models on full dataset with synthetic rankings"""
    # Create models directory if it doesn't exist
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # Load perfume data
    perfumes_df = load_perfume_data()
    if perfumes_df is None or len(perfumes_df) == 0:
        print("No perfume data available for pre-training")
        return False
    
    # Generate synthetic rankings
    synthetic_rankings = generate_synthetic_rankings(perfumes_df)
    
    # Get perfume embeddings
    embeddings = generate_embeddings(perfumes_df)
    
    # Group rankings by user
    user_rankings = {}
    for user_id, perfume_id, rank in synthetic_rankings:
        if user_id not in user_rankings:
            user_rankings[user_id] = []
        user_rankings[user_id].append((perfume_id, rank))
    
    # Split into train/test sets (80/20 split)
    train_users, test_users = train_test_split(list(user_rankings.keys()), test_size=0.2, random_state=42)
    print(f"Split users into {len(train_users)} training and {len(test_users)} testing users")
    
    # Initialize models
    print("Initializing models...")
    ranknet_model = RankNetModel()
    dpl_model = DPLModel()
    bpr_model = BayesianPersonalizedRanking()
    
    # Train models with synthetic user data
    print("Training models with synthetic user data...")
    train_count = 0
    for user_id in train_users:
        rankings = user_rankings[user_id]
        perfume_ids = [r[0] for r in rankings]
        
        # Get embeddings for perfumes
        feature_vectors = []
        for pid in perfume_ids:
            if pid in embeddings:
                feature_vectors.append(embeddings[pid])
            else:
                # Default vector if embedding not found
                feature_vectors.append(np.zeros(300))
        
        perfume_features = np.array(feature_vectors)
        
        # Train each model
        ranknet_model.train(perfume_features, rankings)
        dpl_model.train(perfume_features, rankings)
        bpr_model.train(perfume_features, rankings)
        
        train_count += 1
        if train_count % 10 == 0:
            print(f"Trained on {train_count}/{len(train_users)} users")
    
    # Create a dictionary with all models
    models = {
        'RankNet': ranknet_model,
        'DPL': dpl_model,
        'BPR': bpr_model
    }
    
    # Evaluate models on test set
    print("Evaluating models...")
    test_data = {user_id: user_rankings[user_id] for user_id in test_users}
    evaluation_results = evaluate_models(models, test_data, embeddings)
    
    # Print evaluation results
    print("\n=== Model Evaluation Results ===")
    print("NDCG Scores (higher is better):")
    for model_name, score in evaluation_results.items():
        print(f"{model_name}: {score:.4f}")
    
    # Calculate average response time for each model
    print("\n=== Model Response Times ===")
    # Create sample test data for timing
    sample_features = np.array([np.random.rand(300) for _ in range(10)])
    
    for name, model in models.items():
        # Time prediction
        start_time = time.time()
        model.predict(sample_features)
        end_time = time.time()
        avg_time = (end_time - start_time) * 1000 / len(sample_features)  # ms per prediction
        print(f"{name}: {avg_time:.2f} ms per prediction")
    
    # Save pre-trained models
    print("\nSaving pre-trained models...")
    with open(RANKNET_PATH, 'wb') as f:
        pickle.dump(ranknet_model, f)
    
    with open(DPL_PATH, 'wb') as f:
        pickle.dump(dpl_model, f)
    
    with open(BPR_PATH, 'wb') as f:
        pickle.dump(bpr_model, f)
    
    print(f"Models saved to {MODELS_DIR}")
    
    # Print model size information
    print("\n=== Model Size Information ===")
    ranknet_size = os.path.getsize(RANKNET_PATH) / (1024 * 1024)
    dpl_size = os.path.getsize(DPL_PATH) / (1024 * 1024)
    bpr_size = os.path.getsize(BPR_PATH) / (1024 * 1024)
    
    print(f"RankNet model size: {ranknet_size:.2f} MB")
    print(f"DPL model size: {dpl_size:.2f} MB")
    print(f"BPR model size: {bpr_size:.2f} MB")
    
    print("\nPre-training complete!")
    return True

if __name__ == "__main__":
    print("Starting standalone model pre-training...")
    success = pretrain_models()
    
    if success:
        print("Pre-training completed successfully!")
    else:
        print("Pre-training failed!")
        sys.exit(1) 