import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class RankNetModel:
    """Implementation of RankNet for learning to rank"""
    def __init__(self, feature_dim=300, hidden_dim=128):
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.model = None
        self._build_model()
        
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
        
    def train(self, features, rankings, n_iterations=100):
        """Train BPR model on ranking data"""
        if len(rankings) < 2:
            return False
            
        # Create item mapping
        self.item_map = {rankings[i][0]: i for i in range(len(rankings))}
        
        # Initialize item embeddings using PCA-reduced features
        self.item_embeddings = np.random.normal(size=(len(rankings), self.embedding_dim))
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
            for pos_idx, neg_idx in positive_pairs:
                # Calculate scores
                pos_score = np.dot(features[pos_idx], self.item_embeddings[pos_idx]) + self.item_biases[pos_idx]
                neg_score = np.dot(features[neg_idx], self.item_embeddings[neg_idx]) + self.item_biases[neg_idx]
                
                # Calculate sigmoid of difference
                diff = pos_score - neg_score
                sigmoid = 1.0 / (1.0 + np.exp(-diff))
                
                # Calculate gradients
                grad = 1.0 - sigmoid
                
                # Update embeddings with gradient descent
                self.item_embeddings[pos_idx] += self.learning_rate * (grad * features[pos_idx] - self.reg * self.item_embeddings[pos_idx])
                self.item_embeddings[neg_idx] += self.learning_rate * (-grad * features[neg_idx] - self.reg * self.item_embeddings[neg_idx])
                
                # Update biases
                self.item_biases[pos_idx] += self.learning_rate * (grad - self.reg * self.item_biases[pos_idx])
                self.item_biases[neg_idx] += self.learning_rate * (-grad - self.reg * self.item_biases[neg_idx])
                
        return True
        
    def predict(self, features):
        """Predict scores for all items"""
        if self.item_embeddings is None:
            return np.zeros(len(features))
            
        # For new items not seen during training, we'll estimate scores
        # based on similarity to the items we've seen
        scores = np.zeros(len(features))
        
        for i in range(len(features)):
            # For each item, calculate similarity with all training items
            similarities = np.dot(features[i], self.item_embeddings.T)
            
            # Get weighted average of item biases
            norm_similarities = similarities / (np.sum(similarities) + 1e-10)
            scores[i] = np.sum(norm_similarities * self.item_biases)
            
        return scores 