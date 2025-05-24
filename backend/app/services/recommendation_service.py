import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import pickle
import time
from app import mongo
from bson.objectid import ObjectId
from app.models.ml_models import RankNetModel, DPLModel, BayesianPersonalizedRanking
from app.utils.embedding_utils import get_perfume_embeddings

# Paths for pre-trained models
MODELS_DIR = 'app/data/models'
RANKNET_PATH = f'{MODELS_DIR}/ranknet_pretrained.pkl'
DPL_PATH = f'{MODELS_DIR}/dpl_pretrained.pkl'
BPR_PATH = f'{MODELS_DIR}/bpr_pretrained.pkl'

class RecommendationService:
    def __init__(self):
        # Try to load pre-trained models first
        self.ranknet_model = self._load_pretrained_model(RANKNET_PATH)
        self.dpl_model = self._load_pretrained_model(DPL_PATH)
        self.bpr_model = self._load_pretrained_model(BPR_PATH)
        self.embeddings = {}  # Cache for perfume embeddings
        self.model_weights = {
            'ranknet': 0.33,
            'dpl': 0.33,
            'bpr': 0.34
        }
        print("Recommendation service initialized with pre-trained models")
        
    def _load_pretrained_model(self, path):
        """Load pre-trained model"""
        try:
            if os.path.exists(path):
                print(f"Loading pre-trained model from {path}")
                with open(path, 'rb') as f:
                    return pickle.load(f)
            else:
                print(f"WARNING: Pre-trained model not found at {path}")
                return None
        except Exception as e:
            print(f"Error loading pre-trained model: {e}")
            return None
    
    def train_models(self, user_id):
        """Fine-tune recommendation models based on user rankings"""
        # Get user rankings
        rankings = list(mongo.db.rankings.find({'user_id': user_id}))
        
        if not rankings or len(rankings) < 2:
            print(f"Not enough rankings for user {user_id} to train models")
            return False
            
        # Prepare training data
        ranked_perfumes = [(str(r['perfume_id']), r['rank']) for r in rankings]
        ranked_perfumes.sort(key=lambda x: x[1])  # Sort by rank
        
        # Get perfume features/embeddings
        perfume_ids = [p[0] for p in ranked_perfumes]
        perfume_features = self._get_perfume_features(perfume_ids)
        
        if len(perfume_features) < 2:
            print("Not enough valid perfume features for training")
            return False
        
        # Fine-tune pre-trained models with user's personal rankings
        # Use fewer epochs for fine-tuning since we're starting with pre-trained models
        success = True
        
        if self.ranknet_model:
            try:
                print(f"Fine-tuning RankNet model for user {user_id}")
                self.ranknet_model.train(perfume_features, ranked_perfumes, epochs=20)
            except Exception as e:
                print(f"Error fine-tuning RankNet model: {e}")
                success = False
        
        if self.dpl_model:
            try:
                print(f"Fine-tuning DPL model for user {user_id}")
                self.dpl_model.train(perfume_features, ranked_perfumes, epochs=20)
            except Exception as e:
                print(f"Error fine-tuning DPL model: {e}")
                success = False
        
        if self.bpr_model:
            try:
                print(f"Fine-tuning BPR model for user {user_id}")
                self.bpr_model.train(perfume_features, ranked_perfumes, n_iterations=40)
            except Exception as e:
                print(f"Error fine-tuning BPR model: {e}")
                success = False
        
        # Update model weights based on personalization
        self._update_model_weights(user_id, rankings)
        
        return success
    
    def _update_model_weights(self, user_id, rankings):
        """Update model weights based on user data"""
        # If we have enough rankings, we can adjust weights
        if len(rankings) >= 5:
            # For users with more rankings, give more weight to RankNet and DPL
            self.model_weights = {
                'ranknet': 0.4,
                'dpl': 0.4,
                'bpr': 0.2
            }
        else:
            # For users with fewer rankings, rely more on BPR which works better with sparse data
            self.model_weights = {
                'ranknet': 0.3,
                'dpl': 0.3,
                'bpr': 0.4
            }
    
    def generate_recommendations(self, user_id, top_n=10):
        """Generate perfume recommendations using ensemble of models"""
        # First check if we need to fine-tune models with user's rankings
        has_rankings = False
        if user_id:
            try:
                has_rankings = self.train_models(user_id)
            except Exception as e:
                print(f"Error training models for user {user_id}: {e}")
        
        # Get all perfumes for scoring
        all_perfumes = list(mongo.db.perfumes.find())
        
        if not all_perfumes:
            print("No perfumes found in database")
            return []
            
        all_perfume_ids = [str(p['_id']) for p in all_perfumes]
        
        # Get features for all perfumes
        all_perfume_features = self._get_perfume_features(all_perfume_ids)
        
        if len(all_perfume_features) == 0:
            print("No valid perfume features found")
            return []
        
        # Initialize scores
        ensemble_scores = np.zeros(len(all_perfume_ids))
        
        # Get and combine scores from each model
        if self.ranknet_model:
            try:
                start_time = time.time()
                ranknet_scores = self.ranknet_model.predict(all_perfume_features)
                ranknet_scores_norm = self._normalize_scores(ranknet_scores)
                ensemble_scores += ranknet_scores_norm * self.model_weights['ranknet']
                print(f"RankNet prediction time: {(time.time() - start_time)*1000:.2f} ms")
            except Exception as e:
                print(f"Error getting RankNet predictions: {e}")
        
        if self.dpl_model:
            try:
                start_time = time.time()
                dpl_scores = self.dpl_model.predict(all_perfume_features)
                dpl_scores_norm = self._normalize_scores(dpl_scores)
                ensemble_scores += dpl_scores_norm * self.model_weights['dpl']
                print(f"DPL prediction time: {(time.time() - start_time)*1000:.2f} ms")
            except Exception as e:
                print(f"Error getting DPL predictions: {e}")
        
        if self.bpr_model:
            try:
                start_time = time.time()
                bpr_scores = self.bpr_model.predict(all_perfume_features)
                bpr_scores_norm = self._normalize_scores(bpr_scores)
                ensemble_scores += bpr_scores_norm * self.model_weights['bpr']
                print(f"BPR prediction time: {(time.time() - start_time)*1000:.2f} ms")
            except Exception as e:
                print(f"Error getting BPR predictions: {e}")
        
        # Create (perfume_id, score) pairs and sort by score
        perfume_scores = list(zip(all_perfume_ids, ensemble_scores))
        perfume_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Get user's already ranked perfumes to exclude from recommendations
        ranked_perfume_ids = []
        if has_rankings and user_id:
            user_rankings = list(mongo.db.rankings.find({'user_id': user_id}))
            ranked_perfume_ids = [str(r['perfume_id']) for r in user_rankings]
            
            # Filter out perfumes the user has already ranked
            perfume_scores = [(pid, score) for pid, score in perfume_scores 
                              if pid not in ranked_perfume_ids]
        
        # Convert string IDs back to ObjectId
        return [ObjectId(ps[0]) for ps in perfume_scores[:top_n]]
    
    def get_cold_start_recommendations(self, top_n=10):
        """Get recommendations for new users without any rankings"""
        return self.generate_recommendations(None, top_n)
    
    def _get_perfume_features(self, perfume_ids):
        """Get feature vectors for perfumes"""
        # Check if embeddings are cached
        if not self.embeddings:
            self.embeddings = get_perfume_embeddings()
            
        # Collect feature vectors for given perfume IDs
        feature_vectors = []
        for pid in perfume_ids:
            if pid in self.embeddings:
                feature_vectors.append(self.embeddings[pid])
            else:
                # Use a default vector if embedding not found
                feature_vectors.append(np.zeros(300))  # Assuming 300-dim embeddings
                
        return np.array(feature_vectors)
    
    def _normalize_scores(self, scores):
        """Normalize scores to [0, 1] range"""
        min_score = np.min(scores)
        max_score = np.max(scores)
        
        if max_score == min_score:
            return np.ones_like(scores)
            
        return (scores - min_score) / (max_score - min_score)

# Create a singleton instance
recommendation_service = RecommendationService()

def generate_recommendations(user_id, top_n=10):
    """Function to generate recommendations for a user"""
    return recommendation_service.generate_recommendations(user_id, top_n)

def get_cold_start_recommendations(top_n=10):
    """Function to get recommendations for new users"""
    return recommendation_service.get_cold_start_recommendations(top_n) 