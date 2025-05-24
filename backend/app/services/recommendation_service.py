import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
import pickle
from app import mongo
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
        self.ranknet_model = self._load_pretrained_model(RANKNET_PATH, RankNetModel)
        self.dpl_model = self._load_pretrained_model(DPL_PATH, DPLModel)
        self.bpr_model = self._load_pretrained_model(BPR_PATH, BayesianPersonalizedRanking)
        self.embeddings = {}  # Cache for perfume embeddings
        
    def _load_pretrained_model(self, path, model_class):
        """Load pre-trained model if available, otherwise create a new one"""
        try:
            if os.path.exists(path):
                print(f"Loading pre-trained model from {path}")
                with open(path, 'rb') as f:
                    return pickle.load(f)
            else:
                print(f"Pre-trained model not found at {path}, initializing new model")
                return model_class()
        except Exception as e:
            print(f"Error loading pre-trained model: {e}")
            return model_class()
    
    def train_models(self, user_id):
        """Fine-tune recommendation models based on user rankings"""
        # Get user rankings
        rankings = list(mongo.db.rankings.find({'user_id': user_id}))
        
        if not rankings:
            return False
            
        # Prepare training data
        ranked_perfumes = [(r['perfume_id'], r['rank']) for r in rankings]
        ranked_perfumes.sort(key=lambda x: x[1])  # Sort by rank
        
        # Get perfume features/embeddings
        perfume_ids = [p[0] for p in ranked_perfumes]
        perfume_features = self._get_perfume_features(perfume_ids)
        
        # Fine-tune pre-trained models with user's personal rankings
        # Use fewer epochs for fine-tuning since we're starting with pre-trained models
        self.ranknet_model.train(perfume_features, ranked_perfumes, epochs=20)
        self.dpl_model.train(perfume_features, ranked_perfumes, epochs=20)
        self.bpr_model.train(perfume_features, ranked_perfumes, n_iterations=40)
        
        return True
    
    def generate_recommendations(self, user_id, top_n=10):
        """Generate perfume recommendations using ensemble of models"""
        # First fine-tune models with user's rankings
        has_rankings = self.train_models(user_id)
        
        # Get all perfumes for scoring
        all_perfumes = list(mongo.db.perfumes.find())
        all_perfume_ids = [p['_id'] for p in all_perfumes]
        
        # Get features for all perfumes
        all_perfume_features = self._get_perfume_features(all_perfume_ids)
        
        # Get scores from each model
        ranknet_scores = self.ranknet_model.predict(all_perfume_features)
        dpl_scores = self.dpl_model.predict(all_perfume_features)
        bpr_scores = self.bpr_model.predict(all_perfume_features)
        
        # Normalize scores from each model
        ranknet_scores_norm = self._normalize_scores(ranknet_scores)
        dpl_scores_norm = self._normalize_scores(dpl_scores)
        bpr_scores_norm = self._normalize_scores(bpr_scores)
        
        # Combine scores with adjusted weights based on whether user has rankings
        if has_rankings:
            # If user has rankings, give more weight to personalized results
            ensemble_scores = (ranknet_scores_norm * 0.4 + 
                              dpl_scores_norm * 0.4 + 
                              bpr_scores_norm * 0.2)
        else:
            # If no user rankings, rely more on pre-trained models
            ensemble_scores = (ranknet_scores_norm * 0.33 + 
                              dpl_scores_norm * 0.33 + 
                              bpr_scores_norm * 0.34)
        
        # Create (perfume_id, score) pairs and sort by score
        perfume_scores = list(zip(all_perfume_ids, ensemble_scores))
        perfume_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Get user's already ranked perfumes to exclude from recommendations
        if has_rankings:
            user_rankings = list(mongo.db.rankings.find({'user_id': user_id}))
            ranked_perfume_ids = [r['perfume_id'] for r in user_rankings]
            
            # Filter out perfumes the user has already ranked
            perfume_scores = [(pid, score) for pid, score in perfume_scores 
                              if pid not in ranked_perfume_ids]
        
        # Return top N perfume IDs
        return [ps[0] for ps in perfume_scores[:top_n]]
    
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