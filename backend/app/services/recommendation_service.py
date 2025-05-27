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
        
        # If pre-trained models failed to load, create new ones
        if not self.ranknet_model:
            print("Creating new RankNet model")
            self.ranknet_model = RankNetModel()
        if not self.dpl_model:
            print("Creating new DPL model")
            self.dpl_model = DPLModel()
        if not self.bpr_model:
            print("Creating new BPR model")
            self.bpr_model = BayesianPersonalizedRanking()
            
        self.embeddings = {}  # Cache for perfume embeddings
        self.model_weights = {
            'ranknet': 0.33,
            'dpl': 0.33,
            'bpr': 0.34
        }
        print("Recommendation service initialized")
        
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
    
    def _enhance_recommendations_with_diversity(self, perfume_scores, all_perfumes, top_n=10, diversity_weight=0.3):
        """
        Enhance recommendation scores by incorporating diversity factors into the ranking algorithm
        
        Args:
            perfume_scores: List of (perfume_id, score) tuples sorted by score
            all_perfumes: List of all perfume documents from database
            top_n: Number of recommendations to return
            diversity_weight: Weight for diversity vs relevance (0.0 = pure relevance, 1.0 = pure diversity)
        
        Returns:
            List of enhanced recommendation IDs
        """
        if not perfume_scores:
            return []
        
        # Create a mapping from perfume_id to perfume document
        perfume_map = {str(p['_id']): p for p in all_perfumes}
        
        # Initialize result list and tracking sets for diversity enhancement
        enhanced_recommendations = []
        selected_brands = set()
        selected_note_groups = set()
        
        # Define note groups for diversity (group similar notes together)
        note_groups = {
            'citrus': ['lemon', 'orange', 'bergamot', 'grapefruit', 'lime', 'mandarin', 'citrus'],
            'floral': ['rose', 'jasmine', 'lily', 'violet', 'peony', 'iris', 'gardenia', 'tuberose', 'ylang-ylang'],
            'woody': ['sandalwood', 'cedar', 'oak', 'pine', 'birch', 'rosewood', 'teak'],
            'oriental': ['amber', 'musk', 'oud', 'incense', 'myrrh', 'frankincense'],
            'fresh': ['mint', 'eucalyptus', 'marine', 'ozone', 'cucumber', 'green'],
            'spicy': ['pepper', 'cinnamon', 'cardamom', 'ginger', 'clove', 'nutmeg', 'saffron'],
            'sweet': ['vanilla', 'caramel', 'honey', 'chocolate', 'sugar', 'praline'],
            'fruity': ['apple', 'pear', 'peach', 'berry', 'cherry', 'plum', 'grape'],
            'herbal': ['lavender', 'rosemary', 'thyme', 'sage', 'basil', 'oregano'],
            'gourmand': ['coffee', 'cocoa', 'almond', 'coconut', 'cream', 'milk']
        }
        
        def get_note_groups(notes_text):
            """Extract note groups from perfume notes"""
            if not notes_text:
                return set()
            
            notes_lower = notes_text.lower()
            found_groups = set()
            
            for group, keywords in note_groups.items():
                for keyword in keywords:
                    if keyword in notes_lower:
                        found_groups.add(group)
                        break
            
            return found_groups
        
        def calculate_diversity_bonus(perfume, selected_brands, selected_note_groups):
            """Calculate diversity bonus to enhance recommendation scores"""
            bonus = 0.0
            
            # Brand diversity bonus - reward different brands
            brand = (perfume.get('Brand', '') or perfume.get('brand', '')).lower().strip()
            if brand and brand not in selected_brands:
                bonus += 0.3  # Bonus for new brand
            
            # Note group diversity bonus - reward different scent profiles
            notes = perfume.get('Notes', '') or perfume.get('notes', '')
            perfume_note_groups = get_note_groups(notes)
            
            if perfume_note_groups:
                new_groups = perfume_note_groups - selected_note_groups
                if new_groups:
                    bonus += 0.2 * (len(new_groups) / len(perfume_note_groups))
            
            return bonus
        
        # Create a pool of high-scoring candidates (top 3x the requested amount)
        candidate_pool_size = min(len(perfume_scores), top_n * 3)
        candidate_pool = perfume_scores[:candidate_pool_size].copy()
        
        # Iteratively select recommendations with diversity enhancement
        for _ in range(top_n):
            if not candidate_pool:
                break
            
            best_idx = 0
            best_score = -float('inf')
            
            for i, (perfume_id, relevance_score) in enumerate(candidate_pool):
                perfume = perfume_map.get(perfume_id)
                if not perfume:
                    continue
                
                # Calculate diversity bonus to enhance the recommendation score
                diversity_bonus = calculate_diversity_bonus(perfume, selected_brands, selected_note_groups)
                
                # Enhance the relevance score with diversity bonus
                enhanced_score = relevance_score + diversity_weight * diversity_bonus
                
                if enhanced_score > best_score:
                    best_score = enhanced_score
                    best_idx = i
            
            # Select the best enhanced candidate
            selected_perfume_id, _ = candidate_pool.pop(best_idx)
            selected_perfume = perfume_map.get(selected_perfume_id)
            
            if selected_perfume:
                enhanced_recommendations.append(ObjectId(selected_perfume_id))
                
                # Update tracking sets for next iteration
                brand = (selected_perfume.get('Brand', '') or selected_perfume.get('brand', '')).lower().strip()
                if brand:
                    selected_brands.add(brand)
                
                notes = selected_perfume.get('Notes', '') or selected_perfume.get('notes', '')
                perfume_note_groups = get_note_groups(notes)
                selected_note_groups.update(perfume_note_groups)
        
        return enhanced_recommendations

    def generate_recommendations(self, user_id, top_n=10, enable_diversity=True, diversity_weight=0.3):
        """Generate perfume recommendations using ensemble of models"""
        try:
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
                print("No valid perfume features found, using simple recommendations")
                return self._get_simple_recommendations(user_id, top_n, enable_diversity)
            
            # Initialize scores
            ensemble_scores = np.zeros(len(all_perfume_ids))
            models_working = 0
            
            # Get and combine scores from each model
            if self.ranknet_model:
                try:
                    start_time = time.time()
                    ranknet_scores = self.ranknet_model.predict(all_perfume_features)
                    ranknet_scores_norm = self._normalize_scores(ranknet_scores)
                    ensemble_scores += ranknet_scores_norm * self.model_weights['ranknet']
                    models_working += 1
                    print(f"RankNet prediction time: {(time.time() - start_time)*1000:.2f} ms")
                except Exception as e:
                    print(f"Error getting RankNet predictions: {e}")
            
            if self.dpl_model:
                try:
                    start_time = time.time()
                    dpl_scores = self.dpl_model.predict(all_perfume_features)
                    dpl_scores_norm = self._normalize_scores(dpl_scores)
                    ensemble_scores += dpl_scores_norm * self.model_weights['dpl']
                    models_working += 1
                    print(f"DPL prediction time: {(time.time() - start_time)*1000:.2f} ms")
                except Exception as e:
                    print(f"Error getting DPL predictions: {e}")
            
            if self.bpr_model:
                try:
                    start_time = time.time()
                    bpr_scores = self.bpr_model.predict(all_perfume_features)
                    bpr_scores_norm = self._normalize_scores(bpr_scores)
                    ensemble_scores += bpr_scores_norm * self.model_weights['bpr']
                    models_working += 1
                    print(f"BPR prediction time: {(time.time() - start_time)*1000:.2f} ms")
                except Exception as e:
                    print(f"Error getting BPR predictions: {e}")
            
            # If no models are working, use simple recommendations
            if models_working == 0:
                print("No ML models working, using simple content-based recommendations")
                return self._get_simple_recommendations(user_id, top_n, enable_diversity)
            
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
            
            # Apply diversity enhancement if enabled
            if enable_diversity:
                return self._enhance_recommendations_with_diversity(perfume_scores, all_perfumes, top_n, diversity_weight)
            else:
                # Return top recommendations based purely on ML model scores
                return [ObjectId(ps[0]) for ps in perfume_scores[:top_n]]
            
        except Exception as e:
            print(f"Error in generate_recommendations: {e}")
            # Final fallback to simple recommendations
            return self._get_simple_recommendations(user_id, top_n, enable_diversity)
    
    def get_cold_start_recommendations(self, top_n=10, enable_diversity=True, diversity_weight=0.3):
        """Function to get recommendations for new users"""
        return self.generate_recommendations(None, top_n, enable_diversity, diversity_weight)
    
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
    
    def _get_simple_recommendations(self, user_id=None, top_n=10, enable_diversity=True):
        """Simple content-based recommendations as fallback"""
        try:
            # Get all perfumes
            all_perfumes = list(mongo.db.perfumes.find())
            
            if not all_perfumes:
                return []
            
            # If user has rankings, use them for similarity
            if user_id:
                user_rankings = list(mongo.db.rankings.find({'user_id': user_id}))
                if user_rankings:
                    # Get user's top-ranked perfumes
                    top_ranked = sorted(user_rankings, key=lambda x: x['rank'])[:3]
                    liked_perfume_ids = [r['perfume_id'] for r in top_ranked]
                    
                    # Find similar perfumes based on brand and notes
                    liked_perfumes = list(mongo.db.perfumes.find({'_id': {'$in': liked_perfume_ids}}))
                    
                    # Extract brands and notes from liked perfumes
                    liked_brands = set()
                    liked_notes = set()
                    
                    for perfume in liked_perfumes:
                        if perfume.get('Brand'):
                            liked_brands.add(perfume['Brand'].lower())
                        if perfume.get('brand'):
                            liked_brands.add(perfume['brand'].lower())
                        
                        notes_text = perfume.get('Notes', '') or perfume.get('notes', '')
                        if notes_text:
                            # Simple keyword extraction
                            notes_words = [word.strip().lower() for word in notes_text.split(',')]
                            liked_notes.update(notes_words)
                    
                    # Score perfumes based on similarity
                    scored_perfumes = []
                    for perfume in all_perfumes:
                        if perfume['_id'] in liked_perfume_ids:
                            continue  # Skip already ranked perfumes
                        
                        score = 0
                        
                        # Brand similarity
                        perfume_brand = (perfume.get('Brand', '') or perfume.get('brand', '')).lower()
                        if perfume_brand in liked_brands:
                            score += 3
                        
                        # Notes similarity
                        perfume_notes = perfume.get('Notes', '') or perfume.get('notes', '')
                        if perfume_notes:
                            notes_words = [word.strip().lower() for word in perfume_notes.split(',')]
                            common_notes = len(set(notes_words) & liked_notes)
                            score += common_notes
                        
                        scored_perfumes.append((perfume['_id'], score))
                    
                    # Sort by score and return top recommendations
                    scored_perfumes.sort(key=lambda x: x[1], reverse=True)
                    
                    if enable_diversity:
                        # Apply diversity enhancement to simple recommendations too
                        return self._enhance_recommendations_with_diversity(scored_perfumes, all_perfumes, top_n, 0.4)
                    else:
                        return [ObjectId(p[0]) for p in scored_perfumes[:top_n]]
            
            # Fallback: return random popular perfumes
            import random
            random.shuffle(all_perfumes)
            return [ObjectId(p['_id']) for p in all_perfumes[:top_n]]
            
        except Exception as e:
            print(f"Error in simple recommendations: {e}")
            return []

# Create a singleton instance
recommendation_service = RecommendationService()

def generate_recommendations(user_id, top_n=10, enable_diversity=True, diversity_weight=0.3):
    """Function to generate recommendations for a user"""
    return recommendation_service.generate_recommendations(user_id, top_n, enable_diversity, diversity_weight)

def get_cold_start_recommendations(top_n=10, enable_diversity=True, diversity_weight=0.3):
    """Function to get recommendations for new users"""
    return recommendation_service.generate_recommendations(None, top_n, enable_diversity, diversity_weight) 