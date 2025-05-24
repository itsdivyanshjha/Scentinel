import os
import pandas as pd
import numpy as np
import torch
import pickle
from app.models.ml_models import RankNetModel, DPLModel, BayesianPersonalizedRanking
from app.utils.embedding_utils import get_perfume_embeddings
from sklearn.model_selection import train_test_split

# Paths for saving pre-trained models
MODELS_DIR = 'app/data/models'
RANKNET_PATH = f'{MODELS_DIR}/ranknet_pretrained.pkl'
DPL_PATH = f'{MODELS_DIR}/dpl_pretrained.pkl'
BPR_PATH = f'{MODELS_DIR}/bpr_pretrained.pkl'

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
    data_path = '../../data/perfume_data.csv'
    
    try:
        df = pd.read_csv(data_path)
        print(f"Loaded {len(df)} perfumes from CSV")
        return df
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return None

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
    print("Generating perfume embeddings...")
    embeddings = get_perfume_embeddings()
    
    # Group rankings by user
    user_rankings = {}
    for user_id, perfume_id, rank in synthetic_rankings:
        if user_id not in user_rankings:
            user_rankings[user_id] = []
        user_rankings[user_id].append((perfume_id, rank))
    
    # Split into train/test sets
    train_users, test_users = train_test_split(list(user_rankings.keys()), test_size=0.2)
    
    # Initialize models
    print("Initializing models...")
    ranknet_model = RankNetModel()
    dpl_model = DPLModel()
    bpr_model = BayesianPersonalizedRanking()
    
    # Train models with all synthetic user data
    print("Training models...")
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
    
    # Evaluate models (optional)
    print("Evaluating models...")
    # Simple evaluation on test users could be added here
    
    # Save pre-trained models
    print("Saving pre-trained models...")
    with open(RANKNET_PATH, 'wb') as f:
        pickle.dump(ranknet_model, f)
    
    with open(DPL_PATH, 'wb') as f:
        pickle.dump(dpl_model, f)
    
    with open(BPR_PATH, 'wb') as f:
        pickle.dump(bpr_model, f)
    
    print("Pre-training complete!")
    return True

if __name__ == "__main__":
    pretrain_models() 