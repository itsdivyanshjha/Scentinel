import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.preprocessing import LabelEncoder
import os
import pickle
from app import mongo

# Path to save embeddings
EMBEDDING_CACHE_PATH = 'app/data/perfume_embeddings.pkl'

def preprocess_text(text):
    """Preprocess text data"""
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase and remove special characters
    text = text.lower()
    # Split into words
    words = text.split()
    
    return words

def generate_embeddings_from_csv():
    """Generate embeddings from CSV data"""
    # Path to perfume data
    data_path = '../../data/perfume_data.csv'
    
    # Check if file exists
    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}")
        return {}
    
    # Load data
    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return {}
    
    # Clean data
    df = df.fillna('')
    
    # Prepare text corpus for Word2Vec
    corpus = []
    
    # Extract text features
    text_columns = ['notes', 'description', 'brand']
    for _, row in df.iterrows():
        doc = []
        for col in text_columns:
            if col in df.columns and isinstance(row[col], str):
                doc.extend(preprocess_text(row[col]))
        corpus.append(doc)
    
    # Filter out empty documents
    corpus = [doc for doc in corpus if len(doc) > 0]
    
    if not corpus:
        print("Warning: No valid documents for Word2Vec training")
        return {}
    
    # Train Word2Vec model with explicit build_vocab step
    w2v_model = Word2Vec(vector_size=300, window=5, min_count=1, workers=4)
    w2v_model.build_vocab(corpus)
    w2v_model.train(corpus, total_examples=len(corpus), epochs=10)
    
    # Generate embeddings for each perfume
    perfume_embeddings = {}
    
    for idx, row in df.iterrows():
        perfume_id = row.get('id', idx)
        
        # Initialize embedding vector
        embedding = np.zeros(300)
        word_count = 0
        
        # Combine text from all text columns
        for col in text_columns:
            if col in df.columns and isinstance(row[col], str):
                words = preprocess_text(row[col])
                for word in words:
                    if word in w2v_model.wv:
                        embedding += w2v_model.wv[word]
                        word_count += 1
        
        # Average word vectors
        if word_count > 0:
            embedding /= word_count
        
        # Keep embedding at 300 dimensions for model compatibility
        # Gender information is already captured in the text features
        
        perfume_embeddings[perfume_id] = embedding
    
    # Save embeddings to cache
    os.makedirs(os.path.dirname(EMBEDDING_CACHE_PATH), exist_ok=True)
    with open(EMBEDDING_CACHE_PATH, 'wb') as f:
        pickle.dump(perfume_embeddings, f)
    
    return perfume_embeddings

def generate_embeddings_from_db():
    """Generate embeddings from MongoDB data"""
    # Get perfumes from database
    perfumes = list(mongo.db.perfumes.find())
    
    if not perfumes:
        print("No perfumes found in database")
        return {}
    
    # Prepare text corpus for Word2Vec
    corpus = []
    
    # Extract text features - check for both field name variations
    text_fields = ['Notes', 'notes', 'Description', 'description', 'Brand', 'brand']
    for perfume in perfumes:
        doc = []
        for field in text_fields:
            if field in perfume and isinstance(perfume[field], str):
                doc.extend(preprocess_text(perfume[field]))
        corpus.append(doc)
    
    # Filter out empty documents
    corpus = [doc for doc in corpus if len(doc) > 0]
    
    if not corpus:
        print("Warning: No valid documents for Word2Vec training from database")
        return {}
    
    # Train Word2Vec model with explicit build_vocab step
    w2v_model = Word2Vec(vector_size=300, window=5, min_count=1, workers=4)
    w2v_model.build_vocab(corpus)
    w2v_model.train(corpus, total_examples=len(corpus), epochs=10)
    
    # Generate embeddings for each perfume
    perfume_embeddings = {}
    
    for perfume in perfumes:
        perfume_id = str(perfume['_id'])  # Convert ObjectId to string for consistent mapping
        
        # Initialize embedding vector
        embedding = np.zeros(300)
        word_count = 0
        
        # Combine text from all text fields
        for field in text_fields:
            if field in perfume and isinstance(perfume[field], str):
                words = preprocess_text(perfume[field])
                for word in words:
                    if word in w2v_model.wv:
                        embedding += w2v_model.wv[word]
                        word_count += 1
        
        # Average word vectors
        if word_count > 0:
            embedding /= word_count
        
        # Keep embedding at 300 dimensions for model compatibility
        # Gender information is already captured in the text features
        
        perfume_embeddings[perfume_id] = embedding
    
    # Save embeddings to cache
    os.makedirs(os.path.dirname(EMBEDDING_CACHE_PATH), exist_ok=True)
    with open(EMBEDDING_CACHE_PATH, 'wb') as f:
        pickle.dump(perfume_embeddings, f)
    
    return perfume_embeddings

def get_perfume_embeddings():
    """Get perfume embeddings, either from cache or generate new ones"""
    # Check if embeddings are cached
    if os.path.exists(EMBEDDING_CACHE_PATH):
        try:
            with open(EMBEDDING_CACHE_PATH, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error loading cached embeddings: {e}")
    
    # Try generating from database first
    embeddings = generate_embeddings_from_db()
    
    # If no embeddings from db, try from CSV
    if not embeddings:
        embeddings = generate_embeddings_from_csv()
    
    return embeddings 