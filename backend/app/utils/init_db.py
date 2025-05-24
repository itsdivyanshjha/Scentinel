import pandas as pd
import os
from pymongo import MongoClient
import json
from bson import ObjectId

def clean_data(df):
    """Clean and preprocess the perfume data"""
    # Fill missing values
    df = df.fillna('')
    
    # Convert text columns to lowercase
    text_columns = ['name', 'brand', 'notes', 'description']
    for col in text_columns:
        if col in df.columns:
            df[col] = df[col].str.lower()
    
    return df

def init_database():
    """Initialize the MongoDB database with perfume data"""
    # MongoDB connection
    mongo_uri = os.environ.get('MONGO_URI', 'mongodb://localhost:27017/scentinel')
    client = MongoClient(mongo_uri)
    db = client.get_database()
    
    # Check if perfumes collection already has data
    if db.perfumes.count_documents({}) > 0:
        print("Database already initialized. Skipping.")
        return
    
    # Path to CSV data
    csv_path = '../../data/perfume_data.csv'
    
    if not os.path.exists(csv_path):
        print(f"Data file not found: {csv_path}")
        return
    
    try:
        # Load and clean data
        df = pd.read_csv(csv_path)
        df = clean_data(df)
        
        # Convert DataFrame to list of dictionaries
        perfumes = json.loads(df.to_json(orient='records'))
        
        # Add ObjectId to each perfume
        for perfume in perfumes:
            perfume['_id'] = ObjectId()
        
        # Insert data into MongoDB
        db.perfumes.insert_many(perfumes)
        
        print(f"Initialized database with {len(perfumes)} perfumes")
    except Exception as e:
        print(f"Error initializing database: {e}")

if __name__ == "__main__":
    init_database() 