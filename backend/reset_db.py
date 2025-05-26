#!/usr/bin/env python
"""
Database reset script.
This script will clear the existing perfume data and reload it with correct field names.
"""

import pandas as pd
import os
from pymongo import MongoClient
import json
from bson import ObjectId

def clean_data(df):
    """Clean and preprocess the perfume data"""
    # Fill missing values
    df = df.fillna('')
    
    # Rename columns to lowercase to match frontend expectations
    column_mapping = {
        'Name': 'name',
        'Brand': 'brand', 
        'Notes': 'notes',
        'Description': 'description',
        'Image URL': 'image_url'
    }
    
    # Rename columns if they exist
    df = df.rename(columns=column_mapping)
    
    return df

def reset_database():
    """Reset and reinitialize the MongoDB database with perfume data"""
    # MongoDB connection - use container name for Docker networking
    mongo_uri = "mongodb://db:27017/scentinel"
    
    client = MongoClient(mongo_uri)
    db = client.scentinel
    
    # Clear existing perfume data
    print("Clearing existing perfume data...")
    db.perfumes.delete_many({})
    
    # Path to CSV data
    csv_path = '/app/perfume_data.csv'
    
    if not os.path.exists(csv_path):
        csv_path = 'perfume_data.csv'  # Try current directory
    
    if not os.path.exists(csv_path):
        print(f"Data file not found: {csv_path}")
        return
    
    try:
        # Load and clean data with proper encoding
        print("Loading CSV data...")
        df = pd.read_csv(csv_path, encoding='latin-1')
        print(f"Loaded {len(df)} rows from CSV")
        
        print("Original columns:", df.columns.tolist())
        
        df = clean_data(df)
        
        print("Renamed columns:", df.columns.tolist())
        print("Sample data:", df.head(1).to_dict('records'))
        
        # Convert DataFrame to list of dictionaries
        perfumes = json.loads(df.to_json(orient='records'))
        
        # Add ObjectId to each perfume
        for perfume in perfumes:
            perfume['_id'] = ObjectId()
        
        # Insert data into MongoDB
        print("Inserting perfumes into MongoDB...")
        db.perfumes.insert_many(perfumes)
        
        print(f"✅ Successfully reset database with {len(perfumes)} perfumes")
        
        # Verify the data
        sample = db.perfumes.find_one()
        print("Sample perfume from database:", sample)
        
    except Exception as e:
        print(f"Error resetting database: {e}")

if __name__ == "__main__":
    reset_database() 