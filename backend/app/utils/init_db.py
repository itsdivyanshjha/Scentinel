import pandas as pd
import os
from pymongo import MongoClient
import json
from bson import ObjectId
import urllib.parse

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
    # MongoDB connection with new credentials
    username = "divyanshapp"
    password = "Divyansh@2025"
    mongo_uri = f"mongodb+srv://{username}:{urllib.parse.quote_plus(password)}@scentinelcluster.apxy5nv.mongodb.net/scentinel?retryWrites=true&w=majority"
    
    client = MongoClient(mongo_uri)
    db = client.get_database()
    
    # Check if perfumes collection already has data
    if db.perfumes.count_documents({}) > 0:
        print("Database already initialized. Skipping.")
        return
    
    # Path to CSV data - corrected path
    csv_path = '../perfume_data.csv'
    
    if not os.path.exists(csv_path):
        print(f"Data file not found: {csv_path}")
        return
    
    try:
        # Load and clean data with proper encoding
        print("Loading CSV data...")
        df = pd.read_csv(csv_path, encoding='latin-1')  # Try latin-1 encoding first
        print(f"Loaded {len(df)} rows from CSV")
        
        df = clean_data(df)
        
        # Convert DataFrame to list of dictionaries
        perfumes = json.loads(df.to_json(orient='records'))
        
        # Add ObjectId to each perfume
        for perfume in perfumes:
            perfume['_id'] = ObjectId()
        
        # Insert data into MongoDB
        print("Inserting perfumes into MongoDB...")
        db.perfumes.insert_many(perfumes)
        
        print(f"✅ Successfully initialized database with {len(perfumes)} perfumes")
    except UnicodeDecodeError:
        # Try different encoding if latin-1 fails
        try:
            print("Trying UTF-8 with error handling...")
            df = pd.read_csv(csv_path, encoding='utf-8', errors='ignore')
            df = clean_data(df)
            perfumes = json.loads(df.to_json(orient='records'))
            
            for perfume in perfumes:
                perfume['_id'] = ObjectId()
            
            db.perfumes.insert_many(perfumes)
            print(f"✅ Successfully initialized database with {len(perfumes)} perfumes")
        except Exception as e:
            print(f"Error with UTF-8 encoding: {e}")
    except Exception as e:
        print(f"Error initializing database: {e}")

if __name__ == "__main__":
    init_database() 