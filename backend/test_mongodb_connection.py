#!/usr/bin/env python
"""
Test MongoDB connection.
Run this script to verify the connection to your MongoDB Atlas cluster.
"""

import os
from dotenv import load_dotenv
from pymongo import MongoClient
import urllib.parse

def test_connection():
    # Load environment variables
    load_dotenv()
    
    # Make sure password is URL encoded
    username = "jhadivyansh29"
    password = urllib.parse.quote_plus("pu3ib1WQzUiQGcC0")
    
    # Explicitly specify the database in the connection string
    mongo_uri = f"mongodb+srv://{username}:{password}@scentinelcluster.apxy5nv.mongodb.net/scentinel?retryWrites=true&w=majority&appName=ScentinelCluster"
    
    print(f"Using connection string: {mongo_uri}")
    
    try:
        # Connect to MongoDB
        print("Connecting to MongoDB...")
        client = MongoClient(mongo_uri)
        
        # Test connection by pinging the server
        print("Testing connection with ping...")
        ping_result = client.admin.command('ping')
        if ping_result.get('ok') == 1.0:
            print("Ping successful!")
        
        # Test connection by listing databases
        print("Listing databases:")
        dbs = client.list_database_names()
        for db in dbs:
            print(f"- {db}")
        
        # Get reference to scentinel database
        db = client.scentinel
        
        # Create collections if they don't exist
        collections = ["users", "perfumes", "recommendations", "rankings"]
        existing_collections = db.list_collection_names()
        
        for collection in collections:
            if collection not in existing_collections:
                # Create collection by inserting and removing a dummy document
                print(f"Creating collection: {collection}")
                db[collection].insert_one({"test": "test"})
                db[collection].delete_one({"test": "test"})
            else:
                print(f"Collection already exists: {collection}")
                print(f"Documents count: {db[collection].count_documents({})}")
        
        return True
    except Exception as e:
        print(f"Error connecting to MongoDB: {e}")
        return False

if __name__ == "__main__":
    success = test_connection()
    if success:
        print("\nMongoDB connection test completed successfully!")
    else:
        print("\nMongoDB connection test failed. Check your connection string and network.")
        print("\nTroubleshooting steps:")
        print("1. Verify your MongoDB Atlas username and password")
        print("2. Check if your IP address is whitelisted in Atlas Network Access settings")
        print("3. Ensure your database user has the correct permissions (atlasAdmin recommended)")
        print("4. Try resetting your database user password in the Atlas dashboard")
        print("5. Check if your MongoDB Atlas cluster is up and running") 