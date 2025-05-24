#!/usr/bin/env python
"""
Database initialization script.
Run this script to load perfume data from CSV into MongoDB.
"""

import os
import sys
from dotenv import load_dotenv
from app.utils.init_db import init_database

if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    
    # Run database initialization
    print("Initializing database...")
    init_database()
    print("Database initialization complete.") 