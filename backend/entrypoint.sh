#!/bin/bash

# Wait for MongoDB to be available
echo "Waiting for MongoDB..."
sleep 10

# Initialize the database with perfume data
echo "Initializing database..."
python init_db.py

# Check if models exist, if not suggest pretraining
if [ ! -f "/tmp/models_exist_flag" ]; then
    echo "No pre-trained models found. For better recommendations, run pretraining:"
    echo "  python /app/../pretraining/standalone_pretrain.py"
    echo "  OR from project root: ./pretraining/pretrain.sh"
fi

# Start the Flask application
echo "Starting Flask application..."
python run.py