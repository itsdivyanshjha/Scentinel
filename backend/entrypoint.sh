#!/bin/sh

echo "Waiting for MongoDB to start..."
sleep 5

echo "Initializing database..."
python init_db.py

echo "Pre-training recommendation models..."
python pretrain.py

echo "Starting Flask server..."
python run.py