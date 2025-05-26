#!/bin/sh

echo "Waiting for MongoDB to start..."
sleep 5

echo "Initializing database..."
python init_db.py

echo "Checking for pre-trained models..."
if [ -f "app/data/models/ranknet_pretrained.pkl" ]; then
    echo "Pre-trained models found, skipping pre-training..."
else
    echo "No pre-trained models found. Please run standalone_pretrain.py first."
fi

echo "Starting Flask server..."
python run.py