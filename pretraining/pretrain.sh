#!/bin/bash
# Pre-training script for Scentinel
# This script sets up a virtual environment and runs the pre-training

# Check if Python is installed
if ! command -v python3 &> /dev/null
then
    echo "Python 3 could not be found. Please install Python 3."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install pandas numpy torch scikit-learn gensim python-dotenv

# Run the pre-training
echo "Running pre-training..."
python standalone_pretrain.py

# Check if pre-training was successful
if [ $? -eq 0 ]; then
    echo "Pre-training completed successfully!"
    echo "Pre-trained models are saved in backend/app/data/models/"
else
    echo "Pre-training failed! Check the error messages above."
fi

# Deactivate virtual environment
deactivate
echo "Virtual environment deactivated." 