#!/usr/bin/env python
"""
Model pre-training script.
Run this script to pre-train recommendation models on the full perfume dataset.

NOTE: For better compatibility with different environments, it's recommended
to use the standalone_pretrain.py script from the project root directory
with a virtual environment.
"""

import os
import sys
from dotenv import load_dotenv

def main():
    print("=" * 80)
    print("NOTE: For better compatibility, use the standalone pre-training script:")
    print("1. Create virtual environment: python -m venv venv")
    print("2. Activate: source venv/bin/activate (Unix/Mac) or venv\\Scripts\\activate (Windows)")
    print("3. Install dependencies: pip install pandas numpy torch scikit-learn gensim python-dotenv")
    print("4. Run: python standalone_pretrain.py")
    print("=" * 80)
    
    response = input("Do you want to continue with this script instead? (y/n): ")
    if response.lower() != 'y':
        print("Exiting. Please use the standalone script as recommended.")
        return False
    
    # Load environment variables
    load_dotenv()
    
    # Import after confirmation to avoid unnecessary imports
    try:
        from app.utils.pretrain_models import pretrain_models
        
        # Run pre-training
        print("Starting model pre-training...")
        success = pretrain_models()
        
        if success:
            print("Pre-training completed successfully!")
            return True
        else:
            print("Pre-training failed!")
            return False
    except ImportError as e:
        print(f"Import error: {e}")
        print("This confirms why the standalone script is recommended.")
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1) 