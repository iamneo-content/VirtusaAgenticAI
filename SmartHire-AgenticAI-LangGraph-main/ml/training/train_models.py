#!/usr/bin/env python3
"""
Training script for SmartHire ML models
Run this script to pre-train the models before using the application

This script uses the training functions from ml/training/
"""

import os
import sys

# Add parent directory to path to import training modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

from ml.training.predict_experience import train_experience_model
from ml.training.score_resume import train_resume_score_model

def main():
    """Train all ML models"""

    print("🚀 Starting SmartHire Model Training...")
    print("=" * 50)

    # Create models directory
    os.makedirs("ml/models", exist_ok=True)

    # Train experience level model
    print("\n1️⃣ Training Experience Level Model...")
    try:
        if train_experience_model():
            print("✅ Experience Level Model trained successfully!")
        else:
            print("❌ Experience Level Model training failed!")
    except Exception as e:
        print(f"❌ Error training Experience Level Model: {e}")

    # Train resume score model
    print("\n2️⃣ Training Resume Score Model...")
    try:
        if train_resume_score_model():
            print("✅ Resume Score Model trained successfully!")
        else:
            print("❌ Resume Score Model training failed!")
    except Exception as e:
        print(f"❌ Error training Resume Score Model: {e}")

    print("\n" + "=" * 50)
    print("🎉 Model training completed!")
    print("\nModels saved in: ml/models/")
    print("\n📋 Next steps:")
    print("1. Set up your .env file with Gemini API keys")
    print("2. Run: streamlit run main.py")
    print("3. Start evaluating candidates!")

if __name__ == "__main__":
    main()
