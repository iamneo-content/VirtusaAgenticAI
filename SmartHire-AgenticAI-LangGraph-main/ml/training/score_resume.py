"""
Resume Scoring Node
Scores resume quality (0-10) using ML model.
"""

import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from state import CandidateState


def train_resume_score_model():
    """Train resume scoring model using the uploaded dataset"""

    dataset_path = "data/resume_score_training_dataset.csv"
    model_path = "ml/models/resume_score_model.pkl"
    scaler_path = "ml/models/resume_score_scaler.pkl"

    if not os.path.exists(dataset_path):
        print(f"Dataset not found at {dataset_path}")
        return False

    try:
        # Load and prepare data
        df = pd.read_csv(dataset_path)

        if df.empty:
            print("Dataset is empty")
            return False

        print(f"📊 Loaded dataset with {len(df)} rows and columns: {list(df.columns)}")

        # Create models directory if it doesn't exist
        os.makedirs("ml/models", exist_ok=True)

        # Check required columns
        required_features = [
            'total_experience_years', 'skills_count', 'project_count',
            'certification_count', 'leadership_experience', 'has_research_work'
        ]
        available_features = []

        for col in required_features:
            if col in df.columns:
                available_features.append(col)
                df[col] = df[col].fillna(0)
            else:
                print(f"⚠️ Column '{col}' not found in dataset")

        if not available_features:
            print("❌ No required feature columns found")
            return False

        # Prepare target variable
        if 'resume_score' not in df.columns:
            print("❌ 'resume_score' column not found in dataset")
            return False

        X = df[available_features]
        y = df['resume_score']

        print(f"🎯 Training with features: {available_features}")
        print(f"🎯 Score range: {y.min():.1f} - {y.max():.1f}")

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_scaled, y)

        # Save model, scaler, and feature list
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)
        joblib.dump(available_features, "ml/models/resume_score_features.pkl")

        print("✅ Resume score model trained and saved successfully!")
        return True

    except Exception as e:
        print(f"❌ Error training resume score model: {str(e)}")
        return False


def predict_resume_score(state: CandidateState) -> CandidateState:
    """Predict resume score using ML model or rule-based fallback"""

    try:
        model_path = "ml/models/resume_score_model.pkl"
        scaler_path = "ml/models/resume_score_scaler.pkl"
        features_path = "ml/models/resume_score_features.pkl"

        # Try to load trained model
        if os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(features_path):
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            available_features = joblib.load(features_path)

            features = state["resume_features"]

            # Prepare ML input based on available features
            ml_input = {}
            if 'total_experience_years' in available_features:
                ml_input['total_experience_years'] = float(features.get("total_experience_years", 0))
            if 'skills_count' in available_features:
                ml_input['skills_count'] = len(features.get("skills", []))
            if 'project_count' in available_features:
                ml_input['project_count'] = len(features.get("projects", []))
            if 'certification_count' in available_features:
                ml_input['certification_count'] = len(features.get("certifications", []))
            if 'leadership_experience' in available_features:
                ml_input['leadership_experience'] = int(features.get("leadership_experience", 0))
            if 'has_research_work' in available_features:
                ml_input['has_research_work'] = int(features.get("has_research_work", 0))

            # Create feature vector in the same order as training
            feature_vector = [[ml_input.get(feat, 0) for feat in available_features]]

            # Scale and predict
            feature_vector_scaled = scaler.transform(feature_vector)
            prediction = model.predict(feature_vector_scaled)[0]

            # Ensure score is between 0 and 10
            prediction = max(0, min(10, prediction))

            print(f"🎯 ML Resume Score: {prediction:.1f}")

        else:
            # Train model if it doesn't exist
            print("🔄 Resume score model not found, attempting to train...")
            if train_resume_score_model():
                # Retry prediction after training
                return predict_resume_score(state)
            else:
                # Fallback to rule-based scoring
                print("⚠️ Using rule-based fallback for resume scoring")
                features = state["resume_features"]

                score = 0
                # Experience points (0-3)
                exp_years = float(features.get("total_experience_years", 0))
                score += min(3, exp_years * 0.5)

                # Skills points (0-3)
                skills_count = len(features.get("skills", []))
                score += min(3, skills_count * 0.2)

                # Projects points (0-2)
                project_count = len(features.get("projects", []))
                score += min(2, project_count * 0.5)

                # Certifications points (0-1)
                cert_count = len(features.get("certifications", []))
                score += min(1, cert_count * 0.3)

                # Leadership points (0-1)
                leadership = int(features.get("leadership_experience", 0))
                score += min(1, leadership)

                prediction = round(score, 2)
                print(f"🎯 Rule-based Resume Score: {prediction}")

        state["resume_score"] = prediction

    except Exception as e:
        state["errors"].append(f"Resume scoring failed: {str(e)}")
        print(f"❌ Resume scoring error: {str(e)}")
        # Fallback score
        state["resume_score"] = 5.0

    return state


# Auto-train model if this file is run directly
if __name__ == "__main__":
    train_resume_score_model()
