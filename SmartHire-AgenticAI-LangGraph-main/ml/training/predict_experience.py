"""
Experience Level Prediction Node
Predicts candidate experience level (Junior/Mid-level/Senior) using ML model.
"""

import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from state import CandidateState


def train_experience_model():
    """Train experience level model using the uploaded dataset"""

    dataset_path = "data/experience_level_training_dataset.csv"
    model_path = "ml/models/experience_level_model.pkl"
    encoder_path = "ml/models/experience_level_encoder.pkl"

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
        required_features = ['total_experience_years', 'skills_count', 'project_count', 'leadership_experience']
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
        if 'experience_level' not in df.columns:
            print("❌ 'experience_level' column not found in dataset")
            return False

        # Encode target labels
        label_encoder = LabelEncoder()
        df['experience_level_encoded'] = label_encoder.fit_transform(df['experience_level'])

        # Prepare features and target
        X = df[available_features]
        y = df['experience_level_encoded']

        print(f"🎯 Training with features: {available_features}")
        print(f"🎯 Target classes: {list(label_encoder.classes_)}")

        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)

        # Save model and encoder
        joblib.dump(model, model_path)
        joblib.dump(label_encoder, encoder_path)
        joblib.dump(available_features, "ml/models/experience_features.pkl")  # Save feature list

        print("✅ Experience level model trained and saved successfully!")
        return True

    except Exception as e:
        print(f"❌ Error training model: {str(e)}")
        return False


def predict_experience(state: CandidateState) -> CandidateState:
    """Predict candidate experience level using ML model or rule-based fallback"""

    try:
        model_path = "ml/models/experience_level_model.pkl"
        encoder_path = "ml/models/experience_level_encoder.pkl"
        features_path = "ml/models/experience_features.pkl"

        # Try to load trained model
        if os.path.exists(model_path) and os.path.exists(encoder_path) and os.path.exists(features_path):
            model = joblib.load(model_path)
            label_encoder = joblib.load(encoder_path)
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
            if 'leadership_experience' in available_features:
                ml_input['leadership_experience'] = int(features.get("leadership_experience", 0))

            # Create feature vector in the same order as training
            feature_vector = [[ml_input.get(feat, 0) for feat in available_features]]

            # Predict and decode
            prediction_encoded = model.predict(feature_vector)[0]
            prediction = label_encoder.inverse_transform([prediction_encoded])[0]

            print(f"🎯 ML Prediction: {prediction}")

        else:
            # Train model if it doesn't exist
            print("🔄 Model not found, attempting to train...")
            if train_experience_model():
                # Retry prediction after training
                return predict_experience(state)
            else:
                # Fallback to rule-based prediction
                print("⚠️ Using rule-based fallback for experience prediction")
                exp_years = float(state["resume_features"].get("total_experience_years", 0))
                if exp_years < 2:
                    prediction = "Junior"
                elif exp_years < 5:
                    prediction = "Mid-level"
                else:
                    prediction = "Senior"

        state["experience_level"] = prediction
        state["current_step"] = "experience_predicted"

    except Exception as e:
        state["errors"].append(f"Experience prediction failed: {str(e)}")
        print(f"❌ Experience prediction error: {str(e)}")
        # Fallback prediction
        exp_years = float(state["resume_features"].get("total_experience_years", 0))
        if exp_years < 2:
            state["experience_level"] = "Junior"
        elif exp_years < 5:
            state["experience_level"] = "Mid-level"
        else:
            state["experience_level"] = "Senior"
        state["current_step"] = "experience_predicted"

    return state


# Auto-train model if this file is run directly
if __name__ == "__main__":
    train_experience_model()
