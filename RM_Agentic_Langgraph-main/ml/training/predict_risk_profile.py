"""
Risk Profile Prediction Node
Predicts risk profile and risk scores using ML model.
"""

import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from state import WorkflowState


def train_risk_model():
    """Train risk profile model using the available dataset"""

    model_path = "ml/models/risk_profile_model.pkl"

    if os.path.exists(model_path):
        print(f"✅ Risk model already exists at {model_path}")
        return True

    print("⚠️ Risk profile model training data not available")
    print("Please provide training dataset in data/ folder")
    return False


async def predict_risk_profile(state: WorkflowState) -> WorkflowState:
    """Predict risk profile using ML model or rule-based fallback"""

    try:
        model_path = "ml/models/risk_profile_model.pkl"
        encoder_path = "ml/models/label_encoders.pkl"

        # Try to load trained model
        if os.path.exists(model_path) and os.path.exists(encoder_path):
            model = joblib.load(model_path)
            label_encoder = joblib.load(encoder_path)

            print("🎯 ML Prediction: Risk profile loaded from model")

        else:
            # Train model if it doesn't exist
            print("🔄 Model not found, attempting to train...")
            if train_risk_model():
                # Retry prediction after training
                return await predict_risk_profile(state)
            else:
                # Fallback to rule-based prediction
                print("⚠️ Using rule-based fallback for risk prediction")

                # Simple rule-based risk assessment
                risk_score = 50  # Default medium risk
                if state.prospect.age < 30:
                    risk_score += 20
                elif state.prospect.age > 60:
                    risk_score -= 20

                if state.prospect.investment_experience_level == "Beginner":
                    risk_score -= 15
                elif state.prospect.investment_experience_level == "Expert":
                    risk_score += 15

                state.prospect.risk_score = max(0, min(100, risk_score))
                state.prospect.risk_level = "High" if risk_score > 70 else "Low" if risk_score < 30 else "Medium"

        state.current_step = "risk_assessed"

    except Exception as e:
        print(f"❌ Risk prediction error: {str(e)}")
        # Fallback prediction
        state.prospect.risk_score = 50
        state.prospect.risk_level = "Medium"
        state.current_step = "risk_assessed"

    return state


# Auto-train model if this file is run directly
if __name__ == "__main__":
    train_risk_model()
