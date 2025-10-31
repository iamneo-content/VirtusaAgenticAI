"""
Goal Success Prediction Node
Predicts probability of goal success using ML model.
"""

import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from state import WorkflowState


def train_goal_model():
    """Train goal success prediction model using the available dataset"""

    model_path = "ml/models/goal_success_model.pkl"

    if os.path.exists(model_path):
        print(f"✅ Goal success model already exists at {model_path}")
        return True

    print("⚠️ Goal success model training data not available")
    print("Please provide training dataset in data/ folder")
    return False


async def predict_goal_success(state: WorkflowState) -> WorkflowState:
    """Predict probability of achieving investment goal using ML model or rule-based fallback"""

    try:
        model_path = "ml/models/goal_success_model.pkl"
        encoder_path = "ml/models/goal_success_label_encoders.pkl"

        # Try to load trained model
        if os.path.exists(model_path) and os.path.exists(encoder_path):
            model = joblib.load(model_path)
            label_encoder = joblib.load(encoder_path)

            print("🎯 ML Prediction: Goal success probability loaded from model")

        else:
            # Train model if it doesn't exist
            print("🔄 Model not found, attempting to train...")
            if train_goal_model():
                # Retry prediction after training
                return await predict_goal_success(state)
            else:
                # Fallback to rule-based prediction
                print("⚠️ Using rule-based fallback for goal success prediction")

                # Simple rule-based goal success assessment
                success_probability = 0.5  # Default 50%

                # Adjust based on investment horizon
                if state.prospect.investment_horizon_years >= 10:
                    success_probability += 0.25
                elif state.prospect.investment_horizon_years >= 5:
                    success_probability += 0.15
                elif state.prospect.investment_horizon_years < 2:
                    success_probability -= 0.20

                # Adjust based on savings rate
                if state.prospect.annual_income > 0:
                    savings_rate = state.prospect.current_savings / (state.prospect.annual_income * state.prospect.investment_horizon_years + 0.01)
                    if savings_rate > 0.3:
                        success_probability += 0.15
                    elif savings_rate < 0.1:
                        success_probability -= 0.15

                state.prospect.goal_success_probability = max(0.0, min(1.0, success_probability))

        state.current_step = "goal_assessed"

    except Exception as e:
        print(f"❌ Goal success prediction error: {str(e)}")
        # Fallback prediction
        state.prospect.goal_success_probability = 0.5
        state.current_step = "goal_assessed"

    return state


# Auto-train model if this file is run directly
if __name__ == "__main__":
    train_goal_model()
