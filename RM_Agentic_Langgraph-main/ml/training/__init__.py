"""
ML Training Module
Contains model training and prediction nodes for the Prospect Analysis workflow.
"""

from ml.training.predict_risk_profile import predict_risk_profile, train_risk_model
from ml.training.predict_goal_success import predict_goal_success, train_goal_model
from ml.training.train_models import main as train_all_models

__all__ = [
    # Prediction nodes
    'predict_risk_profile',
    'predict_goal_success',
    # Training functions
    'train_risk_model',
    'train_goal_model',
    'train_all_models',
]

__version__ = '2.0.0'
__doc__ = """
ML Training Module - Machine Learning Components for Prospect Analysis

Architecture:
    - training/: ML prediction nodes and training utilities
      - predict_risk_profile: Risk profile prediction node
      - predict_goal_success: Goal success prediction node
      - train_models: Training orchestration script
    - models/: Trained model files (generated after training)
"""
