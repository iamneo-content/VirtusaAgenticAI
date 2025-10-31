"""
ML Module - Machine Learning Components for Prospect Analysis

This module contains all ML-related components including:
- Model training and prediction (ml.training)
- Training utilities (ml.training)
- Trained models (ml/models/)

ML Prediction Nodes:
  The actual ML prediction nodes are located in:
  - ml.training.predict_risk_profile: Risk profile prediction
  - ml.training.predict_goal_success: Goal success prediction
"""

# Import training utilities and prediction nodes
from ml.training.train_models import main as train_all_models
from ml.training.predict_risk_profile import predict_risk_profile, train_risk_model
from ml.training.predict_goal_success import predict_goal_success, train_goal_model

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
