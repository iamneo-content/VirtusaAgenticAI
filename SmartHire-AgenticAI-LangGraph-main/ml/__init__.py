"""
ML Module - Machine Learning Components for SmartHire

This module contains all ML-related components including:
- Model training and prediction (ml.training)
- Training utilities (ml.training)
- Trained models (ml/models/)

ML Prediction Nodes:
  The actual ML prediction nodes are located in:
  - ml.training.predict_experience: Experience level prediction
  - ml.training.score_resume: Resume quality scoring
"""

# Import training utilities and prediction nodes
from .training.train_models import main as train_all_models
from .training.predict_experience import predict_experience, train_experience_model
from .training.score_resume import predict_resume_score, train_resume_score_model

__all__ = [
    # Prediction nodes
    'predict_experience',
    'predict_resume_score',
    # Training functions
    'train_experience_model',
    'train_resume_score_model',
    'train_all_models',
]

__version__ = '2.0.0'
__doc__ = """
ML Module - Machine Learning Components for SmartHire

Architecture:
    - training/: ML prediction nodes and training utilities
      - predict_experience: Experience level prediction node
      - score_resume: Resume scoring node
      - train_models: Training orchestration script
    - models/: Trained model files (generated after training)
"""
