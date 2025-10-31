"""
HLD Quality Prediction - Inference Module
Production-ready inference for quality predictions
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from .generate_dataset import SyntheticDatasetGenerator
from .train_large_model import LargeScaleMLTrainer


class HLDQualityPredictor:
    """Production-ready quality prediction interface"""

    def __init__(self):
        """Initialize predictor"""
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_columns = SyntheticDatasetGenerator.get_feature_columns()

    def train_models_from_scratch(self, dataset_path='ml/training/synthetic_hld_dataset.csv'):
        """
        Train models for inference

        Parameters:
            dataset_path (str): Path to training dataset

        Returns:
            None (trains and stores models)
        """
        print("Training models from scratch...")
        trainer = LargeScaleMLTrainer()

        # Load or generate dataset
        if Path(dataset_path).exists():
            trainer.load_dataset(dataset_path)
        else:
            trainer.load_dataset()

        # Prepare data
        trainer.prepare_data(trainer.df)

        # Train models
        trainer.train_models()

        # Store models and scaler
        self.models = trainer.models
        self.scaler = trainer.scaler

        print("Models trained and ready for inference!")

    def load_models_from_disk(self, model_dir='ml/models'):
        """
        Load pre-trained models from disk

        Parameters:
            model_dir (str): Directory containing model files

        Returns:
            bool: True if models loaded successfully
        """
        model_path = Path(model_dir)

        try:
            # Load models
            for model_name in ['Random Forest', 'Gradient Boosting', 'Linear Regression']:
                filename = model_path / f"{model_name.lower().replace(' ', '_')}_model.pkl"
                if filename.exists():
                    with open(filename, 'rb') as f:
                        self.models[model_name] = pickle.load(f)

            # Load scaler
            scaler_path = model_path / "scaler.pkl"
            if scaler_path.exists():
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)

            return len(self.models) > 0
        except Exception as e:
            print(f"Error loading models: {e}")
            return False

    def predict(self, features_dict):
        """
        Predict quality score for single document

        Parameters:
            features_dict (dict): Dictionary with 38 feature values

        Returns:
            dict: Predictions from each model (0-100 range)
        """
        if not self.models:
            return {'error': 'Models not trained'}

        try:
            # Extract features in correct order
            feature_values = [features_dict.get(col, 0) for col in self.feature_columns]
            X = np.array(feature_values).reshape(1, -1)

            # Scale features
            X_scaled = self.scaler.transform(X)

            # Make predictions
            predictions = {}
            for model_name, model in self.models.items():
                if model:
                    pred = model.predict(X_scaled)[0]
                    # Clamp to 0-100 range
                    pred = np.clip(pred, 0, 100)
                    predictions[model_name] = float(pred)

            # Calculate ensemble average
            if predictions:
                predictions['ensemble_average'] = float(np.mean(list(predictions.values())))

            return predictions

        except Exception as e:
            return {'error': str(e)}

    def predict_batch(self, features_list):
        """
        Predict quality for multiple documents

        Parameters:
            features_list (list): List of feature dictionaries

        Returns:
            DataFrame: Predictions from all models
        """
        results = []

        for features_dict in features_list:
            pred = self.predict(features_dict)
            pred['sample_id'] = len(results)
            results.append(pred)

        return pd.DataFrame(results)

    def print_feature_guide(self):
        """
        Display feature value ranges

        Parameters:
            None

        Returns:
            None (prints guide)
        """
        print("="*70)
        print("FEATURE VALUE RANGES AND GUIDANCE")
        print("="*70)

        # Feature value ranges based on generation logic
        ranges = {
            'word_count': (500, 5000, 'Document word count'),
            'sentence_count': (50, 500, 'Number of sentences'),
            'avg_sentence_length': (10, 25, 'Average words per sentence'),
            'avg_word_length': (4, 6, 'Average characters per word'),

            'header_count': (5, 40, 'Number of section headers'),
            'code_block_count': (0, 20, 'Number of code examples'),
            'table_count': (0, 15, 'Number of tables'),
            'list_count': (5, 30, 'Number of lists'),
            'diagram_count': (0, 10, 'Number of diagrams'),

            'completeness_score': (0, 100, 'Overall content completeness (0-100)'),
            'security_mentions': (0, 20, 'Count of security-related mentions'),
            'scalability_mentions': (0, 20, 'Count of scalability mentions'),
            'api_mentions': (0, 25, 'Count of API-related mentions'),
            'database_mentions': (0, 15, 'Count of database mentions'),
            'performance_mentions': (0, 20, 'Count of performance mentions'),
            'monitoring_mentions': (0, 15, 'Count of monitoring mentions'),

            'duplicate_headers': (0, 10, 'Number of duplicate headers'),
            'header_coverage': (0.3, 1.0, 'Ratio of covered sections (0-1)'),
            'code_coverage': (0, 1.0, 'Proportion of document with code (0-1)'),

            'keyword_density': (0.01, 0.1, 'Ratio of important keywords'),
            'section_density': (0.1, 0.8, 'Ratio of sections to total content'),

            'has_architecture_section': (0, 1, 'Binary: has architecture section'),
            'has_security_section': (0, 1, 'Binary: has security section'),
            'has_scalability_section': (0, 1, 'Binary: has scalability section'),
            'has_deployment_section': (0, 1, 'Binary: has deployment section'),
            'has_monitoring_section': (0, 1, 'Binary: has monitoring section'),
            'has_api_spec': (0, 1, 'Binary: has API specification'),
            'has_data_model': (0, 1, 'Binary: has data model'),

            'service_count': (1, 15, 'Number of microservices'),
            'entity_count': (5, 40, 'Number of domain entities'),
            'api_endpoint_count': (0, 50, 'Total API endpoints'),

            'readability_score': (20, 100, 'Readability metric (0-100)'),
            'completeness_index': (0, 1, 'Completeness ratio (0-1)'),
            'consistency_index': (0, 1, 'Internal consistency (0-1)'),
            'documentation_quality': (0, 100, 'Quality metric (0-100)'),

            'technical_terms_density': (0, 0.3, 'Ratio of technical terminology'),
            'acronym_count': (0, 30, 'Number of acronyms used')
        }

        for feature in self.feature_columns:
            if feature in ranges:
                min_val, max_val, desc = ranges[feature]
                print(f"{feature:<30} Range: {min_val:>8} - {max_val:>8}  ({desc})")


def demo_predictions():
    """Demonstrate quality predictions"""
    print("="*70)
    print("HLD QUALITY PREDICTION - INFERENCE SCRIPT")
    print("="*70)

    print("\nThis script demonstrates how to use trained models for predictions.")
    print("Models are trained on 30,000 row dataset with 38 features.\n")

    predictor = HLDQualityPredictor()

    # Try to load models
    if not predictor.load_models_from_disk():
        print("Models not found on disk. Training from scratch...")
        predictor.train_models_from_scratch()

    # Demo scenarios
    print("\nDEMO: Single Document Prediction\n")

    # Excellent HLD
    excellent_features = {
        'word_count': 4500,
        'sentence_count': 400,
        'avg_sentence_length': 20,
        'avg_word_length': 5.5,
        'header_count': 35,
        'code_block_count': 15,
        'table_count': 10,
        'list_count': 25,
        'diagram_count': 8,
        'completeness_score': 95,
        'security_mentions': 18,
        'scalability_mentions': 17,
        'api_mentions': 22,
        'database_mentions': 14,
        'performance_mentions': 16,
        'monitoring_mentions': 13,
        'duplicate_headers': 1,
        'header_coverage': 0.95,
        'code_coverage': 0.7,
        'keyword_density': 0.08,
        'section_density': 0.7,
        'has_architecture_section': 1,
        'has_security_section': 1,
        'has_scalability_section': 1,
        'has_deployment_section': 1,
        'has_monitoring_section': 1,
        'has_api_spec': 1,
        'has_data_model': 1,
        'service_count': 12,
        'entity_count': 35,
        'api_endpoint_count': 45,
        'readability_score': 90,
        'completeness_index': 0.95,
        'consistency_index': 0.92,
        'documentation_quality': 92,
        'technical_terms_density': 0.25,
        'acronym_count': 25
    }

    predictions = predictor.predict(excellent_features)
    print("Document: Excellent HLD")
    for model, score in predictions.items():
        print(f"  {model:<20}: {score:>6.2f}/100")

    # Poor HLD
    poor_features = {
        'word_count': 800,
        'sentence_count': 80,
        'avg_sentence_length': 12,
        'avg_word_length': 4.5,
        'header_count': 8,
        'code_block_count': 2,
        'table_count': 1,
        'list_count': 5,
        'diagram_count': 0,
        'completeness_score': 25,
        'security_mentions': 2,
        'scalability_mentions': 1,
        'api_mentions': 3,
        'database_mentions': 2,
        'performance_mentions': 1,
        'monitoring_mentions': 0,
        'duplicate_headers': 5,
        'header_coverage': 0.4,
        'code_coverage': 0.1,
        'keyword_density': 0.02,
        'section_density': 0.2,
        'has_architecture_section': 0,
        'has_security_section': 0,
        'has_scalability_section': 0,
        'has_deployment_section': 0,
        'has_monitoring_section': 0,
        'has_api_spec': 0,
        'has_data_model': 0,
        'service_count': 2,
        'entity_count': 5,
        'api_endpoint_count': 5,
        'readability_score': 30,
        'completeness_index': 0.25,
        'consistency_index': 0.35,
        'documentation_quality': 20,
        'technical_terms_density': 0.05,
        'acronym_count': 3
    }

    predictions = predictor.predict(poor_features)
    print("\nDocument: Poor HLD")
    for model, score in predictions.items():
        print(f"  {model:<20}: {score:>6.2f}/100")


def main():
    """Main entry point"""
    demo_predictions()


if __name__ == "__main__":
    main()
