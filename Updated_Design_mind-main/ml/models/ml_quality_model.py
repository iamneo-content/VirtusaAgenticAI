"""
ML Quality Model Classes
Encapsulates model logic and evaluation functionality
"""

import numpy as np
import pickle
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error


class MLQualityModel:
    """Base ML quality model"""

    def __init__(self, model_type='random_forest'):
        """
        Initialize ML model

        Parameters:
            model_type (str): Type of model ('random_forest', 'gradient_boosting', 'linear_regression')
        """
        self.model_type = model_type
        self.model = self._create_model()
        self.feature_importance = None
        self.trained = False

    def _create_model(self):
        """Create the underlying sklearn model"""
        if self.model_type == 'random_forest':
            return RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == 'gradient_boosting':
            return GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        elif self.model_type == 'linear_regression':
            return LinearRegression()
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def train(self, X_train, y_train):
        """
        Train the model

        Parameters:
            X_train: Training feature matrix
            y_train: Training target values

        Returns:
            None (fits model)
        """
        self.model.fit(X_train, y_train)
        self.trained = True

        # Extract feature importance if available
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = self.model.feature_importances_

    def predict(self, X):
        """
        Make predictions

        Parameters:
            X: Feature matrix

        Returns:
            ndarray: Predictions
        """
        if not self.trained:
            raise ValueError("Model not trained yet")

        predictions = self.model.predict(X)
        # Clamp predictions to 0-100 range
        return np.clip(predictions, 0, 100)

    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance

        Parameters:
            X_test: Test feature matrix
            y_test: Test target values

        Returns:
            dict: Evaluation metrics
        """
        if not self.trained:
            raise ValueError("Model not trained yet")

        y_pred = self.predict(X_test)

        metrics = {
            'r2': r2_score(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'mape': mean_absolute_percentage_error(y_test, y_pred)
        }

        return metrics

    def get_feature_importance(self, feature_names=None, top_n=20):
        """
        Get feature importance

        Parameters:
            feature_names (list): Feature column names
            top_n (int): Number of top features

        Returns:
            dict: Feature importance scores
        """
        if not hasattr(self.model, 'feature_importances_'):
            return {}

        importances = self.model.feature_importances_
        indices = np.argsort(importances)[-top_n:][::-1]

        result = {}
        for idx in indices:
            if feature_names:
                result[feature_names[idx]] = float(importances[idx])
            else:
                result[f"feature_{idx}"] = float(importances[idx])

        return result

    def save_model(self, filepath):
        """
        Save trained model

        Parameters:
            filepath (str): Output file path

        Returns:
            None
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        """
        Load previously trained model

        Parameters:
            filepath (str): Input file path

        Returns:
            None (loads model)
        """
        with open(filepath, 'rb') as f:
            loaded = pickle.load(f)
            self.model = loaded.model
            self.trained = loaded.trained
            self.feature_importance = loaded.feature_importance

        print(f"Model loaded from {filepath}")

    def get_hyperparameters(self):
        """
        Get model hyperparameters

        Returns:
            dict: Hyperparameters
        """
        return self.model.get_params()

    def set_hyperparameters(self, **kwargs):
        """
        Set model hyperparameters

        Parameters:
            **kwargs: Hyperparameter key-value pairs

        Returns:
            None
        """
        self.model.set_params(**kwargs)


class EnsembleQualityModel:
    """Ensemble model combining multiple algorithms"""

    def __init__(self, model_types=None):
        """
        Initialize ensemble model

        Parameters:
            model_types (list): List of model types to include
        """
        if model_types is None:
            model_types = ['random_forest', 'gradient_boosting', 'linear_regression']

        self.models = {
            model_type: MLQualityModel(model_type)
            for model_type in model_types
        }
        self.weights = {model_type: 1.0 for model_type in model_types}

    def train(self, X_train, y_train):
        """
        Train all models

        Parameters:
            X_train: Training feature matrix
            y_train: Training target values

        Returns:
            None
        """
        for model_type, model in self.models.items():
            print(f"Training {model_type}...")
            model.train(X_train, y_train)

    def predict(self, X):
        """
        Make ensemble predictions

        Parameters:
            X: Feature matrix

        Returns:
            dict: Predictions from each model
        """
        predictions = {}
        for model_type, model in self.models.items():
            predictions[model_type] = model.predict(X)

        return predictions

    def predict_average(self, X):
        """
        Make weighted average predictions

        Parameters:
            X: Feature matrix

        Returns:
            ndarray: Ensemble average predictions
        """
        predictions = self.predict(X)

        # Calculate weighted average
        weighted_sum = np.zeros(len(X))
        weight_total = sum(self.weights.values())

        for model_type, preds in predictions.items():
            weight = self.weights.get(model_type, 1.0)
            weighted_sum += preds * weight

        return weighted_sum / weight_total

    def evaluate(self, X_test, y_test):
        """
        Evaluate all models

        Parameters:
            X_test: Test feature matrix
            y_test: Test target values

        Returns:
            dict: Evaluation metrics for each model
        """
        results = {}
        for model_type, model in self.models.items():
            results[model_type] = model.evaluate(X_test, y_test)

        # Ensemble evaluation
        ensemble_pred = self.predict_average(X_test)
        results['ensemble'] = {
            'r2': r2_score(y_test, ensemble_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, ensemble_pred)),
            'mae': mean_absolute_error(y_test, ensemble_pred),
            'mape': mean_absolute_percentage_error(y_test, ensemble_pred)
        }

        return results

    def set_weights(self, weights_dict):
        """
        Set ensemble weights

        Parameters:
            weights_dict (dict): Model type -> weight mapping

        Returns:
            None
        """
        self.weights.update(weights_dict)

    def save_models(self, output_dir='ml/models'):
        """
        Save all models

        Parameters:
            output_dir (str): Output directory

        Returns:
            None
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for model_type, model in self.models.items():
            filename = output_path / f"{model_type}_model.pkl"
            model.save_model(str(filename))

    def load_models(self, output_dir='ml/models'):
        """
        Load all models

        Parameters:
            output_dir (str): Input directory

        Returns:
            None
        """
        output_path = Path(output_dir)

        for model_type in self.models.keys():
            filename = output_path / f"{model_type}_model.pkl"
            if filename.exists():
                self.models[model_type].load_model(str(filename))
