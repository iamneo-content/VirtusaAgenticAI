"""
Large-Scale ML Training Pipeline
Trains three algorithms with proper train/validation/test split methodology
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import time

from .generate_dataset import SyntheticDatasetGenerator


class LargeScaleMLTrainer:
    """ML training pipeline with multiple algorithms"""

    def __init__(self):
        """Initialize trainer"""
        self.df = None
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.scaler = StandardScaler()

        # Models
        self.models = {
            'Random Forest': None,
            'Gradient Boosting': None,
            'Linear Regression': None
        }

        # Results
        self.results = {}

    def load_dataset(self, filepath=None):
        """
        Load or generate dataset

        Parameters:
            filepath (str): Optional path to CSV file

        Returns:
            DataFrame: Loaded or generated dataset
        """
        if filepath and Path(filepath).exists():
            print(f"Loading dataset from {filepath}...")
            self.df = SyntheticDatasetGenerator.load_dataset(filepath)
            print(f"Dataset loaded: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
        else:
            print("Generating new dataset...")
            generator = SyntheticDatasetGenerator(random_state=42)
            self.df = generator.generate(n_samples=30000)

        return self.df

    def prepare_data(self, df):
        """
        Split and scale data for training

        Parameters:
            df (DataFrame): Dataset with features and target

        Returns:
            None (updates internal state)
        """
        print("\nSPLITTING DATA")
        print("=" * 50)

        # Get feature columns
        feature_cols = SyntheticDatasetGenerator.get_feature_columns()
        X = df[feature_cols].copy()
        y = df['quality_score'].copy()

        # First split: 80% train+val, 20% test
        X_temp, self.X_test, y_temp, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Second split: 75% train, 25% validation (of 80%)
        # This gives us 60% train, 20% validation, 20% test
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_temp, y_temp, test_size=0.25, random_state=42
        )

        print(f"Training set:   {len(self.X_train):>6,} samples (60%)")
        print(f"Validation set: {len(self.X_val):>6,} samples (20%)")
        print(f"Test set:       {len(self.X_test):>6,} samples (20%)")
        print(f"Features: {len(feature_cols)}")

        # Fit scaler on training data and transform all
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_val = self.scaler.transform(self.X_val)
        self.X_test = self.scaler.transform(self.X_test)

    def train_models(self):
        """
        Train three ML models

        Parameters:
            None

        Returns:
            None (stores models internally)
        """
        print("\nTRAINING MODELS")
        print("=" * 50)

        # 1. Random Forest
        print("1. Training Random Forest...")
        start = time.time()
        rf = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )
        rf.fit(self.X_train, self.y_train)
        self.models['Random Forest'] = rf
        print(f"   Trained in {time.time() - start:.2f}s")

        # 2. Gradient Boosting
        print("2. Training Gradient Boosting...")
        start = time.time()
        gb = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        gb.fit(self.X_train, self.y_train)
        self.models['Gradient Boosting'] = gb
        print(f"   Trained in {time.time() - start:.2f}s")

        # 3. Linear Regression
        print("3. Training Linear Regression (Baseline)...")
        start = time.time()
        lr = LinearRegression()
        lr.fit(self.X_train, self.y_train)
        self.models['Linear Regression'] = lr
        print(f"   Trained in {time.time() - start:.2f}s")

    def evaluate_models(self):
        """
        Evaluate all models on train/val/test sets

        Parameters:
            None

        Returns:
            None (stores results internally)
        """
        print("\nMODEL EVALUATION")
        print("=" * 50)

        for model_name, model in self.models.items():
            print(f"\n{model_name}")

            # Predictions on all sets
            y_train_pred = model.predict(self.X_train)
            y_val_pred = model.predict(self.X_val)
            y_test_pred = model.predict(self.X_test)

            # Calculate metrics
            metrics = {
                'Train': {
                    'R2': r2_score(self.y_train, y_train_pred),
                    'RMSE': np.sqrt(mean_squared_error(self.y_train, y_train_pred)),
                    'MAE': mean_absolute_error(self.y_train, y_train_pred),
                    'MAPE': mean_absolute_percentage_error(self.y_train, y_train_pred)
                },
                'Validation': {
                    'R2': r2_score(self.y_val, y_val_pred),
                    'RMSE': np.sqrt(mean_squared_error(self.y_val, y_val_pred)),
                    'MAE': mean_absolute_error(self.y_val, y_val_pred),
                    'MAPE': mean_absolute_percentage_error(self.y_val, y_val_pred)
                },
                'Test': {
                    'R2': r2_score(self.y_test, y_test_pred),
                    'RMSE': np.sqrt(mean_squared_error(self.y_test, y_test_pred)),
                    'MAE': mean_absolute_error(self.y_test, y_test_pred),
                    'MAPE': mean_absolute_percentage_error(self.y_test, y_test_pred)
                }
            }

            self.results[model_name] = metrics

            # Print table
            print(f"{'Metric':<15} {'Train':<15} {'Validation':<15} {'Test':<15}")
            print("-" * 60)
            for metric in ['R2', 'RMSE', 'MAE', 'MAPE']:
                train_val = metrics['Train'][metric]
                val_val = metrics['Validation'][metric]
                test_val = metrics['Test'][metric]
                print(f"{metric:<15} {train_val:<15.4f} {val_val:<15.4f} {test_val:<15.4f}")

    def get_feature_importance(self, model_name, top_n=20):
        """
        Get feature importance from tree models

        Parameters:
            model_name (str): Model name ('Random Forest' or 'Gradient Boosting')
            top_n (int): Number of top features (default 20)

        Returns:
            dict: Feature importance scores
        """
        model = self.models.get(model_name)
        if not model or not hasattr(model, 'feature_importances_'):
            return {}

        feature_cols = SyntheticDatasetGenerator.get_feature_columns()
        importances = model.feature_importances_

        # Get top N features
        indices = np.argsort(importances)[-top_n:][::-1]

        result = {}
        for idx in indices:
            result[feature_cols[idx]] = float(importances[idx])

        return result

    def print_feature_importance(self):
        """Print feature importance analysis"""
        print("\n" + "="*50)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("="*50)

        for model_name in ['Random Forest', 'Gradient Boosting']:
            importance = self.get_feature_importance(model_name, top_n=20)
            if importance:
                print(f"\n{model_name} - Top 20 Features:")
                print("-" * 50)
                for i, (feature, score) in enumerate(importance.items(), 1):
                    print(f"{i:2d}. {feature:<30} {score*100:6.2f}%")

    def print_summary(self):
        """Print training summary"""
        print("\n" + "="*50)
        print("TRAINING SUMMARY")
        print("="*50)

        print("\nModel Performance Summary:")
        print("-" * 80)
        print(f"{'Model':<20} {'Test R2':<12} {'Test RMSE':<12} {'Test MAE':<12}")
        print("-" * 80)

        for model_name, metrics in self.results.items():
            r2 = metrics['Test']['R2']
            rmse = metrics['Test']['RMSE']
            mae = metrics['Test']['MAE']
            print(f"{model_name:<20} {r2:<12.4f} {rmse:<12.4f} {mae:<12.4f}")

    def save_models(self, output_dir='ml/models'):
        """
        Save trained models to disk

        Parameters:
            output_dir (str): Output directory for model files
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        for model_name, model in self.models.items():
            if model:
                filename = output_path / f"{model_name.lower().replace(' ', '_')}_model.pkl"
                with open(filename, 'wb') as f:
                    pickle.dump(model, f)
                print(f"Saved {model_name} to {filename}")

        # Also save scaler
        scaler_path = output_path / "scaler.pkl"
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"Saved scaler to {scaler_path}")

    def load_models(self, output_dir='ml/models'):
        """
        Load trained models from disk

        Parameters:
            output_dir (str): Directory containing model files
        """
        output_path = Path(output_dir)

        for model_name in self.models.keys():
            filename = output_path / f"{model_name.lower().replace(' ', '_')}_model.pkl"
            if filename.exists():
                with open(filename, 'rb') as f:
                    self.models[model_name] = pickle.load(f)
                print(f"Loaded {model_name} from {filename}")

        # Load scaler
        scaler_path = output_path / "scaler.pkl"
        if scaler_path.exists():
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            print(f"Loaded scaler from {scaler_path}")


def main():
    """Main execution - full training pipeline"""
    print("="*50)
    print("LARGE-SCALE ML TRAINING PIPELINE")
    print("Dataset: 30,000 HLD Records with 38 Features")
    print("="*50)

    trainer = LargeScaleMLTrainer()

    # Load dataset
    dataset_path = Path(__file__).parent / "synthetic_hld_dataset.csv"
    trainer.load_dataset(str(dataset_path))

    # Prepare data
    trainer.prepare_data(trainer.df)

    # Train models
    trainer.train_models()

    # Evaluate models
    trainer.evaluate_models()

    # Print analysis
    trainer.print_feature_importance()
    trainer.print_summary()

    # Save models
    trainer.save_models()


if __name__ == "__main__":
    main()
