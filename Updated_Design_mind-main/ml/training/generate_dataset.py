"""
Synthetic HLD Dataset Generator
Generates 30,000 synthetic HLD samples with 38 features for ML training
"""

import numpy as np
import pandas as pd
from pathlib import Path
import random
from tqdm import tqdm


class SyntheticDatasetGenerator:
    """Generate synthetic HLD dataset with 38 features and quality scores"""

    # Feature names (38 total)
    FEATURE_NAMES = [
        # Text metrics (4 features)
        'word_count', 'sentence_count', 'avg_sentence_length', 'avg_word_length',

        # Structure (5 features)
        'header_count', 'code_block_count', 'table_count', 'list_count', 'diagram_count',

        # Semantic indicators (7 features)
        'completeness_score', 'security_mentions', 'scalability_mentions', 'api_mentions',
        'database_mentions', 'performance_mentions', 'monitoring_mentions',

        # Consistency metrics (3 features)
        'duplicate_headers', 'header_coverage', 'code_coverage',

        # Density metrics (2 features)
        'keyword_density', 'section_density',

        # Document properties (7 features)
        'has_architecture_section', 'has_security_section', 'has_scalability_section',
        'has_deployment_section', 'has_monitoring_section', 'has_api_spec', 'has_data_model',

        # Complexity metrics (3 features)
        'service_count', 'entity_count', 'api_endpoint_count',

        # Quality indicators (4 features)
        'readability_score', 'completeness_index', 'consistency_index', 'documentation_quality',

        # Text features (2 features)
        'technical_terms_density', 'acronym_count'
    ]

    def __init__(self, random_state=42):
        """Initialize the dataset generator"""
        self.random_state = random_state
        np.random.seed(random_state)
        random.seed(random_state)

    def generate(self, n_samples=30000):
        """
        Generate synthetic HLD dataset

        Parameters:
            n_samples (int): Number of samples to generate (default 30000)

        Returns:
            DataFrame: Generated dataset with 38 features + quality_score
        """
        print(f"Generating {n_samples} synthetic HLD records...")

        data = {
            # Text metrics (4)
            'word_count': np.random.randint(500, 5000, n_samples),
            'sentence_count': np.random.randint(50, 500, n_samples),
            'avg_sentence_length': np.random.uniform(10, 25, n_samples),
            'avg_word_length': np.random.uniform(4, 6, n_samples),

            # Structure (5)
            'header_count': np.random.randint(5, 40, n_samples),
            'code_block_count': np.random.randint(0, 20, n_samples),
            'table_count': np.random.randint(0, 15, n_samples),
            'list_count': np.random.randint(5, 30, n_samples),
            'diagram_count': np.random.randint(0, 10, n_samples),

            # Semantic indicators (7)
            'completeness_score': np.random.uniform(0, 100, n_samples),
            'security_mentions': np.random.randint(0, 20, n_samples),
            'scalability_mentions': np.random.randint(0, 20, n_samples),
            'api_mentions': np.random.randint(0, 25, n_samples),
            'database_mentions': np.random.randint(0, 15, n_samples),
            'performance_mentions': np.random.randint(0, 20, n_samples),
            'monitoring_mentions': np.random.randint(0, 15, n_samples),

            # Consistency metrics (3)
            'duplicate_headers': np.random.randint(0, 10, n_samples),
            'header_coverage': np.random.uniform(0.3, 1.0, n_samples),
            'code_coverage': np.random.uniform(0, 1.0, n_samples),

            # Density metrics (2)
            'keyword_density': np.random.uniform(0.01, 0.1, n_samples),
            'section_density': np.random.uniform(0.1, 0.8, n_samples),

            # Document properties (7) - binary
            'has_architecture_section': np.random.randint(0, 2, n_samples),
            'has_security_section': np.random.randint(0, 2, n_samples),
            'has_scalability_section': np.random.randint(0, 2, n_samples),
            'has_deployment_section': np.random.randint(0, 2, n_samples),
            'has_monitoring_section': np.random.randint(0, 2, n_samples),
            'has_api_spec': np.random.randint(0, 2, n_samples),
            'has_data_model': np.random.randint(0, 2, n_samples),

            # Complexity metrics (3)
            'service_count': np.random.randint(1, 15, n_samples),
            'entity_count': np.random.randint(5, 40, n_samples),
            'api_endpoint_count': np.random.randint(0, 50, n_samples),

            # Quality indicators (4)
            'readability_score': np.random.uniform(20, 100, n_samples),
            'completeness_index': np.random.uniform(0, 1, n_samples),
            'consistency_index': np.random.uniform(0, 1, n_samples),
            'documentation_quality': np.random.uniform(0, 100, n_samples),

            # Text features (2)
            'technical_terms_density': np.random.uniform(0, 0.3, n_samples),
            'acronym_count': np.random.randint(0, 30, n_samples),
        }

        df = pd.DataFrame(data)

        # Calculate quality score
        df['quality_score'] = self._calculate_quality_score(df)

        print(f"Dataset shape: {df.shape}")
        print(f"Features: {len(self.FEATURE_NAMES)}")
        print(f"Samples: {n_samples}")

        return df

    def _calculate_quality_score(self, df):
        """
        Calculate quality score from features

        Parameters:
            df (DataFrame): DataFrame with feature columns

        Returns:
            ndarray: Quality scores (0-100 range)
        """
        # Normalized feature weights
        score = (
            # Text quality indicators (20%)
            (df['avg_sentence_length'] / 25) * 0.05 +
            (df['readability_score'] / 100) * 0.1 +
            (df['avg_word_length'] / 6) * 0.05 +

            # Structure quality (20%)
            (df['header_count'] / 40) * 0.08 +
            (df['table_count'] / 15) * 0.06 +
            (df['diagram_count'] / 10) * 0.06 +

            # Semantic quality (25%)
            (df['completeness_score'] / 100) * 0.10 +
            (df['security_mentions'] / 20) * 0.05 +
            (df['scalability_mentions'] / 20) * 0.05 +
            (df['api_mentions'] / 25) * 0.05 +

            # Consistency (15%)
            (1 - df['duplicate_headers'] / 10) * 0.08 +
            df['header_coverage'] * 0.04 +
            df['code_coverage'] * 0.03 +

            # Document properties (15%)
            df['has_architecture_section'] * 0.04 +
            df['has_security_section'] * 0.03 +
            df['has_scalability_section'] * 0.03 +
            df['has_deployment_section'] * 0.025 +
            df['has_monitoring_section'] * 0.025 +
            df['has_api_spec'] * 0.03 +
            df['has_data_model'] * 0.025 +

            # Other quality metrics (5%)
            (df['documentation_quality'] / 100) * 0.03 +
            (df['completeness_index']) * 0.02
        )

        # Scale to 0-100 range and add noise
        score = score * 100
        noise = np.random.normal(0, 2, len(df))  # Gaussian noise
        score = np.clip(score + noise, 29, 96)  # Clip to realistic range

        return score

    def save_dataset(self, df, filepath):
        """
        Save dataset to CSV file

        Parameters:
            df (DataFrame): Dataset to save
            filepath (str): Output file path
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(filepath, index=False)
        print(f"Dataset saved to: {filepath}")

    @staticmethod
    def load_dataset(filepath):
        """
        Load dataset from CSV file

        Parameters:
            filepath (str): Path to CSV file

        Returns:
            DataFrame: Loaded dataset
        """
        return pd.read_csv(filepath)

    @staticmethod
    def get_feature_columns():
        """
        Get list of feature column names

        Returns:
            list: Feature names (excluding target)
        """
        return SyntheticDatasetGenerator.FEATURE_NAMES


def main():
    """Main execution - generate and save dataset"""
    generator = SyntheticDatasetGenerator(random_state=42)
    df = generator.generate(n_samples=30000)

    # Print summary statistics
    print("\n" + "="*50)
    print("DATASET SUMMARY")
    print("="*50)
    print(f"Mean quality score: {df['quality_score'].mean():.2f}")
    print(f"Std dev: {df['quality_score'].std():.2f}")
    print(f"Min: {df['quality_score'].min():.2f}")
    print(f"Max: {df['quality_score'].max():.2f}")
    print(f"Median: {df['quality_score'].median():.2f}")

    # Save dataset
    output_path = Path(__file__).parent / "synthetic_hld_dataset.csv"
    generator.save_dataset(df, str(output_path))


if __name__ == "__main__":
    main()
