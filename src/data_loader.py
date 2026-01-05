"""
Data loading and preprocessing utilities for fairness datasets
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path


class DataLoader:
    """
    Dataset loader for COMPAS, Adult, and German Credit datasets
    """
    
    def __init__(self, dataset_name="compas", data_dir="data/raw"):
        """
        Initialize data loader
        
        Args:
            dataset_name: Name of dataset ('compas', 'adult', 'german')
            data_dir: Directory containing raw data
        """
        self.dataset_name = dataset_name.lower()
        self.data_dir = Path(data_dir)
        self.scaler = StandardScaler()
        
    def load_compas(self):
        """
        Load and preprocess COMPAS recidivism dataset
        
        Returns:
            X: Features (n_samples, n_features)
            y: Labels (n_samples,) - 1: recidivate, 0: not recidivate
            z: Sensitive attribute (n_samples,) - 1: African-American, 0: others
        """
        # Load data
        file_path = self.data_dir / "compas" / "compas-scores-two-years.csv"
        
        if not file_path.exists():
            raise FileNotFoundError(
                f"COMPAS dataset not found at {file_path}\n"
                "Please download from: https://github.com/propublica/compas-analysis"
            )
        
        df = pd.read_csv(file_path)
        
        # Filter data (same as ProPublica analysis)
        df = df[
            (df['days_b_screening_arrest'] <= 30) &
            (df['days_b_screening_arrest'] >= -30) &
            (df['is_recid'] != -1) &
            (df['c_charge_degree'] != 'O') &
            (df['score_text'] != 'N/A')
        ]
        
        # Select features
        feature_cols = [
            'age', 
            'priors_count', 
            'juv_fel_count', 
            'juv_misd_count', 
            'juv_other_count'
        ]
        X = df[feature_cols].values.astype(np.float32)
        
        # Label: two-year recidivism
        y = df['two_year_recid'].values.astype(np.int32)
        
        # Sensitive attribute: race (African-American vs others)
        z = (df['race'] == 'African-American').astype(np.int32).values
        
        print(f"COMPAS dataset loaded:")
        print(f"  Samples: {len(X)}")
        print(f"  Features: {X.shape[1]}")
        print(f"  Positive rate: {y.mean():.2%}")
        print(f"  African-American: {z.mean():.2%}")
        
        return X, y, z
    
    def load_adult(self):
        """
        Load and preprocess Adult Income dataset
        
        Returns:
            X: Features
            y: Labels (1: >50K, 0: <=50K)
            z: Sensitive attribute (1: female, 0: male)
        """
        # Load training data
        file_path = self.data_dir / "adult" / "adult.data"
        
        if not file_path.exists():
            raise FileNotFoundError(
                f"Adult dataset not found at {file_path}\n"
                "Please download from: https://archive.ics.uci.edu/ml/datasets/adult"
            )
        
        # Column names (UCI Adult dataset has no header)
        columns = [
            'age', 'workclass', 'fnlwgt', 'education', 'education-num',
            'marital-status', 'occupation', 'relationship', 'race', 'sex',
            'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income'
        ]
        
        df = pd.read_csv(file_path, names=columns, sep=', ', engine='python', na_values='?')
        
        # Drop rows with missing values
        df = df.dropna()
        
        # Select numerical features only (for simplicity)
        feature_cols = [
            'age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week'
        ]
        X = df[feature_cols].values.astype(np.float32)
        
        # Label: income >50K
        y = (df['income'] == '>50K').astype(np.int32).values
        
        # Sensitive attribute: sex (1=Female, 0=Male)
        z = (df['sex'] == 'Female').astype(np.int32).values
        
        print(f"Adult dataset loaded:")
        print(f"  Samples: {len(X)}")
        print(f"  Features: {X.shape[1]}")
        print(f"  Positive rate (>50K): {y.mean():.2%}")
        print(f"  Female: {z.mean():.2%}")
        
        return X, y, z
    
    def load_german(self):
        """
        Load and preprocess German Credit dataset
        
        Returns:
            X: Features
            y: Labels (1: bad credit, 0: good credit)
            z: Sensitive attribute (1: age>=25, 0: age<25)
        """
        # Load data
        file_path = self.data_dir / "german" / "german.data"
        
        if not file_path.exists():
            raise FileNotFoundError(
                f"German dataset not found at {file_path}\n"
                "Please download from: https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)"
            )
        
        # German dataset has no header, space-separated
        df = pd.read_csv(file_path, sep='\s+', header=None)
        
        # Select numerical features (columns 1, 4, 7, 12, 15, 17 are numerical)
        # For simplicity, use key numerical features
        feature_indices = [1, 4, 7, 12, 15, 17]  # age, duration, credit_amount, etc.
        X = df.iloc[:, feature_indices].values.astype(np.float32)
        
        # Label: column 20 (1=good, 2=bad) -> convert to (0=good, 1=bad)
        y = (df.iloc[:, 20] == 2).astype(np.int32).values
        
        # Sensitive attribute: age (column 1) >= 25
        age = df.iloc[:, 1].values
        z = (age >= 25).astype(np.int32)
        
        print(f"German Credit dataset loaded:")
        print(f"  Samples: {len(X)}")
        print(f"  Features: {X.shape[1]}")
        print(f"  Positive rate (bad credit): {y.mean():.2%}")
        print(f"  Age>=25: {z.mean():.2%}")
        
        return X, y, z
    
    def load_dataset(self):
        """Load dataset based on dataset_name"""
        if self.dataset_name == "compas":
            return self.load_compas()
        elif self.dataset_name == "adult":
            return self.load_adult()
        elif self.dataset_name == "german":
            return self.load_german()
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
    
    def preprocess(self, X_train, X_test):
        """
        Standardize features (zero mean, unit variance)
        
        Args:
            X_train: Training features
            X_test: Test features
            
        Returns:
            X_train_scaled, X_test_scaled
        """
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)
        return X_train, X_test
    
    def add_label_noise(self, y, noise_rate=0.1, noise_type='random', z=None, seed=None):
        """
        Add label noise to training data
        
        Args:
            y: Clean labels
            noise_rate: Fraction of labels to flip (0.0 to 1.0)
            noise_type: Type of noise ('random', 'group_targeted')
            z: Sensitive attributes (required for group_targeted noise)
            seed: Random seed for reproducibility
            
        Returns:
            y_noisy: Labels with injected noise
        """
        if seed is not None:
            np.random.seed(seed)
        
        n = len(y)
        y_noisy = y.copy()
        
        if noise_type == 'random':
            # Symmetric noise: flip labels uniformly at random
            n_flip = int(n * noise_rate)
            flip_indices = np.random.choice(n, size=n_flip, replace=False)
            y_noisy[flip_indices] = 1 - y_noisy[flip_indices]
            
        elif noise_type == 'group_targeted':
            # Asymmetric noise: more noise in minority group (z=1)
            if z is None:
                raise ValueError("z (sensitive attribute) required for group_targeted noise")
            
            # Apply 1.5x more noise to minority group
            minority_indices = np.where(z == 1)[0]
            majority_indices = np.where(z == 0)[0]
            
            n_flip_minority = int(len(minority_indices) * noise_rate * 1.5)
            n_flip_majority = int(len(majority_indices) * noise_rate * 0.5)
            
            flip_minority = np.random.choice(
                minority_indices, 
                size=min(n_flip_minority, len(minority_indices)), 
                replace=False
            )
            flip_majority = np.random.choice(
                majority_indices, 
                size=min(n_flip_majority, len(majority_indices)), 
                replace=False
            )
            
            flip_indices = np.concatenate([flip_minority, flip_majority])
            y_noisy[flip_indices] = 1 - y_noisy[flip_indices]
            
        else:
            raise ValueError(f"Unknown noise type: {noise_type}")
        
        actual_noise_rate = (y != y_noisy).mean()
        print(f"Noise injection complete:")
        print(f"  Target noise rate: {noise_rate:.1%}")
        print(f"  Actual noise rate: {actual_noise_rate:.1%}")
        print(f"  Noise type: {noise_type}")
        
        return y_noisy
    
    def load_and_prepare(self, noise_rate=0.1, noise_type='random', 
                         test_size=0.3, seed=42):
        """
        Complete data loading pipeline
        
        Args:
            noise_rate: Fraction of training labels to corrupt
            noise_type: Type of label noise ('random', 'group_targeted')
            test_size: Fraction of data for testing
            seed: Random seed
            
        Returns:
            data: Dictionary containing:
                - X_train: Training features (scaled)
                - y_train: Clean training labels (for evaluation only)
                - y_train_noisy: Noisy training labels (for training)
                - z_train: Training sensitive attributes
                - X_test: Test features (scaled)
                - y_test: Test labels
                - z_test: Test sensitive attributes
        """
        # Load raw data
        print(f"\nLoading {self.dataset_name.upper()} dataset...")
        X, y, z = self.load_dataset()
        
        # Train-test split (stratified by label to maintain class balance)
        print(f"\nSplitting data (test_size={test_size})...")
        X_train, X_test, y_train, y_test, z_train, z_test = train_test_split(
            X, y, z, 
            test_size=test_size, 
            random_state=seed, 
            stratify=y
        )
        
        print(f"  Train samples: {len(X_train)}")
        print(f"  Test samples: {len(X_test)}")
        
        # Standardize features
        print(f"\nStandardizing features...")
        X_train, X_test = self.preprocess(X_train, X_test)
        
        # Add label noise to training set
        print(f"\nAdding label noise...")
        y_train_noisy = self.add_label_noise(
            y_train, 
            noise_rate=noise_rate, 
            noise_type=noise_type, 
            z=z_train,
            seed=seed
        )
        
        return {
            'X_train': X_train,
            'y_train': y_train,  # Clean labels (for evaluation only!)
            'y_train_noisy': y_train_noisy,  # Noisy labels (for training)
            'z_train': z_train,
            'X_test': X_test,
            'y_test': y_test,
            'z_test': z_test,
            'noise_rate': noise_rate,
            'noise_type': noise_type
        }


# Test the data loader
if __name__ == "__main__":
    print("="*60)
    print("Testing DataLoader")
    print("="*60)
    
    # Initialize loader
    loader = DataLoader(dataset_name="compas", data_dir="data/raw")
    
    # Load and prepare data
    try:
        data = loader.load_and_prepare(
            noise_rate=0.1, 
            noise_type='random',
            test_size=0.3,
            seed=42
        )
        
        print("\n" + "="*60)
        print("Data prepared successfully!")
        print("="*60)
        print(f"\nDataset statistics:")
        print(f"  Train samples: {len(data['y_train'])}")
        print(f"  Test samples: {len(data['y_test'])}")
        print(f"  Features: {data['X_train'].shape[1]}")
        print(f"  Noise rate: {(data['y_train'] != data['y_train_noisy']).mean():.2%}")
        print(f"  Train minority %: {data['z_train'].mean():.2%}")
        print(f"  Test minority %: {data['z_test'].mean():.2%}")
        
    except FileNotFoundError as e:
        print(f"\n⚠️  {e}")
        print("\nTo download COMPAS dataset:")
        print("1. Visit: https://github.com/propublica/compas-analysis")
        print("2. Download 'compas-scores-two-years.csv'")
        print("3. Save to: data/raw/compas/compas-scores-two-years.csv")
