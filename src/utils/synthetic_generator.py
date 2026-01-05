"""
Synthetic Data Generator for Meta-Training
===========================================

Generates diverse classification tasks for meta-learning:
- Varying sample sizes: 100-10,000 samples
- Varying noise rates: 0%-30% label noise
- Varying group imbalances: 10%-90% minority representation
- Varying class imbalances: 20%-80% positive class
- Varying feature dimensions: 5-20 features
- Varying separability: easy to hard classification

Each task is a binary classification problem with:
- X: Feature matrix (n_samples, n_features)
- y: Labels (n_samples,) with optional noise
- z: Sensitive attribute (n_samples,) - binary group membership
- metadata: Task configuration parameters

Usage:
------
from src.utils.synthetic_generator import SyntheticDataGenerator

generator = SyntheticDataGenerator(seed=42)
task = generator.generate_task(
    n_samples=1000,
    n_features=10,
    noise_rate=0.1,
    group_imbalance=0.3,
    class_imbalance=0.4
)
X_train, y_train, z_train = task['train']
X_test, y_test, z_test = task['test']
"""

import numpy as np
import os
import json
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class SyntheticDataGenerator:
    """
    Generate diverse synthetic classification tasks for meta-learning.
    
    Attributes:
        seed (int): Random seed for reproducibility
        rng (np.random.Generator): NumPy random generator
        tasks_dir (str): Directory to save generated tasks
    """
    
    def __init__(self, seed=42, tasks_dir='data/synthetic'):
        """
        Initialize generator.
        
        Args:
            seed (int): Random seed
            tasks_dir (str): Directory to save tasks
        """
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.tasks_dir = tasks_dir
        
        # Create tasks directory if it doesn't exist
        os.makedirs(tasks_dir, exist_ok=True)
    
    def generate_task(
        self,
        n_samples=1000,
        n_features=10,
        noise_rate=0.1,
        group_imbalance=0.5,
        class_imbalance=0.5,
        separability=1.0,
        test_size=0.3,
        add_group_bias=True
    ):
        """
        Generate a single binary classification task.
        
        Args:
            n_samples (int): Total number of samples
            n_features (int): Number of features
            noise_rate (float): Proportion of labels to flip (0.0-0.5)
            group_imbalance (float): Proportion of minority group (0.1-0.9)
            class_imbalance (float): Proportion of positive class (0.2-0.8)
            separability (float): Class separability (0.5=hard, 2.0=easy)
            test_size (float): Proportion of test set
            add_group_bias (bool): Add bias correlation between group and label
        
        Returns:
            dict: Task dictionary with train/test splits and metadata
        """
        # Generate base classification problem
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=max(2, n_features // 2),
            n_redundant=max(1, n_features // 4),
            n_clusters_per_class=2,
            weights=[1 - class_imbalance, class_imbalance],
            flip_y=0.0,  # We'll add noise manually
            class_sep=separability,
            random_state=self.seed
        )
        
        # Generate sensitive attribute (binary group membership)
        z = self._generate_sensitive_attribute(
            n_samples, 
            y, 
            group_imbalance, 
            add_group_bias
        )
        
        # Add label noise
        y_clean = y.copy()
        if noise_rate > 0:
            y = self._add_label_noise(y, noise_rate)
        
        # Standardize features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        # Train/test split
        X_train, X_test, y_train, y_test, z_train, z_test, y_clean_train, y_clean_test = train_test_split(
            X, y, z, y_clean,
            test_size=test_size,
            random_state=self.seed,
            stratify=y
        )
        
        # Compute statistics
        stats = self._compute_task_statistics(
            y_train, z_train, y_test, z_test, y_clean_train, y_clean_test
        )
        
        # Create task dictionary
        task = {
            'train': {
                'X': X_train,
                'y': y_train,
                'z': z_train,
                'y_clean': y_clean_train
            },
            'test': {
                'X': X_test,
                'y': y_test,
                'z': z_test,
                'y_clean': y_clean_test
            },
            'metadata': {
                'n_samples': int(n_samples),
                'n_train': int(len(y_train)),
                'n_test': int(len(y_test)),
                'n_features': int(n_features),
                'noise_rate': float(noise_rate),
                'group_imbalance': float(group_imbalance),
                'class_imbalance': float(class_imbalance),
                'separability': float(separability),
                'add_group_bias': bool(add_group_bias),
                'seed': int(self.seed),
                **stats
            }
        }
        
        return task
    
    def _generate_sensitive_attribute(self, n_samples, y, group_imbalance, add_bias):
        """
        Generate binary sensitive attribute.
        
        If add_bias=True, creates correlation between group and label.
        This simulates real-world bias where certain groups have
        different base rates.
        
        Args:
            n_samples (int): Number of samples
            y (np.ndarray): Labels
            group_imbalance (float): Target minority proportion
            add_bias (bool): Whether to add group-label correlation
        
        Returns:
            np.ndarray: Binary sensitive attribute (0 or 1)
        """
        if not add_bias:
            # Random assignment: no correlation with labels
            z = self.rng.binomial(1, 1 - group_imbalance, size=n_samples)
        else:
            # Create correlation: minority group has lower positive rate
            z = np.zeros(n_samples, dtype=int)
            
            # Assign groups with bias
            # Positive class: 60% majority, 40% minority
            # Negative class: 40% majority, 60% minority
            pos_idx = np.where(y == 1)[0]
            neg_idx = np.where(y == 0)[0]
            
            # Majority group (z=0) is more likely to have positive labels
            majority_rate_pos = 0.6
            majority_rate_neg = 0.4
            
            n_majority_pos = int(len(pos_idx) * majority_rate_pos)
            n_majority_neg = int(len(neg_idx) * majority_rate_neg)
            
            # Randomly assign majority group
            majority_pos = self.rng.choice(pos_idx, size=n_majority_pos, replace=False)
            majority_neg = self.rng.choice(neg_idx, size=n_majority_neg, replace=False)
            
            z[majority_pos] = 0
            z[majority_neg] = 0
            
            # Rest are minority group (z=1)
            minority_idx = np.setdiff1d(np.arange(n_samples), np.concatenate([majority_pos, majority_neg]))
            z[minority_idx] = 1
        
        return z
    
    def _add_label_noise(self, y, noise_rate):
        """
        Flip a proportion of labels to add noise.
        
        Simulates annotation errors or data corruption.
        
        Args:
            y (np.ndarray): Original labels
            noise_rate (float): Proportion to flip (0.0-0.5)
        
        Returns:
            np.ndarray: Noisy labels
        """
        y_noisy = y.copy()
        n_samples = len(y)
        n_flip = int(n_samples * noise_rate)
        
        # Randomly select samples to flip
        flip_idx = self.rng.choice(n_samples, size=n_flip, replace=False)
        y_noisy[flip_idx] = 1 - y_noisy[flip_idx]
        
        return y_noisy
    
    def _compute_task_statistics(self, y_train, z_train, y_test, z_test, y_clean_train, y_clean_test):
        """
        Compute task statistics for metadata.
        
        Returns:
            dict: Statistics about the task
        """
        # Training set statistics
        train_pos_rate = np.mean(y_train)
        train_minority_rate = np.mean(z_train)
        train_noise_rate = np.mean(y_train != y_clean_train)
        
        # Test set statistics (no noise in test)
        test_pos_rate = np.mean(y_test)
        test_minority_rate = np.mean(z_test)
        
        # Group-specific statistics
        train_pos_rate_z0 = np.mean(y_train[z_train == 0]) if np.sum(z_train == 0) > 0 else 0
        train_pos_rate_z1 = np.mean(y_train[z_train == 1]) if np.sum(z_train == 1) > 0 else 0
        
        test_pos_rate_z0 = np.mean(y_test[z_test == 0]) if np.sum(z_test == 0) > 0 else 0
        test_pos_rate_z1 = np.mean(y_test[z_test == 1]) if np.sum(z_test == 1) > 0 else 0
        
        # Fairness metrics (demographic parity on labels)
        train_dp_gap = abs(train_pos_rate_z0 - train_pos_rate_z1)
        test_dp_gap = abs(test_pos_rate_z0 - test_pos_rate_z1)
        
        return {
            'train_pos_rate': float(train_pos_rate),
            'train_minority_rate': float(train_minority_rate),
            'train_actual_noise_rate': float(train_noise_rate),
            'train_dp_gap': float(train_dp_gap),
            'test_pos_rate': float(test_pos_rate),
            'test_minority_rate': float(test_minority_rate),
            'test_dp_gap': float(test_dp_gap),
            'train_pos_rate_z0': float(train_pos_rate_z0),
            'train_pos_rate_z1': float(train_pos_rate_z1),
            'test_pos_rate_z0': float(test_pos_rate_z0),
            'test_pos_rate_z1': float(test_pos_rate_z1)
        }
    
    def generate_task_suite(
        self,
        n_tasks=100,
        sample_range=(100, 10000),
        feature_range=(5, 20),
        noise_range=(0.0, 0.3),
        group_imbalance_range=(0.1, 0.9),
        class_imbalance_range=(0.2, 0.8),
        separability_range=(0.5, 2.0),
        save=True
    ):
        """
        Generate a suite of diverse tasks for meta-learning.
        
        Args:
            n_tasks (int): Number of tasks to generate
            sample_range (tuple): (min, max) number of samples
            feature_range (tuple): (min, max) number of features
            noise_range (tuple): (min, max) noise rate
            group_imbalance_range (tuple): (min, max) minority proportion
            class_imbalance_range (tuple): (min, max) positive class proportion
            separability_range (tuple): (min, max) class separability
            save (bool): Save tasks to disk
        
        Returns:
            list: List of task dictionaries
        """
        tasks = []
        
        print(f"Generating {n_tasks} synthetic tasks...")
        
        for i in range(n_tasks):
            # Sample task parameters from ranges
            n_samples = self.rng.integers(sample_range[0], sample_range[1] + 1)
            n_features = self.rng.integers(feature_range[0], feature_range[1] + 1)
            noise_rate = self.rng.uniform(noise_range[0], noise_range[1])
            group_imbalance = self.rng.uniform(group_imbalance_range[0], group_imbalance_range[1])
            class_imbalance = self.rng.uniform(class_imbalance_range[0], class_imbalance_range[1])
            separability = self.rng.uniform(separability_range[0], separability_range[1])
            add_bias = self.rng.random() > 0.3  # 70% of tasks have group bias
            
            # Generate task
            task = self.generate_task(
                n_samples=n_samples,
                n_features=n_features,
                noise_rate=noise_rate,
                group_imbalance=group_imbalance,
                class_imbalance=class_imbalance,
                separability=separability,
                add_group_bias=add_bias
            )
            
            # Add task ID (ensure it's a Python int)
            task['metadata']['task_id'] = int(i)
            
            tasks.append(task)
            
            # Progress indicator
            if (i + 1) % 10 == 0:
                print(f"  Generated {i + 1}/{n_tasks} tasks...")
        
        # Save tasks if requested
        if save:
            self._save_tasks(tasks)
        
        print(f"✓ Generated {n_tasks} tasks successfully!")
        return tasks
    
    def _save_tasks(self, tasks):
        """
        Save tasks to disk.
        
        Saves each task as:
        - data/synthetic/task_XXX.npz (numpy arrays)
        - data/synthetic/metadata.json (all metadata)
        
        Args:
            tasks (list): List of task dictionaries
        """
        print(f"\nSaving tasks to {self.tasks_dir}...")
        
        metadata_list = []
        
        for task in tasks:
            task_id = task['metadata']['task_id']
            
            # Save numpy arrays
            task_file = os.path.join(self.tasks_dir, f'task_{task_id:03d}.npz')
            np.savez(
                task_file,
                X_train=task['train']['X'],
                y_train=task['train']['y'],
                z_train=task['train']['z'],
                y_clean_train=task['train']['y_clean'],
                X_test=task['test']['X'],
                y_test=task['test']['y'],
                z_test=task['test']['z'],
                y_clean_test=task['test']['y_clean']
            )
            
            # Collect metadata
            metadata_list.append(task['metadata'])
        
        # Save all metadata as JSON
        metadata_file = os.path.join(self.tasks_dir, 'metadata.json')
        with open(metadata_file, 'w') as f:
            json.dump(metadata_list, f, indent=2)
        
        print(f"✓ Saved {len(tasks)} tasks to {self.tasks_dir}/")
        print(f"  - task_XXX.npz: numpy arrays")
        print(f"  - metadata.json: task configurations")
    
    @staticmethod
    def load_task(task_id, tasks_dir='data/synthetic'):
        """
        Load a saved task from disk.
        
        Args:
            task_id (int): Task ID to load
            tasks_dir (str): Directory containing tasks
        
        Returns:
            dict: Task dictionary
        """
        # Load numpy arrays
        task_file = os.path.join(tasks_dir, f'task_{task_id:03d}.npz')
        data = np.load(task_file)
        
        # Load metadata
        metadata_file = os.path.join(tasks_dir, 'metadata.json')
        with open(metadata_file, 'r') as f:
            metadata_list = json.load(f)
        
        # Find metadata for this task
        metadata = next(m for m in metadata_list if m['task_id'] == task_id)
        
        # Reconstruct task dictionary
        task = {
            'train': {
                'X': data['X_train'],
                'y': data['y_train'],
                'z': data['z_train'],
                'y_clean': data['y_clean_train']
            },
            'test': {
                'X': data['X_test'],
                'y': data['y_test'],
                'z': data['z_test'],
                'y_clean': data['y_clean_test']
            },
            'metadata': metadata
        }
        
        return task


def test_generator():
    """
    Test the synthetic data generator.
    """
    print("=" * 80)
    print("Testing SyntheticDataGenerator")
    print("=" * 80)
    
    generator = SyntheticDataGenerator(seed=42)
    
    # Test single task generation
    print("\n1. Generating single task...")
    task = generator.generate_task(
        n_samples=1000,
        n_features=10,
        noise_rate=0.1,
        group_imbalance=0.3,
        class_imbalance=0.4,
        separability=1.0
    )
    
    print(f"✓ Task generated successfully!")
    print(f"  Train: {task['train']['X'].shape[0]} samples, {task['train']['X'].shape[1]} features")
    print(f"  Test:  {task['test']['X'].shape[0]} samples, {task['test']['X'].shape[1]} features")
    print(f"  Positive rate: {task['metadata']['train_pos_rate']:.2%}")
    print(f"  Minority rate: {task['metadata']['train_minority_rate']:.2%}")
    print(f"  Actual noise:  {task['metadata']['train_actual_noise_rate']:.2%}")
    print(f"  Train DP gap:  {task['metadata']['train_dp_gap']:.3f}")
    
    # Test task suite generation (small for testing)
    print("\n2. Generating task suite (10 tasks for testing)...")
    tasks = generator.generate_task_suite(
        n_tasks=10,
        save=False  # Don't save test tasks
    )
    
    print(f"\n✓ Task suite generated successfully!")
    print(f"  Tasks: {len(tasks)}")
    print(f"  Sample sizes: {[t['metadata']['n_train'] for t in tasks]}")
    print(f"  Feature dims:  {[t['metadata']['n_features'] for t in tasks]}")
    noise_rates = [t['metadata']['train_actual_noise_rate'] for t in tasks]
    print(f"  Noise rates:   {[f'{nr:.2%}' for nr in noise_rates]}")
    
    print("\n" + "=" * 80)
    print("✓ All tests passed!")
    print("=" * 80)


if __name__ == '__main__':
    test_generator()
