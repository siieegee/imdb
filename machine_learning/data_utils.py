"""
Data loading and preprocessing utilities for IMDB dataset.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from pathlib import Path


def load_imdb_data(data_dir="../dataset/cleaned_datasets"):
    """
    Load cleaned IMDB datasets from CSV files.
    
    Args:
        data_dir: Path to the cleaned datasets directory
        
    Returns:
        Dictionary containing loaded dataframes
    """
    data_path = Path(data_dir)
    
    datasets = {}
    csv_files = ['title.basics.csv', 'title.ratings.csv', 'name.basics.csv']
    
    for file in csv_files:
        file_path = data_path / file
        if file_path.exists():
            key = file.replace('.csv', '').replace('.', '_')
            datasets[key] = pd.read_csv(file_path)
            print(f"✓ Loaded {file}: {datasets[key].shape}")
        else:
            print(f"⚠ Warning: {file} not found at {file_path}")
    
    return datasets


def prepare_rating_prediction_data(datasets, test_size=0.2):
    """
    Prepare data for rating prediction task (supervised learning).
    
    Args:
        datasets: Dictionary of loaded dataframes
        test_size: Proportion of data for testing
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test, feature_names)
    """
    from sklearn.model_selection import train_test_split
    
    if 'title_basics' not in datasets or 'title_ratings' not in datasets:
        raise ValueError("Required datasets not found")
    
    # Merge ratings with title basics
    titles = datasets['title_basics'].copy()
    ratings = datasets['title_ratings'].copy()
    
    merged = titles.merge(ratings, on='tconst', how='inner')
    
    # Select features for prediction
    feature_cols = []
    
    # Numeric features
    if 'runtimeMinutes' in merged.columns:
        merged['runtimeMinutes'] = pd.to_numeric(merged['runtimeMinutes'], errors='coerce')
        feature_cols.append('runtimeMinutes')
    
    if 'startYear' in merged.columns:
        merged['startYear'] = pd.to_numeric(merged['startYear'], errors='coerce')
        feature_cols.append('startYear')
    
    if 'numVotes' in merged.columns:
        merged['numVotes'] = pd.to_numeric(merged['numVotes'], errors='coerce')
        feature_cols.append('numVotes')
    
    # Categorical features - title type
    if 'titleType' in merged.columns:
        feature_cols.append('titleType')
    
    # Drop rows with missing values
    merged = merged[feature_cols + ['averageRating']].dropna()
    
    if len(merged) == 0:
        raise ValueError("No valid data after preprocessing")
    
    # Encode categorical variables
    le_dict = {}
    for col in feature_cols:
        if merged[col].dtype == 'object':
            le = LabelEncoder()
            merged[col] = le.fit_transform(merged[col])
            le_dict[col] = le
    
    X = merged[feature_cols].values
    y = merged['averageRating'].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    print(f"\n✓ Prepared rating prediction data")
    print(f"  Training samples: {len(X_train)}, Testing samples: {len(X_test)}")
    print(f"  Features: {len(feature_cols)}")
    
    return X_train, X_test, y_train, y_test, feature_cols


def prepare_clustering_data(datasets, n_samples=5000):
    """
    Prepare data for clustering (unsupervised learning).
    
    Args:
        datasets: Dictionary of loaded dataframes
        n_samples: Number of samples to use for clustering
        
    Returns:
        Tuple of (X, feature_names)
    """
    if 'title_basics' not in datasets or 'title_ratings' not in datasets:
        raise ValueError("Required datasets not found")
    
    # Merge datasets
    titles = datasets['title_basics'].copy()
    ratings = datasets['title_ratings'].copy()
    
    merged = titles.merge(ratings, on='tconst', how='inner')
    
    # Select features for clustering
    feature_cols = []
    
    if 'runtimeMinutes' in merged.columns:
        merged['runtimeMinutes'] = pd.to_numeric(merged['runtimeMinutes'], errors='coerce')
        feature_cols.append('runtimeMinutes')
    
    if 'startYear' in merged.columns:
        merged['startYear'] = pd.to_numeric(merged['startYear'], errors='coerce')
        feature_cols.append('startYear')
    
    if 'numVotes' in merged.columns:
        merged['numVotes'] = pd.to_numeric(merged['numVotes'], errors='coerce')
        feature_cols.append('numVotes')
    
    if 'averageRating' in merged.columns:
        feature_cols.append('averageRating')
    
    # Drop missing values and sample
    merged = merged[feature_cols].dropna()
    
    if len(merged) > n_samples:
        merged = merged.sample(n=n_samples, random_state=42)
    
    X = merged[feature_cols].values
    
    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    print(f"\n✓ Prepared clustering data")
    print(f"  Samples: {len(X)}, Features: {len(feature_cols)}")
    
    return X, feature_cols
