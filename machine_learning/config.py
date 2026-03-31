"""
Configuration file for Machine Learning experiments on IMDB dataset.
Edit these parameters to customize model training.
"""

# Data Configuration
DATA_CONFIG = {
    'data_dir': '../dataset/cleaned_datasets',
    'test_size': 0.2,
    'random_seed': 42,
    'clustering_samples': 5000,
}

# Supervised Learning Configuration
SUPERVISED_CONFIG = {
    'models': {
        'linear_regression': {
            'enabled': True,
        },
        'decision_tree': {
            'enabled': True,
            'max_depth': 10,
        },
        'random_forest': {
            'enabled': True,
            'n_estimators': 100,
            'max_depth': 10,
        },
        'neural_network': {
            'enabled': True,
            'epochs': 50,
            'batch_size': 32,
            'architecture': {
                'hidden_layers': [128, 64, 32],
                'activation': 'relu',
                'dropout': 0.2,
            }
        }
    }
}

# Unsupervised Learning Configuration
UNSUPERVISED_CONFIG = {
    'clustering': {
        'kmeans': {
            'enabled': True,
            'k_values': [3, 5, 7],
        },
        'hierarchical': {
            'enabled': True,
            'n_clusters': 5,
            'linkage': 'ward',
        },
        'dbscan': {
            'enabled': True,
            'eps': 0.5,
            'min_samples': 5,
        }
    }
}

# Output Configuration
OUTPUT_CONFIG = {
    'results_dir': '../outputs/ml_models',
    'save_plots': True,
    'plot_format': 'png',
    'plot_dpi': 150,
    'save_models': False,  # Set to True to save trained models
}

# Feature Configuration for Supervised Learning
SUPERVISED_FEATURES = {
    'numeric': ['runtimeMinutes', 'startYear', 'numVotes'],
    'categorical': ['titleType'],
    'target': 'averageRating'
}

# Feature Configuration for Unsupervised Learning
UNSUPERVISED_FEATURES = [
    'runtimeMinutes',
    'startYear',
    'numVotes',
    'averageRating'
]

# Logging Configuration
LOGGING_CONFIG = {
    'verbose': True,
    'log_file': None,  # Set to filepath to save logs
}
