# Machine Learning on IMDB Dataset

This folder contains machine learning implementations using **TensorFlow/Keras** and **scikit-learn** on the IMDB movie dataset.

## Overview

The project demonstrates both **supervised** and **unsupervised** learning algorithms:

### Supervised Learning (Rating Prediction)
- **Linear Regression** - Baseline linear model
- **Decision Tree Regressor** - Tree-based prediction
- **Random Forest Regressor** - Ensemble of decision trees
- **Neural Network** - Deep learning with TensorFlow/Keras

### Unsupervised Learning (Clustering)
- **K-Means** - Partitioning clustering with multiple k values
- **Hierarchical Clustering** - Agglomerative clustering
- **DBSCAN** - Density-based clustering

## Project Structure

```
machine_learning/
├── requirements.txt              # Python dependencies
├── data_utils.py                # Data loading and preprocessing
├── supervised_learning.py        # Supervised learning models
├── unsupervised_learning.py     # Unsupervised learning models
├── model_evaluation.py          # Evaluation metrics and visualization
├── main.py                      # Main orchestration script
└── README.md                    # This file
```

## Installation

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

Or individually:
```bash
pip install tensorflow scikit-learn pandas numpy matplotlib seaborn
```

2. **Ensure the dataset is available** at `../dataset/cleaned_datasets/`

## Quick Start

Run all models:
```bash
python main.py
```

This will:
1. Load cleaned IMDB datasets
2. Train all supervised learning models for rating prediction
3. Train all unsupervised learning models for clustering
4. Generate evaluation metrics and visualizations
5. Save results to `../outputs/ml_models/`

## Detailed Usage

### Run Supervised Learning Only

```python
from data_utils import load_imdb_data, prepare_rating_prediction_data
from supervised_learning import SupervisedLearningModels

# Load data
datasets = load_imdb_data()
X_train, X_test, y_train, y_test, features = prepare_rating_prediction_data(datasets)

# Train models
sl_models = SupervisedLearningModels()
sl_models.train_random_forest(X_train, X_test, y_train, y_test)
sl_models.train_neural_network(X_train, X_test, y_train, y_test, epochs=50)

# Save results
sl_models.save_results('../outputs/ml_models/supervised_results.json')

# Get best model
best_name, best_model = sl_models.get_best_model()
```

### Run Unsupervised Learning Only

```python
from data_utils import load_imdb_data, prepare_clustering_data
from unsupervised_learning import UnsupervisedLearningModels

# Load data
datasets = load_imdb_data()
X, features = prepare_clustering_data(datasets)

# Train clustering models
ul_models = UnsupervisedLearningModels()
kmeans_model, kmeans_labels, kmeans_metrics = ul_models.train_kmeans(X, n_clusters=5)
hier_model, hier_labels, hier_metrics = ul_models.train_hierarchical(X, n_clusters=5)

# Dimensionality reduction
pca_model, X_pca, pca_metrics = ul_models.train_pca(X)
```

### Custom Model Training

```python
from supervised_learning import SupervisedLearningModels

# Train custom models
sl = SupervisedLearningModels()

# Gradient Boosting with custom parameters
sl.train_gradient_boosting(
    X_train, X_test, y_train, y_test,
    n_estimators=200,
    learning_rate=0.05
)

# Neural Network with custom architecture
# (Edit the model definition in the class)
sl.train_neural_network(
    X_train, X_test, y_train, y_test,
    epochs=100,
    batch_size=16
)
```

## Output Files

After running the models, the following files are generated in `../outputs/ml_models/`:

### Results
- `supervised_learning_results.json` - Metrics for all supervised models (MSE, RMSE, MAE, R²)
- `unsupervised_learning_results.json` - Metrics for clustering models (Silhouette, Davies-Bouldin, etc.)

### Visualizations
- `*_regression.png` - Actual vs Predicted and Residual plots for each supervised model
- `*_clusters.png` - 2D PCA visualization of clusters for each clustering model
- `model_comparison.png` - Comparison of all models by R² score

## Evaluation Metrics

### Supervised Learning Metrics
- **MSE** (Mean Squared Error) - Average squared differences between predicted and actual values
- **RMSE** (Root Mean Squared Error) - Square root of MSE
- **MAE** (Mean Absolute Error) - Average absolute differences
- **R²** (Coefficient of Determination) - Proportion of variance explained (0-1, higher is better)

### Unsupervised Learning Metrics
- **Silhouette Score** - Measures how similar points are to their own cluster vs other clusters (-1 to 1)
- **Davies-Bouldin Index** - Ratio of within-cluster to between-cluster distances (lower is better)
- **Calinski-Harabasz Index** - Ratio of between-cluster to within-cluster dispersion (higher is better)

## Dataset Features Used

### For Rating Prediction (Supervised)
- `runtimeMinutes` - Movie duration in minutes
- `startYear` - Year movie was released
- `numVotes` - Number of votes received
- `titleType` - Type of title (short, feature, etc.)
- **Target**: `averageRating` - Average IMDB rating

### For Clustering (Unsupervised)
- `runtimeMinutes` - Movie duration
- `startYear` - Release year
- `numVotes` - Number of votes
- `averageRating` - Average rating score

## Customization

### Modify Data Preparation

Edit `data_utils.py` to change:
- Feature selection
- Test/train split ratio
- Scaling methods
- Sample size for clustering

### Add New Models

1. Edit `supervised_learning.py` or `unsupervised_learning.py`
2. Add a new training method (e.g., `train_svm`, `train_gmm`)
3. Add it to the model list in `main.py`

### Adjust Hyperparameters

Edit the parameter values in the training functions:
```python
# Example: Change neural network architecture
model = keras.Sequential([
    keras.layers.Dense(256, activation='relu', input_dim=X_train.shape[1]),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(128, activation='relu'),
    # ... add more layers
])
```

## Troubleshooting

### Memory Issues
Reduce `n_samples` in `prepare_clustering_data()`:
```python
X, features = prepare_clustering_data(datasets, n_samples=2000)
```

### Slow Neural Network Training
Reduce `epochs` or increase `batch_size`:
```python
sl_models.train_neural_network(X_train, X_test, y_train, y_test, epochs=20, batch_size=64)
```

### Missing Data
- Ensure `../dataset/cleaned_datasets/` contains the cleaned CSV files
- Check data quality with `../analysis.py`

## Requirements

- Python 3.8+
- TensorFlow >= 2.14.0
- scikit-learn >= 1.3.2
- pandas >= 2.1.1
- numpy >= 1.24.3
- matplotlib >= 3.8.0
- seaborn >= 0.13.0

## References

- [TensorFlow Documentation](https://www.tensorflow.org)
- [scikit-learn Documentation](https://scikit-learn.org)
- [Supervised Learning Guide](https://scikit-learn.org/stable/supervised_learning.html)
- [Unsupervised Learning Guide](https://scikit-learn.org/stable/unsupervised_learning.html)

## Notes

- PCA analysis identifies the most important dimensions in your data
- K-Means clustering with different k values helps find optimal number of clusters
- DBSCAN is better for non-spherical clusters and automatic noise detection
- Neural Networks may require GPU for faster training (enable GPU in TensorFlow)
