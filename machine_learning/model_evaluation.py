"""
Model Evaluation and Visualization utilities.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import os


class ModelEvaluator:
    """Utilities for evaluating and visualizing model performance."""
    
    def __init__(self, output_dir="../outputs/ml_models"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        sns.set_style("whitegrid")
    
    def plot_regression_results(self, y_true, y_pred, model_name):
        """Plot regression model results: actual vs predicted."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Actual vs Predicted
        axes[0].scatter(y_true, y_pred, alpha=0.5, s=20)
        axes[0].plot([y_true.min(), y_true.max()], 
                     [y_true.min(), y_true.max()], 
                     'r--', lw=2)
        axes[0].set_xlabel('Actual Values')
        axes[0].set_ylabel('Predicted Values')
        axes[0].set_title(f'{model_name}: Actual vs Predicted')
        axes[0].grid(True, alpha=0.3)
        
        # Residuals
        residuals = y_true - y_pred
        axes[1].scatter(y_pred, residuals, alpha=0.5, s=20)
        axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[1].set_xlabel('Predicted Values')
        axes[1].set_ylabel('Residuals')
        axes[1].set_title(f'{model_name}: Residual Plot')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'{model_name.lower()}_regression.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved regression plot: {model_name.lower()}_regression.png")
    
    def plot_clustering_results(self, X, labels, model_name, n_components=2):
        """Plot clustering results using PCA for visualization."""
        from sklearn.decomposition import PCA
        
        if X.shape[1] > 2:
            pca = PCA(n_components=2)
            X_vis = pca.fit_transform(X)
        else:
            X_vis = X
        
        plt.figure(figsize=(10, 7))
        
        # Plot clusters
        unique_labels = np.unique(labels)
        colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
        
        for label, color in zip(unique_labels, colors):
            if label == -1:  # Noise points in DBSCAN
                color = [0, 0, 0]  # Black
                marker = 'x'
            else:
                marker = 'o'
            
            mask = labels == label
            plt.scatter(X_vis[mask, 0], X_vis[mask, 1], 
                       c=[color], label=f'Cluster {label}',
                       marker=marker, s=50, alpha=0.7)
        
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
        plt.title(f'{model_name}: Clustering Visualization')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(os.path.join(self.output_dir, f'{model_name.lower()}_clusters.png'),
                   dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved clustering plot: {model_name.lower()}_clusters.png")
    
    def plot_feature_importance(self, feature_names, importances, model_name):
        """Plot feature importance from tree-based models."""
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(importances)), importances[indices])
        plt.xticks(range(len(importances)), 
                  [feature_names[i] for i in indices], 
                  rotation=45, ha='right')
        plt.ylabel('Importance')
        plt.title(f'{model_name}: Feature Importance')
        plt.tight_layout()
        
        plt.savefig(os.path.join(self.output_dir, f'{model_name.lower()}_importance.png'),
                   dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved importance plot: {model_name.lower()}_importance.png")
    

        """Compare metrics across multiple models."""
        models = list(results.keys())
        metrics = [results[m].get(metric, 0) for m in models]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(models, metrics, color='steelblue')
        
        # Add value labels on bars
        for bar, metric_val in zip(bars, metrics):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{metric_val:.4f}', ha='center', va='bottom')
        
        plt.ylabel(metric.replace('_', ' ').title())
        plt.title(f'Model Comparison: {metric.replace("_", " ").title()}')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        plt.savefig(os.path.join(self.output_dir, 'model_comparison.png'),
                   dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved model comparison plot: model_comparison.png")
    
    
    def plot_model_comparison(self, results, metric='r2_score'):
        """Compare metrics across multiple models."""
        models = list(results.keys())
        metrics = [results[m].get(metric, 0) for m in models]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(models, metrics, color='steelblue')
        
        # Add value labels on bars
        for bar, metric_val in zip(bars, metrics):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{metric_val:.4f}', ha='center', va='bottom')
        
        plt.ylabel(metric.replace('_', ' ').title())
        plt.title(f'Model Comparison: {metric.replace("_", " ").title()}')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        plt.savefig(os.path.join(self.output_dir, 'model_comparison.png'),
                   dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved model comparison plot: model_comparison.png")
    
    @staticmethod
    def print_model_summary(results):
        """Print summary of all model results."""
        print("\n" + "="*60)
        print("MODEL PERFORMANCE SUMMARY")
        print("="*60)
        for model_name, metrics in results.items():
            print(f"\n{model_name.upper()}:")
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)):
                    print(f"  {metric_name}: {value:.4f}")
                else:
                    print(f"  {metric_name}: {value}")
