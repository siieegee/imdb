"""
Unsupervised Learning Models for IMDB Dataset.
- K-Means Clustering
- Hierarchical Clustering
- DBSCAN
- Principal Component Analysis (PCA)
- t-SNE (for visualization)
"""

import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, calinski_harabasz_score
)
import json


class UnsupervisedLearningModels:
    """Collection of unsupervised learning models for clustering and dimensionality reduction."""
    
    def __init__(self):
        self.models = {}
        self.results = {}
        self.clusters = {}
    
    def train_kmeans(self, X, n_clusters=3):
        """Train K-Means clustering model."""
        print("\n" + "="*60)
        print("K-MEANS CLUSTERING")
        print("="*60)
        
        model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = model.fit_predict(X)
        
        metrics = self._evaluate_clustering(X, labels, model, "K-Means")
        
        self.models['kmeans'] = model
        self.clusters['kmeans'] = labels
        self.results['kmeans'] = metrics
        return model, labels, metrics
    
    def train_hierarchical(self, X, n_clusters=3, linkage='ward'):
        """Train Hierarchical Clustering model."""
        print("\n" + "="*60)
        print("HIERARCHICAL CLUSTERING")
        print("="*60)
        
        model = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage=linkage
        )
        labels = model.fit_predict(X)
        
        metrics = self._evaluate_clustering(X, labels, None, f"Hierarchical ({linkage})")
        
        self.models['hierarchical'] = model
        self.clusters['hierarchical'] = labels
        self.results['hierarchical'] = metrics
        return model, labels, metrics
    
    def train_dbscan(self, X, eps=0.5, min_samples=5):
        """Train DBSCAN clustering model."""
        print("\n" + "="*60)
        print("DBSCAN CLUSTERING")
        print("="*60)
        
        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(X)
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        print(f"  Clusters found: {n_clusters}")
        print(f"  Noise points: {n_noise}")
        
        if n_clusters > 1:
            silhouette = silhouette_score(X, labels)
            davies_bouldin = davies_bouldin_score(X, labels)
            print(f"  Silhouette Score: {silhouette:.4f}")
            print(f"  Davies-Bouldin Index: {davies_bouldin:.4f}")
            
            metrics = {
                'n_clusters': n_clusters,
                'n_noise_points': n_noise,
                'silhouette_score': float(silhouette),
                'davies_bouldin_index': float(davies_bouldin)
            }
        else:
            print("  ⚠ Warning: Not enough clusters for meaningful metrics")
            metrics = {
                'n_clusters': n_clusters,
                'n_noise_points': n_noise
            }
        
        self.models['dbscan'] = model
        self.clusters['dbscan'] = labels
        self.results['dbscan'] = metrics
        return model, labels, metrics
    

    
    def _evaluate_clustering(self, X, labels, model, model_name):
        """Calculate clustering evaluation metrics."""
        metrics = {}
        
        # Filter out noise points for metrics (if any)
        valid_mask = labels != -1
        X_valid = X[valid_mask]
        labels_valid = labels[valid_mask]
        
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        
        if n_clusters > 1 and len(np.unique(labels_valid)) > 1:
            silhouette = silhouette_score(X_valid, labels_valid)
            davies_bouldin = davies_bouldin_score(X_valid, labels_valid)
            calinski_harabasz = calinski_harabasz_score(X_valid, labels_valid)
            
            metrics = {
                'n_clusters': n_clusters,
                'silhouette_score': float(silhouette),
                'davies_bouldin_index': float(davies_bouldin),
                'calinski_harabasz_index': float(calinski_harabasz)
            }
            
            print(f"\n{model_name} Metrics:")
            print(f"  Clusters: {n_clusters}")
            print(f"  Silhouette Score: {silhouette:.4f} (higher is better)")
            print(f"  Davies-Bouldin Index: {davies_bouldin:.4f} (lower is better)")
            print(f"  Calinski-Harabasz Index: {calinski_harabasz:.4f} (higher is better)")
        else:
            print(f"\n⚠ {model_name}: Insufficient clusters for meaningful metrics")
            metrics = {'n_clusters': n_clusters}
        
        return metrics
    
    def save_results(self, output_path):
        """Save clustering results to JSON."""
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\n✓ Results saved to {output_path}")
    
    def get_cluster_summary(self):
        """Print summary of all clustering models."""
        print("\n" + "="*60)
        print("CLUSTERING SUMMARY")
        print("="*60)
        for model_name, metrics in self.results.items():
            print(f"\n{model_name.upper()}:")
            for key, value in metrics.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")
