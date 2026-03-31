"""
Main orchestration script for Machine Learning models on IMDB dataset.
Runs both supervised and unsupervised learning algorithms.
"""

import os
import sys
import json
from pathlib import Path

# Import modules
from data_utils import load_imdb_data, prepare_rating_prediction_data, prepare_clustering_data
from supervised_learning import SupervisedLearningModels
from unsupervised_learning import UnsupervisedLearningModels
from model_evaluation import ModelEvaluator


def run_supervised_learning():
    """Run all supervised learning models."""
    print("\n" + "="*80)
    print("SUPERVISED LEARNING: RATING PREDICTION")
    print("="*80)
    
    try:
        # Load and prepare data
        datasets = load_imdb_data()
        X_train, X_test, y_train, y_test, features = prepare_rating_prediction_data(datasets)
        
        # Initialize models and evaluator
        sl_models = SupervisedLearningModels()
        evaluator = ModelEvaluator()
        
        # Train all models
        models_to_train = [
            ('linear_regression', lambda: sl_models.train_linear_regression(X_train, X_test, y_train, y_test)),
            ('decision_tree', lambda: sl_models.train_decision_tree(X_train, X_test, y_train, y_test)),
            ('random_forest', lambda: sl_models.train_random_forest(X_train, X_test, y_train, y_test)),
            ('neural_network', lambda: sl_models.train_neural_network(X_train, X_test, y_train, y_test, epochs=30))
        ]
        
        trained_models = {}
        for model_name, train_func in models_to_train:
            try:
                model, metrics = train_func()
                trained_models[model_name] = (model, metrics)
            except Exception as e:
                print(f"⚠ Error training {model_name}: {e}")
        
        # Plot results for each model
        print("\n" + "="*80)
        print("GENERATING VISUALIZATIONS")
        print("="*80)
        
        for model_name, (model, metrics) in trained_models.items():
            try:
                # Get predictions for visualization
                y_pred = model.predict(X_test)
                if hasattr(y_pred, 'flatten'):
                    y_pred = y_pred.flatten()
                evaluator.plot_regression_results(y_test, y_pred, model_name.replace('_', ' ').title())
            except Exception as e:
                print(f"⚠ Could not plot {model_name}: {e}")
        
        # Save results
        os.makedirs('../outputs/ml_models', exist_ok=True)
        sl_models.save_results('../outputs/ml_models/supervised_learning_results.json')
        
        # Print and plot summary
        evaluator.print_model_summary(sl_models.results)
        evaluator.plot_model_comparison(sl_models.results, metric='r2_score')
        
        # Get best model
        best_name, best_model = sl_models.get_best_model()
        
        return sl_models, best_model
        
    except Exception as e:
        print(f"✗ Error in supervised learning: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def run_unsupervised_learning():
    """Run all unsupervised learning models."""
    print("\n" + "="*80)
    print("UNSUPERVISED LEARNING: CLUSTERING & DIMENSIONALITY REDUCTION")
    print("="*80)
    
    try:
        # Load and prepare data
        datasets = load_imdb_data()
        X, features = prepare_clustering_data(datasets, n_samples=5000)
        
        # Initialize models and evaluator
        ul_models = UnsupervisedLearningModels()
        evaluator = ModelEvaluator()
        
        # Train clustering models
        print("\n" + "-"*80)
        print("CLUSTERING MODELS")
        print("-"*80)
        
        # K-Means with different k values
        for k in [3, 5, 7]:
            try:
                print(f"\nK-Means with k={k}:")
                model, labels, metrics = ul_models.train_kmeans(X, n_clusters=k)
                evaluator.plot_clustering_results(X, labels, f"KMeans (k={k})")
            except Exception as e:
                print(f"⚠ Error with K-Means k={k}: {e}")
        
        # Hierarchical clustering
        try:
            model, labels, metrics = ul_models.train_hierarchical(X, n_clusters=5)
            evaluator.plot_clustering_results(X, labels, "Hierarchical Clustering")
        except Exception as e:
            print(f"⚠ Error in hierarchical clustering: {e}")
        
        # DBSCAN
        try:
            model, labels, metrics = ul_models.train_dbscan(X, eps=1.0, min_samples=5)
            unique_labels = len(set(labels)) - (1 if -1 in labels else 0)
            if unique_labels > 1:
                evaluator.plot_clustering_results(X, labels, "DBSCAN")
        except Exception as e:
            print(f"⚠ Error in DBSCAN: {e}")
        
        # Print clustering summary
        ul_models.get_cluster_summary()
        
        # Save results
        os.makedirs('../outputs/ml_models', exist_ok=True)
        ul_models.save_results('../outputs/ml_models/unsupervised_learning_results.json')
        
        return ul_models
        
    except Exception as e:
        print(f"✗ Error in unsupervised learning: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main execution function."""
    print("\n" + "█"*80)
    print("█ MACHINE LEARNING ON IMDB DATASET")
    print("█ Supervised & Unsupervised Algorithms with TensorFlow & scikit-learn")
    print("█"*80)
    
    # Run supervised learning
    sl_models, best_model = run_supervised_learning()
    
    # Run unsupervised learning
    ul_models = run_unsupervised_learning()
    
    # Final summary
    print("\n" + "="*80)
    print("EXECUTION COMPLETED")
    print("="*80)
    print("\n✓ Output files saved to: ../outputs/ml_models/")
    print("\nFiles generated:")
    print("  - Supervised learning results: supervised_learning_results.json")
    print("  - Unsupervised learning results: unsupervised_learning_results.json")
    print("  - Regression plots: *_regression.png")
    print("  - Clustering visualizations: *_clusters.png")
    print("  - PCA variance analysis: pca_variance.png")
    print("  - Model comparison: model_comparison.png")


if __name__ == "__main__":
    main()
