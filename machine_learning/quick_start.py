"""
Quick Start Script - Run individual components of the ML pipeline.
This is useful for testing and exploring specific models.
"""

import sys
from data_utils import load_imdb_data, prepare_rating_prediction_data, prepare_clustering_data
from supervised_learning import SupervisedLearningModels
from unsupervised_learning import UnsupervisedLearningModels
from model_evaluation import ModelEvaluator


def test_data_loading():
    """Test if data can be loaded successfully."""
    print("\n" + "="*60)
    print("TEST: Loading IMDB Data")
    print("="*60)
    try:
        datasets = load_imdb_data()
        print(f"✓ Successfully loaded data")
        print(f"  Available datasets: {list(datasets.keys())}")
        return datasets
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return None


def test_supervised_learning():
    """Test supervised learning model training."""
    print("\n" + "="*60)
    print("TEST: Supervised Learning (Single Model)")
    print("="*60)
    try:
        datasets = load_imdb_data()
        X_train, X_test, y_train, y_test, features = prepare_rating_prediction_data(datasets)
        
        sl_models = SupervisedLearningModels()
        model, metrics = sl_models.train_random_forest(X_train, X_test, y_train, y_test, n_estimators=50)
        
        print(f"✓ Random Forest model trained successfully")
        print(f"  R² Score: {metrics['r2_score']:.4f}")
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_unsupervised_learning():
    """Test unsupervised learning model training."""
    print("\n" + "="*60)
    print("TEST: Unsupervised Learning (K-Means)")
    print("="*60)
    try:
        datasets = load_imdb_data()
        X, features = prepare_clustering_data(datasets, n_samples=1000)
        
        ul_models = UnsupervisedLearningModels()
        model, labels, metrics = ul_models.train_kmeans(X, n_clusters=5)
        
        print(f"✓ K-Means clustering trained successfully")
        print(f"  Silhouette Score: {metrics['silhouette_score']:.4f}")
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_pca():
    """Test PCA dimensionality reduction."""
    print("\n" + "="*60)
    print("TEST: PCA Dimensionality Reduction")
    print("="*60)
    try:
        datasets = load_imdb_data()
        X, features = prepare_clustering_data(datasets, n_samples=1000)
        
        ul_models = UnsupervisedLearningModels()
        model, X_transformed, metrics = ul_models.train_pca(X, n_components=5)
        
        print(f"✓ PCA completed successfully")
        print(f"  Variance explained: {metrics['total_variance_explained']:.4f}")
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_neural_network():
    """Test Neural Network training (warning: may be slow)."""
    print("\n" + "="*60)
    print("TEST: Neural Network Training")
    print("="*60)
    print("⚠ WARNING: This test may take a while...")
    try:
        datasets = load_imdb_data()
        X_train, X_test, y_train, y_test, features = prepare_rating_prediction_data(datasets)
        
        # Use smaller subset for quick test
        X_train = X_train[:1000]
        X_test = X_test[:300]
        y_train = y_train[:1000]
        y_test = y_test[:300]
        
        sl_models = SupervisedLearningModels()
        model, metrics = sl_models.train_neural_network(
            X_train, X_test, y_train, y_test, 
            epochs=10,  # Fewer epochs for quick test
            batch_size=32
        )
        
        print(f"✓ Neural Network trained successfully")
        print(f"  R² Score: {metrics['r2_score']:.4f}")
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_full_test_suite():
    """Run all tests."""
    print("\n" + "█"*60)
    print("█ MACHINE LEARNING - QUICK START TESTS")
    print("█"*60)
    
    tests = [
        ("Data Loading", test_data_loading),
        ("Supervised Learning", test_supervised_learning),
        ("Unsupervised Learning", test_unsupervised_learning),
        ("PCA", test_pca),
        # Uncomment to test neural network (may be slow)
        # ("Neural Network", test_neural_network),
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = "✓ PASSED" if result else "✗ FAILED"
        except Exception as e:
            results[test_name] = f"✗ FAILED: {str(e)[:50]}"
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    for test_name, result in results.items():
        print(f"{test_name:.<40} {result}")
    
    print("\n✓ Quick start tests completed!")
    print("\nNext steps:")
    print("  1. Review the results above")
    print("  2. Run the full pipeline: python main.py")
    print("  3. Check outputs in: ../outputs/ml_models/")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_name = sys.argv[1].lower()
        if test_name == 'data':
            test_data_loading()
        elif test_name == 'supervised':
            test_supervised_learning()
        elif test_name == 'unsupervised':
            test_unsupervised_learning()
        elif test_name == 'pca':
            test_pca()
        elif test_name == 'nn':
            test_neural_network()
        else:
            print(f"Unknown test: {test_name}")
            print("Available tests: data, supervised, unsupervised, pca, nn")
    else:
        run_full_test_suite()
