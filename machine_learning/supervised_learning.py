"""
Supervised Learning Models for IMDB Dataset.
- Linear Regression & Ridge Regression (predicting ratings)
- Decision Tree, Random Forest, Gradient Boosting (classification/regression)
- Neural Network with TensorFlow (rating prediction)
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import tensorflow as tf
from tensorflow import keras
import json


class SupervisedLearningModels:
    """Collection of supervised learning models for rating prediction."""
    
    def __init__(self):
        self.models = {}
        self.results = {}
    
    def train_linear_regression(self, X_train, X_test, y_train, y_test):
        """Train Linear Regression model."""
        print("\n" + "="*60)
        print("LINEAR REGRESSION")
        print("="*60)
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        metrics = self._evaluate(y_test, y_pred, "Linear Regression")
        
        self.models['linear_regression'] = model
        self.results['linear_regression'] = metrics
        return model, metrics
    

    
    def train_decision_tree(self, X_train, X_test, y_train, y_test, max_depth=10):
        """Train Decision Tree Regressor."""
        print("\n" + "="*60)
        print("DECISION TREE REGRESSOR")
        print("="*60)
        
        model = DecisionTreeRegressor(max_depth=max_depth, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        metrics = self._evaluate(y_test, y_pred, "Decision Tree")
        
        self.models['decision_tree'] = model
        self.results['decision_tree'] = metrics
        return model, metrics
    
    def train_random_forest(self, X_train, X_test, y_train, y_test, 
                           n_estimators=100, max_depth=10):
        """Train Random Forest Regressor."""
        print("\n" + "="*60)
        print("RANDOM FOREST REGRESSOR")
        print("="*60)
        
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        metrics = self._evaluate(y_test, y_pred, "Random Forest")
        
        # Feature importance
        if hasattr(model, 'feature_importances_'):
            print(f"\nFeature Importance:")
            for i, importance in enumerate(model.feature_importances_):
                print(f"  Feature {i}: {importance:.4f}")
        
        self.models['random_forest'] = model
        self.results['random_forest'] = metrics
        return model, metrics
    

    
    def train_neural_network(self, X_train, X_test, y_train, y_test,
                            epochs=50, batch_size=32):
        """Train Neural Network using TensorFlow/Keras."""
        print("\n" + "="*60)
        print("NEURAL NETWORK (TensorFlow/Keras)")
        print("="*60)
        
        model = keras.Sequential([
            keras.layers.Dense(128, activation='relu', input_dim=X_train.shape[1]),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(1)  # Output layer for regression
        ])
        
        model.compile(
            optimizer='adam',
            loss='mean_squared_error',
            metrics=['mae']
        )
        
        print(f"Training Neural Network...")
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            verbose=1
        )
        
        y_pred = model.predict(X_test, verbose=0)
        metrics = self._evaluate(y_test, y_pred.flatten(), "Neural Network")
        
        self.models['neural_network'] = model
        self.results['neural_network'] = metrics
        return model, metrics
    
    def _evaluate(self, y_true, y_pred, model_name):
        """Calculate evaluation metrics."""
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        metrics = {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'r2_score': float(r2)
        }
        
        print(f"\n{model_name} Metrics:")
        print(f"  MSE:  {mse:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE:  {mae:.4f}")
        print(f"  R²:   {r2:.4f}")
        
        return metrics
    
    def save_results(self, output_path):
        """Save model results to JSON."""
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\n✓ Results saved to {output_path}")
    
    def get_best_model(self):
        """Get the model with the best R² score."""
        best_model = max(self.results.items(), key=lambda x: x[1]['r2_score'])
        print(f"\n✓ Best model: {best_model[0].upper()} with R² = {best_model[1]['r2_score']:.4f}")
        return best_model[0], self.models[best_model[0]]
