"""Model training and recommendation generation."""

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

try:
    from tensorflow import keras
except ImportError:
    keras = None


def train_models(X, y, random_seed=42):
    """Train scikit-learn and TensorFlow (or fallback) ranking models."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_seed
    )

    models = {}
    metrics = {}

    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=14,
        random_state=random_seed,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    models["random_forest"] = rf
    metrics["random_forest"] = _regression_metrics(y_test, rf_pred)

    if keras is not None:
        nn = keras.Sequential(
            [
                keras.layers.Dense(128, activation="relu", input_dim=X_train.shape[1]),
                keras.layers.Dropout(0.2),
                keras.layers.Dense(64, activation="relu"),
                keras.layers.Dense(1),
            ]
        )
        nn.compile(optimizer="adam", loss="mse", metrics=["mae"])
        nn.fit(X_train, y_train, epochs=20, batch_size=64, validation_split=0.2, verbose=0)
        nn_pred = nn.predict(X_test, verbose=0).flatten()
        models["tensorflow_nn"] = nn
        metrics["tensorflow_nn"] = _regression_metrics(y_test, nn_pred)
    else:
        mlp = MLPRegressor(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            random_state=random_seed,
            max_iter=300,
        )
        mlp.fit(X_train, y_train)
        mlp_pred = mlp.predict(X_test)
        models["sklearn_mlp_fallback"] = mlp
        metrics["sklearn_mlp_fallback"] = _regression_metrics(y_test, mlp_pred)

    return models, metrics


def recommend_top_n(model, X, movie_meta, top_n=20):
    """Score all items and return top-N movie recommendations."""
    if hasattr(model, "predict"):
        try:
            scores = model.predict(X, verbose=0).flatten()
        except TypeError:
            scores = model.predict(X)
    else:
        raise ValueError("Model does not support prediction")

    recs = movie_meta.copy()
    recs["predicted_score"] = scores
    recs = recs.sort_values("predicted_score", ascending=False).head(top_n).reset_index(drop=True)
    return recs


def _regression_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return {
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "r2_score": float(r2_score(y_true, y_pred)),
    }
