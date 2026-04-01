"""Data loading and feature engineering for movie recommendation."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def resolve_data_dir(candidates):
    """Return the first existing directory that contains required CSV files."""
    required = {"title.basics.csv", "title.ratings.csv"}
    for candidate in candidates:
        path = Path(candidate)
        if not path.exists():
            continue
        files = {p.name for p in path.glob("*.csv")}
        if required.issubset(files):
            return path
    raise FileNotFoundError("Could not find dataset directory with title.basics.csv and title.ratings.csv")


def load_imdb_tables(data_dir):
    """Load IMDB basics and ratings tables."""
    basics = pd.read_csv(Path(data_dir) / "title.basics.csv")
    ratings = pd.read_csv(Path(data_dir) / "title.ratings.csv")
    return basics, ratings


def build_feature_matrix(basics, ratings, sample_size=10000, min_votes=100, random_seed=42):
    """Create model features and recommendation target from movie metadata."""
    df = basics.merge(ratings, on="tconst", how="inner")

    df["runtimeMinutes"] = pd.to_numeric(df["runtimeMinutes"], errors="coerce")
    df["startYear"] = pd.to_numeric(df["startYear"], errors="coerce")
    df["numVotes"] = pd.to_numeric(df["numVotes"], errors="coerce")
    df["averageRating"] = pd.to_numeric(df["averageRating"], errors="coerce")

    df = df.dropna(subset=["runtimeMinutes", "startYear", "numVotes", "averageRating", "primaryTitle"])
    df = df[df["numVotes"] >= min_votes].copy()

    if len(df) == 0:
        raise ValueError("No rows available after filtering. Lower MIN_VOTES or check source data.")

    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=random_seed)

    genre_features = df["genres"].fillna("").str.get_dummies(sep=",")
    numeric = df[["runtimeMinutes", "startYear", "numVotes", "averageRating"]].copy()
    numeric["log_numVotes"] = np.log1p(numeric["numVotes"])
    feature_df = pd.concat([numeric, genre_features], axis=1)

    scaler = StandardScaler()
    X = scaler.fit_transform(feature_df.values)

    # Smoothed target score for recommendation ranking.
    y = 0.7 * _minmax(df["averageRating"].values) + 0.3 * _minmax(np.log1p(df["numVotes"].values))

    meta_cols = ["tconst", "primaryTitle", "genres", "averageRating", "numVotes", "startYear"]
    movie_meta = df[meta_cols].reset_index(drop=True)

    return X, y, movie_meta


def _minmax(values):
    min_val = float(np.min(values))
    max_val = float(np.max(values))
    if min_val == max_val:
        return np.zeros_like(values, dtype=float)
    return (values - min_val) / (max_val - min_val)
