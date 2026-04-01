"""Run AI/ML movie recommendation sample pipeline."""

import json

from config import (
    DATA_DIR_CANDIDATES,
    MIN_VOTES,
    OUTPUT_DIR,
    RANDOM_SEED,
    SAMPLE_SIZE,
    TOP_N,
)
from data_pipeline import build_feature_matrix, load_imdb_tables, resolve_data_dir
from recommender import recommend_top_n, train_models


def main():
    data_dir = resolve_data_dir(DATA_DIR_CANDIDATES)
    print(f"Using dataset directory: {data_dir}")

    basics, ratings = load_imdb_tables(data_dir)
    X, y, movie_meta = build_feature_matrix(
        basics,
        ratings,
        sample_size=SAMPLE_SIZE,
        min_votes=MIN_VOTES,
        random_seed=RANDOM_SEED,
    )

    models, metrics = train_models(X, y, random_seed=RANDOM_SEED)
    best_name = max(metrics.items(), key=lambda item: item[1]["r2_score"])[0]
    recommendations = recommend_top_n(models[best_name], X, movie_meta, top_n=TOP_N)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    metrics_path = OUTPUT_DIR / "ai_ml_sample_metrics.json"
    recs_path = OUTPUT_DIR / "ai_ml_sample_recommendations.json"

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    payload = {
        "best_model": best_name,
        "top_n": TOP_N,
        "recommendations": recommendations.to_dict(orient="records"),
    }
    with open(recs_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved recommendations to: {recs_path}")


if __name__ == "__main__":
    main()
