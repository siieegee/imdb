import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_INPUT_DIR = Path("dataset/cleaned_datasets")
DEFAULT_OUTPUT_DIR = Path("outputs/analysis")


STANDALONE_TITLE_TYPES = {"movie", "tvSeries", "tvMiniSeries", "tvMovie"}


def _require_columns(df: pd.DataFrame, required: set[str], label: str) -> None:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{label} missing columns: {sorted(missing)}")


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def filter_standalone_titles(title_basics: pd.DataFrame) -> pd.DataFrame:
    out = title_basics.copy()
    out["titleType"] = out["titleType"].astype(str).str.strip()
    return out[out["titleType"].isin(STANDALONE_TITLE_TYPES)]


def analyze_title_type_distribution(title_basics: pd.DataFrame) -> pd.DataFrame:
    if "titleType" not in title_basics.columns:
        return pd.DataFrame(columns=["titleType", "count"])
    counts = title_basics["titleType"].value_counts(dropna=True).head(15)
    return counts.rename_axis("titleType").reset_index(name="count")


def analyze_titles_per_year(title_basics: pd.DataFrame) -> pd.DataFrame:
    if "startYear" not in title_basics.columns:
        return pd.DataFrame(columns=["startYear", "count"])
    series = pd.to_numeric(title_basics["startYear"], errors="coerce").dropna().astype(int)
    series = series[(series >= 1870) & (series <= 2100)]
    if series.empty:
        return pd.DataFrame(columns=["startYear", "count"])
    counts = series.value_counts().sort_index()
    return counts.rename_axis("startYear").reset_index(name="count")


def analyze_rating_distribution(title_ratings: pd.DataFrame) -> dict[str, Any]:
    if "averageRating" not in title_ratings.columns:
        return {}
    ratings = pd.to_numeric(title_ratings["averageRating"], errors="coerce").dropna()
    ratings = ratings[(ratings >= 0) & (ratings <= 10)]
    if ratings.empty:
        return {}

    # visualization.py uses: plt.hist(ratings, bins=20)
    counts, edges = np.histogram(ratings.to_numpy(), bins=20)
    quantiles = ratings.quantile([0.5, 0.9, 0.95, 0.99]).to_dict()
    quantiles = {str(k): float(v) for k, v in quantiles.items()}
    return {
        "count": int(len(ratings)),
        "min": float(ratings.min()),
        "max": float(ratings.max()),
        "bin_edges": edges.astype(float).tolist(),
        "bin_counts": counts.astype(int).tolist(),
        "quantiles": quantiles,
    }


def analyze_votes_distribution(
    title_basics: pd.DataFrame,
    title_ratings: pd.DataFrame,
    min_vote_floor: float = 0.0,
    num_bins: int = 40,
) -> dict[str, Any]:
    """
    Matches visualization.py logic for votes_distribution:
    - filter to show/movie types using title_basics.titleType
    - votes > 0
    - log-spaced bins between log10(min) and log10(max) with `num_bins`
    - histogram counts use those bins (y is linear in the current visualization)
    """
    _require_columns(title_basics, {"tconst", "titleType"}, "title.basics.csv")
    _require_columns(title_ratings, {"tconst", "numVotes"}, "title.ratings.csv")

    standalone_basics = filter_standalone_titles(title_basics)
    merged = title_ratings.merge(standalone_basics[["tconst", "titleType"]], on="tconst", how="inner")

    votes = pd.to_numeric(merged["numVotes"], errors="coerce").dropna()
    votes = votes[votes > min_vote_floor]
    if votes.empty:
        return {}

    # visualization.py: log_bins = np.logspace(np.log10(votes.min()), np.log10(votes.max()), num=40)
    log_bins = np.logspace(np.log10(votes.min()), np.log10(votes.max()), num=num_bins)
    counts, edges = np.histogram(votes.to_numpy(), bins=log_bins)
    quantiles = votes.quantile([0.5, 0.9, 0.95, 0.99, 0.999]).to_dict()
    quantiles = {str(k): float(v) for k, v in quantiles.items()}

    return {
        "count": int(len(votes)),
        "min": float(votes.min()),
        "max": float(votes.max()),
        "bin_edges": edges.astype(float).tolist(),
        "bin_counts": counts.astype(int).tolist(),
        "quantiles": quantiles,
    }


def analyze_top_genres(title_basics: pd.DataFrame) -> pd.DataFrame:
    if "genres" not in title_basics.columns:
        return pd.DataFrame(columns=["genre", "count"])
    genre_series = (
        title_basics["genres"]
        .dropna()
        .astype(str)
        .str.split(",")
        .explode()
        .str.strip()
    )
    genre_series = genre_series[genre_series != ""]
    if genre_series.empty:
        return pd.DataFrame(columns=["genre", "count"])
    counts = genre_series.value_counts().head(15)
    return counts.rename_axis("genre").reset_index(name="count")


def analyze_median_rating_by_title_type(title_basics: pd.DataFrame, title_ratings: pd.DataFrame) -> pd.DataFrame:
    required_basics = {"tconst", "titleType"}
    required_ratings = {"tconst", "averageRating", "numVotes"}
    _require_columns(title_basics, required_basics, "title.basics.csv")
    _require_columns(title_ratings, required_ratings, "title.ratings.csv")

    merged = title_basics.merge(title_ratings[["tconst", "averageRating", "numVotes"]], on="tconst", how="inner")
    merged["averageRating"] = pd.to_numeric(merged["averageRating"], errors="coerce")
    merged["numVotes"] = pd.to_numeric(merged["numVotes"], errors="coerce")
    merged = merged.dropna(subset=["titleType", "averageRating", "numVotes"])
    merged = merged[merged["numVotes"] >= 100]
    if merged.empty:
        return pd.DataFrame(columns=["titleType", "medianAverageRating"])

    grouped = (
        merged.groupby("titleType")["averageRating"].median().sort_values(ascending=False).head(15)
    )
    out = grouped.reset_index().rename(columns={"averageRating": "medianAverageRating", "titleType": "titleType"})
    return out


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


def run_reports_analysis(input_dir: Path, output_dir: Path) -> None:
    _ensure_dir(output_dir)

    title_basics_path = input_dir / "title.basics.csv"
    title_ratings_path = input_dir / "title.ratings.csv"
    if not title_basics_path.exists():
        raise FileNotFoundError(f"Missing: {title_basics_path}")
    if not title_ratings_path.exists():
        raise FileNotFoundError(f"Missing: {title_ratings_path}")

    title_basics = load_csv(title_basics_path)
    title_ratings = load_csv(title_ratings_path)

    title_type_counts = analyze_title_type_distribution(title_basics)
    titles_per_year = analyze_titles_per_year(title_basics)
    rating_distribution = analyze_rating_distribution(title_ratings)
    votes_distribution = analyze_votes_distribution(title_basics, title_ratings)
    top_genres = analyze_top_genres(title_basics)
    median_rating_by_title_type = analyze_median_rating_by_title_type(title_basics, title_ratings)

    # Save machine-readable report
    report = {
        "title_type_distribution_top15": {
            "rows": len(title_type_counts),
            "data": title_type_counts.to_dict(orient="records"),
        },
        "titles_per_year_1870_2100": {
            "rows": len(titles_per_year),
            "data": titles_per_year.to_dict(orient="records"),
        },
        "rating_distribution": rating_distribution,
        "votes_distribution": votes_distribution,
        "top_genres_top15": {
            "rows": len(top_genres),
            "data": top_genres.to_dict(orient="records"),
        },
        "median_rating_by_title_type_top15": {
            "rows": len(median_rating_by_title_type),
            "data": median_rating_by_title_type.to_dict(orient="records"),
        },
    }

    # Keep output naming consistent with the script name.
    report_path = output_dir / "report_analysis.json"
    with open(report_path, "w", encoding="utf-8") as fp:
        json.dump(report, fp, indent=2)

    # Also save a few CSVs for quick inspection
    title_type_counts.to_csv(output_dir / "title_type_distribution_top15.csv", index=False)
    titles_per_year.to_csv(output_dir / "titles_per_year_counts.csv", index=False)
    top_genres.to_csv(output_dir / "top_genres_top15.csv", index=False)
    median_rating_by_title_type.to_csv(output_dir / "median_rating_by_title_type_top15.csv", index=False)

    if rating_distribution:
        pd.DataFrame(
            {
                "bin_left": rating_distribution["bin_edges"][:-1],
                "bin_right": rating_distribution["bin_edges"][1:],
                "count": rating_distribution["bin_counts"],
            }
        ).to_csv(output_dir / "rating_distribution_bins.csv", index=False)

    if votes_distribution:
        pd.DataFrame(
            {
                "bin_left": votes_distribution["bin_edges"][:-1],
                "bin_right": votes_distribution["bin_edges"][1:],
                "count": votes_distribution["bin_counts"],
            }
        ).to_csv(output_dir / "votes_distribution_bins.csv", index=False)

    # Print a short human summary
    print(f"Wrote: {report_path}")
    if not title_type_counts.empty:
        print("Top title types:", ", ".join(title_type_counts["titleType"].astype(str).head(5).tolist()))
    if not median_rating_by_title_type.empty:
        top_row = median_rating_by_title_type.iloc[0]
        print(f"Highest median rating type: {top_row['titleType']} ({top_row['medianAverageRating']:.3f})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze data underlying visualization charts.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Directory containing cleaned CSV files (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to save analysis outputs (default: {DEFAULT_OUTPUT_DIR})",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_reports_analysis(args.input_dir, args.output_dir)

