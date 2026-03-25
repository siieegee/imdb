import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_INPUT_DIR = Path("dataset/cleaned_datasets")
DEFAULT_OUTPUT_DIR = Path("outputs/visualizations")


def plot_title_type_distribution(df: pd.DataFrame, output_dir: Path) -> None:
    if "titleType" not in df.columns:
        return
    counts = df["titleType"].value_counts(dropna=True).head(15)
    plt.figure(figsize=(12, 6))
    counts.plot(kind="bar")
    plt.title("Top 15 Title Types")
    plt.xlabel("Title Type")
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "title_type_distribution.png", dpi=150)
    plt.close()


def plot_start_year_trend(df: pd.DataFrame, output_dir: Path) -> None:
    if "startYear" not in df.columns:
        return
    series = pd.to_numeric(df["startYear"], errors="coerce").dropna().astype(int)
    series = series[(series >= 1870) & (series <= 2100)]
    counts = series.value_counts().sort_index()
    if counts.empty:
        return
    plt.figure(figsize=(12, 6))
    counts.plot(kind="line")
    plt.title("Titles Released Per Year")
    plt.xlabel("Year")
    plt.ylabel("Number of Titles")
    plt.tight_layout()
    plt.savefig(output_dir / "titles_per_year.png", dpi=150)
    plt.close()


def plot_rating_distribution(df: pd.DataFrame, output_dir: Path) -> None:
    if "averageRating" not in df.columns:
        return
    ratings = pd.to_numeric(df["averageRating"], errors="coerce").dropna()
    ratings = ratings[(ratings >= 0) & (ratings <= 10)]
    if ratings.empty:
        return
    plt.figure(figsize=(10, 6))
    plt.hist(ratings, bins=20)
    plt.title("Average Rating Distribution")
    plt.xlabel("Average Rating")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(output_dir / "rating_distribution.png", dpi=150)
    plt.close()


def plot_votes_distribution(df: pd.DataFrame, output_dir: Path) -> None:
    if "numVotes" not in df.columns:
        return
    votes = pd.to_numeric(df["numVotes"], errors="coerce").dropna()
    votes = votes[votes > 0]
    if votes.empty:
        return
    plt.figure(figsize=(10, 6))
    plt.hist(votes, bins=30, log=True)
    plt.title("Votes Distribution (Log Frequency)")
    plt.xlabel("Number of Votes")
    plt.ylabel("Frequency (log scale)")
    plt.tight_layout()
    plt.savefig(output_dir / "votes_distribution.png", dpi=150)
    plt.close()


def plot_runtime_distribution(df: pd.DataFrame, output_dir: Path) -> None:
    if "runtimeMinutes" not in df.columns:
        return
    runtime = pd.to_numeric(df["runtimeMinutes"], errors="coerce").dropna()
    runtime = runtime[(runtime > 0) & (runtime <= 300)]
    if runtime.empty:
        return
    plt.figure(figsize=(10, 6))
    plt.hist(runtime, bins=30)
    plt.title("Runtime Distribution (<= 300 mins)")
    plt.xlabel("Runtime (minutes)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(output_dir / "runtime_distribution.png", dpi=150)
    plt.close()


def plot_top_genres(df: pd.DataFrame, output_dir: Path) -> None:
    if "genres" not in df.columns:
        return
    genre_series = (
        df["genres"]
        .dropna()
        .astype(str)
        .str.split(",")
        .explode()
        .str.strip()
    )
    genre_series = genre_series[genre_series != ""]
    if genre_series.empty:
        return
    counts = genre_series.value_counts().head(15)
    plt.figure(figsize=(12, 6))
    counts.plot(kind="bar")
    plt.title("Top 15 Genres")
    plt.xlabel("Genre")
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "top_genres.png", dpi=150)
    plt.close()


def plot_top_rated_titles(merged_df: pd.DataFrame, output_dir: Path, min_votes: int = 1000) -> None:
    required = {"primaryTitle", "averageRating", "numVotes"}
    if not required.issubset(merged_df.columns):
        return

    frame = merged_df.copy()
    frame["averageRating"] = pd.to_numeric(frame["averageRating"], errors="coerce")
    frame["numVotes"] = pd.to_numeric(frame["numVotes"], errors="coerce")
    frame = frame.dropna(subset=["primaryTitle", "averageRating", "numVotes"])
    frame = frame[frame["numVotes"] >= min_votes]
    frame = frame.sort_values(["averageRating", "numVotes"], ascending=[False, False]).head(15)
    if frame.empty:
        return

    labels = frame["primaryTitle"].astype(str).str.slice(0, 40)
    plt.figure(figsize=(12, 8))
    plt.barh(labels[::-1], frame["averageRating"][::-1])
    plt.title(f"Top Rated Titles (min {min_votes} votes)")
    plt.xlabel("Average Rating")
    plt.ylabel("Title")
    plt.tight_layout()
    plt.savefig(output_dir / "top_rated_titles.png", dpi=150)
    plt.close()


def plot_median_rating_by_title_type(merged_df: pd.DataFrame, output_dir: Path) -> None:
    required = {"titleType", "averageRating", "numVotes"}
    if not required.issubset(merged_df.columns):
        return
    frame = merged_df.copy()
    frame["averageRating"] = pd.to_numeric(frame["averageRating"], errors="coerce")
    frame["numVotes"] = pd.to_numeric(frame["numVotes"], errors="coerce")
    frame = frame.dropna(subset=["titleType", "averageRating", "numVotes"])
    frame = frame[frame["numVotes"] >= 100]
    if frame.empty:
        return
    grouped = frame.groupby("titleType")["averageRating"].median().sort_values(ascending=False).head(15)
    plt.figure(figsize=(12, 6))
    grouped.plot(kind="bar")
    plt.title("Median Rating by Title Type (min 100 votes)")
    plt.xlabel("Title Type")
    plt.ylabel("Median Average Rating")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "median_rating_by_title_type.png", dpi=150)
    plt.close()


def run_visualization(input_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    title_basics_path = input_dir / "title.basics.csv"
    title_ratings_path = input_dir / "title.ratings.csv"
    title_basics = None
    title_ratings = None

    if title_basics_path.exists():
        title_basics = pd.read_csv(title_basics_path, low_memory=False)
        plot_title_type_distribution(title_basics, output_dir)
        plot_start_year_trend(title_basics, output_dir)
        plot_top_genres(title_basics, output_dir)
        plot_runtime_distribution(title_basics, output_dir)

    if title_ratings_path.exists():
        title_ratings = pd.read_csv(title_ratings_path, low_memory=False)
        plot_rating_distribution(title_ratings, output_dir)
        plot_votes_distribution(title_ratings, output_dir)

    if title_basics is not None and title_ratings is not None:
        merged = title_basics.merge(
            title_ratings[["tconst", "averageRating", "numVotes"]],
            on="tconst",
            how="inner",
        )
        plot_top_rated_titles(merged, output_dir)
        plot_median_rating_by_title_type(merged, output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate IMDb dataset visualizations.")
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
        help=f"Directory to save chart images (default: {DEFAULT_OUTPUT_DIR})",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_visualization(args.input_dir, args.output_dir)
