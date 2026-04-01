import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------

INPUT_DIR = Path("dataset/imdb_datasets")
OUTPUT_DIR = Path("dataset/cleaned_datasets")
REPORT_PATH = Path("dataset/cleaning_report.json")
ANALYSIS_OUTPUT_DIR = Path("outputs/analysis")
VISUALIZATION_OUTPUT_DIR = Path("outputs/visualizations")


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

STANDALONE_TITLE_TYPES = {"movie", "tvSeries", "tvMiniSeries", "tvMovie"}


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def _require_columns(df: pd.DataFrame, required: set[str], label: str) -> None:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{label} missing columns: {sorted(missing)}")


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

def normalize_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize common missing value markers to pandas NA."""
    return df.replace({"": pd.NA, "\\N": pd.NA, "nan": pd.NA, "None": pd.NA})


def clean_title_basics(df: pd.DataFrame) -> pd.DataFrame:
    """Clean `title.basics` specific columns and value ranges."""
    numeric_cols = ["isAdult", "startYear", "endYear", "runtimeMinutes"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "isAdult" in df.columns:
        df["isAdult"] = df["isAdult"].fillna(0).astype("Int64")
        df["isAdult"] = df["isAdult"].clip(lower=0, upper=1)

    if "startYear" in df.columns:
        df.loc[df["startYear"] < 1870, "startYear"] = pd.NA
        df.loc[df["startYear"] > 2100, "startYear"] = pd.NA

    if "runtimeMinutes" in df.columns:
        df.loc[df["runtimeMinutes"] <= 0, "runtimeMinutes"] = pd.NA
        df.loc[df["runtimeMinutes"] > 1500, "runtimeMinutes"] = pd.NA

    if "genres" in df.columns:
        df["genres"] = df["genres"].astype("string").str.strip()

    return df


def clean_title_ratings(df: pd.DataFrame) -> pd.DataFrame:
    """Clean `title.ratings` by enforcing rating/vote validity."""
    for col in ["averageRating", "numVotes"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "averageRating" in df.columns:
        df.loc[(df["averageRating"] < 0) | (df["averageRating"] > 10), "averageRating"] = pd.NA

    if "numVotes" in df.columns:
        df.loc[df["numVotes"] < 0, "numVotes"] = pd.NA
        df["numVotes"] = df["numVotes"].astype("Int64")

    return df


def clean_name_basics(df: pd.DataFrame) -> pd.DataFrame:
    """Clean `name.basics` year fields and remove impossible timelines."""
    for col in ["birthYear", "deathYear"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "birthYear" in df.columns:
        df.loc[df["birthYear"] < 1850, "birthYear"] = pd.NA
        df.loc[df["birthYear"] > 2100, "birthYear"] = pd.NA

    if "deathYear" in df.columns:
        df.loc[df["deathYear"] < 1850, "deathYear"] = pd.NA
        df.loc[df["deathYear"] > 2100, "deathYear"] = pd.NA

    if {"birthYear", "deathYear"}.issubset(df.columns):
        invalid = (
            df["birthYear"].notna()
            & df["deathYear"].notna()
            & (df["deathYear"] < df["birthYear"])
        )
        df.loc[invalid, "deathYear"] = pd.NA

    return df


def clean_title_episode(df: pd.DataFrame) -> pd.DataFrame:
    """Clean `title.episode` season and episode numbers."""
    for col in ["seasonNumber", "episodeNumber"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df.loc[df[col] <= 0, col] = pd.NA

    return df


def clean_title_akas(df: pd.DataFrame) -> pd.DataFrame:
    """Clean `title.akas` ordering flags and original-title indicator."""
    for col in ["ordering", "isOriginalTitle"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "ordering" in df.columns:
        df.loc[df["ordering"] <= 0, "ordering"] = pd.NA
        df["ordering"] = df["ordering"].astype("Int64")

    if "isOriginalTitle" in df.columns:
        df["isOriginalTitle"] = df["isOriginalTitle"].fillna(0).astype("Int64")
        df["isOriginalTitle"] = df["isOriginalTitle"].clip(lower=0, upper=1)

    return df


def clean_title_principals(df: pd.DataFrame) -> pd.DataFrame:
    """Clean `title.principals` cast/crew ordering values."""
    if "ordering" in df.columns:
        df["ordering"] = pd.to_numeric(df["ordering"], errors="coerce")
        df.loc[df["ordering"] <= 0, "ordering"] = pd.NA
        df["ordering"] = df["ordering"].astype("Int64")
    return df


def clean_common_strings(df: pd.DataFrame) -> pd.DataFrame:
    """Trim whitespace across all string-like columns."""
    string_cols = df.select_dtypes(include=["object", "string"]).columns
    for col in string_cols:
        df[col] = df[col].astype("string").str.strip()
    return df


def clean_dataset(df: pd.DataFrame, filename: str) -> pd.DataFrame:
    """Apply shared cleaning steps, then route to table-specific rules."""
    df = normalize_nulls(df)
    df = clean_common_strings(df)
    df = df.drop_duplicates()

    if filename == "title.basics.csv":
        df = clean_title_basics(df)
    elif filename == "title.ratings.csv":
        df = clean_title_ratings(df)
    elif filename == "name.basics.csv":
        df = clean_name_basics(df)
    elif filename == "title.episode.csv":
        df = clean_title_episode(df)
    elif filename == "title.akas.csv":
        df = clean_title_akas(df)
    elif filename == "title.principals.csv":
        df = clean_title_principals(df)

    return df


def build_report_entry(before: pd.DataFrame, after: pd.DataFrame) -> dict:
    """Build a per-file summary for audit/debugging after a run."""
    return {
        "rows_before": int(len(before)),
        "rows_after": int(len(after)),
        "rows_removed": int(len(before) - len(after)),
        "missing_before": {k: int(v) for k, v in before.isna().sum().to_dict().items()},
        "missing_after": {k: int(v) for k, v in after.isna().sum().to_dict().items()},
        "dtypes_after": {k: str(v) for k, v in after.dtypes.to_dict().items()},
    }


def safe_describe(df: pd.DataFrame) -> dict:
    """Build numeric summary stats for a dataframe."""
    numeric = df.select_dtypes(include=["number"])
    if numeric.empty:
        return {}
    desc = numeric.describe().transpose()
    return {
        column: {k: float(v) for k, v in stats.to_dict().items()}
        for column, stats in desc.iterrows()
    }


def analyze_file(file_path: Path) -> dict:
    """Analyze one cleaned CSV file and return metadata/stats."""
    df = pd.read_csv(file_path, low_memory=False)
    return {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": list(df.columns),
        "missing_values": {k: int(v) for k, v in df.isna().sum().to_dict().items()},
        "duplicate_rows": int(df.duplicated().sum()),
        "numeric_summary": safe_describe(df),
    }


def run_cleaning(input_dir: Path, output_dir: Path, report_path: Path) -> None:
    """Run cleaning for every CSV file in `input_dir`."""
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    report = {}
    csv_files = sorted(input_dir.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {input_dir}")

    for file_path in csv_files:
        df_raw = pd.read_csv(file_path, low_memory=False)
        df_clean = clean_dataset(df_raw.copy(), file_path.name)
        out_file = output_dir / file_path.name
        df_clean.to_csv(out_file, index=False)
        report[file_path.name] = build_report_entry(df_raw, df_clean)

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


def run_cleaning_analysis(input_dir: Path, output_dir: Path) -> None:
    """Analyze all cleaned CSV files and write analysis artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)

    report = {}
    csv_files = sorted(input_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {input_dir}")

    for csv_file in csv_files:
        report[csv_file.name] = analyze_file(csv_file)

    report_path = output_dir / "analysis_report.json"
    with open(report_path, "w", encoding="utf-8") as fp:
        json.dump(report, fp, indent=2)

    summary_rows = []
    for filename, details in report.items():
        summary_rows.append(
            {
                "file": filename,
                "rows": details["rows"],
                "columns": details["columns"],
                "duplicate_rows": details["duplicate_rows"],
            }
        )
    pd.DataFrame(summary_rows).to_csv(output_dir / "dataset_summary.csv", index=False)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

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

    # visualization uses: plt.hist(ratings, bins=20)
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
    Matches visualization logic for votes_distribution:
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

    # visualization: log_bins = np.logspace(np.log10(votes.min()), np.log10(votes.max()), num=40)
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

    report_path = output_dir / "report_analysis.json"
    with open(report_path, "w", encoding="utf-8") as fp:
        json.dump(report, fp, indent=2)

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

    print(f"Wrote: {report_path}")
    if not title_type_counts.empty:
        print("Top title types:", ", ".join(title_type_counts["titleType"].astype(str).head(5).tolist()))
    if not median_rating_by_title_type.empty:
        top_row = median_rating_by_title_type.iloc[0]
        print(f"Highest median rating type: {top_row['titleType']} ({top_row['medianAverageRating']:.3f})")


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

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
    # Annotate where data becomes incomplete — IMDb entries for recent years
    # are added retroactively, so counts drop sharply and should not be
    # interpreted as a real-world decline in production.
    plt.axvline(x=2023, color="red", linestyle="--", alpha=0.7)
    plt.text(
        2023.5,
        counts.max() * 0.9,
        "Data incomplete\nbeyond this point",
        color="red",
        fontsize=9,
    )
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


def plot_votes_distribution(
    ratings_df: pd.DataFrame,
    output_dir: Path,
    title_basics: pd.DataFrame | None = None,
) -> None:
    """
    Plot votes distribution for show/movie titles only.

    If `title_basics` is provided, `tvEpisode` is filtered out using `titleType`
    so this chart is consistent with the other "top rated titles" charts.
    """
    if "numVotes" not in ratings_df.columns:
        return

    votes_source = ratings_df.copy()

    if title_basics is not None and {"tconst", "titleType"}.issubset(title_basics.columns) and {"tconst", "numVotes"}.issubset(votes_source.columns):
        # Filter out episodes by keeping only show/movie titleTypes.
        votes_source = votes_source.merge(
            title_basics[["tconst", "titleType"]],
            on="tconst",
            how="left",
        )
        votes_source["titleType"] = votes_source["titleType"].astype(str).str.strip()
        votes_source = votes_source[votes_source["titleType"].isin(STANDALONE_TITLE_TYPES)]

    votes = pd.to_numeric(votes_source["numVotes"], errors="coerce").dropna()
    votes = votes[votes > 0]
    if votes.empty:
        return
    log_bins = np.logspace(
        np.log10(votes.min()), np.log10(votes.max()), num=40
    )
    # Use log-spaced bins for `x` but keep y on a linear scale.
    # This avoids the confusing "log-log" effect where both axes are powers of 10.
    plt.figure(figsize=(10, 6))
    plt.hist(votes, bins=log_bins, log=False)
    plt.xscale("log")
    plt.title("Votes Distribution (Shows/Movies only)")
    plt.xlabel("Number of Votes (log scale)")
    plt.ylabel("Frequency (count)")
    plt.tight_layout()
    plt.savefig(output_dir / "votes_distribution.png", dpi=150)
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

    if title_ratings_path.exists():
        title_ratings = pd.read_csv(title_ratings_path, low_memory=False)
        plot_rating_distribution(title_ratings, output_dir)
        # Keep this chart consistent with other "titles" charts by filtering
        # out episode entries (tvEpisode).
        plot_votes_distribution(title_ratings, output_dir, title_basics=title_basics)

    if title_basics is not None and title_ratings is not None:
        merged = title_basics.merge(
            title_ratings[["tconst", "averageRating", "numVotes"]],
            on="tconst",
            how="inner",
        )
        plot_median_rating_by_title_type(merged, output_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="IMDb pipeline: clean datasets, run analysis, and generate visualizations."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=INPUT_DIR,
        help=f"Input directory containing raw CSV files (default: {INPUT_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help=f"Output directory for cleaned CSV files (default: {OUTPUT_DIR})",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=REPORT_PATH,
        help=f"Path to write JSON cleaning report (default: {REPORT_PATH})",
    )
    parser.add_argument(
        "--analysis-output-dir",
        type=Path,
        default=ANALYSIS_OUTPUT_DIR,
        help=f"Directory to write analysis outputs (default: {ANALYSIS_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--visualization-output-dir",
        type=Path,
        default=VISUALIZATION_OUTPUT_DIR,
        help=f"Directory to save chart images (default: {VISUALIZATION_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--skip-cleaning-analysis",
        action="store_true",
        help="Skip the post-cleaning dataset analysis step.",
    )
    parser.add_argument(
        "--skip-reports-analysis",
        action="store_true",
        help="Skip the reports analysis step.",
    )
    parser.add_argument(
        "--skip-visualization",
        action="store_true",
        help="Skip generating visualization charts.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Step 1: Clean raw datasets
    run_cleaning(args.input_dir, args.output_dir, args.report_path)

    # Step 2: Analyze cleaned datasets (dataset_summary.csv, analysis_report.json)
    if not args.skip_cleaning_analysis:
        run_cleaning_analysis(args.output_dir, args.analysis_output_dir)

    # Step 3: Run reports analysis (report_analysis.json, CSVs, bin files)
    if not args.skip_reports_analysis:
        run_reports_analysis(args.output_dir, args.analysis_output_dir)

    # Step 4: Generate visualization charts
    if not args.skip_visualization:
        run_visualization(args.output_dir, args.visualization_output_dir)