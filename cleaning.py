import argparse
import json
from pathlib import Path

import pandas as pd


# Default locations for raw inputs, cleaned outputs, and the run report.
INPUT_DIR = Path("dataset/imdb_datasets")
OUTPUT_DIR = Path("dataset/cleaned_datasets")
REPORT_PATH = Path("dataset/cleaning_report.json")
ANALYSIS_OUTPUT_DIR = Path("outputs/analysis")


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
    # Shared baseline cleaning for every dataset.
    df = normalize_nulls(df)
    df = clean_common_strings(df)
    df = df.drop_duplicates()

    # Dataset-specific rules keep behavior explicit and easy to extend.
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


def run_cleaning(input_dir: Path, output_dir: Path, report_path: Path) -> None:
    """Run cleaning for every CSV file in `input_dir`."""
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    report = {}
    csv_files = sorted(input_dir.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in: {input_dir}")

    for file_path in csv_files:
        # Read raw data, clean it, then persist both output and run metrics.
        df_raw = pd.read_csv(file_path, low_memory=False)
        df_clean = clean_dataset(df_raw.copy(), file_path.name)
        out_file = output_dir / file_path.name
        df_clean.to_csv(out_file, index=False)
        report[file_path.name] = build_report_entry(df_raw, df_clean)

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


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


def run_analysis(input_dir: Path, output_dir: Path) -> None:
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


def parse_args() -> argparse.Namespace:
    """Parse command-line options for input/output/report paths."""
    parser = argparse.ArgumentParser(description="Clean IMDb CSV datasets.")
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
        "--skip-analysis",
        action="store_true",
        help="Skip running post-cleaning analysis step.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_cleaning(args.input_dir, args.output_dir, args.report_path)
    if not args.skip_analysis:
        # Analyze the freshly cleaned datasets.
        run_analysis(args.output_dir, args.analysis_output_dir)
