import argparse
import json
from pathlib import Path

import pandas as pd


DEFAULT_INPUT_DIR = Path("dataset/cleaned_datasets")
DEFAULT_OUTPUT_DIR = Path("outputs/analysis")


def safe_describe(df: pd.DataFrame) -> dict:
    numeric = df.select_dtypes(include=["number"])
    if numeric.empty:
        return {}
    desc = numeric.describe().transpose()
    return {
        column: {k: float(v) for k, v in stats.to_dict().items()}
        for column, stats in desc.iterrows()
    }


def analyze_file(file_path: Path) -> dict:
    df = pd.read_csv(file_path, low_memory=False)
    analysis = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": list(df.columns),
        "missing_values": {k: int(v) for k, v in df.isna().sum().to_dict().items()},
        "duplicate_rows": int(df.duplicated().sum()),
        "numeric_summary": safe_describe(df),
    }
    return analysis


def run_analysis(input_dir: Path, output_dir: Path) -> None:
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

    # Save a flat summary table for quick inspection.
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
    parser = argparse.ArgumentParser(description="Analyze cleaned IMDb datasets.")
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
        help=f"Directory for analysis outputs (default: {DEFAULT_OUTPUT_DIR})",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_analysis(args.input_dir, args.output_dir)
