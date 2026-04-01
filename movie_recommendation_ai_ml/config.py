"""Configuration for the standalone AI/ML movie recommendation sample."""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

DATA_DIR_CANDIDATES = [
    PROJECT_ROOT / "dataset" / "imdb_datasets",
    PROJECT_ROOT / "dataset" / "cleaned_datasets",
]

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "ml_models"

RANDOM_SEED = 42
SAMPLE_SIZE = 10000
MIN_VOTES = 100
TOP_N = 20
