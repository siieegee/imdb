# Movie Recommendation AI/ML Sample

This folder contains a standalone AI/ML implementation for movie recommendation using:

- Python
- scikit-learn
- TensorFlow/Keras

It is designed for the Kaggle dataset:
[IMDB Dataset](https://www.kaggle.com/datasets/ahmedosamamath/imdb-dataset/)

## What this sample does

1. Loads IMDB title and ratings data.
2. Builds movie features from:
   - runtime
   - release year
   - vote count
   - genres (multi-hot encoded)
3. Trains two ranking models to score movies:
   - Random Forest Regressor (scikit-learn)
   - Neural Network Regressor (TensorFlow/Keras, optional fallback if unavailable)
4. Produces:
   - model metrics JSON
   - top-N recommendations JSON

## Folder structure

- `main.py` - run full pipeline
- `recommender.py` - model training and recommendation logic
- `data_pipeline.py` - loading and feature engineering
- `config.py` - runtime configuration
- `requirements.txt` - dependencies

## Dataset setup

Place Kaggle CSV files in one of these folders:

- `../dataset/imdb_datasets`
- `../dataset/cleaned_datasets`

Expected files:

- `title.basics.csv`
- `title.ratings.csv`

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

## Outputs

Files are saved to:

- `../outputs/ml_models/ai_ml_sample_metrics.json`
- `../outputs/ml_models/ai_ml_sample_recommendations.json`

## Notes

- If TensorFlow is not installed, the script falls back to `MLPRegressor` from scikit-learn.
- This is a sample baseline recommender; for production, add user-item interactions and hybrid ranking.
