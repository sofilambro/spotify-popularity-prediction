# Spotify Song Popularity Prediction (Audio Features + Genre Momentum)

This project studies what drives Spotify song popularity and builds models to predict it from a Kaggle dataset sourced from Spotify’s API. The dataset contains 4,831 tracks and 30 features spanning audio descriptors (e.g., energy, danceability), genre/subgenre information, and time-related variables. Two subsets are provided: "high-popularity" (popularity ≥ 68) and "low-popularity" (popularity < 68). 

## Questions
1. What factors determine how popular a song becomes?
2. Which audio or genre-related features are most predictive of a hit?

## Methods Overview
### Popularity as a continuous target (Regression)
After cleaning, scaling, and exploratory analysis, we compared three supervised models for predicting the continuous popularity score:
- Linear Regression
- k-Nearest Neighbors (k = 5)
- Random Forest Regressor

Models were evaluated on the same train–test split using RMSE, MAE, and R². Random Forest achieved the best performance and was selected as the base model (RMSE ≈ 13.9).

### Hyperparameter tuning
We used GridSearchCV with 3-fold cross-validation to tune:
- number of trees: {50, 100, 200}
- maximum depth: {None, 10, 20}

The best configuration was 200 trees with unlimited depth. On the test set, the tuned model achieved RMSE ≈ 13.86, meaning predictions are typically within ~14 popularity points of the true score.

### Interpretability (Classification)
To interpret which variables push a song toward high popularity, we also trained a logistic regression classifier. Coefficients suggest:
- negative effects: instrumentalness, release year
- positive effects: loudness, energy, danceability

These results align with the idea that highly popular songs tend to be vocal, energetic, and professionally produced.

## Genre Momentum Subanalysis (Historical Trend Features)
A second analysis tests the hypothesis that popularity depends strongly on the historical momentum of a genre/subgenre rather than only on a song’s individual attributes.

We computed 5-year-window log-odds hit rates for each genre and subgenre, merged them back into the dataset, and trained classification models. Random Forest achieved strong performance (AUC ≈ 0.955). Feature importance showed that historical momentum dominates prediction:
- logodds_sub_5yr (strongest)
- logodds_genre_5yr (second)

## Results Summary
- Random Forest is the best overall predictor for popularity.
- Audio features matter (energy, danceability, loudness), but the most powerful predictors come from long-term genre/subgenre historical momentum.
- Limitations include missing non-audio drivers (marketing, playlisting) and the fact that popularity is a snapshot rather than a full time series.

## Repository Structure (suggested)
- `notebooks/` : Jupyter notebooks (EDA, modeling, tuning, momentum features)
- `data/` : raw and processed data (or instructions to download if too large)
- `figures/` : exported plots used in the report
- `src/` : reusable Python modules (cleaning, feature engineering, training)
- `requirements.txt` : dependencies
- `README.md` : this file

## How to Run
1. Clone the repository
2. Create a virtual environment
3. Install dependencies
4. Run notebooks in `notebooks/`

## Dataset
The dataset is from Kaggle and is derived from Spotify’s API. If the data cannot be redistributed, this repo provides instructions to download it directly from Kaggle.

## Authors
(Insert names / contributions if you want)
