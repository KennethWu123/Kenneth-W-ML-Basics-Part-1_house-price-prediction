
# House Price Prediction with Realistic Dataset (King County, WA)
# -----------------------------------------------------------------------------
# Data source:
#   Original: Kaggle – "House Sales in King County, USA" (kc_house_data.csv)
#   https://www.kaggle.com/datasets/shivachandel/kc-house-data  # Kaggle landing page
#   Mirror (convenience raw CSV on GitHub; use if you don't want to sign in to Kaggle):
#   https://raw.githubusercontent.com/PashaArrighi/King_County_House_Sales_Dashboard/main/kc_house_data.csv
#   NOTE: See your course announcements on how to access datasets. If external download
#   is restricted, place a local copy of `kc_house_data.csv` in the project folder.
# -----------------------------------------------------------------------------
# This script:
#   1) Loads 21,613 real home sales with price, sqft, and zipcode (100+ records ✔).
#   2) Uses OneHotEncoder(drop='first') for zipcode; LinearRegression for modeling.
#   3) Evaluates with train/test split (R^2, MAE, RMSE) and optional 5-fold CV.
#   4) Predicts price for a 2000 sq ft home in a chosen zipcode (falls back to the most
#      common zipcode in the dataset if the chosen one is unseen).
#   5) Prints an interpretable coefficient table showing per-sq-ft effect and location offsets.
#   6) Implements small improvements vs. the starter: clean pipeline, identifiability via
#      drop='first', proper evaluation metrics, robust handling of unknown zips.
# -----------------------------------------------------------------------------

import sys
import io
import math
import warnings
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings('ignore', category=FutureWarning)

# -------------------------------
# Config
# -------------------------------
RAW_URLS = [
    # GitHub raw mirror (no auth)
    'https://raw.githubusercontent.com/PashaArrighi/King_County_House_Sales_Dashboard/main/kc_house_data.csv',
]
LOCAL_FALLBACK = 'kc_house_data.csv'  # place locally if downloads are blocked
CHOSEN_ZIP_EXAMPLE = '98101'  # Downtown Seattle; will fallback to most frequent if unseen
TEST_SIZE = 0.2
RANDOM_STATE = 42
DO_CV = True
CV_FOLDS = 5

# -------------------------------
# Load data helper
# -------------------------------
def load_kc_data() -> pd.DataFrame:
    import os
    import pathlib

    # Try local file first (useful if running offline)
    if os.path.exists(LOCAL_FALLBACK):
        return pd.read_csv(LOCAL_FALLBACK)

    # Try remote mirrors
    for url in RAW_URLS:
        try:
            df = pd.read_csv(url)
            return df
        except Exception as e:
            print(f"Warning: failed to read {url}: {e}")
            continue
    raise FileNotFoundError(
        "Could not load kc_house_data.csv. Place it locally or update RAW_URLS.")

# -------------------------------
# Main
# -------------------------------
def main():
    # 1) Load
    df = load_kc_data()

    # Keep only fields we need
    # Map to the starter-code naming convention
    df = df.rename(columns={'sqft_living': 'square_footage', 'zipcode': 'location'})
    cols_needed = ['price', 'square_footage', 'location']
    df = df[cols_needed].dropna()

    # Ensure location is string/categorical
    df['location'] = df['location'].astype(str)

    # Sanity filter: non-zero/positive square footage and price
    df = df[(df['square_footage'] > 100) & (df['price'] > 1000)]

    # 2) Train/test split
    X = df[['square_footage', 'location']]
    y = df['price']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    # 3) Preprocess + model pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('location', OneHotEncoder(drop='first', handle_unknown='ignore',
                                       sparse_output=False), ['location'])
        ],
        remainder='passthrough'  # keeps square_footage as the last column
    )

    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])

    # 4) Fit
    model.fit(X_train, y_train)

    # 5) Evaluate on test set
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)

    # Optional cross-validation (on the full dataset)
    cv_msg = ''
    if DO_CV:
        kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        cv_scores = cross_val_score(model, X, y, cv=kf, scoring='r2')
        cv_msg = f"CV R^2 (mean ± std over {CV_FOLDS} folds): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}"

    # 6) Coefficients (interpretable)
    #    With drop='first', the omitted (baseline) location is the first category in sorted order.
    ohe = model.named_steps['preprocessor'].named_transformers_['location']
    location_feature_names = ohe.get_feature_names_out(['location']).tolist()
    feature_names = location_feature_names + ['square_footage']

    lr = model.named_steps['regressor']
    coefs = lr.coef_
    intercept = lr.intercept_

    coef_df = pd.DataFrame({
        'feature': feature_names,
        'coefficient': coefs
    }).sort_values(by='coefficient', key=lambda s: s.abs(), ascending=False).reset_index(drop=True)

    # 7) Predict price for a 2000 sq ft house in a zipcode
    chosen_zip = CHOSEN_ZIP_EXAMPLE
    # Fallback to the most frequent zipcode in the training data if the chosen one is unseen
    train_zips = X_train['location'].astype(str)
    if chosen_zip not in set(train_zips.unique()):
        chosen_zip = train_zips.mode().iloc[0]

    new_house = pd.DataFrame({'square_footage': [2000], 'location': [chosen_zip]})
    pred_price = float(model.predict(new_house)[0])

    # 8) Print outputs
    print("
===== DATA SUMMARY =====")
    print(df.describe(include='all'))

    print("
===== EVALUATION =====")
    print(f"Test R^2:  {r2:.3f}")
    print(f"Test MAE:  ${mae:,.0f}")
    print(f"Test RMSE: ${rmse:,.0f}")
    if cv_msg:
        print(cv_msg)

    print("
===== INTERPRETATION =====")
    print(f"Intercept: ${intercept:,.2f}")
    per_sqft = coef_df.loc[coef_df['feature'] == 'square_footage', 'coefficient'].values
    if len(per_sqft):
        print(f"Per-square-foot effect (average across zips): ${per_sqft[0]:.2f} per sq ft")
    print("
Top coefficients by absolute magnitude:")
    print(coef_df.head(15).to_string(index=False))

    print("
===== EXAMPLE PREDICTION =====")
    print(f"Predicted price for a 2000 sq ft house in zipcode {chosen_zip}: ${pred_price:,.2f}")

    print("
NOTE")
    print("- Location coefficients are relative to the baseline zipcode omitted by OneHotEncoder(drop='first').")
    print("- Coefficients reflect linear additive effects. In reality, price-per-sq-ft may vary by zip;")
    print("  to capture that, you can add interaction terms between square_footage and location dummies.")

if __name__ == '__main__':
    main()
