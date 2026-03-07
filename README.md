# House Price Prediction Assignment

This repository contains my improved script using a real dataset (21,613 rows).

## Files
- `house_price_kc_linear.py` — Main solution script

## Dataset Source
- Kaggle: King County House Sales dataset (kc_house_data)
- GitHub raw mirror used inside the script

## How to Run
In Google Colab:

!wget -O house_price_kc_linear.py "https://raw.githubusercontent.com/<yourusername>/house-price-prediction/main/house_price_kc_linear.py"
%run house_price_kc_linear.py

Part 1 — House Price Prediction (Linear Regression)
1) Dataset & Setup

Dataset: [Name the dataset + link or file path] (≥ 100 rows).
Source citation: [URL or “synthetic dataset generated for coursework”] (include this same citation in your code comments).
Features used:

price (target)
square_footage (numeric)
location (categorical: e.g., Downtown, Suburb, Rural)


Preprocessing: location encoded with OneHotEncoder so the linear model can use categorical location signals.
