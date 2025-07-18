# NYC Taxi Fare and Tip Prediction

This project focuses on building regression models to predict taxi **fare amount** and **tip amount** based on ride features. The dataset is derived from New York City taxi ride data.

## Project Overview

We aim to:
- Predict **fare amount** and **tip amount** using historical taxi ride features.
- Apply rigorous **data preprocessing**, **feature engineering**, and **hyperparameter tuning**.
- Evaluate models based on **Root Mean Squared Error (RMSE)**.
- Analyze **feature importance** to interpret model decisions.

---

## Execution Pipeline

### STEP 1: Load and Split Data
- Load `taxi-train.csv` for training, validation, and testing.
- Split: **60% train**, **20% validation**, **20% test**.
- Load `taxi-test.csv` for final prediction. This set is **kept untouched** during training and tuning.

---

### STEP 2.1: Data Exploration
(Only based on `train_df` to avoid leakage)
- General statistics (mean, std, min, max).
- Count of nulls, distincts, and zeros.
- Clustering map of pickup/drop-off points (3 clusters identified).
- Geographic visualization using binning for long/lat.

---

### STEP 2.2: Data Preprocessing
- Use **KMeans clustering** on coordinates (train only).
- Extract date-time features: `pickup_hour`, `pickup_day`, `trip_duration`, `trip_speed`, etc.
- Bucketize rare categorical values (based on train frequency).
- Remove invalid rows (e.g., zero distance, negative fare).
- Drop unused or derived columns (e.g., `tip_paid`).

---

### STEP 3: Hyperparameter Tuning
- Define parameter grid for:
  - `RandomForestRegressor`
  - `AdaBoostRegressor`
  - `GradientBoostingRegressor`
- Use `RandomizedSearchCV` with sampling strategy (100,000 rows) and 3-fold CV.
- Evaluate top 2 candidates on full data.

---

### STEP 4: Final Evaluation & Prediction
- Evaluate top models on **test set**.
- Print **best parameters** and **RMSE** scores.

#### Fare Amount RMSE:
| Model             | RMSE  | Best Params |
|------------------|-------|-------------|
| GradientBoost    | 1.272 | `n_estimators=100, max_depth=5, learning_rate=0.1` |
| RandomForest     | 1.338 | `n_estimators=200, max_depth=10` |
| AdaBoost         | 1.729 | `n_estimators=200, learning_rate=0.01` |

#### Tip Amount RMSE:
| Model             | RMSE  | Best Params |
|------------------|-------|-------------|
| GradientBoost    | 0.624 | `n_estimators=50, max_depth=5, learning_rate=0.1` |
| RandomForest     | 0.634 | `n_estimators=200, max_depth=10` |
| AdaBoost         | 0.649 | `n_estimators=100, learning_rate=0.01` |

#### Sample Prediction Output:
- **Fare Amount**: `[7.05, 15.15, 7.49, 9.05, 8.50, ...]`
- **Tip Amount**: `[1.45, 2.56, 0.00, 1.78, 0.00, ...]`

---

### STEP 5: Feature Importance Analysis

#### For `fare_amount`:
- `trip_duration` and `trip_distance` contribute >80%.
- Long/lat info (pickup/dropoff) adds up to 10%.

#### For `tip_amount`:
- `payment_type_CRD` dominates (~0.6 importance).
- Distance and duration consistently rank top 5.

---

## Liberaries Used
- Python, Scikit-learn, Pandas, Matplotlib, Seaborn

---

## Repository Structure

Taxi-Fare-Tip-Prediction  
├── taxi_tip_pred_main_script.ipynb # Main notebook  
├── taxi-train.csv # Training dataset  
├── taxi-test.csv # Final prediction dataset  
└── README.md # This file  
