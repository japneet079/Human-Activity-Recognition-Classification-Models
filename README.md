# Human Activity Recognition & Classification Models

> **Evaluating multiple machine learning algorithms to classify human activity based on sensor data.**

## Overview
This repository contains a collection of Python scripts designed to train, evaluate, and compare various classification models on a Human Activity Recognition dataset. The primary objective is to accurately predict the `Activity` category of a user while excluding specific identifiers like the `subject` column[cite: 1, 2, 3, 4, 5]. 

## Tech Stack
**Python** | **pandas** | **NumPy** | **scikit-learn** | **XGBoost** | **matplotlib** | **seaborn**

## Models & Scripts
This project implements several different machine learning algorithms to compare performance:

*   **XGBoost (`xgboost_model.py`):** Utilizes `XGBClassifier` with 100 estimators and encodes the target variable using `LabelEncoder`[cite: 1].
*   **Decision Tree (`decision_tree.py`):** Trains a standard `DecisionTreeClassifier` directly on the dataset[cite: 2].
*   **Gradient Boosting (`gradient_boosting.py`):** Employs `GradientBoostingClassifier` with 100 estimators for sequential ensemble learning[cite: 3].
*   **K-Nearest Neighbors (`knn_model.py`):** Uses `KNeighborsClassifier` configured with 5 neighbors. This script also scales the features using `StandardScaler` for distance-based uniformity[cite: 4].
*   **Random Forest (`random_forest.py`):** Implements an ensemble `RandomForestClassifier` to predict the activity categories[cite: 5].

*Note: The project reads training and testing data from local files, specifically referencing `train.csv`[cite: 1, 2, 3, 5] and `test.csv`[cite: 4].*

## Data Utility Scripts
The repository also includes a data processing script for an independent market basket analysis project:
*   **Data Reduction (`data_reducer.py`):** Reads order data, samples exactly 166 rows per `Category` to create a perfectly balanced subset, shuffles the results, and outputs them to `reduced_orders_data.csv`[cite: 6]. 

## Evaluation Metrics
Each classification script evaluates the model against a 20% test split[cite: 1, 2, 3, 4, 5] and outputs:
*   **Overall Accuracy Score**
*   **Detailed Classification Report** (Precision, Recall, F1-Score)
*   **Data Visualizations:** Scripts contain commented-out `matplotlib` code designed to plot Actual vs. Predicted sensor data for the first 40 rows to visualize model performance visually[cite: 1, 2, 3, 4, 5].

## How to Run
1. Clone this repository to your local machine.
2. Ensure you have the required datasets (`train.csv` and `test.csv`) in your working directory.
3. Install the necessary dependencies (e.g., `pip install pandas scikit-learn xgboost matplotlib seaborn numpy`).
4. Run any of the individual model scripts via the terminal:
   ```bash
   python xgboost_model.py
