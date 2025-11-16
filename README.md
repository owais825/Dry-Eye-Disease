Dry Eye Disease Prediction
📌 Introduction

Dry Eye Disease (DED) is a common ocular surface condition in which the tear film becomes unstable, leading to eye discomfort and visual disturbances.
Symptoms can include burning, itching, redness, and blurred vision as the eyes fail to produce sufficient lubrication.
DED is a leading reason for eye care visits and can significantly impact quality of life.

This project applies machine learning to patient lifestyle and health data to predict the presence of dry eye disease.
The goal is to identify key factors associated with DED and build predictive models to help screen at-risk individuals.

📁 Dataset

The dataset used is Dry_Eye_Dataset.csv, containing patient records with features such as:

Demographics: Gender, Age

Health & Lifestyle: Sleep duration/quality, Stress level, Blood Pressure, Heart Rate

Activity: Daily steps, Physical activity

Habits: Caffeine use, Smoking, Alcohol consumption

Ocular symptoms: Eye strain, Redness, Itchiness

Target variable: Dry Eye Disease (Yes/No)

Preprocessing Steps

No missing or duplicate values found

One-hot encoding applied to categorical variables (e.g., BP category, sleep category)

Selected numerical columns standardized

Multicollinearity checked using VIF; highly correlated features (e.g., Diastolic BP) removed

🧠 Methodology
1. Train/Test Split

Data split into 70% training and 30% testing

Stratified by the target variable

2. Preprocessing

Numeric features scaled using StandardScaler

Categorical features converted to dummies

Removed constant or highly collinear features

3. Feature Selection

Used Recursive Feature Elimination (RFE)

Identified the most relevant predictors, including demographics, sleep metrics, stress, vital signs, and symptoms

4. Models Trained

Logistic Regression

Decision Tree (and tuned Decision Tree)

Random Forest

AdaBoost

Gradient Boosting

XGBoost

LightGBM

Voting Classifier (ensemble)

5. Hyperparameter Tuning

Grid search or manual tuning for Decision Tree, AdaBoost, etc.

Adjusted depth, entropy, and base estimators

6. Evaluation Metrics

Accuracy

Precision

Recall

F1-score

Confusion matrices

ROC-AUC and Precision-Recall curves

▶️ Usage

Ensure the dataset (Dry_Eye_Dataset.csv) is in the working directory, then install dependencies:

# Install required Python libraries
pip install numpy pandas matplotlib seaborn scikit-learn xgboost lightgbm shap


Launch Jupyter Notebook:

jupyter notebook


Then open Dry_eye_disease.ipynb and run all cells.

📦 Dependencies

Key Python libraries:

numpy, pandas — Data manipulation

matplotlib, seaborn — Visualization

scikit-learn — Preprocessing & ML models

XGBoost — XGBoostClassifier

LightGBM — LGBMClassifier

statsmodels — Statistical analysis

SHAP — Model explainability

A requirements.txt can be generated from these packages if needed.

📊 Results

The Random Forest classifier performed best, achieving:

Accuracy: ~70.4%

Recall (DED cases): ~93.6%

Precision: ~71.2%

Weighted F1-score: ~0.66

Other models (e.g., tuned Decision Tree, XGBoost, Voting ensemble) achieved similar accuracy (upper 60% to ~70%).

These results indicate:

Features carry meaningful predictive signal

Class imbalance affects precision

Further improvements are possible with more data or additional features

✅ Conclusion

This project demonstrates a complete machine-learning pipeline for predicting Dry Eye Disease from health and lifestyle survey data.

Key insights:

Random Forest performed best, capturing most true DED cases

Moderate overall accuracy due to class imbalance

Lifestyle factors (sleep, blood pressure, screen time) correlate with DED risk

Additional clinical data could improve precision and reduce false positives

Future improvements may include advanced feature engineering, oversampling techniques, integration of clinical measures, or deep learning models.
