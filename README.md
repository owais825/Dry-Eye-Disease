Dry Eye Disease Prediction
Introduction

Dry Eye Disease (DED) is a common ocular surface condition in which the tear film becomes unstable, leading to eye discomfort and visual disturbance
ncbi.nlm.nih.gov
my.clevelandclinic.org
. Symptoms can include burning, itching, redness, and blurred vision as the eyes fail to produce sufficient lubrication
my.clevelandclinic.org
ncbi.nlm.nih.gov
. DED is a leading reason for eye care visits and can significantly impact quality of life. This project applies machine learning to patient lifestyle and health data to predict the presence
of dry eye disease. The goal is to identify key factors associated with DED and build predictive models that could help screen at-risk individuals.

Dataset

The data are provided in the file Dry_Eye_Dataset.csv (loaded in the notebook). It contains patient records with features including demographics and health/lifestyle factors.
Example features include Gender, Age, Sleep duration/quality, Stress level, Blood pressure, Heart rate, Daily steps, Physical activity, and Lifestyle factors (caffeine use, smoking, alcohol, etc.),
as well as ocular symptoms (eye strain, redness, itchiness). The target column is “Dry Eye Disease” (e.g. Yes/No diagnosis). In the notebook, the data are examined (no duplicate or missing values found)
and preprocessed: categorical variables (like blood pressure category and sleep category) are one-hot encoded, and selected numerical columns are standardized. Multicollinearity is checked (e.g. by VIF),
and highly correlated variables (such as diastolic blood pressure) are dropped before modeling.

Methodology

The analysis proceeds as follows:

Train/Test Split: The processed data are split into training (70%) and test (30%) sets (stratified by the target).

Preprocessing: Numeric features (e.g. age, sleep hours) are scaled with StandardScaler. Categorical features (BP category, sleep category, screen time category) are converted to dummy indicators. Any constant or collinear features are removed.

Feature Selection: Recursive Feature Elimination (RFE) is applied to select the most relevant predictors for each model. This yields a subset of features (e.g. gender, age, sleep metrics, stress, vital signs, and symptom indicators) used in final models.

Models: Various classification algorithms are trained and compared, including Logistic Regression, Decision Tree (and tuned Decision Tree), Random Forest, AdaBoost, Gradient Boosting, XGBoost, LightGBM, and an ensemble (Voting classifier) combining multiple models.

Hyperparameter Tuning: Grid search (or manual tuning) is used for key models (e.g. tuning the Decision Tree entropy criterion and depth, and the AdaBoost base estimator).

Evaluation: Models are evaluated on accuracy, precision, recall, and F1-score. The notebook computes classification reports and plots confusion matrices for both train and test sets. ROC-AUC and precision-recall curves are also available, focusing on detecting positive (Dry Eye) cases.

Usage

To reproduce the analysis, ensure the dataset file (Dry_Eye_Dataset.csv) is in the working directory and run the notebook in a Python environment. For example:

# Install required Python libraries (if not already installed)
pip install numpy pandas matplotlib seaborn scikit-learn xgboost lightgbm shap

# Launch Jupyter Notebook and open the analysis
jupyter notebook
# In the browser interface, open Dry_eye_disease.ipynb and run all cells


Alternatively, you can execute the notebook cells sequentially using any compatible environment (e.g., JupyterLab or VSCode). The notebook is fully self-contained and will load the data, perform EDA, train models, and output results.

Dependencies

Key Python libraries used in this project include:

numpy, pandas – for data manipulation

matplotlib, seaborn – for plotting

scikit-learn – for preprocessing, modeling (LogisticRegression, DecisionTree, RandomForest, AdaBoost, GradientBoosting, SVM, metrics, etc.)

XGBoost – for XGBoostClassifier

LightGBM – for LGBMClassifier

statsmodels – (imported for statistical tests)

SHAP – for model interpretability (SHAP values)

warnings – to suppress non-critical warnings

A requirements.txt can be generated from the imports above, or install packages as needed.

Results

The best-performing model in this analysis was a Random Forest classifier, which achieved about 70.4% accuracy on the test set. This model had high sensitivity (recall ≈ 93.6%) for detecting dry eye cases, though its precision was moderate (≈ 71.2%) – indicating it predicted many positive cases (catching most true cases) at the cost of some false positives. The weighted F1-score was around 0.66. Other models (e.g. tuned Decision Tree, XGBoost, or the Voting ensemble) achieved similar overall accuracy (typically in the high 60% to ~70% range). These results suggest that the available features carry predictive signal but also highlight class imbalance (many more non-DED cases) and room for improvement.

Conclusion

This study demonstrates a pipeline for predicting Dry Eye Disease from survey/health data. The models show moderate predictive ability, with the Random Forest capturing most true dry eye cases but with some false alarms. These findings imply that lifestyle and health factors (sleep quality, blood pressure, screen time, etc.) are indeed correlated with DED risk, but additional data or features may be needed for more accurate predictions. In practice, such models could help flag at-risk patients for further eye examinations or preventive measures. Further work could refine feature engineering, address class imbalance, or incorporate clinical measurements to improve accuracy.
