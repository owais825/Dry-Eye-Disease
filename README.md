<div style="font-family: Arial, sans-serif; padding: 20px; line-height: 1.6;">

<h1 style="color:#2c3e50; text-align:center;">🔵 Dry Eye Disease Prediction</h1>

<hr style="border:1px solid #dcdcdc;">

<h2 style="color:#34495e;">📌 Introduction</h2>
<p>
Dry Eye Disease (DED) is a common ocular surface condition characterized by an unstable tear film that leads to discomfort and visual disturbances.
Symptoms include burning, itching, redness, and blurred vision. DED is a major reason for eye care visits and can significantly impact quality of life.
</p>

<p>
This project applies <strong>machine learning</strong> to health and lifestyle data to predict whether a patient has Dry Eye Disease, with the goal
of identifying at-risk individuals for screening or preventive care.
</p>

<hr style="border:1px solid #dcdcdc; margin-top:30px;">

<h2 style="color:#34495e;">📁 Dataset</h2>
<p>
The dataset <strong>Dry_Eye_Dataset.csv</strong> contains health, lifestyle, and ocular symptom data.  
Key features include:
</p>

<ul style="margin-left:20px;">
  <li><strong>Demographics:</strong> Age, Gender</li>
  <li><strong>Lifestyle:</strong> Sleep quality, stress, caffeine intake, smoking, alcohol</li>
  <li><strong>Vitals:</strong> Heart rate, blood pressure</li>
  <li><strong>Activity:</strong> Daily steps, physical activity</li>
  <li><strong>Ocular symptoms:</strong> Redness, itchiness, eye strain</li>
  <li><strong>Target:</strong> Dry Eye Disease (Yes/No)</li>
</ul>

<p>The data was cleaned, one-hot encoded, standardized, and checked for multicollinearity (e.g., VIF scores).</p>

<hr style="border:1px solid #dcdcdc; margin-top:30px;">

<h2 style="color:#34495e;">🧠 Methodology</h2>

<h3 style="color:#2c3e50;">1. Train/Test Split</h3>
<p>Data split into <strong>70% train</strong> and <strong>30% test</strong> sets, stratified by diagnosis.</p>

<h3 style="color:#2c3e50;">2. Preprocessing</h3>
<ul>
  <li>StandardScaler applied to numeric features</li>
  <li>Categorical variables encoded using one-hot encoding</li>
  <li>Highly correlated variables removed</li>
</ul>

<h3 style="color:#2c3e50;">3. Feature Selection</h3>
<p><strong>Recursive Feature Elimination (RFE)</strong> used to identify top predictors.</p>

<h3 style="color:#2c3e50;">4. Models Used</h3>
<ul>
  <li>Logistic Regression</li>
  <li>Decision Tree (+ tuned version)</li>
  <li>Random Forest</li>
  <li>AdaBoost</li>
  <li>Gradient Boosting</li>
  <li>XGBoost</li>
  <li>LightGBM</li>
  <li>Voting Classifier</li>
</ul>

<h3 style="color:#2c3e50;">5. Hyperparameter Tuning</h3>
<ul>
  <li>Grid Search for Decision Tree, AdaBoost, and others</li>
</ul>

<h3 style="color:#2c3e50;">6. Evaluation Metrics</h3>
<ul>
  <li>Accuracy</li>
  <li>Precision</li>
  <li>Recall</li>
  <li>F1 Score</li>
  <li>Confusion Matrix</li>
  <li>ROC-AUC, Precision-Recall Curves</li>
</ul>

<hr style="border:1px solid #dcdcdc; margin-top:30px;">

<h2 style="color:#34495e;">▶️ Usage</h2>

<pre style="background:#f7f7f7; padding:15px; border-radius:5px; border:1px solid #ddd; font-size:14px;">
# Install dependencies
pip install numpy pandas matplotlib seaborn scikit-learn xgboost lightgbm shap

# Launch Jupyter Notebook
jupyter notebook
</pre>

<p>Open <strong>Dry_eye_disease.ipynb</strong> and run all cells.</p>

<hr style="border:1px solid #dcdcdc; margin-top:30px;">

<h2 style="color:#34495e;">📦 Dependencies</h2>
<ul>
  <li>numpy, pandas</li>
  <li>matplotlib, seaborn</li>
  <li>scikit-learn</li>
  <li>XGBoost, LightGBM</li>
  <li>statsmodels</li>
  <li>SHAP</li>
</ul>

<hr style="border:1px solid #dcdcdc; margin-top:30px;">

<h2 style="color:#34495e;">📊 Results</h2>

<div style="background:#ecf0f1; padding:15px; border-left:4px solid #3498db; margin-bottom:20px;">
  <p><strong>Best Model: SVM</strong></p>
  <ul>
    <li><strong>Accuracy:</strong> ~70.8%</li>
    <li><strong>Recall (DED cases):</strong> ~96.7%</li>
    <li><strong>Precision:</strong> ~70%</li>
    <li><strong>F1 Score:</strong> ~0.81</li>
  </ul>
</div>

<p>
Other models performed similarly (high 60% to ~70% accuracy).  
The dataset shows moderate signal but noticeable class imbalance.
</p>

<hr style="border:1px solid #dcdcdc; margin-top:30px;">

<h2 style="color:#34495e;">✅ Conclusion</h2>

<p>
This project builds a machine-learning pipeline to predict Dry Eye Disease using health and lifestyle data.  
Random Forest performed best, detecting most true dry eye cases but with moderate precision.  
Lifestyle factors (sleep, stress, BP, screen time) show meaningful correlation with DED risk.
</p>

<p>
Future improvements could include:
</p>
<ul>
  <li>More clinical data</li>
  <li>Handling class imbalance</li>
  <li>Advanced feature engineering</li>
  <li>Deep learning models</li>
</ul>

</div>
