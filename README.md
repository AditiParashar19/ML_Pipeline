# ⚡ ML Pipeline Studio

An interactive end-to-end Machine Learning pipeline builder built using Streamlit.  
This tool allows users to seamlessly go from data upload → preprocessing → model training → evaluation → prediction, all in a guided UI.

---

## 🚀 Features

- 🧠 Problem Type Selection  
  - Classification & Regression support  

- 📂 Data Input  
  - Upload your own CSV dataset  
  - Use built-in sample datasets (Iris, Breast Cancer, etc.)  

- 📊 Exploratory Data Analysis (EDA)  
  - Feature distributions  
  - Correlation matrix  
  - Box plots & target analysis  

- 🛠️ Data Preprocessing  
  - Missing value imputation (Mean, Median, Mode, Zero)  
  - Outlier detection (IQR, Isolation Forest, DBSCAN, OPTICS)  

- 🎯 Feature Selection  
  - Variance Threshold  
  - Correlation-based selection  
  - Mutual Information (Information Gain)  

- 🔀 Train-Test Split  
  - Custom split ratio  
  - Stratified sampling (for classification)  

- 🤖 Model Selection  
  - Logistic Regression  
  - SVM (Classifier & Regressor)  
  - Random Forest  
  - KNN / Ridge / Linear Regression  

- 📈 Model Training  
  - K-Fold Cross Validation  
  - Performance visualization  

- 📊 Evaluation Metrics  
  - Classification: Accuracy, Precision, Recall, F1-score  
  - Regression: RMSE, MAE, R²  

- 🎨 Interactive UI  
  - Dark/Light theme toggle  
  - Step-by-step guided pipeline  

---

## 🏗️ Project Structure
ML-Pipeline-Studio
│── streamlit_app.py
│── requirements.txt
│── README.md
