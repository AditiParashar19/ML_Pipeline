# ⚡ ML Pipeline Studio

> An interactive, GUI-based end-to-end Machine Learning pipeline builder powered by Streamlit.

ML Pipeline Studio provides a **visual (GUI-driven) interface** to build, train, and evaluate machine learning models without writing extensive code. It integrates **Machine Learning, Data Visualization, and UI/UX design** into a single streamlined application.

---

## 🚀 Key Highlights

- 🖥️ Fully interactive **Graphical User Interface (GUI)**  
- 🔄 End-to-end ML workflow with step-by-step navigation  
- 🧠 Supports **Classification** and **Regression**  
- 📊 Rich **interactive visualizations** using Plotly  
- 🛠️ Built-in preprocessing & feature engineering tools  
- 🤖 Multiple ML models with hyperparameter tuning  
- 📈 K-Fold Cross Validation  
- 🎨 Modern UI with **Dark/Light theme toggle**

---

## 🧠 Skills & Technologies

### 💻 Programming & Machine Learning
- Python  
- Scikit-learn (Model Training, Evaluation, Optimization)

### 📊 Data Analysis & Visualization
- Pandas, NumPy  
- Plotly (Interactive Dashboards & Graphs)

### 🖥️ GUI / Frontend Development
- Streamlit (Interactive Web Applications)  
- GUI-based ML Pipeline Design  
- Dashboard Development  
- UI/UX Design (Dark/Light Theme, Responsive Layouts)  
- Data Visualization Interfaces  

### ⚙️ Core ML Concepts
- Data Preprocessing (Missing Values, Outliers)  
- Feature Selection & Engineering  
- Model Selection & Hyperparameter Tuning  
- K-Fold Cross Validation  
- Performance Evaluation (Classification & Regression)  

---

## 🧩 Features

### 🔹 Interactive Pipeline
- Step-by-step guided ML workflow  
- No coding required for end users  

### 🔹 Data Input
- Upload custom CSV datasets  
- Use built-in datasets (Iris, Breast Cancer, etc.)  

### 🔹 Exploratory Data Analysis (EDA)
- Histograms & distributions  
- Correlation heatmaps  
- Box plots for outlier detection  
- Target variable analysis  

### 🔹 Data Preprocessing
- Missing value handling (Mean, Median, Mode, Zero)  
- Outlier detection:
  - IQR  
  - Isolation Forest  
  - DBSCAN  
  - OPTICS  

### 🔹 Feature Selection
- Variance Threshold  
- Correlation filtering  
- Mutual Information  

### 🔹 Model Training
- Logistic Regression  
- Linear Regression  
- Ridge Regression  
- SVM (Classifier & Regressor)  
- Random Forest  
- K-Nearest Neighbors  

### 🔹 Model Evaluation
- **Classification:** Accuracy, Precision, Recall, F1-score  
- **Regression:** RMSE, MAE, R² Score  

---

## 🏗️ Project Structure
ML-Pipeline-Studio/
│── streamlit_app.py # Main application
│── requirements.txt # Dependencies
│── README.md # Documentation


---

## ⚙️ Installation

```bash
git clone https://github.com/AditiParashar19/ML-Pipeline.git
cd ml-pipeline-studio
pip install -r requirements.txt

---

## Run the Application
streamlit run streamlit_app.py

---

## Workflow

Problem Selection → Data Upload → EDA → Preprocessing → Feature Selection
→ Train/Test Split → Model Selection → Training → Evaluation → Prediction
