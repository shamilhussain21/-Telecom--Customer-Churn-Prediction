# 📊 Telecom Customer Churn Prediction — Machine Learning Project

---

## 📝 Problem Statement

Customer attrition (churn) occurs when customers stop doing business with a company. High churn directly impacts revenue, making it critical for telecom companies to identify customers at risk and proactively take retention actions.

In this project, I leveraged data analytics and machine learning to build a predictive model using the **Telco Customer Churn dataset**. By analyzing customer demographics, service usage, and account details, I aimed to classify customers who are likely to churn and provide insights for targeted retention strategies.

---

## 🎯 Objectives

- Analyze customer data to understand patterns contributing to churn.
- Build a predictive model to classify customers likely to churn.
- Evaluate model performance using appropriate metrics.
- Generate actionable business insights to reduce churn rates.

---

## 🔧 Tech Stack

- **Programming Language**: Python  
- **Data Manipulation**: Pandas, NumPy  
- **Visualization**: Matplotlib, Seaborn  
- **Machine Learning**: Scikit-learn, XGBoost  
- **Imbalance Handling**: SMOTE (Synthetic Minority Oversampling Technique)  
- **Hyperparameter Tuning**: GridSearchCV, RandomizedSearchCV

---

## 🔄 Project Workflow

### 1️⃣ Data Loading & Initial Exploration
- Imported necessary libraries.
- Loaded and inspected the Telco Customer Churn dataset.

### 2️⃣ Data Preprocessing
- Handled missing values.
- Removed outliers using the **IQR method**.
- Cleaned and transformed data for modeling.
- Applied **One-Hot Encoding** for categorical variables.
- Rearranged columns for better structure.

### 3️⃣ Feature Engineering & Scaling
- Scaled numerical features for better model performance using StandardScaler.
- Applied feature selection to retain important predictors.

### 4️⃣ Handling Imbalanced Data
- Target variable was highly imbalanced.
- Applied **SMOTE** technique to balance classes and avoid biased models.

### 5️⃣ Model Development & Comparison
- Trained multiple classification algorithms:
  - Logistic Regression
  - Decision Tree Classifier
  - Random Forest Classifier
  - XGBoost Classifier
  - Support Vector Classifier (SVC)
  - K-Nearest Neighbors (KNN)
  
- Evaluated models using **F1-Score** for balanced performance comparison.

### 6️⃣ Hyperparameter Tuning
- Tuned Random Forest & XGBoost models using:
  - **GridSearchCV**
  - **RandomizedSearchCV**

### 7️⃣ Final Model Selection
- **Random Forest Classifier** delivered the best overall performance after tuning.

---

## 📊 Evaluation Metrics

- Accuracy  
- Precision  
- Recall  
- F1-Score  
- ROC-AUC Score

---

## 📈 Key Insights

- Key features influencing churn include: **Contract Type, Tenure, Monthly Charges, Internet Service Type, and Payment Method**.
- Imbalance handling with SMOTE significantly improved model generalization.
- Hyperparameter tuning helped boost model performance beyond default settings.

---

## 📌 Future Improvements

- Deploy model using Flask or FastAPI for real-time churn prediction.
- Implement feature importance visualizations.
- Explore cost-sensitive learning or advanced ensemble techniques.

---

## 🧠 Key Learnings

- Built a full end-to-end machine learning pipeline.  
- Strengthened skills in data preprocessing, class imbalance handling, model comparison, and tuning.  
- Gained business understanding of customer behavior analytics in the telecom industry.


