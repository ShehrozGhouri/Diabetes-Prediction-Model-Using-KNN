# 🩺 Pima Indians Diabetes Prediction Using K-Nearest Neighbors (KNN)

This repository contains an end-to-end Binary Classification machine learning pipeline built to predict whether a patient has diabetes based on diagnostic and medical measurements. The project focuses heavily on rigorous data preprocessing, invalid data imputation, feature scaling, and performance evaluation using the **K-Nearest Neighbors (KNN)** algorithm.

---

## 📌 Project Overview
Predicting chronic conditions like diabetes accurately requires clean data and well-calibrated machine learning models. Because the KNN algorithm evaluates classifications based on the geometric distance between data points, standardizing features and correcting missing/corrupt data records is critical. 

This project demonstrates a production-ready approach to handling hidden missing values, scaling feature spaces, training a KNN classifier, and evaluating performance using a confusion matrix and classification metrics.

---

## 📊 Dataset Features
The dataset used is the widely recognized **Pima Indians Diabetes Dataset**. It contains 8 clinical features (independent variables) and 1 target label:

* **Pregnancies:** Number of times pregnant.
* **Glucose:** Plasma glucose concentration (2 hours in an oral glucose tolerance test).
* **BloodPressure:** Diastolic blood pressure ($mm$ $Hg$).
* **SkinThickness:** Triceps skin fold thickness ($mm$).
* **Insulin:** 2-Hour serum insulin ($mu$ $U/ml$).
* **BMI:** Body Mass Index ($weight$ in $kg / (height$ in $m)^2$).
* **DiabetesPedigreeFunction:** A genetic score assessing diabetes likelihood based on family history.
* **Age:** Patient age in years.
* **Outcome (Target Variable):** `0` for Non-Diabetic, `1` for Diabetic.

---

## 🛠️ Data Preprocessing Pipeline
To maximize the reliability of the distance calculations within the KNN model, the data underwent the following processing workflow:

1.  **Handling Implicit Null Values:** Columns like `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, and `BMI` contained invalid $0$ entries (e.g., a blood pressure of 0 is medically impossible). These records were replaced with `NaN` values and imputed using the **median** of their respective columns to maintain distribution sanity.
2.  **Feature Scaling:** Features with high ranges (like `Insulin` up to 800) would inherently dominate features with smaller ranges (like `DiabetesPedigreeFunction` under 2.5). **StandardScaler** was applied to give all features equal weight by centering them around a mean of 0 with a standard deviation of 1.
3.  **Data Partitioning:** The dataset was split into an **80% Training Set** (for fitting the model and learning feature scales) and a **20% Test Set** (held out completely for unbiased model evaluation).

---

## 📈 Model Performance & Evaluation
The model was evaluated on the unseen testing split using a baseline configuration of $K = 5$ neighbors. 

### 🔹 Summary Metrics
* **Model Accuracy:** `~72%`
* **Class 0 (Non-Diabetic) Precision:** `0.80`
* **Class 1 (Diabetic) Precision:** `0.60`
* **Class 1 (Diabetic) Recall:** `0.67`

### 🔹 Confusion Matrix
The confusion matrix highlights exactly how the model's predictions match up against actual medical outcomes:

| | Predicted Non-Diabetic (0) | Predicted Diabetic (1) |
| :--- | :---: | :---: |
| **Actual Non-Diabetic (0)** | **74** (True Negatives) | **25** (False Positives) |
| **Actual Diabetic (1)** | **18** (False Negatives) | **37** (True Positives) |

> 💡 **Key Insight:** The model demonstrates robust performance in isolating healthy individuals (with an 80% precision rate) while catching nearly two-thirds of all true positive diabetes cases in the dataset.

---

## 🚀 Technologies & Libraries Used
* **Python 3.12+**
* **Pandas & NumPy:** Advanced data cleaning, array operations, and structure tracking.
* **Matplotlib & Seaborn:** Custom matrix plotting and exploratory feature visualization.
* **Scikit-Learn:** Implementation of data splits (`train_test_split`), normalization (`StandardScaler`), model algorithms (`KNeighborsClassifier`), and validation analytics.

---

## 💻 How to Use and Run the Project

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/your-username/diabetes-knn-prediction.git](https://github.com/your-username/diabetes-knn-prediction.git)
    cd diabetes-knn-prediction
    ```

2.  **Install Required Dependencies:**
    ```bash
    pip install pandas numpy scikit-learn matplotlib seaborn
    ```

3.  **Run Predictions with New Data:**
    Always ensure new raw diagnostic samples are transformed through the trained `StandardScaler` pipeline instance before sending them to `knn.predict()` to ensure classification accuracy.
