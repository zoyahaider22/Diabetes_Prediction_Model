
# 🩺 PIMA Diabetes Prediction Using Machine Learning

This project focuses on predicting the onset of diabetes in female Pima Indian patients using machine learning techniques. The objective is to build a robust classifier that can assist in early diagnosis of diabetes using medical diagnostic measurements.

## 📊 Dataset

- **Name:** PIMA Indians Diabetes Dataset
- **Source:** [UCI Machine Learning Repository](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- **Description:** The dataset contains diagnostic data for 768 patients, including attributes like glucose level, blood pressure, BMI, age, etc., along with a binary outcome indicating diabetes (1) or not (0).

## 🧠 Project Goals

- Handle missing and invalid values appropriately
- Address class imbalance using advanced resampling techniques
- Standardize the data for optimal model performance
- Train and evaluate multiple classifiers: SVM, Random Forest, and XGBoost
- Perform hyperparameter tuning using GridSearchCV
- Compare performance metrics to identify the best model

## 🔧 Tools & Technologies

- Python
- Pandas, NumPy
- Scikit-learn
- Imbalanced-learn (`SMOTEENN`)
- XGBoost
- Matplotlib, Seaborn

## 🧪 Data Preprocessing

- Replaced 0 values in columns like `Glucose`, `BloodPressure`, etc., with `NaN`
- Imputed missing values using **mean imputation**
- Performed **feature scaling** using `StandardScaler`
- Used **Stratified Train-Test Split** to maintain class balance

## ⚖️ Handling Class Imbalance

Applied **SMOTEENN** (Synthetic Minority Oversampling + Edited Nearest Neighbors) to balance the dataset and reduce noise.

## 🤖 Models Trained

1. **Support Vector Machine (SVM)**
   - Tuned `C`, `kernel`, and `gamma`
   - Best Accuracy: `0.7208`

2. **Random Forest Classifier**
   - Tuned `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `bootstrap`
   - Best Accuracy: `0.7468`
   - **Best overall performance**

3. **XGBoost Classifier**
   - Tuned `n_estimators`, `learning_rate`, `max_depth`, `subsample`, `colsample_bytree`
   - Best Accuracy: `0.7273`

## 📈 Evaluation Metrics

| Model        | Accuracy | Precision (1) | Recall (1) | F1-score (1) |
|--------------|----------|---------------|------------|--------------|
| **SVM**      | 0.7208   | 0.58          | 0.76       | 0.66         |
| **Random Forest** | **0.7468**   | 0.60          | **0.85**       | **0.70**         |
| **XGBoost**  | 0.7273   | **0.58**      | 0.83       | 0.68         |

📌 **Note:** Metrics are for class `1` (positive diabetes diagnosis), which is the primary focus in medical applications.

## 📊 Visualizations

- Correlation Heatmap
- Confusion Matrices for each model
- Feature Importance for Random Forest & XGBoost *(optional)*

## 🏁 Conclusion

The **Random Forest** classifier provided the best performance in terms of accuracy and F1-score, especially for the diabetic class (1). With appropriate preprocessing, resampling, and tuning, machine learning models can assist effectively in medical diagnostics.

## 🚀 Future Work

- Incorporate feature selection to improve performance
- Test other ensemble models like LightGBM or CatBoost
- Build a web-based interface for diabetes prediction
- Deploy the model using Flask or Streamlit

## 📂 Folder Structure

```bash
.
├── data/                   # Raw dataset
├── notebooks/              # Jupyter notebooks
├── models/                 # Saved models (.pkl)
├── diabetes_prediction.py  # Script version of pipeline
├── README.md
└── requirements.txt
````

## 📌 Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
```

## 👩‍💻 Author

**Zoya Haider**
M.Tech (Data Science), Jamia Hamdard University
📫 [LinkedIn](https://www.linkedin.com/in/zoya-haider-13b14b262) | 📧 [zoyahaidersusz@gmail.com](mailto:zoyahaidersusz@gmail.com)

---

⭐️ Feel free to star this repository if you found it helpful!

```


