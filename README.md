# 📱 Smartphone Addiction Detection Using Machine Learning

[![Live App](https://img.shields.io/badge/🚀%20Live%20App-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://smartphone-addiction-detection-using-ml.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)

> A machine learning web app that predicts smartphone addiction levels (Low / Medium / High) in students based on behavioral, emotional, and academic indicators — deployed live with Streamlit.

---

## 🌐 Live Demo

🔗 **[Click here to try the app](https://smartphone-addiction-detection-using-ml.streamlit.app/)**

Enter a student's daily usage hours, sleep pattern, anxiety level, and academic performance to get an instant addiction risk prediction with confidence scores.

---

## 🔍 Problem Statement

Smartphone overuse among teenagers is a growing public health concern linked to:
- Declining academic performance
- Sleep deprivation and anxiety
- Reduced social interaction

This project builds an end-to-end ML pipeline to **classify a student's addiction risk level** — helping schools, counselors, and parents identify at-risk students early and take preventive action.

---

## 📁 Project Structure

```
smartphone-addiction-detection-using-ml/
├── app.py                        # Streamlit web application
├── train.py                      # Model training pipeline
├── requirements.txt              # Python dependencies
├── data/
│   └── phone_addiction.csv       # Dataset
├── model/
│   └── best_model.pkl            # Saved best model (sklearn Pipeline)
├── Code files/                   # Jupyter notebooks & experiments
└── Document/
    ├── PROJECT DOCUMENTATION ML.docx
    └── Teenage Phone Addiction Prediction.pptx
```

---

## 🛠 Tech Stack

| Layer | Tools |
|---|---|
| Language | Python 3.10 |
| ML Models | Logistic Regression, Random Forest |
| ML Pipeline | Scikit-learn (Pipeline, ColumnTransformer) |
| Preprocessing | StandardScaler, OneHotEncoder, SimpleImputer |
| Model Selection | StratifiedKFold Cross-Validation |
| Deployment | Streamlit Community Cloud |
| Data Handling | Pandas, NumPy |
| Model Storage | Joblib |

---

## 📊 Methodology

### 1. Data Preprocessing
- Handled missing values using median imputation (numeric) and mode imputation (categorical)
- Scaled numeric features using `StandardScaler`
- Encoded `Gender` using `OneHotEncoder`
- Created 3-class target variable from continuous `Addiction_Level` score:

| Class | Score Range | Meaning |
|---|---|---|
| Low | 0 – 4 | Healthy usage pattern |
| Medium | 4 – 7 | Moderate risk, monitor closely |
| High | 7 – 10 | High risk, intervention recommended |

### 2. Model Training
Built a full sklearn `Pipeline` for each model (preprocessing + classifier):

```
Input Features → ColumnTransformer → Classifier → Prediction
```

Evaluated using **5-Fold Stratified Cross-Validation** with F1-Weighted scoring.

### 3. Model Comparison

| Model | CV F1-Weighted Score |
|---|---|
| Logistic Regression | ~0.78 |
| **Random Forest** | **~0.91** |

**Random Forest** was selected as the best model and saved as `best_model.pkl`.

### 4. Deployment
- Saved full pipeline (preprocessing + model) as a single `.pkl` file
- Built interactive Streamlit UI with real-time prediction and confidence chart
- Deployed on Streamlit Community Cloud

---

## 🔑 Key Findings

- **Daily screen time > 6 hours** was present in over 90% of High-addiction predictions — the strongest single predictor
- **Students with less than 6 hours of sleep** were significantly more likely to fall in the High-addiction class
- **Low academic performance (below 50/100)** strongly correlated with Medium and High addiction levels
- **Random Forest outperformed Logistic Regression** by ~13% on F1-weighted score, capturing non-linear interactions between features

---

## 🎯 Input Features

| Feature | Type | Range |
|---|---|---|
| Daily Usage Hours | Numeric | 0 – 24 hrs |
| Sleep Hours | Numeric | 0 – 15 hrs |
| Anxiety Level | Numeric | 0 – 10 |
| Academic Performance | Numeric | 0 – 100 |
| Age | Numeric | 5 – 25 |
| Gender | Categorical | Male / Female / Other |

---

## 🚀 Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/KrishnaJ4F/smartphone-addiction-detection-using-ml.git
cd smartphone-addiction-detection-using-ml

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the app
streamlit run app.py
```

---

## 🔮 Future Improvements

- Add SHAP explainability to show which features drove each prediction
- Collect real survey data from students for a more robust dataset
- Build a school-level dashboard showing class-wide addiction risk distribution
- Add XGBoost and compare against Random Forest
- Send automated alert emails for High-risk predictions

---

## 👤 Author

**Krishna Kumar**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0077B5?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/krishna-kumar-41b84633b/)
[![GitHub](https://img.shields.io/badge/GitHub-KrishnaJ4F-181717?style=flat&logo=github&logoColor=white)](https://github.com/KrishnaJ4F)

---

*If you found this project useful, please consider giving it a ⭐ on GitHub!*
